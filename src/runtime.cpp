#include "laya/runtime.hpp"
#include "laya/precision.hpp"
#include "vulkan_precision.hpp"
#include "vulkan_rotary.hpp"
#include "vulkan_status.hpp"
#include "vulkan/gelu_rocm_patches.hpp"
#include "vulkan/gelu_tables.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if LAYA_CUDA
#include "ggml-cuda.h"
const char* laya_cuda_bf16_compatibility_error();
#endif
#if LAYA_VULKAN
#include "ggml-vulkan.h"
#endif
#if LAYA_COREML
#include "laya/coreml.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <span>
#include <unordered_map>

namespace laya {
result<serving_config> serving_config::parse(const json& document) {
    auto integer = [&](std::string_view key, int fallback) {
        const json& value = field(document, key);
        return value.is_null() ? fallback : value.is_number() ? value.get<int>() : 0;
    };
    serving_config config;
    config.variant = field(document, "model_name") == "laya-typed-decisions" ? model_variant::typed_decisions
                   : field(document, "encoder") == "jhu-clsp/mmBERT-base" ? model_variant::multilingual : model_variant::english;
    config.max_len = integer("max_len", 512);
    config.head_max_len = integer("head_max_len", 192);
    if ((config.max_len != 512 && config.max_len != 1024) || config.head_max_len < 1 || config.head_max_len >= config.max_len)
        return fail(errc::model, "Unsupported serving sequence limits");
    // Checkpoints name ({"escalate": 0.5}) or list the action costs; the act_head shape is validated with the weights.
    const json& costs = field(document, "act_costs");
    if (costs.is_null()) return fail(errc::model, "Missing act_costs");
    config.actions = int(costs.size()) + 1;
    const json &base = field(document, "temperature"), &by_options = field(document, "temperature_by_options");
    constexpr const char* buckets[] = {"2", "3-5", "6-10", "11+"};
    for (std::size_t type = 0; type < 3; ++type)
        for (std::size_t bucket = 0; bucket < 4; ++bucket) {
            json value = field(by_options, std::string(question_types[type]) + ":" + buckets[bucket]);
            if (value.is_null()) value = base.is_null() ? json(1) : base.is_array() && type < base.size() ? base[type] : json();
            if (!value.is_number()) return fail(errc::model, "Unsupported temperature configuration");
            config.temperature[type][bucket] = std::max(1e-3, value.get<double>());
        }
    return config;
}

result<checkpoint> checkpoint::load(const std::filesystem::path& directory) {
    LAYA_TRY(config, read_json(directory / "rl_agent_config.json"));
    LAYA_TRY(encoder, read_json(directory / "encoder/config.json"));
    auto integer = [&](std::string_view key) {
        const json& value = field(*encoder, key);
        return value.is_number() ? value.get<int>() : 0;
    };
    architecture arch{integer("hidden_size"), integer("num_attention_heads"), integer("num_hidden_layers"),
                      integer("intermediate_size"), integer("vocab_size"), 0};
    arch.local_rope = arch.width == multilingual_encoder.width ? multilingual_encoder.local_rope : large_encoder.local_rope;
    if (arch != large_encoder && arch != multilingual_encoder) return fail(errc::model, "Unsupported encoder architecture");
    static const json supported{{"model_type", "modernbert"}, {"local_attention", 128}, {"global_attn_every_n_layers", 3},
            {"norm_bias", false}, {"attention_bias", false}, {"mlp_bias", false}, {"hidden_activation", "gelu"}};
    for (const auto& [key, expected] : supported.items())
        if (field(*encoder, key) != expected) return fail(errc::model, "Unsupported encoder field: " + key);
    if (field(*config, "head_layers") != 2 || field(*config, "amp_dtype") != "bf16")
        return fail(errc::model, "Expected two head layers and BF16 model configuration");
    if (const json& epsilon = field(*encoder, "norm_eps"); !epsilon.is_null() && epsilon != 1e-5)
        return fail(errc::model, "Unsupported normalization epsilon");
    LAYA_TRY(serving, serving_config::parse(*config));
    const json& types = field(*encoder, "layer_types");
    for (int i = 0; i < arch.layers; ++i)
        if (!types.is_array() || i >= int(types.size()) || types[i] != (global_layer(i) ? "full_attention" : "sliding_attention"))
            return fail(errc::model, "Unsupported attention schedule");
    const json& rope = field(*encoder, "rope_parameters");
    for (auto [kind, base] : {std::pair{"full_attention", architecture::global_rope}, std::pair{"sliding_attention", arch.local_rope}})
        if (field(field(rope, kind), "rope_type") != "default" || field(field(rope, kind), "rope_theta") != double(base))
            return fail(errc::model, "Unsupported rotary configuration");
    return checkpoint{directory, std::move(*serving), arch};
}

namespace {
using uint = std::uint32_t;
#include "vulkan/bf16_range.glsl"

using tensor = ggml_tensor;
template<auto Free> struct release {
    template<class T> void operator()(T* handle) const noexcept { Free(handle); }
};
using context_handle = std::unique_ptr<ggml_context, release<ggml_free>>;
using backend_handle = std::unique_ptr<ggml_backend, release<ggml_backend_free>>;
using buffer_handle = std::unique_ptr<ggml_backend_buffer, release<ggml_backend_buffer_free>>;
using allocator_handle = std::unique_ptr<ggml_gallocr, release<ggml_gallocr_free>>;

// Backends compiled into this library: only their modes are instantiated.
constexpr mode built = feature::cpu | feature::cuda * LAYA_CUDA | feature::vulkan * LAYA_VULKAN | feature::coreml * LAYA_COREML;
// Every supported mode; vendor bits only refine Vulkan mixed precision.
constexpr auto modes = [] {
    using namespace feature;
    return std::to_array<mode>({
        cpu, cpu | flash, coreml,
        cuda, cuda | flash, cuda | compensated, cuda | compensated | flash, cuda | bf16 | flash,
        vulkan, vulkan | compensated,
        vulkan | bf16, vulkan | bf16 | flash, vulkan | fp16, vulkan | fp16 | flash,
        vulkan | bf16 | nvidia, vulkan | bf16 | flash | nvidia, vulkan | fp16 | nvidia, vulkan | fp16 | flash | nvidia,
        vulkan | bf16 | amd, vulkan | bf16 | flash | amd, vulkan | fp16 | amd, vulkan | fp16 | flash | amd,
    });
}();
static_assert(std::ranges::none_of(modes, rejection));

// The only place a runtime mode selects code: calls f.operator()<M>() for the
// mode M == m, which runtime::load admitted as a supported mode of a built backend.
template<class F> using dispatched = decltype(std::declval<F>().template operator()<feature::cpu>());
template<std::size_t I = 0, class F> dispatched<F> dispatch(mode m, F&& f) {
    if constexpr (I == modes.size()) {
        std::unreachable();
    } else {
        if constexpr ((modes[I] & built) != 0)
            if (m == modes[I]) return f.template operator()<modes[I]>();
        return dispatch<I + 1>(m, f);
    }
}

template<class Values> void put(tensor* t, const Values& values) { ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t)); }

// How a checkpoint tensor is stored on the device; the mode maps each role to a type.
enum class role : std::uint8_t {
    table,       // embeddings and normalization parameters: FP32
    bias,        // projection bias: FP32 holding product-precision values
    projection,  // decision-head matrix: product precision
    trunk,       // encoder and head-layer matrix: product precision or compensated FP16
};
struct storage {
    ggml_type trunk = GGML_TYPE_F32, projection = GGML_TYPE_F32;
    ggml_type bias = GGML_TYPE_F32;    // biases hold values rounded through this type
    bool f16_checkpoint = false;       // trunk matrices must already be F16 in the checkpoint
    bool amd_bf16_range = false;       // BF16 matrices must convert exactly in AMD Vulkan kernels
    ggml_type gelu = GGML_TYPE_COUNT;  // Vulkan mixed-precision GELU lookup values, if any
    bool rocm_gelu = false;            // apply the ROCm GELU table patches
};

// Checkpoint tensors by their safetensors names: "encoder.layers.3.attn.Wqkv.weight"
// is w.encoder.layers[3].attn.Wqkv.weight and "scorer.3.bias" is w.scorer[3].bias.
struct weights {
    struct linear { tensor *weight = nullptr, *bias = nullptr; };  // also LayerNorm parameters
    struct encoder_layer {
        linear attn_norm, mlp_norm;  // layer 0 has no attn_norm
        struct { linear Wqkv, Wo; } attn;
        struct { linear Wi, Wo; } mlp;
    };
    struct head_layer {
        struct { tensor *in_proj_weight = nullptr, *in_proj_bias = nullptr; linear out_proj; } self_attn;
        linear linear1, linear2, norm1, norm2;
    };
    struct {
        struct { linear tok_embeddings, norm; } embeddings;
        std::array<encoder_layer, 28> layers;
        linear final_norm;
    } encoder;
    linear type_emb;
    tensor* temperature = nullptr;
    struct { std::array<head_layer, 2> layers; } head;
    std::array<linear, 4> scorer;    // LayerNorm, Linear, GELU, Linear
    std::array<linear, 3> act_head;  // Linear, GELU, Linear
    tensor* gelu_table = nullptr;    // device-generated: Vulkan mixed-precision GELU lookup
};

// A checkpoint tensor, its PyTorch shape ({rows} for vectors) and its slot.
struct entry {
    std::string name;
    role kind;
    std::vector<std::int64_t> shape;
    tensor** slot;
};
// Every tensor a checkpoint of this geometry holds, in validation order.
std::vector<entry> entries(weights& w, const architecture& arch, int actions) {
    const std::int64_t W = arch.width, I = arch.intermediate;
    std::vector<entry> list;
    auto add = [&](std::string name, role kind, std::vector<std::int64_t> shape, tensor*& slot) {
        list.push_back({std::move(name), kind, std::move(shape), &slot});
    };
    // A Linear {rows, columns} or LayerNorm {rows} (no columns) and its bias {rows}.
    auto pair = [&](const std::string& name, role kind, std::int64_t rows, std::int64_t columns, weights::linear& slot) {
        add(name + ".weight", kind, columns ? std::vector{rows, columns} : std::vector{rows}, slot.weight);
        add(name + ".bias", kind == role::table ? kind : role::bias, {rows}, slot.bias);
    };
    // The member path of each tensor spells its checkpoint name.
#define LAYA_TENSOR(path, kind, ...) add(#path, role::kind, {__VA_ARGS__}, w.path)
#define LAYA_LAYER(path, kind, ...) add(prefix + "." #path, role::kind, {__VA_ARGS__}, layer.path)
#define LAYA_PAIR(path, kind, ...) pair(prefix + "." #path, role::kind, __VA_ARGS__, layer.path)
    LAYA_TENSOR(encoder.embeddings.tok_embeddings.weight, table, arch.vocabulary, W);
    LAYA_TENSOR(encoder.embeddings.norm.weight, table, W);
    LAYA_TENSOR(encoder.final_norm.weight, table, W);
    LAYA_TENSOR(type_emb.weight, table, 3, W);
    LAYA_TENSOR(temperature, table, 3);
    for (int i = 0; i < arch.layers; ++i) {
        auto& layer = w.encoder.layers[i];
        const auto prefix = "encoder.layers." + std::to_string(i);
        if (i) LAYA_LAYER(attn_norm.weight, table, W);
        LAYA_LAYER(mlp_norm.weight, table, W);
        LAYA_LAYER(attn.Wqkv.weight, trunk, 3 * W, W);
        LAYA_LAYER(attn.Wo.weight, trunk, W, W);
        LAYA_LAYER(mlp.Wi.weight, trunk, 2 * I, W);
        LAYA_LAYER(mlp.Wo.weight, trunk, W, I);
    }
    for (int i = 0; i < 2; ++i) {
        auto& layer = w.head.layers[i];
        const auto prefix = "head.layers." + std::to_string(i);
        LAYA_LAYER(self_attn.in_proj_weight, trunk, 3 * W, W);
        LAYA_LAYER(self_attn.in_proj_bias, bias, 3 * W);
        LAYA_PAIR(self_attn.out_proj, trunk, W, W);
        LAYA_PAIR(linear1, trunk, 4 * W, W);
        LAYA_PAIR(linear2, trunk, W, 4 * W);
        LAYA_PAIR(norm1, table, W, 0);
        LAYA_PAIR(norm2, table, W, 0);
    }
#undef LAYA_PAIR
#undef LAYA_LAYER
#undef LAYA_TENSOR
    pair("scorer.0", role::table, W, 0, w.scorer[0]);
    pair("scorer.1", role::projection, W, W, w.scorer[1]);
    pair("scorer.3", role::projection, 1, W, w.scorer[3]);
    pair("act_head.0", role::projection, 256, W + 4, w.act_head[0]);
    pair("act_head.2", role::projection, actions, 256, w.act_head[2]);
    return list;
}

// A graph value; Stored marks values already rounded to the storage precision
// that mixed-precision projections consume, so they are not rounded again.
template<bool Stored> struct operand { tensor* t; };
using wide = operand<false>;

// Where an attention block sits, for kernels selected by shape.
struct attention_site {
    bool head, global, padding;
    int length, batch;
    std::int64_t width;
    std::int64_t tokens() const { return std::int64_t(length) * batch; }
    // Fused mixed-precision attention names the masks its kernels must apply.
    const char* masked_kernel() const {
        if (!head && !global && length >= architecture::local_window) return "laya.sdpa-local";
        return head || padding ? "laya.sdpa-masked" : nullptr;
    }
};

tensor* fused_attention(ggml_context* ctx, tensor* q, tensor* k, tensor* v, tensor* mask, const char* kernel, const attention_site& at) {
    auto* value = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / 8.0f, 0, 0);
    if (kernel) ggml_set_name(value, kernel);
    ggml_prec_set_acc(value, GGML_PREC_F32);
    return ggml_reshape_2d(ctx, ggml_is_contiguous(value) ? value : ggml_cont(ctx, value), at.width, at.tokens());
}

// Scores and probabilities as separate products. AMD-matched numerics scale
// both factors by 8^-1/2 before the product, as the ROCm reference does.
template<bool Amd>
tensor* explicit_attention(ggml_context* ctx, tensor* q, tensor* k, tensor* v, tensor* mask, const attention_site& at) {
    if constexpr (Amd) q = ggml_scale(ctx, q, std::sqrt(1.0f / 8.0f)), k = ggml_scale(ctx, k, std::sqrt(1.0f / 8.0f));
    auto* scores = ggml_mul_mat(ctx, k, q);
    ggml_prec_set_acc(scores, GGML_PREC_F32);
    auto* probabilities = ggml_soft_max_ext(ctx, scores, mask, Amd ? 1.0f : 1.0f / 8.0f, 0);
    auto* value = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), probabilities);
    ggml_prec_set_acc(value, GGML_PREC_F32);
    if constexpr (Amd) {
        ggml_set_name(scores, "laya.amd-low-qk");
        ggml_set_name(probabilities, "laya.amd-low-softmax");
        ggml_set_name(value, "laya.amd-low-pv");
    }
    return ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, value, 0, 2, 1, 3)), at.width, at.tokens());
}

// Numerics policies: how each mode rounds, projects, normalizes and attends.
template<mode M> struct numerics;

// Strict FP32: products accumulate in FP32 from FP32 operands. Compensated
// modes store trunk matrices in FP16 and split each FP32 operand into a high
// and a scaled low FP16 half.
template<mode M> requires(!(M & (feature::low | feature::coreml)))
struct numerics<M> {
    static constexpr bool vulkan = has(M, feature::vulkan), compensated = has(M, feature::compensated);
    static constexpr storage layout{.trunk = compensated ? GGML_TYPE_F16 : GGML_TYPE_F32, .f16_checkpoint = compensated};
    static constexpr bool mixed = false, fused_residual = false, transposed_head = false;
    static constexpr bool split_mlp = compensated && !vulkan;  // CUDA fuses the compensated MLP

    template<bool Compact> static wide norm(ggml_context* ctx, tensor* x, const weights::linear& affine) {
        auto* value = ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), affine.weight);
        return {affine.bias ? ggml_add(ctx, value, affine.bias) : value};
    }
    template<role Kind, bool Compact = false, bool Stored>
    static wide linear(ggml_context* ctx, operand<Stored> input, tensor* weight, tensor* bias, tensor* = nullptr) {
        constexpr bool split = compensated && Kind == role::trunk;
        auto* x = input.t;
        const std::int64_t columns = x->ne[1];
        // ggml's small-matrix CUDA kernel uses TF32 even for F32 weights; padding
        // keeps tiny strict products, including the decision heads, on cuBLAS.
        const bool pad = !vulkan && !split && columns <= 16;
        if (pad) x = ggml_pad(ctx, x, 0, int(17 - columns), 0, 0);
        tensor* value;
        if constexpr (split && vulkan) {
            value = ggml_mul_mat(ctx, weight, vulkan_precision::split_half(ctx, x));
            ggml_prec_set_acc(value, GGML_PREC_F32);
            value = vulkan_precision::merge_half(ctx, value);
        } else if constexpr (split) {
            value = merge_f16(ctx, ggml_mul_mat(ctx, weight, split_f16(ctx, x)));
        } else {
            value = ggml_mul_mat(ctx, weight, x);
            ggml_prec_set_acc(value, GGML_PREC_F32);
        }
        if (bias) value = ggml_add(ctx, value, bias);
        if (pad) value = ggml_cont(ctx, ggml_view_2d(ctx, value, value->ne[0], columns, value->nb[1], 0));
        return {value};
    }
    static wide gelu(ggml_context* ctx, tensor*, wide x) { return {ggml_gelu_erf(ctx, x.t)}; }
    // GELU of the first half of each row times its second half.
    template<bool Stored> static wide gated_gelu(ggml_context* ctx, tensor*, operand<Stored> gated) {
        auto* x = gated.t;
        if constexpr (vulkan) return {ggml_geglu_erf(ctx, x)};
        const std::int64_t width = x->ne[0] / 2;
        auto* first = ggml_cont(ctx, ggml_view_2d(ctx, x, width, x->ne[1], x->nb[1], 0));
        auto* second = ggml_cont(ctx, ggml_view_2d(ctx, x, width, x->ne[1], x->nb[1], width * sizeof(float)));
        return {ggml_mul(ctx, ggml_gelu_erf(ctx, first), second)};
    }
    static tensor* round(ggml_context*, tensor* x) { return x; }
    static tensor* pack(ggml_context* ctx, tensor* qkv, tensor* cosine, tensor* sine, int length, int batch) {
        if constexpr (vulkan) return vulkan_precision::pack_qkv(ctx, qkv, cosine, sine, length, batch, GGML_TYPE_F32);
        else return pack_qkv(ctx, qkv, cosine, sine, length, batch, false);
    }
    static tensor* query(ggml_context*, tensor* q) { return q; }
    static tensor* attend(ggml_context* ctx, tensor* q, tensor* k, tensor* v, tensor* mask, const attention_site& at) {
        // Fused FP32 attention is exact only while its tiles cover the sequence.
        if (has(M, feature::flash) && at.length <= 128) return fused_attention(ctx, q, k, v, mask, nullptr, at);
        return explicit_attention<false>(ctx, q, k, v, mask, at);
    }
    static std::pair<float, float> rotary(int, float inverse, int position, int) {
        const float angle = float(position) * inverse;
        return {std::cos(angle), std::sin(angle)};
    }
};

// Mixed BF16 on CUDA: custom kernels reproduce PyTorch autocast rounding.
// Compact kernels store BF16 outputs that projections consume directly.
template<mode M> requires(has(M, feature::cuda | feature::bf16))
struct numerics<M> {
    static constexpr ggml_type low = GGML_TYPE_BF16;
    static constexpr storage layout{.trunk = low, .projection = low, .bias = low};
    static constexpr bool mixed = true, fused_residual = true, transposed_head = false, split_mlp = false;

    template<bool Compact> static operand<Compact> norm(ggml_context* ctx, tensor* x, const weights::linear& affine) {
        return {norm_bf16(ctx, x, affine.weight, affine.bias, Compact)};
    }
    template<role, bool Compact = false, bool Stored>
    static operand<Compact> linear(ggml_context* ctx, operand<Stored> input, tensor* weight, tensor* bias, tensor* residual = nullptr) {
        auto* x = input.t;
        if (!Stored && x->type != low) x = ggml_cast(ctx, x, low);
        return {linear_bf16(ctx, x, weight, bias, Compact, residual)};
    }
    static wide gelu(ggml_context* ctx, tensor*, wide x) { return {gelu_bf16(ctx, x.t)}; }
    template<bool Stored> static operand<true> gated_gelu(ggml_context* ctx, tensor*, operand<Stored> gated) { return {mlp_bf16(ctx, gated.t)}; }
    static tensor* round(ggml_context* ctx, tensor* x) { return ggml_cast(ctx, ggml_cast(ctx, x, low), GGML_TYPE_F32); }
    static tensor* pack(ggml_context* ctx, tensor* qkv, tensor* cosine, tensor* sine, int length, int batch) {
        return pack_qkv(ctx, qkv, cosine, sine, length, batch, true);
    }
    static tensor* query(ggml_context* ctx, tensor* q) { return ggml_cast(ctx, q, GGML_TYPE_F32); }
    static tensor* attend(ggml_context* ctx, tensor* q, tensor* k, tensor* v, tensor* mask, const attention_site& at) {
        return fused_attention(ctx, q, k, v, mask, at.masked_kernel(), at);
    }
    // The CUDA rotary kernel evaluates the cosine itself from the raw angle.
    static std::pair<float, float> rotary(int, float inverse, int position, int) {
        const float angle = float(position) * inverse;
        return {angle, std::sin(angle)};
    }
};

// Mixed FP16/BF16 on Vulkan: portable kernels keep the storage rounding points
// of autocast. Vendor bits select the device whose reference numerics are matched.
template<mode M> requires(has(M, feature::vulkan) && (M & feature::low) != 0)
struct numerics<M> {
    static constexpr ggml_type low = has(M, feature::fp16) ? GGML_TYPE_F16 : GGML_TYPE_BF16;
    static constexpr bool bf16 = low == GGML_TYPE_BF16, amd = has(M, feature::amd), nvidia = has(M, feature::nvidia);
    static constexpr storage layout{.trunk = low, .projection = low, .bias = low, .amd_bf16_range = amd && bf16,
                                    .gelu = low, .rocm_gelu = amd};
    static constexpr bool mixed = true, fused_residual = true, split_mlp = false;
    static constexpr bool transposed_head = amd;  // AMD batched heads project in a sequence-major layout

    template<bool Compact> static operand<Compact> norm(ggml_context* ctx, tensor* x, const weights::linear& affine) {
        return {vulkan_precision::norm(ctx, x, affine.weight, affine.bias, Compact ? low : GGML_TYPE_F32)};
    }
    template<role, bool Compact = false, bool Stored>
    static wide linear(ggml_context* ctx, operand<Stored> input, tensor* weight, tensor* bias, tensor* residual = nullptr) {
        auto* x = input.t;
        // Round in the F32 layout the matrix kernels read, without
        // materializing an intermediate 16-bit tensor.
        if (!Stored && x->type != low)
            x = x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) ? vulkan_precision::finish_projection(ctx, x, nullptr, nullptr, low)
                                                                  : ggml_cast(ctx, x, low);
        vulkan_precision::projection_plan plan{};
        if constexpr (nvidia) {
            plan = vulkan_precision::select_projection_plan(weight->ne[0], weight->ne[1], x->ne[1], bias != nullptr);
            // Bound cancellation in batched scalar heads with short FP32 dot
            // products before the final rounding.
            if (bf16 && weight->ne[1] == 1 && x->ne[1] > 1 && !plan.chunk) plan.chunk = 64;
        }
        return {vulkan_precision::linear(ctx, x, weight, bias, residual, low, plan, false, amd)};
    }
    static wide gelu(ggml_context* ctx, tensor* table, wide x) { return {vulkan_precision::activation(ctx, x.t, table, false, bf16)}; }
    template<bool Stored> static wide gated_gelu(ggml_context* ctx, tensor* table, operand<Stored> gated) {
        return {vulkan_precision::activation(ctx, gated.t, table, true, bf16)};
    }
    static tensor* round(ggml_context* ctx, tensor* x) { return vulkan_precision::round(ctx, x, low); }
    static tensor* pack(ggml_context* ctx, tensor* qkv, tensor* cosine, tensor* sine, int length, int batch) {
        return vulkan_precision::pack_qkv(ctx, qkv, cosine, sine, length, batch, low);
    }
    static tensor* query(ggml_context* ctx, tensor* q) { return ggml_cast(ctx, q, GGML_TYPE_F32); }
    static tensor* attend(ggml_context* ctx, tensor* q, tensor* k, tensor* v, tensor* mask, const attention_site& at) {
        if constexpr (has(M, feature::flash) && !amd) {
            // Unpadded global (or fully windowed) attention needs no mask.
            if (!at.head && !at.padding && (at.global || at.length < architecture::local_window)) mask = nullptr;
            k = ggml_cast(ctx, k, low);
            v = ggml_cast(ctx, v, low);
            return fused_attention(ctx, q, k, v, mask, mask ? at.masked_kernel() : "laya.sdpa-flash", at);
        } else {
            return explicit_attention<amd>(ctx, q, k, v, mask, at);
        }
    }
    // Rotary tables reproduce the rounded values of the matched device library.
    static std::pair<float, float> rotary(int base, float, int position, int i) {
        return {vulkan_precision::rotary(amd, base, position, i, false), vulkan_precision::rotary(amd, base, position, i, true)};
    }
};

// Graph shapes are fixed per compiled graph; mixed precision also depends on
// whether any row is shorter than the batch length.
struct graph_shape {
    int batch = 0, length = 0, options = 0;
    bool padding = false;
    friend bool operator==(const graph_shape&, const graph_shape&) = default;
};
using trace_list = std::vector<std::pair<std::string, tensor*>>;
struct compiled_graph {
    context_handle context;
    allocator_handle allocator;
    ggml_cgraph* graph = nullptr;
};
struct encoder_graph : compiled_graph {
    graph_shape shape;
    tensor *ids = nullptr, *types = nullptr, *markers = nullptr, *cls = nullptr, *lengths = nullptr;
    tensor *global_mask = nullptr, *local_mask = nullptr, *pooled = nullptr, *logits = nullptr;
    std::array<tensor*, 2> cosine{}, sine{};  // global and local rotary tables
    trace_list traces;
};
struct action_graph : compiled_graph {
    int batch = 0;
    tensor *input = nullptr, *output = nullptr;
};

result<void> begin(compiled_graph& g) {
    g.context.reset(ggml_init({ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false), nullptr, true}));
    if (!g.context) return fail(errc::backend, "Cannot allocate graph metadata");
    g.graph = ggml_new_graph_custom(g.context.get(), 8192, false);
    return {};
}
// Checks backend support and allocates every intermediate of a built graph.
result<void> allocate(compiled_graph& g, ggml_backend_t backend) {
    for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i)
        if (const auto* node = ggml_graph_node(g.graph, i); !ggml_backend_supports_op(backend, node))
            return fail(errc::unsupported, std::string("Requested backend does not support ") + ggml_op_name(node->op));
    g.allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    if (!g.allocator || !ggml_gallocr_alloc_graph(g.allocator.get(), g.graph)) return fail(errc::backend, "Insufficient memory for this batch");
    return {};
}
// Keeps a graph value for LAYA_TRACE_DIR, named scope + layer + suffix.
void trace(trace_list* traces, tensor* value, std::string_view scope, int layer = -1, std::string_view suffix = {}) {
    if (!traces) return;
    ggml_set_output(value);
    traces->emplace_back(std::string(scope).append(layer < 0 ? "" : std::to_string(layer)).append(suffix), value);
}

// The Laya topology: ModernBERT encoder, two decision-head layers, option
// scorer and action head. numerics<M> supplies the arithmetic.
template<mode M> struct network {
    using N = numerics<M>;
    static constexpr bool fused = N::fused_residual;  // projections add their residual
    ggml_context* ctx;
    const weights& w;
    const architecture& arch;
    trace_list* traces;  // null unless tracing

    static result<encoder_graph> encoder(ggml_backend_t backend, const weights& w, const architecture& arch, graph_shape shape, bool trace) {
        encoder_graph g;
        g.shape = shape;
        LAYA_CHECK(begin(g));
        network{g.context.get(), w, arch, trace ? &g.traces : nullptr}.build(g);
        LAYA_CHECK(allocate(g, backend));
        // Rotary tables depend on the shape only; as outputs they stay allocated,
        // so each graph uploads them once.
        std::vector<float> cosine(64 * std::size_t(shape.length)), sine(cosine.size());
        for (int kind = 0; kind < 2; ++kind) {
            const int base = kind == 0 || arch.local_rope == architecture::global_rope ? 0 : 1;
            for (int i = 0; i < 32; ++i) {
                const float inverse = 1.0f / std::pow(kind == 0 ? architecture::global_rope : arch.local_rope, float(2 * i) / 64.0f);
                for (int position = 0; position < shape.length; ++position) {
                    const auto [c, s] = N::rotary(base, inverse, position, i);
                    cosine[position * 64 + i] = cosine[position * 64 + i + 32] = c;
                    sine[position * 64 + i] = sine[position * 64 + i + 32] = s;
                }
            }
            put(g.cosine[kind], cosine);
            put(g.sine[kind], sine);
        }
        return g;
    }

    // The action head over pooled features and option statistics.
    static result<action_graph> actions(ggml_backend_t backend, const weights& w, const architecture& arch, int batch) {
        action_graph g;
        g.batch = batch;
        LAYA_CHECK(begin(g));
        network n{g.context.get(), w, arch, nullptr};
        g.input = n.input(GGML_TYPE_F32, {arch.width + 4, batch});
        const auto& head = w.act_head;
        auto hidden = N::gelu(n.ctx, w.gelu_table, N::template linear<role::projection>(n.ctx, wide{g.input}, head[0].weight, head[0].bias));
        g.output = N::template linear<role::projection>(n.ctx, hidden, head[2].weight, head[2].bias).t;
        ggml_set_output(g.output);
        ggml_build_forward_expand(g.graph, g.output);
        LAYA_CHECK(allocate(g, backend));
        return g;
    }

    tensor* input(ggml_type type, std::initializer_list<std::int64_t> dimensions) {
        auto* t = ggml_new_tensor(ctx, type, int(dimensions.size()), dimensions.begin());
        ggml_set_input(t);
        return t;
    }

    struct projections { tensor *qkv, *qkv_bias, *out, *out_bias; };
    template<bool Head, bool Stored>
    tensor* attention(operand<Stored> x, const projections& p, const attention_site& at, const char* scope, int layer, tensor* h,
                      const encoder_graph& g) {
        trace(traces, x.t, scope, layer, ".qkv-input");
        // Batched head inputs have a transposed sequence/batch layout in the
        // mixed-precision contract: the product is rounded before this bias.
        const bool separate_bias = N::mixed && Head && at.batch > 1;
        tensor* qkv;
        if (N::transposed_head && Head && at.batch > 1) {
            auto* bx = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, x.t, at.width, at.length, at.batch), 0, 2, 1, 3));
            auto* by = N::template linear<role::trunk>(ctx, wide{bx}, p.qkv, nullptr).t;
            qkv = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, by, 0, 2, 1, 3)), 3 * at.width, at.tokens());
        } else {
            qkv = N::template linear<role::trunk, N::mixed && !Head>(ctx, x, p.qkv, Head && !separate_bias ? p.qkv_bias : nullptr).t;
        }
        if (separate_bias) qkv = N::round(ctx, ggml_add(ctx, qkv, p.qkv_bias));
        trace(traces, qkv, scope, layer, ".attn.Wqkv");
        const int kind = at.global ? 0 : 1;
        auto* packed = N::pack(ctx, qkv, Head ? nullptr : g.cosine[kind], Head ? nullptr : g.sine[kind], at.length, at.batch);
        tensor* split[3];
        for (int i = 0; i < 3; ++i)
            split[i] = ggml_view_4d(ctx, packed, 64, at.length, arch.heads, at.batch, packed->nb[1], packed->nb[2], packed->nb[3],
                                    i * at.batch * packed->nb[3]);
        auto *q = N::query(ctx, split[0]), *k = split[1];
        if (traces) {  // traced views are made contiguous
            trace(traces, q = ggml_cont(ctx, q), scope, layer, ".q");
            trace(traces, k = ggml_cont(ctx, k), scope, layer, ".k");
        }
        auto* value = N::attend(ctx, q, k, split[2], Head || at.global ? g.global_mask : g.local_mask, at);
        trace(traces, value, scope, layer, ".attn.Wo.input");
        auto* output = N::template linear<role::trunk>(ctx, wide{value}, p.out, Head ? p.out_bias : nullptr, fused ? h : nullptr).t;
        trace(traces, output, scope, layer, fused ? ".attn.residual" : ".attn.Wo");
        return output;
    }

    void build(encoder_graph& g) {
        const auto [batch, length, options, padding] = g.shape;
        const std::int64_t tokens = std::int64_t(length) * batch;
        g.ids = input(GGML_TYPE_I32, {tokens});
        g.types = input(GGML_TYPE_I32, {tokens});
        for (int kind = 0; kind < 2; ++kind) {
            ggml_set_output(g.cosine[kind] = input(GGML_TYPE_F32, {64, 1, length, 1}));
            ggml_set_output(g.sine[kind] = input(GGML_TYPE_F32, {64, 1, length, 1}));
        }
        g.markers = input(GGML_TYPE_I32, {std::int64_t(options) * batch});
        g.cls = input(GGML_TYPE_I32, {batch});
        // Fused attention reads mask query rows in multiples of 64.
        constexpr bool flash = has(M, feature::flash);
        const int rows = flash ? (length + 63) / 64 * 64 : length;
        if constexpr (has(M, feature::vulkan)) {
            g.global_mask = input(flash ? GGML_TYPE_F16 : GGML_TYPE_F32, {length, rows, 1, batch});
            g.local_mask = input(flash ? GGML_TYPE_F16 : GGML_TYPE_F32, {length, rows, 1, batch});
        } else {
            g.lengths = input(GGML_TYPE_I32, {batch});
            g.global_mask = attention_mask(ctx, g.lengths, length, rows, false, flash);
            g.local_mask = attention_mask(ctx, g.lengths, length, rows, true, flash);
        }

        const auto& e = w.encoder;
        auto* h = N::template norm<false>(ctx, ggml_get_rows(ctx, e.embeddings.tok_embeddings.weight, g.ids), e.embeddings.norm).t;
        trace(traces, h, "embedding");
        for (int layer = 0; layer < arch.layers; ++layer) {
            const auto& p = e.layers[layer];
            const attention_site at{false, global_layer(layer), padding, length, batch, arch.width};
            const projections attn{p.attn.Wqkv.weight, nullptr, p.attn.Wo.weight, nullptr};
            auto* attended = layer == 0 ? attention<false>(wide{h}, attn, at, "encoder.layers.", layer, h, g)
                                        : attention<false>(N::template norm<true>(ctx, h, p.attn_norm), attn, at, "encoder.layers.", layer, h, g);
            h = fused ? attended : ggml_add(ctx, h, attended);
            if constexpr (N::split_mlp) {
                auto* products = ggml_mul_mat(ctx, p.mlp.Wi.weight, split_f16(ctx, N::template norm<true>(ctx, h, p.mlp_norm).t));
                h = ggml_add(ctx, h, merge_f16(ctx, ggml_mul_mat(ctx, p.mlp.Wo.weight, mlp_split_f16(ctx, products))));
            } else {
                auto normalized = N::template norm<true>(ctx, h, p.mlp_norm);
                trace(traces, normalized.t, "encoder.layers.", layer, ".mlp.Wi.input");
                auto gated = N::template linear<role::trunk, true>(ctx, normalized, p.mlp.Wi.weight, nullptr);
                trace(traces, gated.t, "encoder.layers.", layer, ".mlp.Wi");
                auto activated = N::gated_gelu(ctx, w.gelu_table, gated);
                trace(traces, activated.t, "encoder.layers.", layer, ".mlp.Wo.input");
                auto* projected = N::template linear<role::trunk>(ctx, activated, p.mlp.Wo.weight, nullptr, fused ? h : nullptr).t;
                trace(traces, projected, "encoder.layers.", layer, fused ? ".mlp.residual" : ".mlp.Wo");
                h = fused ? projected : ggml_add(ctx, h, projected);
            }
            trace(traces, h, "encoder-", layer);
        }
        h = N::template norm<false>(ctx, h, e.final_norm).t;
        trace(traces, h, "final-norm");
        h = ggml_add(ctx, h, ggml_get_rows(ctx, w.type_emb.weight, g.types));
        for (int layer = 0; layer < 2; ++layer) {
            const auto& p = w.head.layers[layer];
            const attention_site at{true, global_layer(layer), padding, length, batch, arch.width};
            const projections attn{p.self_attn.in_proj_weight, p.self_attn.in_proj_bias, p.self_attn.out_proj.weight, p.self_attn.out_proj.bias};
            auto* attended = attention<true>(N::template norm<true>(ctx, h, p.norm1), attn, at, "head.layers.", layer, h, g);
            h = fused ? attended : ggml_add(ctx, h, attended);
            auto* first = N::template linear<role::trunk>(ctx, N::template norm<true>(ctx, h, p.norm2), p.linear1.weight, p.linear1.bias).t;
            trace(traces, first, "head.layers.", layer, ".linear1");
            auto* second = N::template linear<role::trunk>(ctx, wide{ggml_relu(ctx, first)}, p.linear2.weight, p.linear2.bias, fused ? h : nullptr).t;
            trace(traces, second, "head.layers.", layer, fused ? ".linear2-residual" : ".linear2");
            h = fused ? second : ggml_add(ctx, h, second);
            trace(traces, h, "head-", layer);
        }
        g.pooled = ggml_get_rows(ctx, h, g.cls);
        auto selected = N::template norm<true>(ctx, ggml_get_rows(ctx, h, g.markers), w.scorer[0]);
        auto hidden = N::gelu(ctx, w.gelu_table, N::template linear<role::projection>(ctx, selected, w.scorer[1].weight, w.scorer[1].bias));
        g.logits = N::template linear<role::projection>(ctx, hidden, w.scorer[3].weight, w.scorer[3].bias).t;
        ggml_set_output(g.pooled);
        ggml_set_output(g.logits);
        ggml_build_forward_expand(g.graph, g.pooled);
        ggml_build_forward_expand(g.graph, g.logits);
    }
};
}

// Model execution in one mode: the device, its resident weights and the graphs
// compiled for the most recent batch shape.
struct runtime::impl {
    checkpoint model;
    mode m;
    std::filesystem::path trace_directory;  // empty unless LAYA_TRACE_DIR is set
    std::string name, description;
    backend_handle backend;
    context_handle context;
    buffer_handle buffer;
    weights w;
#if LAYA_COREML
    std::optional<coreml_runtime> coreml;
#endif
    std::optional<encoder_graph> encoder;
    std::optional<action_graph> head;
    // Host staging reused across calls.
    std::vector<std::int32_t> types, cls;
    std::vector<float> pooled, features, probabilities, global_mask, local_mask;
    std::vector<ggml_fp16_t> global_half, local_half;

    // Initializes the device; Vulkan mixed precision adds the vendor bits of the
    // reference numerics it matches.
    result<void> open() {
        using namespace feature;
        if (has(m, coreml)) {
            name = "coreml", description = "Core ML (all compute units)";
            return {};
        }
        if (has(m, cuda)) {
#if LAYA_CUDA
            // cuBLAS enables TF32 by default; strict FP32 withdraws that permission before CUDA initializes.
#ifdef _WIN32
            if (!(m & low) && _putenv_s("NVIDIA_TF32_OVERRIDE", "0") != 0)
#else
            if (!(m & low) && setenv("NVIDIA_TF32_OVERRIDE", "0", 1) != 0)
#endif
                return fail(errc::backend, "Cannot enforce FP32 CUDA arithmetic");
            backend.reset(ggml_backend_cuda_init(0));
            if (backend && (m & low))
                if (const char* reason = laya_cuda_bf16_compatibility_error()) return fail(errc::unsupported, reason);
#else
            return fail(errc::unsupported, "This build has no CUDA backend");
#endif
        }
#if LAYA_VULKAN
        if (has(m, vulkan)) backend.reset(ggml_backend_vk_init(0));
#else
        if (has(m, vulkan)) return fail(errc::unsupported, "This build has no Vulkan backend");
#endif
        if (has(m, cpu)) backend.reset(ggml_backend_cpu_init());
        if (!backend) return fail(errc::backend, "Cannot initialize requested backend");
        name = ggml_backend_name(backend.get());
        description = ggml_backend_dev_description(ggml_backend_get_device(backend.get()));
        if (has(m, vulkan) && (m & low)) {
            if (description.contains("AMD")) m |= amd;
            else if (description.contains("NVIDIA")) m |= nvidia;
        }
        return {};
    }

#if LAYA_COREML
    // Core ML runs compiled model buckets; the exported model owns precision and accelerator decisions.
    template<mode M> requires(has(M, feature::coreml)) result<void> load() {
        LAYA_TRY(bridge, coreml_runtime::load(model));
        coreml.emplace(std::move(*bridge));
        return {};
    }
    template<mode M> requires(has(M, feature::coreml)) result<raw_result> run(const batch& input) { return coreml->forward(input); }
#endif
    template<mode M> result<void> load() { return load_weights(numerics<M>::layout); }

    result<void> load_weights(const storage& layout) {
        std::ifstream file(model.directory / "model.safetensors", std::ios::binary | std::ios::ate);
        if (!file) return fail(errc::io, "Cannot open model.safetensors");
        const auto file_size = std::uint64_t(file.tellg());
        file.seekg(0);
        std::uint64_t header_size = 0;
        file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
        if (!file || header_size > 16 * 1024 * 1024 || header_size + 8 > file_size) return fail(errc::model, "Invalid safetensors header size");
        std::string text(header_size, '\0');
        file.read(text.data(), std::streamsize(text.size()));
        auto header = parse_json(text);
        if (!header) return fail(errc::model, std::move(header.error().message));

        const auto list = entries(w, model.arch, model.serving.actions);
        std::unordered_map<std::string_view, const entry*> schema;
        for (const auto& e : list) {
            schema.emplace(e.name, &e);
            if (field(field(*header, e.name), "shape") != json(e.shape))
                return fail(errc::model, "Missing or incorrectly shaped checkpoint tensor: " + e.name);
        }
        for (const auto& [key, spec] : header->items())
            if (key != "__metadata__" && !schema.contains(key)) return fail(errc::model, "Unexpected checkpoint tensor: " + key);

        context.reset(ggml_init({(list.size() + 1) * ggml_tensor_overhead() + 1024, nullptr, true}));
        if (!context) return fail(errc::backend, "Cannot allocate weight metadata");
        // Tensors are created in checkpoint order.
        for (const auto& [key, spec] : header->items()) {
            if (key == "__metadata__") continue;
            const auto& e = *schema.find(key)->second;
            if (layout.f16_checkpoint && e.kind == role::trunk && field(spec, "dtype") != "F16")
                return fail(errc::model, "Compensated Tensor Core mode requires F16 stored projections");
            const std::vector<std::int64_t> shape(e.shape.rbegin(), e.shape.rend());
            const ggml_type type = e.kind == role::trunk ? layout.trunk : e.kind == role::projection ? layout.projection : GGML_TYPE_F32;
            *e.slot = ggml_new_tensor(context.get(), type, int(shape.size()), shape.data());
            ggml_set_name(*e.slot, key.c_str());
        }
        if (layout.gelu != GGML_TYPE_COUNT) w.gelu_table = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 65536);
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend.get()));
        if (!buffer) return fail(errc::backend, "Insufficient device memory for model weights");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // Payloads are validated and uploaded in name order.
        std::vector<const entry*> order;
        for (const auto& e : list) order.push_back(&e);
        std::ranges::sort(order, {}, &entry::name);
        const std::uint64_t payload_size = file_size - 8 - header_size;
        std::vector<char> bytes;
        std::vector<float> values;
        for (const auto* e : order) {
            auto* t = *e->slot;
            const auto& spec = field(*header, e->name);
            const auto &offsets = field(spec, "data_offsets"), &dtype = field(spec, "dtype");
            const auto count = std::size_t(ggml_nelements(t));
            const std::size_t stride = dtype == "F16" || dtype == "BF16" ? 2 : dtype == "F32" ? 4 : 0;
            const bool valid = offsets.is_array() && offsets.size() == 2 && offsets[0].is_number_unsigned() && offsets[1].is_number_unsigned();
            const auto begin = valid ? offsets[0].get<std::uint64_t>() : 0, end = valid ? offsets[1].get<std::uint64_t>() : 0;
            if (!valid || !stride || end < begin || end > payload_size || end - begin != std::uint64_t(count) * stride)
                return fail(errc::model, "Invalid safetensors payload: " + e->name);
            bytes.resize(count * stride);
            file.seekg(std::streamoff(8 + header_size + begin));
            file.read(bytes.data(), std::streamsize(bytes.size()));
            if (!file) return fail(errc::model, "Truncated tensor payload: " + e->name);
            values.resize(count);
            if (dtype == "F16") ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(bytes.data()), values.data(), std::int64_t(count));
            else if (dtype == "BF16") ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t*>(bytes.data()), values.data(), std::int64_t(count));
            else std::memcpy(values.data(), bytes.data(), bytes.size());
            if (!std::ranges::all_of(values, [](float x) { return std::isfinite(x); }))
                return fail(errc::model, "Nonfinite checkpoint values: " + e->name);
            if (t->type == GGML_TYPE_BF16) {
                std::vector<ggml_bf16_t> converted(count);
                ggml_fp32_to_bf16_row_ref(values.data(), converted.data(), std::int64_t(count));
                if (layout.amd_bf16_range && !std::ranges::all_of(converted, [](ggml_bf16_t x) { return layaBf16ScaledFitsHalf(x.bits, 0); }))
                    return fail(errc::model, "AMD Vulkan BF16 projection weights exceed exact conversion range: " + e->name);
                put(t, converted);
            } else if (t->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> converted(count);
                ggml_fp32_to_fp16_row(values.data(), converted.data(), std::int64_t(count));
                put(t, converted);
            } else {
                if (e->kind == role::bias && layout.bias == GGML_TYPE_F16)
                    for (auto& x : values) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
                if (e->kind == role::bias && layout.bias == GGML_TYPE_BF16)
                    for (auto& x : values) x = ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
                put(t, values);
            }
        }
        if (auto* table = w.gelu_table) {
            using namespace vulkan_precision;
            const bool half = layout.gelu == GGML_TYPE_F16;
            const auto widen = [half](std::uint16_t bits) { return half ? ggml_fp16_to_fp32(ggml_fp16_t(bits)) : ggml_bf16_to_fp32(ggml_bf16_t{bits}); };
            values.resize(65536);
            std::ranges::transform(half ? std::span(gelu_fp16_nvidia) : std::span(gelu_bf16_nvidia), values.begin(), widen);
            if (layout.rocm_gelu)
                for (const auto patch : half ? std::span<const gelu_patch>(gelu_fp16_rocm) : std::span<const gelu_patch>(gelu_bf16_rocm))
                    values[patch.index] = widen(patch.value);
            put(table, values);
        }
        return {};
    }

    result<void> validate(const batch& input) const {
        if (input.size < 1 || input.length < 1 || input.length > model.serving.max_len || input.options < 2 ||
            input.ids.size() != std::size_t(input.size) * std::size_t(input.length))
            return fail(errc::backend, "Invalid model batch");
        const auto rows = std::size_t(input.size);
        if (input.lengths.size() != rows || input.types.size() != rows || input.counts.size() != rows ||
            input.markers.size() != rows * std::size_t(input.options))
            return fail(errc::backend, "Invalid batch metadata sizes");
        for (int row = 0; row < input.size; ++row) {
            if (input.lengths[row] < 1 || input.lengths[row] > input.length || input.types[row] < 0 || input.types[row] > 2 ||
                input.counts[row] < 2 || input.counts[row] > input.options)
                return fail(errc::backend, "Invalid batch row metadata");
            for (int k = 0; k < input.options; ++k) {
                const auto marker = std::int64_t(input.markers[std::size_t(row) * input.options + k]) - std::int64_t(row) * input.length;
                if (marker < 0 || marker >= input.lengths[row]) return fail(errc::backend, "Option marker outside its sequence");
            }
        }
        if (!std::ranges::all_of(input.ids, [&](int id) { return id >= 0 && id < model.arch.vocabulary; }))
            return fail(errc::backend, "Token ID outside the vocabulary");
        return {};
    }

    void bind(const batch& input) {
        put(encoder->ids, input.ids);
        put(encoder->markers, input.markers);
        types.resize(input.ids.size());
        cls.resize(std::size_t(input.size));
        for (int row = 0; row < input.size; ++row) {
            cls[row] = row * input.length;
            std::fill_n(types.begin() + cls[row], input.length, input.types[row]);
        }
        put(encoder->types, types);
        put(encoder->cls, cls);
    }

    // Vulkan reads host attention masks: 0 where a query may attend a key and
    // -inf elsewhere, including the padded query rows of fused attention.
    template<class T> void upload_masks(const batch& input, std::vector<T>& global, std::vector<T>& local, T open, T closed) {
        const int length = input.length, rows = int(encoder->global_mask->ne[1]), window = architecture::local_window;
        global.assign(std::size_t(length) * rows * input.size, closed);
        local.assign(global.size(), closed);
        for (int row = 0; row < input.size; ++row) {
            const int valid = input.lengths[row];
            for (int query = 0; query < length; ++query) {
                const std::size_t base = (std::size_t(row) * rows + query) * length;
                std::fill_n(global.begin() + base, valid, open);
                const int first = std::max(0, query - window), last = std::min(valid, query + window + 1);
                if (first < last) std::fill(local.begin() + base + first, local.begin() + base + last, open);
                if (query >= valid + window) local[base] = open;  // beyond every valid key's window: attend the first token
            }
        }
        put(encoder->global_mask, global);
        if (encoder->local_mask->buffer) put(encoder->local_mask, local);  // unused by unpadded short fused attention
    }

    // Action-head features per row: pooled state, then the top probability,
    // its margin, normalized entropy and option count of the option softmax.
    // Logits of padded options are set to -1e4 in place.
    void summarize(const batch& input, std::vector<float>& logits) {
        const int width = model.arch.width;
        features.resize(std::size_t(width + 4) * input.size);
        probabilities.resize(std::size_t(input.options));
        for (int row = 0; row < input.size; ++row) {
            auto* out = features.data() + std::size_t(row) * (width + 4);
            std::copy_n(pooled.data() + std::size_t(row) * width, width, out);
            auto* scores = logits.data() + std::size_t(row) * input.options;
            std::fill(scores + input.counts[row], scores + input.options, -1e4f);
            const float maximum = *std::max_element(scores, scores + input.options);
            float total = 0, entropy = 0;
            for (int j = 0; j < input.options; ++j) total += probabilities[j] = std::exp(scores[j] - maximum);
            for (auto& v : probabilities) v /= total, entropy -= v * std::log(std::max(v, 1e-9f));
            std::partial_sort(probabilities.begin(), probabilities.begin() + 2, probabilities.end(), std::greater<float>());
            const int count = std::max(2, input.counts[row]);
            out[width] = probabilities[0];
            out[width + 1] = probabilities[0] - probabilities[1];
            out[width + 2] = entropy / std::log(float(count));
            out[width + 3] = float(count) / 255.0f;
        }
    }

    void write_traces() const {
        std::error_code ignored;
        std::filesystem::create_directories(trace_directory, ignored);
        std::vector<float> values;
        std::vector<ggml_bf16_t> packed;
        for (const auto& [name, t] : encoder->traces) {
            values.resize(std::size_t(ggml_nelements(t)));
            if (t->type == GGML_TYPE_BF16) {
                packed.resize(values.size());
                ggml_backend_tensor_get(t, packed.data(), 0, ggml_nbytes(t));
                ggml_bf16_to_fp32_row(packed.data(), values.data(), std::int64_t(values.size()));
            } else {
                ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
            }
            std::ofstream(trace_directory / (name + ".f32"), std::ios::binary)
                .write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
        }
    }

    template<mode M> result<raw_result> run(const batch& input) {
        // AMD Vulkan BF16 kernels flag nonfinite projection inputs on the device.
        constexpr bool checked = has(M, feature::vulkan | feature::bf16 | feature::amd);
        const auto check = [&](const std::vector<float>& values) -> result<void> {
            if constexpr (checked) {
                if (laya_vk_bf16_status_failed(backend.get())) return fail(errc::backend, "Nonfinite AMD Vulkan BF16 projection input");
                if (!std::ranges::all_of(values, [](float x) { return std::isfinite(x); })) return fail(errc::backend, "Nonfinite AMD Vulkan BF16 output");
            }
            return {};
        };
        LAYA_CHECK(validate(input));
        const graph_shape shape{input.size, input.length, input.options,
                                numerics<M>::mixed && std::ranges::any_of(input.lengths, [&](int n) { return n < input.length; })};
        if (!encoder || encoder->shape != shape) {
            encoder.reset();
            LAYA_TRY(graph, network<M>::encoder(backend.get(), w, model.arch, shape, !trace_directory.empty()));
            encoder.emplace(std::move(*graph));
        }
        if (!head || head->batch != input.size) {
            head.reset();
            LAYA_TRY(graph, network<M>::actions(backend.get(), w, model.arch, input.size));
            head.emplace(std::move(*graph));
        }
        bind(input);
        if constexpr (has(M, feature::vulkan | feature::flash))
            upload_masks(input, global_half, local_half, ggml_fp32_to_fp16(0.0f), ggml_fp32_to_fp16(-INFINITY));
        else if constexpr (has(M, feature::vulkan))
            upload_masks(input, global_mask, local_mask, 0.0f, -INFINITY);
        else
            put(encoder->lengths, input.lengths);

        const auto start = std::chrono::steady_clock::now();
        if constexpr (checked) laya_vk_bf16_status_reset(backend.get());
        if (ggml_backend_graph_compute(backend.get(), encoder->graph) != GGML_STATUS_SUCCESS) return fail(errc::backend, "Encoder computation failed");
        raw_result output;
        output.action_count = model.serving.actions;
        output.logits.resize(std::size_t(input.size) * input.options);
        pooled.resize(std::size_t(model.arch.width) * input.size);
        ggml_backend_tensor_get(encoder->logits, output.logits.data(), 0, ggml_nbytes(encoder->logits));
        ggml_backend_tensor_get(encoder->pooled, pooled.data(), 0, ggml_nbytes(encoder->pooled));
        LAYA_CHECK(check(output.logits));
        LAYA_CHECK(check(pooled));
        if (!trace_directory.empty()) write_traces();

        summarize(input, output.logits);
        put(head->input, features);
        if (ggml_backend_graph_compute(backend.get(), head->graph) != GGML_STATUS_SUCCESS) return fail(errc::backend, "Action computation failed");
        output.actions.resize(std::size_t(input.size) * output.action_count);
        ggml_backend_tensor_get(head->output, output.actions.data(), 0, ggml_nbytes(head->output));
        LAYA_CHECK(check(output.actions));
        output.compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        return output;
    }
};

result<runtime> runtime::load(const std::filesystem::path& directory, mode requested) {
    if (has(requested, feature::coreml) && !(built & feature::coreml))
        return fail(errc::unsupported, "This build has no Core ML backend; rebuild with -DLAYA_COREML=ON on macOS arm64");
    if (const char* reason = rejection(requested)) return fail(errc::unsupported, reason);
    LAYA_TRY(model, checkpoint::load(directory));
    auto state = std::make_unique<impl>(std::move(*model), requested);
    if (const char* traces = std::getenv("LAYA_TRACE_DIR")) state->trace_directory = traces;
    LAYA_CHECK(state->open());
    if (!std::ranges::contains(modes, state->m)) return fail(errc::unsupported, "Unsupported backend and precision");
    LAYA_CHECK(dispatch(state->m, [&]<mode M> { return state->template load<M>(); }));
    return runtime(std::move(state));
}
result<raw_result> runtime::forward(const batch& input) {
    return dispatch(p->m, [&]<mode M> { return p->run<M>(input); });
}
const checkpoint& runtime::model() const { return p->model; }
const std::string& runtime::backend_name() const { return p->name; }
const std::string& runtime::device_name() const { return p->description; }
runtime::runtime(std::unique_ptr<impl> state) : p(std::move(state)) {}
runtime::runtime(runtime&&) noexcept = default;
runtime& runtime::operator=(runtime&&) noexcept = default;
runtime::~runtime() = default;
}
