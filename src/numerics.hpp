#pragma once
#include "device.hpp"
#include "laya/precision.hpp"
#include "vulkan_precision.hpp"
#include "vulkan_rotary.hpp"
#include <cmath>
#include <utility>

// Numerics policies: how each mode rounds, projects, normalizes and attends.
// The network topology is shared; numerics<M> supplies the arithmetic of mode M.
namespace laya {
using tensor = ggml_tensor;

// A graph value; Stored marks values already rounded to the storage precision
// that mixed-precision projections consume, so they are not rounded again.
template<bool Stored>
struct operand {
    tensor* t;
};
using wide = operand<false>;

// Where an attention block sits, for kernels selected by shape.
struct attention_site {
    bool head, global, padding;
    int length, batch;
    std::int64_t width;
    [[nodiscard]] std::int64_t tokens() const { return std::int64_t(length) * batch; }
    // Fused mixed-precision attention names the masks its kernels must apply.
    [[nodiscard]] const char* masked_kernel() const {
        if (!head && !global && length >= architecture::local_window) return "laya.sdpa-local";
        return head || padding ? "laya.sdpa-masked" : nullptr;
    }
};

namespace detail {
inline tensor* fused_attention(ggml_context* ctx, tensor* q, tensor* k, tensor* v, tensor* mask, const char* kernel,
                               const attention_site& at) {
    auto* value = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / 8.0f, 0, 0);
    if (kernel) ggml_set_name(value, kernel);
    ggml_prec_set_acc(value, GGML_PREC_F32);
    return ggml_reshape_2d(ctx, ggml_is_contiguous(value) ? value : ggml_cont(ctx, value), at.width, at.tokens());
}

// Scores and probabilities as separate products. AMD-matched numerics scale
// both factors by 8^-1/2 before the product, as the ROCm reference does.
template<bool Amd>
tensor* explicit_attention(ggml_context* ctx, tensor* q, tensor* k, tensor* v, tensor* mask, const attention_site& at) {
    if constexpr (Amd) {
        q = ggml_scale(ctx, q, std::sqrt(1.0f / 8.0f));
        k = ggml_scale(ctx, k, std::sqrt(1.0f / 8.0f));
    }
    auto* scores = ggml_mul_mat(ctx, k, q);
    if constexpr (Amd) ggml_set_name(scores, "laya.amd-low-qk");
    ggml_prec_set_acc(scores, GGML_PREC_F32);
    auto* probabilities = ggml_soft_max_ext(ctx, scores, mask, Amd ? 1.0f : 1.0f / 8.0f, 0);
    if constexpr (Amd) ggml_set_name(probabilities, "laya.amd-low-softmax");
    auto* value = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), probabilities);
    if constexpr (Amd) ggml_set_name(value, "laya.amd-low-pv");
    ggml_prec_set_acc(value, GGML_PREC_F32);
    return ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, value, 0, 2, 1, 3)), at.width, at.tokens());
}

// Fills the cosine and sine tables of `length` positions. `values(i)` returns
// the position -> {cosine, sine} function of rotary dimension i, which both
// halves of each 64-wide head share.
template<class Values>
void fill_rotary(int length, float* cosine, float* sine, Values values) {
    for (int i = 0; i < 32; ++i) {
        const auto at = values(i);
        for (int position = 0; position < length; ++position) {
            const auto [c, s] = at(position);
            cosine[position * 64 + i] = cosine[position * 64 + i + 32] = c;
            sine[position * 64 + i] = sine[position * 64 + i + 32] = s;
        }
    }
}

// Rotary frequency of dimension i for global (kind 0) or local layers.
inline float rotary_inverse(int kind, float local_rope, int i) {
    return 1.0f / std::pow(kind == 0 ? architecture::global_rope : local_rope, float(2 * i) / 64.0f);
}
}

template<mode M>
struct numerics;

// Strict FP32: products accumulate in FP32 from FP32 operands. Compensated
// modes store trunk matrices in FP16 and split each FP32 operand into a high
// and a scaled low FP16 half.
template<mode M>
    requires(!(M & (feature::low | feature::coreml)))
struct numerics<M> {
    static constexpr bool vulkan = has(M, feature::vulkan);
    static constexpr bool compensated = has(M, feature::compensated);
    static constexpr storage layout{.trunk = compensated ? GGML_TYPE_F16 : GGML_TYPE_F32, .f16_checkpoint = compensated};
    static constexpr bool mixed = false;
    static constexpr bool fused_residual = false;   // projections return the bare product
    static constexpr bool split_mlp = compensated && !vulkan;  // CUDA fuses the compensated MLP
    static constexpr bool transposed_head = false;

    template<bool Compact>
    static wide norm(ggml_context* ctx, tensor* x, const weights::linear& affine) {
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
        if constexpr (split) {
            auto* product = ggml_mul_mat(ctx, weight, split_half(ctx, x));
            if constexpr (vulkan) ggml_prec_set_acc(product, GGML_PREC_F32);
            value = merge_half(ctx, product);
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
    template<bool Stored>
    static wide gated_gelu(ggml_context* ctx, tensor*, operand<Stored> gated) {
        if constexpr (vulkan) {
            return {ggml_geglu_erf(ctx, gated.t)};
        } else {
            auto* x = gated.t;
            const std::int64_t width = x->ne[0] / 2;
            auto* first = ggml_cont(ctx, ggml_view_2d(ctx, x, width, x->ne[1], x->nb[1], 0));
            auto* second = ggml_cont(ctx, ggml_view_2d(ctx, x, width, x->ne[1], x->nb[1], width * sizeof(float)));
            return {ggml_mul(ctx, ggml_gelu_erf(ctx, first), second)};
        }
    }

    static tensor* round(ggml_context*, tensor* x) { return x; }

    static tensor* pack(ggml_context* ctx, tensor* qkv, tensor* cosine, tensor* sine, int length, int batch) {
        if constexpr (vulkan) return vulkan_precision::pack_qkv(ctx, qkv, cosine, sine, length, batch, GGML_TYPE_F32);
        else return pack_qkv(ctx, qkv, cosine, sine, length, batch, false);
    }

    static tensor* query(ggml_context*, tensor* q) { return q; }

    static tensor* attend(ggml_context* ctx, tensor* q, tensor* k, tensor* v, tensor* mask, const attention_site& at) {
        // Fused FP32 attention is exact only while its tiles cover the sequence.
        if constexpr (has(M, feature::flash))
            if (at.length <= 128) return detail::fused_attention(ctx, q, k, v, mask, nullptr, at);
        return detail::explicit_attention<false>(ctx, q, k, v, mask, at);
    }

    static void rotary(int kind, float local_rope, int length, float* cosine, float* sine) {
        detail::fill_rotary(length, cosine, sine, [&](int i) {
            return [inverse = detail::rotary_inverse(kind, local_rope, i)](int position) {
                const float angle = float(position) * inverse;
                return std::pair{std::cos(angle), std::sin(angle)};
            };
        });
    }

private:
    static tensor* split_half(ggml_context* ctx, tensor* x) {
        if constexpr (vulkan) return vulkan_precision::split_half(ctx, x);
        else return split_f16(ctx, x);
    }
    static tensor* merge_half(ggml_context* ctx, tensor* x) {
        if constexpr (vulkan) return vulkan_precision::merge_half(ctx, x);
        else return merge_f16(ctx, x);
    }
};

// Mixed BF16 on CUDA: custom kernels reproduce PyTorch autocast rounding.
// Compact kernels store BF16 outputs that projections consume directly.
template<mode M>
    requires(has(M, feature::cuda | feature::bf16))
struct numerics<M> {
    static constexpr ggml_type low = GGML_TYPE_BF16;
    static constexpr storage layout{.trunk = low, .projection = low, .bias = low};
    static constexpr bool mixed = true;
    static constexpr bool fused_residual = true;  // projections add their FP32 residual
    static constexpr bool split_mlp = false;
    static constexpr bool transposed_head = false;

    template<bool Compact>
    static operand<Compact> norm(ggml_context* ctx, tensor* x, const weights::linear& affine) {
        return {norm_bf16(ctx, x, affine.weight, affine.bias, Compact)};
    }

    template<role, bool Compact = false, bool Stored>
    static operand<Compact> linear(ggml_context* ctx, operand<Stored> input, tensor* weight, tensor* bias,
                                   tensor* residual = nullptr) {
        auto* x = input.t;
        if constexpr (!Stored) x = ggml_cast(ctx, x, low);
        return {linear_bf16(ctx, x, weight, bias, Compact, residual)};
    }

    static wide gelu(ggml_context* ctx, tensor*, wide x) { return {gelu_bf16(ctx, x.t)}; }

    template<bool Stored>
    static operand<true> gated_gelu(ggml_context* ctx, tensor*, operand<Stored> gated) { return {mlp_bf16(ctx, gated.t)}; }

    static tensor* round(ggml_context* ctx, tensor* x) { return ggml_cast(ctx, ggml_cast(ctx, x, low), GGML_TYPE_F32); }

    static tensor* pack(ggml_context* ctx, tensor* qkv, tensor* cosine, tensor* sine, int length, int batch) {
        return pack_qkv(ctx, qkv, cosine, sine, length, batch, true);
    }

    static tensor* query(ggml_context* ctx, tensor* q) { return ggml_cast(ctx, q, GGML_TYPE_F32); }

    static tensor* attend(ggml_context* ctx, tensor* q, tensor* k, tensor* v, tensor* mask, const attention_site& at) {
        return detail::fused_attention(ctx, q, k, v, mask, at.masked_kernel(), at);
    }

    // The CUDA rotary kernel evaluates the cosine itself from the raw angle.
    static void rotary(int kind, float local_rope, int length, float* cosine, float* sine) {
        detail::fill_rotary(length, cosine, sine, [&](int i) {
            return [inverse = detail::rotary_inverse(kind, local_rope, i)](int position) {
                const float angle = float(position) * inverse;
                return std::pair{angle, std::sin(angle)};
            };
        });
    }
};

// Mixed FP16/BF16 on Vulkan: portable kernels keep the storage rounding points
// of autocast. Vendor bits select the device whose reference numerics are matched.
template<mode M>
    requires(has(M, feature::vulkan) && (M & feature::low) != 0)
struct numerics<M> {
    static constexpr ggml_type low = has(M, feature::fp16) ? GGML_TYPE_F16 : GGML_TYPE_BF16;
    static constexpr bool bf16 = low == GGML_TYPE_BF16;
    static constexpr bool amd = has(M, feature::amd), nvidia = has(M, feature::nvidia);
    static constexpr storage layout{.trunk = low, .projection = low, .bias = low, .amd_bf16_range = amd && bf16,
                                    .gelu = low, .rocm_gelu = amd};
    static constexpr bool mixed = true;
    static constexpr bool fused_residual = true;
    static constexpr bool split_mlp = false;
    // AMD batched heads project in a sequence-major layout.
    static constexpr bool transposed_head = amd;

    template<bool Compact>
    static operand<Compact> norm(ggml_context* ctx, tensor* x, const weights::linear& affine) {
        return {vulkan_precision::norm(ctx, x, affine.weight, affine.bias, Compact ? low : GGML_TYPE_F32)};
    }

    template<role, bool Compact = false, bool Stored>
    static wide linear(ggml_context* ctx, operand<Stored> input, tensor* weight, tensor* bias, tensor* residual = nullptr) {
        auto* x = input.t;
        // Round in the F32 layout the matrix kernels read, without
        // materializing an intermediate 16-bit tensor.
        if constexpr (!Stored)
            x = x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) ? vulkan_precision::finish_projection(ctx, x, nullptr, nullptr, low)
                                                                  : ggml_cast(ctx, x, low);
        vulkan_precision::projection_plan plan{};
        if constexpr (nvidia) {
            plan = vulkan_precision::select_projection_plan(weight->ne[0], weight->ne[1], x->ne[1], bias != nullptr);
            // Bound cancellation in batched scalar heads with short FP32 dot
            // products before the final rounding.
            if constexpr (bf16)
                if (weight->ne[1] == 1 && x->ne[1] > 1 && !plan.chunk) plan.chunk = 64;
        }
        return {vulkan_precision::linear(ctx, x, weight, bias, residual, low, plan, false, amd)};
    }

    static wide gelu(ggml_context* ctx, tensor* table, wide x) {
        return {vulkan_precision::activation(ctx, x.t, table, false, bf16)};
    }

    template<bool Stored>
    static wide gated_gelu(ggml_context* ctx, tensor* table, operand<Stored> gated) {
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
            return detail::fused_attention(ctx, q, k, v, mask, mask ? at.masked_kernel() : "laya.sdpa-flash", at);
        } else {
            return detail::explicit_attention<amd>(ctx, q, k, v, mask, at);
        }
    }

    // Rotary tables reproduce the rounded values of the matched device library.
    static void rotary(int kind, float local_rope, int length, float* cosine, float* sine) {
        const int base = kind == 0 || local_rope == architecture::global_rope ? 0 : 1;
        detail::fill_rotary(length, cosine, sine, [base](int i) {
            return [base, i](int position) {
                return std::pair{vulkan_precision::rotary(amd, base, position, i, false),
                                 vulkan_precision::rotary(amd, base, position, i, true)};
            };
        });
    }
};
}
