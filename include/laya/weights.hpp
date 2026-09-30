#pragma once
#include "laya/checkpoint.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>

struct ggml_tensor;

// Resident checkpoint tensors. The members mirror the checkpoint names, so
// "encoder.layers.3.attn.Wqkv.weight" is w.encoder.layers[3].attn.Wqkv.weight
// and the PyTorch Sequential entry "scorer.3.bias" is w.scorer[3].bias.
namespace laya::weights {
using tensor = ggml_tensor;

struct parameter { tensor* weight = nullptr; };
struct affine { tensor* weight = nullptr; tensor* bias = nullptr; };

struct attention { parameter Wqkv, Wo; };
struct mlp { parameter Wi, Wo; };
struct encoder_layer {
    parameter attn_norm;  // absent in layer 0, which reuses the embedding norm
    weights::attention attn;
    parameter mlp_norm;
    weights::mlp mlp;
};
struct embeddings { parameter tok_embeddings, norm; };
struct encoder {
    weights::embeddings embeddings;
    std::array<encoder_layer, max_encoder_layers> layers;
    parameter final_norm;
};

struct self_attention {
    tensor* in_proj_weight = nullptr;
    tensor* in_proj_bias = nullptr;
    affine out_proj;
};
struct head_layer {
    self_attention self_attn;
    affine linear1, linear2, norm1, norm2;
};
struct head { std::array<head_layer, decision_head_layers> layers; };

struct model {
    weights::encoder encoder;
    weights::head head;
    parameter type_emb;
    tensor* temperature = nullptr;
    std::array<affine, 4> scorer;    // LayerNorm, Linear, GELU, Linear
    std::array<affine, 3> act_head;  // Linear, GELU, Linear
};

// How a checkpoint tensor is stored and prepared on the device.
enum class role : std::uint8_t {
    fixed,     // FP32 as stored: embeddings, norms and the temperature vector
    bias,      // FP32 linear bias; mixed precision rounds it to the product type
    trunk,     // encoder and decision-head layer projection
    decision,  // scorer and action-head projection
};
[[nodiscard]] constexpr bool projection(role kind) noexcept { return kind == role::trunk || kind == role::decision; }

// Checkpoint dimensions, resolved against the loaded architecture.
enum class dim : std::uint8_t { one, token_types, width, qkv, feedforward, gated, vocabulary, action_input, action_hidden, actions };
inline constexpr int action_hidden = 256;
inline constexpr int action_features = 4;  // appended to the pooled row

// PyTorch shape order (outermost first), as the safetensors header lists it.
struct extent {
    std::array<dim, 2> dims{};
    int rank = 0;
    constexpr extent(dim size) : dims{size, dim::one}, rank(1) {}
    constexpr extent(dim rows, dim columns) : dims{rows, columns}, rank(2) {}
};

[[nodiscard]] constexpr std::int64_t size(dim d, const architecture& arch, int actions) noexcept {
    switch (d) {
    case dim::one: return 1;
    case dim::token_types: return 3;
    case dim::width: return arch.width;
    case dim::qkv: return 3 * std::int64_t(arch.width);
    case dim::feedforward: return 4 * std::int64_t(arch.width);
    case dim::gated: return 2 * std::int64_t(arch.intermediate);
    case dim::vocabulary: return arch.vocabulary;
    case dim::action_input: return arch.width + action_features;
    case dim::action_hidden: return action_hidden;
    case dim::actions: return actions;
    }
    return 0;
}

template<std::size_t N>
struct fixed_string {
    char text[N]{};
    consteval fixed_string(const char (&value)[N]) { std::copy_n(value, N, text); }
    [[nodiscard]] constexpr std::string_view view() const noexcept { return {text, N - 1}; }
};

// Path steps: data-member pointers and Sequential indices (index<3> is ".3").
template<std::size_t I> struct index_t {};
template<std::size_t I> inline constexpr index_t<I> index{};

template<class Object, class Member, class Class>
[[nodiscard]] constexpr auto& step(Object& object, Member Class::* member) noexcept { return object.*member; }
template<class Object, std::size_t I>
[[nodiscard]] constexpr auto& step(Object& object, index_t<I>) noexcept { return object[I]; }

template<auto First, auto... Rest, class Object>
[[nodiscard]] constexpr auto& walk(Object& object) noexcept {
    if constexpr (sizeof...(Rest) == 0) return step(object, First);
    else return walk<Rest...>(step(object, First));
}

// One named tensor, reached from its module through a compile-time path.
template<fixed_string Name, role Role, extent Shape, auto... Path>
struct leaf {
    static constexpr std::string_view name = Name.view();
    static constexpr role kind = Role;
    static constexpr extent shape = Shape;
    template<class Module>
    [[nodiscard]] static constexpr tensor*& slot(Module& module) noexcept { return walk<Path...>(module); }
};

template<class... Leaves> struct schema {};

using prologue = schema<
    leaf<"encoder.embeddings.tok_embeddings.weight", role::fixed, extent{dim::vocabulary, dim::width},
         &model::encoder, &encoder::embeddings, &embeddings::tok_embeddings, &parameter::weight>,
    leaf<"encoder.embeddings.norm.weight", role::fixed, extent{dim::width},
         &model::encoder, &encoder::embeddings, &embeddings::norm, &parameter::weight>,
    leaf<"encoder.final_norm.weight", role::fixed, extent{dim::width}, &model::encoder, &encoder::final_norm, &parameter::weight>,
    leaf<"type_emb.weight", role::fixed, extent{dim::token_types, dim::width}, &model::type_emb, &parameter::weight>,
    leaf<"temperature", role::fixed, extent{dim::token_types}, &model::temperature>>;

using encoder_attention_norm = schema<
    leaf<"attn_norm.weight", role::fixed, extent{dim::width}, &encoder_layer::attn_norm, &parameter::weight>>;

using encoder_layer_schema = schema<
    leaf<"mlp_norm.weight", role::fixed, extent{dim::width}, &encoder_layer::mlp_norm, &parameter::weight>,
    leaf<"attn.Wqkv.weight", role::trunk, extent{dim::qkv, dim::width}, &encoder_layer::attn, &attention::Wqkv, &parameter::weight>,
    leaf<"attn.Wo.weight", role::trunk, extent{dim::width, dim::width}, &encoder_layer::attn, &attention::Wo, &parameter::weight>,
    leaf<"mlp.Wi.weight", role::trunk, extent{dim::gated, dim::width}, &encoder_layer::mlp, &mlp::Wi, &parameter::weight>,
    leaf<"mlp.Wo.weight", role::trunk, extent{dim::width, dim::feedforward}, &encoder_layer::mlp, &mlp::Wo, &parameter::weight>>;

using head_layer_schema = schema<
    leaf<"self_attn.in_proj_weight", role::trunk, extent{dim::qkv, dim::width}, &head_layer::self_attn, &self_attention::in_proj_weight>,
    leaf<"self_attn.in_proj_bias", role::bias, extent{dim::qkv}, &head_layer::self_attn, &self_attention::in_proj_bias>,
    leaf<"self_attn.out_proj.weight", role::trunk, extent{dim::width, dim::width},
         &head_layer::self_attn, &self_attention::out_proj, &affine::weight>,
    leaf<"self_attn.out_proj.bias", role::bias, extent{dim::width}, &head_layer::self_attn, &self_attention::out_proj, &affine::bias>,
    leaf<"linear1.weight", role::trunk, extent{dim::feedforward, dim::width}, &head_layer::linear1, &affine::weight>,
    leaf<"linear1.bias", role::bias, extent{dim::feedforward}, &head_layer::linear1, &affine::bias>,
    leaf<"linear2.weight", role::trunk, extent{dim::width, dim::feedforward}, &head_layer::linear2, &affine::weight>,
    leaf<"linear2.bias", role::bias, extent{dim::width}, &head_layer::linear2, &affine::bias>,
    leaf<"norm1.weight", role::fixed, extent{dim::width}, &head_layer::norm1, &affine::weight>,
    leaf<"norm1.bias", role::fixed, extent{dim::width}, &head_layer::norm1, &affine::bias>,
    leaf<"norm2.weight", role::fixed, extent{dim::width}, &head_layer::norm2, &affine::weight>,
    leaf<"norm2.bias", role::fixed, extent{dim::width}, &head_layer::norm2, &affine::bias>>;

using epilogue = schema<
    leaf<"scorer.0.weight", role::fixed, extent{dim::width}, &model::scorer, index<0>, &affine::weight>,
    leaf<"scorer.0.bias", role::fixed, extent{dim::width}, &model::scorer, index<0>, &affine::bias>,
    leaf<"scorer.1.weight", role::decision, extent{dim::width, dim::width}, &model::scorer, index<1>, &affine::weight>,
    leaf<"scorer.1.bias", role::bias, extent{dim::width}, &model::scorer, index<1>, &affine::bias>,
    leaf<"scorer.3.weight", role::decision, extent{dim::one, dim::width}, &model::scorer, index<3>, &affine::weight>,
    leaf<"scorer.3.bias", role::bias, extent{dim::one}, &model::scorer, index<3>, &affine::bias>,
    leaf<"act_head.0.weight", role::decision, extent{dim::action_hidden, dim::action_input}, &model::act_head, index<0>, &affine::weight>,
    leaf<"act_head.0.bias", role::bias, extent{dim::action_hidden}, &model::act_head, index<0>, &affine::bias>,
    leaf<"act_head.2.weight", role::decision, extent{dim::actions, dim::action_hidden}, &model::act_head, index<2>, &affine::weight>,
    leaf<"act_head.2.bias", role::bias, extent{dim::actions}, &model::act_head, index<2>, &affine::bias>>;

// A tensor the checkpoint must provide, in PyTorch shape order.
struct expected_tensor {
    std::string name;
    tensor** slot;
    std::array<std::int64_t, 2> shape;
    int rank;
    role kind;
};

template<class... Leaves, class Module, class Visitor>
constexpr void visit(schema<Leaves...>, Module& module, std::string_view prefix, const architecture& arch, int actions,
                     Visitor& visitor) {
    (visitor(expected_tensor{std::string(prefix).append(Leaves::name), &Leaves::slot(module),
                             {size(Leaves::shape.dims[0], arch, actions), size(Leaves::shape.dims[1], arch, actions)},
                             Leaves::shape.rank, Leaves::kind}), ...);
}

// Visits every checkpoint tensor once, in the order the checkpoint is validated.
template<class Visitor>
constexpr void visit(model& w, const architecture& arch, int actions, Visitor&& visitor) {
    visit(prologue{}, w, "", arch, actions, visitor);
    for (int i = 0; i < arch.layers; ++i) {
        const auto prefix = "encoder.layers." + std::to_string(i) + ".";
        if (i) visit(encoder_attention_norm{}, w.encoder.layers[i], prefix, arch, actions, visitor);
        visit(encoder_layer_schema{}, w.encoder.layers[i], prefix, arch, actions, visitor);
    }
    for (int i = 0; i < decision_head_layers; ++i)
        visit(head_layer_schema{}, w.head.layers[i], "head.layers." + std::to_string(i) + ".", arch, actions, visitor);
    visit(epilogue{}, w, "", arch, actions, visitor);
}
}
