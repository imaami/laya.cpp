#pragma once
#include "laya/checkpoint.hpp"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;

namespace laya {
// How a checkpoint tensor is stored on the device; the mode maps each role to a type.
enum class role : std::uint8_t {
    table,       // embeddings and normalization parameters: FP32
    bias,        // projection bias: FP32 holding product-precision values
    projection,  // decision-head matrix: product precision
    trunk,       // encoder and head-layer matrix: product precision or compensated FP16
};

// Checkpoint tensors by their safetensors names: "encoder.layers.3.attn.Wqkv.weight"
// is w.encoder.layers[3].attn.Wqkv.weight and "scorer.3.bias" is w.scorer[3].bias.
struct weights {
    using tensor = ggml_tensor*;
    struct linear { tensor weight = nullptr, bias = nullptr; };  // also LayerNorm parameters
    struct encoder_layer {
        linear attn_norm, mlp_norm;  // layer 0 has no attn_norm
        struct { linear Wqkv, Wo; } attn;
        struct { linear Wi, Wo; } mlp;
    };
    struct head_layer {
        struct { tensor in_proj_weight = nullptr, in_proj_bias = nullptr; linear out_proj; } self_attn;
        linear linear1, linear2, norm1, norm2;
    };
    struct {
        struct { linear tok_embeddings, norm; } embeddings;
        std::array<encoder_layer, 28> layers;
        linear final_norm;
    } encoder;
    linear type_emb;
    tensor temperature = nullptr;
    struct { std::array<head_layer, 2> layers; } head;
    std::array<linear, 4> scorer;    // LayerNorm, Linear, GELU, Linear
    std::array<linear, 3> act_head;  // Linear, GELU, Linear
    tensor gelu_table = nullptr;     // device-generated: Vulkan mixed-precision GELU lookup

    // A checkpoint tensor, its PyTorch shape ({rows} for vectors) and its slot here.
    struct entry {
        std::string name;
        role kind;
        std::vector<std::int64_t> shape;
        tensor* slot;
    };
    // Every tensor a checkpoint of this geometry holds, in validation order.
    [[nodiscard]] std::vector<entry> entries(const architecture& arch, int actions);
};
}
