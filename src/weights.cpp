#include "laya/weights.hpp"

namespace laya {
std::vector<weights::entry> weights::entries(const architecture& arch, int actions) {
    const std::int64_t W = arch.width, I = arch.intermediate;
    std::vector<entry> list;
    auto add = [&](std::string name, role kind, std::vector<std::int64_t> shape, tensor& slot) {
        list.push_back({std::move(name), kind, std::move(shape), &slot});
    };
    // The member path of each tensor spells its checkpoint name.
#define LAYA_TENSOR(path, kind, ...) add(#path, role::kind, {__VA_ARGS__}, path)
#define LAYA_LAYER(path, kind, ...) add(prefix + "." #path, role::kind, {__VA_ARGS__}, layer.path)
    LAYA_TENSOR(encoder.embeddings.tok_embeddings.weight, table, arch.vocabulary, W);
    LAYA_TENSOR(encoder.embeddings.norm.weight, table, W);
    LAYA_TENSOR(encoder.final_norm.weight, table, W);
    LAYA_TENSOR(type_emb.weight, table, 3, W);
    LAYA_TENSOR(temperature, table, 3);
    for (int i = 0; i < arch.layers; ++i) {
        auto& layer = encoder.layers[i];
        const auto prefix = "encoder.layers." + std::to_string(i);
        if (i) LAYA_LAYER(attn_norm.weight, table, W);
        LAYA_LAYER(mlp_norm.weight, table, W);
        LAYA_LAYER(attn.Wqkv.weight, trunk, 3 * W, W);
        LAYA_LAYER(attn.Wo.weight, trunk, W, W);
        LAYA_LAYER(mlp.Wi.weight, trunk, 2 * I, W);
        LAYA_LAYER(mlp.Wo.weight, trunk, W, I);
    }
    for (int i = 0; i < 2; ++i) {
        auto& layer = head.layers[i];
        const auto prefix = "head.layers." + std::to_string(i);
        LAYA_LAYER(self_attn.in_proj_weight, trunk, 3 * W, W);
        LAYA_LAYER(self_attn.in_proj_bias, bias, 3 * W);
        LAYA_LAYER(self_attn.out_proj.weight, trunk, W, W);
        LAYA_LAYER(self_attn.out_proj.bias, bias, W);
        LAYA_LAYER(linear1.weight, trunk, 4 * W, W);
        LAYA_LAYER(linear1.bias, bias, 4 * W);
        LAYA_LAYER(linear2.weight, trunk, W, 4 * W);
        LAYA_LAYER(linear2.bias, bias, W);
        LAYA_LAYER(norm1.weight, table, W);
        LAYA_LAYER(norm1.bias, table, W);
        LAYA_LAYER(norm2.weight, table, W);
        LAYA_LAYER(norm2.bias, table, W);
    }
#undef LAYA_LAYER
#undef LAYA_TENSOR
    add("scorer.0.weight", role::table, {W}, scorer[0].weight);
    add("scorer.0.bias", role::table, {W}, scorer[0].bias);
    add("scorer.1.weight", role::projection, {W, W}, scorer[1].weight);
    add("scorer.1.bias", role::bias, {W}, scorer[1].bias);
    add("scorer.3.weight", role::projection, {1, W}, scorer[3].weight);
    add("scorer.3.bias", role::bias, {1}, scorer[3].bias);
    add("act_head.0.weight", role::projection, {256, W + 4}, act_head[0].weight);
    add("act_head.0.bias", role::bias, {256}, act_head[0].bias);
    add("act_head.2.weight", role::projection, {actions, 256}, act_head[2].weight);
    add("act_head.2.bias", role::bias, {actions}, act_head[2].bias);
    return list;
}
}
