#pragma once
#include "numerics.hpp"
#include <array>
#include <charconv>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// The Laya network topology: ModernBERT encoder, two decision-head layers,
// option scorer and action head. numerics<M> supplies the arithmetic.
namespace laya {
// Graph shapes are fixed per compiled graph; mixed precision also depends on
// whether any row is shorter than the batch length.
struct graph_shape {
    int batch = 0, length = 0, options = 0;
    bool padding = false;
    friend bool operator==(const graph_shape&, const graph_shape&) = default;
};

// One compiled graph with its allocator and input/output tensors.
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
    std::vector<std::pair<std::string, tensor*>> traces;
};

struct action_graph : compiled_graph {
    int batch = 0;
    tensor *input = nullptr, *output = nullptr;
};

namespace detail {
inline void append(std::string& text, std::string_view part) { text += part; }
inline void append(std::string& text, int number) {
    char digits[16];
    text.append(digits, std::to_chars(digits, digits + sizeof(digits), number).ptr);
}

inline result<context_handle> graph_context() {
    context_handle context(ggml_init({ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false), nullptr, true}));
    if (!context) return fail(errc::backend, "Cannot allocate graph metadata");
    return context;
}

// Checks backend support and allocates every intermediate of a built graph.
inline result<void> allocate(compiled_graph& g, ggml_backend_t backend) {
    for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i)
        if (const auto* node = ggml_graph_node(g.graph, i); !ggml_backend_supports_op(backend, node))
            return fail(errc::unsupported, std::string("Requested backend does not support ") + ggml_op_name(node->op));
    g.allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    if (!g.allocator || !ggml_gallocr_alloc_graph(g.allocator.get(), g.graph))
        return fail(errc::backend, "Insufficient memory for this batch");
    return {};
}
}

template<mode M, bool Trace>
class network {
    using N = numerics<M>;
    static constexpr bool fused = N::fused_residual;

public:
    // The encoder, head and scorer graph for one batch shape.
    static result<encoder_graph> encoder(ggml_backend_t backend, const weights& w, const architecture& arch, graph_shape shape) {
        encoder_graph g;
        LAYA_TRY(context, detail::graph_context());
        g.context = std::move(*context);
        g.shape = shape;
        network(g, w, arch).build_encoder(g);
        LAYA_CHECK(detail::allocate(g, backend));
        // Rotary tables depend on the shape only; their inputs stay allocated
        // as outputs, so each graph uploads them once.
        std::vector<float> cosine(64 * std::size_t(shape.length)), sine(cosine.size());
        for (int kind = 0; kind < 2; ++kind) {
            N::rotary(kind, arch.local_rope, shape.length, cosine.data(), sine.data());
            ggml_backend_tensor_set(g.cosine[kind], cosine.data(), 0, ggml_nbytes(g.cosine[kind]));
            ggml_backend_tensor_set(g.sine[kind], sine.data(), 0, ggml_nbytes(g.sine[kind]));
        }
        return g;
    }

    // The action head over pooled features and option statistics.
    static result<action_graph> actions(ggml_backend_t backend, const weights& w, const architecture& arch, int batch) {
        action_graph g;
        LAYA_TRY(context, detail::graph_context());
        g.context = std::move(*context);
        g.batch = batch;
        network(g, w, arch).build_actions(g);
        LAYA_CHECK(detail::allocate(g, backend));
        return g;
    }

private:
    network(compiled_graph& g, const weights& w, const architecture& arch)
        : ctx(g.context.get()), w(w), arch(arch) {
        g.graph = ggml_new_graph_custom(ctx, 8192, false);
    }

    ggml_context* ctx;
    const weights& w;
    const architecture& arch;
    std::vector<std::pair<std::string, tensor*>>* traces = nullptr;

    struct attention_weights {
        tensor *qkv, *qkv_bias, *out, *out_bias;
    };

    tensor* input(ggml_type type, std::initializer_list<std::int64_t> dimensions) {
        auto* t = ggml_new_tensor(ctx, type, int(dimensions.size()), dimensions.begin());
        ggml_set_input(t);
        return t;
    }

    template<class... Parts>
    void trace(tensor* value, const Parts&... name) {
        if constexpr (Trace) {
            ggml_set_output(value);
            std::string text;
            (detail::append(text, name), ...);
            traces->emplace_back(std::move(text), value);
        }
    }

    // Mixed precision adds the residual inside the projection.
    static tensor* residual(tensor* h) { return fused ? h : nullptr; }
    tensor* add_residual(tensor* h, tensor* value) { return fused ? value : ggml_add(ctx, h, value); }

    template<bool Head, bool Stored>
    tensor* attention(operand<Stored> x, const attention_weights& p, const attention_site& at, std::string_view scope,
                      int layer, tensor* h, const encoder_graph& g) {
        trace(x.t, scope, layer, ".qkv-input");
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
        trace(qkv, scope, layer, ".attn.Wqkv");
        const int kind = at.global ? 0 : 1;
        auto* packed = N::pack(ctx, qkv, Head ? nullptr : g.cosine[kind], Head ? nullptr : g.sine[kind], at.length, at.batch);
        std::array<tensor*, 3> split;
        for (int i = 0; i < 3; ++i)
            split[i] = ggml_view_4d(ctx, packed, 64, at.length, arch.heads, at.batch, packed->nb[1], packed->nb[2],
                                    packed->nb[3], i * at.batch * packed->nb[3]);
        auto* q = N::query(ctx, split[0]);
        auto* k = split[1];
        if constexpr (Trace) {
            q = ggml_cont(ctx, q);
            trace(q, scope, layer, ".q");
            k = ggml_cont(ctx, k);
            trace(k, scope, layer, ".k");
        }
        auto* value = N::attend(ctx, q, k, split[2], Head || at.global ? g.global_mask : g.local_mask, at);
        trace(value, scope, layer, ".attn.Wo.input");
        auto* output = N::template linear<role::trunk>(ctx, wide{value}, p.out, Head ? p.out_bias : nullptr, residual(h)).t;
        trace(output, scope, layer, fused ? ".attn.residual" : ".attn.Wo");
        return output;
    }

    void build_encoder(encoder_graph& g) {
        traces = &g.traces;
        const auto [batch, length, options, padding] = g.shape;
        const std::int64_t tokens = std::int64_t(length) * batch;
        g.ids = input(GGML_TYPE_I32, {tokens});
        g.types = input(GGML_TYPE_I32, {tokens});
        for (int kind = 0; kind < 2; ++kind) {
            g.cosine[kind] = input(GGML_TYPE_F32, {64, 1, length, 1});
            g.sine[kind] = input(GGML_TYPE_F32, {64, 1, length, 1});
            ggml_set_output(g.cosine[kind]);
            ggml_set_output(g.sine[kind]);
        }
        g.markers = input(GGML_TYPE_I32, {std::int64_t(options) * batch});
        g.cls = input(GGML_TYPE_I32, {batch});
        // Fused attention reads mask query rows in multiples of 64.
        constexpr bool flash = has(M, feature::flash);
        const int rows = flash ? (length + 63) / 64 * 64 : length;
        if constexpr (has(M, feature::vulkan)) {
            constexpr auto type = flash ? GGML_TYPE_F16 : GGML_TYPE_F32;
            g.global_mask = input(type, {length, rows, 1, batch});
            g.local_mask = input(type, {length, rows, 1, batch});
        } else {
            g.lengths = input(GGML_TYPE_I32, {batch});
            g.global_mask = attention_mask(ctx, g.lengths, length, rows, false, flash);
            g.local_mask = attention_mask(ctx, g.lengths, length, rows, true, flash);
        }

        const auto& encoder = w.encoder;
        auto* h = N::template norm<false>(ctx, ggml_get_rows(ctx, encoder.embeddings.tok_embeddings.weight, g.ids),
                                          encoder.embeddings.norm).t;
        trace(h, "embedding");
        for (int layer = 0; layer < arch.layers; ++layer) {
            const auto& p = encoder.layers[layer];
            const attention_site at{false, global_layer(layer), padding, length, batch, arch.width};
            const attention_weights projections{p.attn.Wqkv.weight, nullptr, p.attn.Wo.weight, nullptr};
            auto* attended = layer == 0
                ? attention<false>(wide{h}, projections, at, "encoder.layers.", layer, h, g)
                : attention<false>(N::template norm<true>(ctx, h, p.attn_norm), projections, at, "encoder.layers.", layer, h, g);
            h = add_residual(h, attended);
            if constexpr (N::split_mlp) {
                auto* normalized = N::template norm<true>(ctx, h, p.mlp_norm).t;
                auto* products = ggml_mul_mat(ctx, p.mlp.Wi.weight, split_f16(ctx, normalized));
                auto* projected = ggml_mul_mat(ctx, p.mlp.Wo.weight, mlp_split_f16(ctx, products));
                h = ggml_add(ctx, h, merge_f16(ctx, projected));
            } else {
                auto normalized = N::template norm<true>(ctx, h, p.mlp_norm);
                trace(normalized.t, "encoder.layers.", layer, ".mlp.Wi.input");
                auto gated = N::template linear<role::trunk, true>(ctx, normalized, p.mlp.Wi.weight, nullptr);
                trace(gated.t, "encoder.layers.", layer, ".mlp.Wi");
                auto activated = N::gated_gelu(ctx, w.gelu_table, gated);
                trace(activated.t, "encoder.layers.", layer, ".mlp.Wo.input");
                auto* projected = N::template linear<role::trunk>(ctx, activated, p.mlp.Wo.weight, nullptr, residual(h)).t;
                trace(projected, "encoder.layers.", layer, fused ? ".mlp.residual" : ".mlp.Wo");
                h = add_residual(h, projected);
            }
            trace(h, "encoder-", layer);
        }
        h = N::template norm<false>(ctx, h, encoder.final_norm).t;
        trace(h, "final-norm");
        h = ggml_add(ctx, h, ggml_get_rows(ctx, w.type_emb.weight, g.types));
        for (int layer = 0; layer < 2; ++layer) {
            const auto& p = w.head.layers[layer];
            const attention_site at{true, global_layer(layer), padding, length, batch, arch.width};
            const attention_weights projections{p.self_attn.in_proj_weight, p.self_attn.in_proj_bias,
                                                p.self_attn.out_proj.weight, p.self_attn.out_proj.bias};
            h = add_residual(h, attention<true>(N::template norm<true>(ctx, h, p.norm1), projections, at, "head.layers.", layer, h, g));
            auto* first = N::template linear<role::trunk>(ctx, N::template norm<true>(ctx, h, p.norm2), p.linear1.weight, p.linear1.bias).t;
            trace(first, "head.layers.", layer, ".linear1");
            auto* second = N::template linear<role::trunk>(ctx, wide{ggml_relu(ctx, first)}, p.linear2.weight, p.linear2.bias, residual(h)).t;
            trace(second, "head.layers.", layer, fused ? ".linear2-residual" : ".linear2");
            h = add_residual(h, second);
            trace(h, "head-", layer);
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

    void build_actions(action_graph& g) {
        g.input = input(GGML_TYPE_F32, {arch.width + 4, g.batch});
        const auto& head = w.act_head;
        auto hidden = N::gelu(ctx, w.gelu_table, N::template linear<role::projection>(ctx, wide{g.input}, head[0].weight, head[0].bias));
        g.output = N::template linear<role::projection>(ctx, hidden, head[2].weight, head[2].bias).t;
        ggml_set_output(g.output);
        ggml_build_forward_expand(g.graph, g.output);
    }
};
}
