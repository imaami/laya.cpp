#pragma once
#include "network.hpp"
#include "vulkan_status.hpp"
#include "laya/batch.hpp"
#if LAYA_COREML
#include "laya/coreml.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace laya {
// Model execution in mode M: the device, its resident weights and the graphs
// compiled for the most recent batch shape. Callers serialize forward().
template<mode M>
class engine {
    using N = numerics<M>;
    static constexpr bool vulkan = has(M, feature::vulkan);
    // AMD Vulkan BF16 kernels flag nonfinite projection inputs on the device.
    static constexpr bool checked_outputs = has(M, feature::vulkan | feature::bf16 | feature::amd);

public:
    [[nodiscard]] static result<engine> load(const checkpoint& model, device&& opened) {
        LAYA_TRY(resident, load_weights(model, opened.backend.get(), N::layout));
        engine loaded(model, std::move(opened), std::move(*resident));
        if (const char* directory = std::getenv("LAYA_TRACE_DIR")) loaded.trace_directory = directory;
        return loaded;
    }

    [[nodiscard]] result<raw_result> forward(const batch& input) {
        LAYA_CHECK(validate(input));
        const graph_shape shape{input.size, input.length, input.options,
                                N::mixed && std::ranges::any_of(input.lengths, [&](int n) { return n < input.length; })};
        if (!encoder || encoder->shape != shape) {
            encoder.reset();
            LAYA_TRY(built, trace_directory ? network<M, true>::encoder(backend(), store.tensors, arch, shape)
                                            : network<M, false>::encoder(backend(), store.tensors, arch, shape));
            encoder.emplace(std::move(*built));
        }
        if (!head || head->batch != input.size) {
            head.reset();
            LAYA_TRY(built, network<M, false>::actions(backend(), store.tensors, arch, input.size));
            head.emplace(std::move(*built));
        }
        auto& g = *encoder;
        put(g.ids, input.ids);
        put(g.markers, input.markers);
        types.resize(input.ids.size());
        cls.resize(std::size_t(input.size));
        for (int row = 0; row < input.size; ++row) {
            cls[row] = row * input.length;
            std::fill_n(types.begin() + cls[row], input.length, input.types[row]);
        }
        put(g.types, types);
        put(g.cls, cls);
        if constexpr (vulkan) upload_masks(g, input);
        else put(g.lengths, input.lengths);

        const auto start = std::chrono::steady_clock::now();
        if constexpr (checked_outputs) laya_vk_bf16_status_reset(backend());
        if (ggml_backend_graph_compute(backend(), g.graph) != GGML_STATUS_SUCCESS)
            return fail(errc::backend, "Encoder computation failed");
        raw_result output;
        output.action_count = actions;
        output.logits.resize(std::size_t(input.size) * input.options);
        pooled.resize(std::size_t(arch.width) * input.size);
        ggml_backend_tensor_get(g.logits, output.logits.data(), 0, ggml_nbytes(g.logits));
        ggml_backend_tensor_get(g.pooled, pooled.data(), 0, ggml_nbytes(g.pooled));
        LAYA_CHECK(check(output.logits));
        LAYA_CHECK(check(pooled));
        if (trace_directory) write_traces(g);

        summarize(input, output.logits);
        put(head->input, features);
        if (ggml_backend_graph_compute(backend(), head->graph) != GGML_STATUS_SUCCESS)
            return fail(errc::backend, "Action computation failed");
        output.actions.resize(std::size_t(input.size) * actions);
        ggml_backend_tensor_get(head->output, output.actions.data(), 0, ggml_nbytes(head->output));
        LAYA_CHECK(check(output.actions));
        output.compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        return output;
    }

    [[nodiscard]] const std::string& backend_name() const { return dev.name; }
    [[nodiscard]] const std::string& device_name() const { return dev.description; }

private:
    engine(const checkpoint& model, device&& opened, resident_weights&& weights)
        : dev(std::move(opened)), store(std::move(weights)), arch(model.arch),
          actions(model.serving.actions), max_length(model.serving.max_len) {}

    device dev;
    resident_weights store;
    architecture arch;
    int actions, max_length;
    std::optional<std::filesystem::path> trace_directory;
    std::optional<encoder_graph> encoder;
    std::optional<action_graph> head;
    // Host staging reused across calls.
    std::vector<std::int32_t> types, cls;
    std::vector<float> pooled, features, probabilities;
    std::vector<float> global_mask, local_mask;
    std::vector<ggml_fp16_t> global_half, local_half;

    ggml_backend_t backend() const { return dev.backend.get(); }

    template<class Values>
    static void put(tensor* t, const Values& values) {
        ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
    }

    result<void> validate(const batch& input) const {
        if (input.size < 1 || input.length < 1 || input.length > max_length || input.options < 2 ||
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
        if (!std::ranges::all_of(input.ids, [&](int id) { return id >= 0 && id < arch.vocabulary; }))
            return fail(errc::backend, "Token ID outside the vocabulary");
        return {};
    }

    // Vulkan reads host attention masks: 0 where a query may attend a key and
    // -inf elsewhere, including the padded query rows of fused attention.
    template<class T>
    static void fill_masks(const encoder_graph& g, const batch& input, std::vector<T>& global, std::vector<T>& local,
                           T open, T closed) {
        const int length = input.length, rows = int(g.global_mask->ne[1]);
        const std::size_t size = std::size_t(length) * rows * input.size;
        global.assign(size, closed);
        local.assign(size, closed);
        constexpr int window = architecture::local_window;
        for (int row = 0; row < input.size; ++row) {
            const int valid = input.lengths[row];
            for (int query = 0; query < length; ++query) {
                const std::size_t base = (std::size_t(row) * rows + query) * length;
                std::fill_n(global.begin() + base, valid, open);
                const int first = std::max(0, query - window), last = std::min(valid, query + window + 1);
                if (first < last) std::fill(local.begin() + base + first, local.begin() + base + last, open);
                // Queries beyond the window of every valid key attend the first token.
                if (query >= valid + window) local[base] = open;
            }
        }
    }

    void upload_masks(const encoder_graph& g, const batch& input) {
        if constexpr (has(M, feature::flash)) {
            fill_masks(g, input, global_half, local_half, ggml_fp32_to_fp16(0.0f), ggml_fp32_to_fp16(-INFINITY));
            put(g.global_mask, global_half);
            if (g.local_mask->buffer) put(g.local_mask, local_half);
        } else {
            fill_masks(g, input, global_mask, local_mask, 0.0f, -INFINITY);
            put(g.global_mask, global_mask);
            put(g.local_mask, local_mask);
        }
    }

    result<void> check(const std::vector<float>& values) const {
        if constexpr (checked_outputs) {
            if (laya_vk_bf16_status_failed(backend())) return fail(errc::backend, "Nonfinite AMD Vulkan BF16 projection input");
            if (!std::ranges::all_of(values, [](float x) { return std::isfinite(x); }))
                return fail(errc::backend, "Nonfinite AMD Vulkan BF16 output");
        }
        return {};
    }

    // Action-head features per row: pooled state, then the top probability,
    // its margin, normalized entropy and option count of the option softmax.
    // Logits of padded options are set to -1e4 in place.
    void summarize(const batch& input, std::vector<float>& logits) {
        const int width = arch.width;
        features.resize(std::size_t(width + 4) * input.size);
        probabilities.resize(std::size_t(input.options));
        for (int row = 0; row < input.size; ++row) {
            auto* out = features.data() + std::size_t(row) * (width + 4);
            std::copy_n(pooled.data() + std::size_t(row) * width, width, out);
            auto* scores = logits.data() + std::size_t(row) * input.options;
            std::fill(scores + input.counts[row], scores + input.options, -1e4f);
            const float maximum = *std::max_element(scores, scores + input.options);
            float total = 0;
            for (int j = 0; j < input.options; ++j) total += probabilities[j] = std::exp(scores[j] - maximum);
            float entropy = 0;
            for (auto& v : probabilities) {
                v /= total;
                entropy -= v * std::log(std::max(v, 1e-9f));
            }
            std::partial_sort(probabilities.begin(), probabilities.begin() + 2, probabilities.end(), std::greater<float>());
            const int count = std::max(2, input.counts[row]);
            out[width] = probabilities[0];
            out[width + 1] = probabilities[0] - probabilities[1];
            out[width + 2] = entropy / std::log(float(count));
            out[width + 3] = float(count) / 255.0f;
        }
    }

    void write_traces(const encoder_graph& g) const {
        std::error_code ignored;
        std::filesystem::create_directories(*trace_directory, ignored);
        std::vector<float> values;
        std::vector<ggml_bf16_t> packed;
        for (const auto& [name, t] : g.traces) {
            values.resize(std::size_t(ggml_nelements(t)));
            if (t->type == GGML_TYPE_BF16) {
                packed.resize(values.size());
                ggml_backend_tensor_get(t, packed.data(), 0, ggml_nbytes(t));
                ggml_bf16_to_fp32_row(packed.data(), values.data(), std::int64_t(values.size()));
            } else {
                ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
            }
            std::ofstream file(*trace_directory / (name + ".f32"), std::ios::binary);
            file.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
        }
    }
};

#if LAYA_COREML
// Core ML runs compiled model buckets; the exported model owns precision and
// accelerator decisions.
template<mode M>
    requires(has(M, feature::coreml))
class engine<M> {
public:
    [[nodiscard]] static result<engine> load(const checkpoint& model, device&& opened) {
        LAYA_TRY(runtime, coreml_runtime::load(model));
        return engine(std::move(*runtime), std::move(opened));
    }
    [[nodiscard]] result<raw_result> forward(const batch& input) { return runtime.forward(input); }
    [[nodiscard]] const std::string& backend_name() const { return dev.name; }
    [[nodiscard]] const std::string& device_name() const { return dev.description; }

private:
    engine(coreml_runtime&& runtime, device&& opened) : runtime(std::move(runtime)), dev(std::move(opened)) {}
    coreml_runtime runtime;
    device dev;
};
#endif
}
