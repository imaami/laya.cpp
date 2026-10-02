#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "vulkan_ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "check.hpp"

// Fused sliding-window attention against the separate AMD-matched score,
// softmax and value products over the same scaled, transposed pack and local
// mask: identical bits on AMD, where those products follow the ROCm order, and
// close values elsewhere. A double-precision reference bounds both.
int main() {
    using namespace laya::vulkan_precision;
    if (!ggml_backend_vk_get_device_count()) return 77;
    auto backend = ggml_backend_vk_init(0);
    if (!backend) return 1;
    const std::string device = ggml_backend_dev_description(ggml_backend_get_device(backend));
    const bool amd = device.find("AMD") != std::string::npos;
    constexpr int heads = 2;
    bool ran = false;
    for (const auto& valid : std::vector<std::vector<int32_t>>{
             {1}, {5, 3}, {64, 1}, {65, 2}, {79, 13}, {106, 41, 106}, {127, 1}, {128, 60}, {129, 65},
             {211, 140, 3}, {787, 700}, {1024, 1}}) {
        const int batches = int(valid.size()), length = *std::ranges::max_element(valid), rows = (length + 63) / 64 * 64;
        const int width = 64 * heads, count = 3 * width * length * batches;
        auto ctx = ggml_init({32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
        auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3 * width, length * batches);
        auto lengths = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batches);
        ggml_set_input(x), ggml_set_input(lengths);
        auto packed = pack_qkv(ctx, x, nullptr, nullptr, length, batches, GGML_TYPE_F16, packed_scaled_qk | packed_transposed_v);
        const auto part = [&](int i, bool transposed) {
            const int64_t row = transposed ? length : 64, count = transposed ? 64 : length;
            return ggml_view_4d(ctx, packed, row, count, heads, batches, row * sizeof(float), packed->nb[2], packed->nb[3],
                                i * batches * packed->nb[3]);
        };
        auto q = part(0, false), k = part(1, false), v = part(2, true);
        auto fused = local_attention(ctx, q, k, v, lengths);
        if (!ggml_backend_supports_op(backend, fused)) {
            std::cout << "Local attention needs " << local_attention_shared << " bytes of workgroup memory on " << device << '\n';
            return 77;
        }
        auto scores = ggml_mul_mat(ctx, k, q);
        ggml_prec_set_acc(scores, GGML_PREC_F32);
        auto probabilities = ggml_soft_max_ext(ctx, scores, attention_mask(ctx, lengths, length, rows, true, true), 1.0f, 0);
        auto separate = ggml_mul_mat(ctx, v, probabilities);
        ggml_prec_set_acc(separate, GGML_PREC_F32);
        ggml_set_name(scores, "laya.amd-low-qk");
        ggml_set_name(probabilities, "laya.amd-low-softmax");
        ggml_set_name(separate, "laya.amd-low-pv");
        auto graph = ggml_new_graph(ctx);
        ggml_set_output(fused), ggml_set_output(separate);
        ggml_build_forward_expand(graph, fused), ggml_build_forward_expand(graph, separate);
        auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(allocator, graph)) laya::test::fail("Local attention allocation failed");
        std::vector<float> input(count), packs(ggml_nelements(packed)), expected(ggml_nelements(fused)), actual(expected.size());
        for (int i = 0; i < count; ++i) input[i] = std::sin(float(i) * 0.0713f + float(length)) * 3.7f;
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(lengths, valid.data(), 0, ggml_nbytes(lengths));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) laya::test::fail("Local attention compute failed");
        ggml_backend_tensor_get(fused, actual.data(), 0, ggml_nbytes(fused));
        ggml_backend_tensor_get(separate, expected.data(), 0, ggml_nbytes(separate));
        ggml_backend_tensor_get(packed, packs.data(), 0, ggml_nbytes(packed));
        const auto at = [&](int i, int b, int h, int t, int d) {  // the pack's Q, K and transposed V
            const size_t head = (size_t(i) * batches + b) * heads + h;
            return double(i == 2 ? packs[(head * 64 + d) * length + t] : packs[(head * length + t) * 64 + d]);
        };
        for (int b = 0; b < batches; ++b) for (int h = 0; h < heads; ++h) for (int query = 0; query < length; ++query) {
            std::vector<double> weights(length, 0.0);
            double peak = -INFINITY, sum = 0.0;
            for (int key = 0; key < length; ++key) {
                const bool open = key < valid[b] && std::abs(query - key) <= 64 || key == 0 && query >= valid[b] + 64;
                if (!open) { weights[key] = -INFINITY; continue; }
                double score = 0.0;
                for (int d = 0; d < 64; ++d) score += at(0, b, h, query, d) * at(1, b, h, key, d);
                peak = std::max(peak, weights[key] = score);
            }
            for (auto& w : weights) sum += w = std::isinf(w) ? 0.0 : std::exp(w - peak);
            for (int d = 0; d < 64; ++d) {
                double reference = 0.0;
                for (int key = 0; key < length; ++key) reference += weights[key] / sum * at(2, b, h, key, d);
                const size_t i = ((size_t(b) * heads + h) * length + query) * 64 + d;
                const auto close = [&](float value) { return std::abs(double(value) - reference) <= 1e-5 * (1.0 + std::abs(reference)); };
                if (!close(actual[i]) || !close(expected[i]))
                    laya::test::fail("Local attention departs from attention over the local window at length " + std::to_string(length));
                if (amd && std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0 && !(actual[i] == 0.0f && expected[i] == 0.0f))
                    laya::test::fail("Local attention changed an AMD-matched rounding at length " + std::to_string(length));
            }
        }
        ran = true;
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return ran ? 0 : 77;
}
