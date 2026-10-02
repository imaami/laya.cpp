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

// ROCm's single-precision division (rocm_math.glsl), with fused multiply-adds
// or, as llvmpipe evaluates them, a rounded product and sum.
template<bool Fused> float rocm_divide(float numerator, float denominator) {
    const auto fma = [](float a, float b, float c) {
        if constexpr (Fused) return std::fma(a, b, c);
        volatile float product = a * b;
        return product + c;
    };
    const float estimate = 1.0f / denominator;
    const float reciprocal = fma(fma(-denominator, estimate, 1.0f), estimate, estimate);
    const float quotient = numerator * reciprocal;
    return fma(fma(-quotient, denominator, numerator), reciprocal, quotient);
}

// Fused sliding-window and global attention against the separate AMD-matched
// score, softmax and value products over the same scaled, transposed pack and
// mask: identical bits on AMD, where those products follow the ROCm order. A
// double-precision reference bounds the fused pass everywhere, and the separate
// products on AMD; elsewhere their stock kernels may round through FP16. The
// second head has zero queries and values that are powers of two: uniform
// probabilities over the open keys weigh them exactly, so on every device the
// fused sums must round as the AMD value product's key order does, whether or
// not the device fuses multiply-adds. A kind whose workgroup memory the device
// lacks is skipped.
int main() {
    using namespace laya::vulkan_precision;
    if (!ggml_backend_vk_get_device_count()) return 77;
    auto backend = ggml_backend_vk_init(0);
    if (!backend) return 1;
    const std::string device = ggml_backend_dev_description(ggml_backend_get_device(backend));
    const bool amd = device.find("AMD") != std::string::npos;
    constexpr int heads = 2;
    bool ran = false;
    for (const bool local : {true, false}) for (const auto& valid : std::vector<std::vector<int32_t>>{
             {1}, {5, 3}, {64, 1}, {65, 2}, {79, 13}, {106, 41, 106}, {127, 1}, {128, 60}, {129, 65}, {211, 140, 3},
             {256, 255}, {257, 130}, {512, 509}, {513, 1}, {787, 700}, {1024, 1}, {1024, 1021}}) {
        const std::string kind = local ? "Local" : "Global";
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
        auto fused = local ? local_attention(ctx, q, k, v, lengths) : global_attention(ctx, q, k, v, lengths);
        if (!ggml_backend_supports_op(backend, fused)) {
            std::cout << kind << " attention at length " << length << " needs " << (local ? local_attention_shared : global_attention_shared)
                      << " bytes of workgroup memory on " << device << '\n';
            ggml_free(ctx);
            continue;
        }
        auto scores = ggml_mul_mat(ctx, k, q);
        ggml_prec_set_acc(scores, GGML_PREC_F32);
        auto probabilities = ggml_soft_max_ext(ctx, scores, attention_mask(ctx, lengths, length, rows, local, true), 1.0f, 0);
        auto separate = ggml_mul_mat(ctx, v, probabilities);
        ggml_prec_set_acc(separate, GGML_PREC_F32);
        ggml_set_name(scores, "laya.amd-low-qk");
        ggml_set_name(probabilities, "laya.amd-low-softmax");
        ggml_set_name(separate, "laya.amd-low-pv");
        auto graph = ggml_new_graph(ctx);
        ggml_set_output(fused), ggml_set_output(separate);
        ggml_build_forward_expand(graph, fused), ggml_build_forward_expand(graph, separate);
        auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(allocator, graph)) laya::test::fail(kind + " attention allocation failed");
        std::vector<float> input(count), packs(ggml_nelements(packed)), expected(ggml_nelements(fused)), actual(expected.size());
        for (int i = 0; i < count; ++i) {
            const float wave = std::sin(float(i) * 0.0713f + float(length));
            const int column = i % (3 * width) / 64;  // Q, K and V of each head
            input[i] = column == 1 ? 0.0f : column == 5 ? std::ldexp(wave < 0.0f ? -1.0f : 1.0f, i % 11 - 5) : wave * 3.7f;
        }
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(lengths, valid.data(), 0, ggml_nbytes(lengths));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) laya::test::fail(kind + " attention compute failed");
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
                const bool open = local ? key < valid[b] && std::abs(query - key) <= 64 || key == 0 && query >= valid[b] + 64
                                        : key < valid[b];
                if (!open) { weights[key] = -INFINITY; continue; }
                double score = 0.0;
                for (int d = 0; d < 64; ++d) score += at(0, b, h, query, d) * at(1, b, h, key, d);
                peak = std::max(peak, weights[key] = score);
            }
            std::vector<bool> opened(length);
            for (int key = 0; key < length; ++key) opened[key] = !std::isinf(weights[key]);
            const float keys = float(std::ranges::count(opened, true));
            for (auto& w : weights) sum += w = std::isinf(w) ? 0.0 : std::exp(w - peak);
            // laya_amd_attention's key order: from key 64 for the upper 32
            // dimensions once 128 keys fill its blocks of 8.
            const int full = length / 8 * 8;
            const auto uniform = [&](int d, float weight) {
                float value = 0.0f;
                const auto add = [&](int from, int to) {
                    for (int key = from; key < to; ++key) value = std::fma(float(at(2, b, h, key, d)), opened[key] ? weight : 0.0f, value);
                };
                if (d >= 32 && full >= 128) add(64, full), add(0, 64), add(full, length);
                else add(0, length);
                return value;
            };
            for (int d = 0; d < 64; ++d) {
                double reference = 0.0;
                for (int key = 0; key < length; ++key) reference += weights[key] / sum * at(2, b, h, key, d);
                const size_t i = ((size_t(b) * heads + h) * length + query) * 64 + d;
                const auto close = [&](float value) { return std::abs(double(value) - reference) <= 1e-5 * (1.0 + std::abs(reference)); };
                if (!close(actual[i]) || amd && !close(expected[i]))
                    laya::test::fail(kind + " attention departs from attention over its mask at length " + std::to_string(length));
                const auto same = [&](float value) {
                    return std::memcmp(&actual[i], &value, sizeof(float)) == 0 || (actual[i] == 0.0f && value == 0.0f);
                };
                if ((amd && !same(expected[i])) ||
                    (h == 1 && !same(uniform(d, rocm_divide<true>(1.0f, keys))) && !same(uniform(d, rocm_divide<false>(1.0f, keys)))))
                    laya::test::fail(kind + " attention changed an AMD-matched rounding at length " + std::to_string(length));
            }
        }
        ran = true;
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return ran ? 0 : 77;
}
