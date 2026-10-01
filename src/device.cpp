#include "device.hpp"
#include "laya/json.hpp"
#include "vulkan/gelu_rocm_patches.hpp"
#include "vulkan/gelu_tables.hpp"
#include "ggml-cpu.h"
#if LAYA_CUDA
#include "ggml-cuda.h"
const char* laya_cuda_bf16_compatibility_error();
#endif
#if LAYA_VULKAN
#include "ggml-vulkan.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ranges>
#include <span>
#include <unordered_map>
#include <vector>

namespace laya {
namespace {
using uint = std::uint32_t;
#include "vulkan/bf16_range.glsl"

ggml_backend_t initialize(mode requested) {
    switch (requested & feature::backend) {
    case feature::cpu: return ggml_backend_cpu_init();
#if LAYA_CUDA
    case feature::cuda: return ggml_backend_cuda_init(0);
#endif
#if LAYA_VULKAN
    case feature::vulkan: return ggml_backend_vk_init(0);
#endif
    default: return nullptr;
    }
}

// Rounds a value through a 16-bit storage type.
float round_through(ggml_type type, float value) {
    return type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(value)) : ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
}
}

result<device> open_device(mode requested) {
    using namespace feature;
    // Core ML schedules its compiled model itself; there is no ggml backend.
    if (has(requested, coreml)) return device{nullptr, requested, "coreml", "Core ML (all compute units)"};
    if (has(requested, cuda)) {
#if LAYA_CUDA
        // cuBLAS enables TF32 by default; strict FP32 withdraws that permission
        // before CUDA initializes.
        if (!(requested & low)) {
#ifdef _WIN32
            const bool failed = _putenv_s("NVIDIA_TF32_OVERRIDE", "0") != 0;
#else
            const bool failed = setenv("NVIDIA_TF32_OVERRIDE", "0", 1) != 0;
#endif
            if (failed) return fail(errc::backend, "Cannot enforce FP32 CUDA arithmetic");
        }
#else
        return fail(errc::unsupported, "This build has no CUDA backend");
#endif
    }
#if !LAYA_VULKAN
    if (has(requested, vulkan)) return fail(errc::unsupported, "This build has no Vulkan backend");
#endif
    backend_handle backend(initialize(requested));
#if LAYA_CUDA
    if (backend && has(requested, cuda) && (requested & low))
        if (const char* reason = laya_cuda_bf16_compatibility_error()) return fail(errc::unsupported, reason);
#endif
    if (!backend) return fail(errc::backend, "Cannot initialize requested backend");
    device opened{std::move(backend), requested, {}, {}};
    opened.name = ggml_backend_name(opened.backend.get());
    opened.description = ggml_backend_dev_description(ggml_backend_get_device(opened.backend.get()));
    if (has(requested, vulkan) && (requested & low)) {
        if (opened.description.find("AMD") != std::string::npos) opened.resolved |= amd;
        else if (opened.description.find("NVIDIA") != std::string::npos) opened.resolved |= nvidia;
    }
    return opened;
}

result<resident_weights> load_weights(const checkpoint& model, ggml_backend_t backend, const storage& layout) {
    std::ifstream file(model.directory / "model.safetensors", std::ios::binary | std::ios::ate);
    if (!file) return fail(errc::io, "Cannot open model.safetensors");
    const auto file_size = std::uint64_t(file.tellg());
    file.seekg(0);
    std::uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!file || header_size > 16 * 1024 * 1024 || header_size + 8 > file_size)
        return fail(errc::model, "Invalid safetensors header size");
    std::string text(header_size, '\0');
    file.read(text.data(), std::streamsize(text.size()));
    auto header = parse_json(text);
    if (!header) return fail(errc::model, std::move(header.error().message));

    resident_weights loaded;
    const auto entries = loaded.tensors.entries(model.arch, model.serving.actions);
    std::unordered_map<std::string_view, const weights::entry*> schema;
    for (const auto& entry : entries) {
        schema.emplace(entry.name, &entry);
        if (field(field(*header, entry.name), "shape") != json(entry.shape))
            return fail(errc::model, "Missing or incorrectly shaped checkpoint tensor: " + entry.name);
    }
    for (const auto& [name, spec] : header->items())
        if (name != "__metadata__" && !schema.contains(name)) return fail(errc::model, "Unexpected checkpoint tensor: " + name);

    loaded.context.reset(ggml_init({(entries.size() + 1) * ggml_tensor_overhead() + 1024, nullptr, true}));
    if (!loaded.context) return fail(errc::backend, "Cannot allocate weight metadata");
    auto* context = loaded.context.get();
    const auto type_of = [&](role kind) {
        return kind == role::trunk ? layout.trunk : kind == role::projection ? layout.projection : GGML_TYPE_F32;
    };
    // Tensors are created in checkpoint order.
    for (const auto& [name, spec] : header->items()) {
        if (name == "__metadata__") continue;
        const auto& entry = *schema.find(name)->second;
        if (layout.f16_checkpoint && entry.kind == role::trunk && field(spec, "dtype") != "F16")
            return fail(errc::model, "Compensated Tensor Core mode requires F16 stored projections");
        std::vector<std::int64_t> shape(entry.shape.rbegin(), entry.shape.rend());
        auto* tensor = ggml_new_tensor(context, type_of(entry.kind), int(shape.size()), shape.data());
        ggml_set_name(tensor, name.c_str());
        *entry.slot = tensor;
    }
    if (layout.gelu != GGML_TYPE_COUNT) loaded.tensors.gelu_table = ggml_new_tensor_1d(context, GGML_TYPE_F32, 65536);
    loaded.buffer.reset(ggml_backend_alloc_ctx_tensors(context, backend));
    if (!loaded.buffer) return fail(errc::backend, "Insufficient device memory for model weights");
    ggml_backend_buffer_set_usage(loaded.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Payloads are validated and uploaded in name order.
    std::vector<const weights::entry*> order;
    for (const auto& entry : entries) order.push_back(&entry);
    std::ranges::sort(order, {}, &weights::entry::name);
    const std::uint64_t payload_size = file_size - 8 - header_size;
    std::vector<char> bytes;
    std::vector<float> values;
    for (const auto* entry : order) {
        const auto& name = entry->name;
        auto* tensor = *entry->slot;
        const auto& spec = field(*header, name);
        const auto& offsets = field(spec, "data_offsets");
        const auto& dtype = field(spec, "dtype");
        const auto count = std::size_t(ggml_nelements(tensor));
        const std::size_t stride = dtype == "F16" || dtype == "BF16" ? 2 : dtype == "F32" ? 4 : 0;
        const bool valid_offsets = offsets.is_array() && offsets.size() == 2 &&
            offsets[0].is_number_unsigned() && offsets[1].is_number_unsigned();
        const auto begin = valid_offsets ? offsets[0].get<std::uint64_t>() : 0;
        const auto end = valid_offsets ? offsets[1].get<std::uint64_t>() : 0;
        if (!valid_offsets || !stride || end < begin || end > payload_size || end - begin != std::uint64_t(count) * stride)
            return fail(errc::model, "Invalid safetensors payload: " + name);
        bytes.resize(count * stride);
        file.seekg(std::streamoff(8 + header_size + begin));
        file.read(bytes.data(), std::streamsize(bytes.size()));
        if (!file) return fail(errc::model, "Truncated tensor payload: " + name);
        values.resize(count);
        if (dtype == "F16") ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(bytes.data()), values.data(), std::int64_t(count));
        else if (dtype == "BF16") ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t*>(bytes.data()), values.data(), std::int64_t(count));
        else std::memcpy(values.data(), bytes.data(), bytes.size());
        if (!std::ranges::all_of(values, [](float x) { return std::isfinite(x); }))
            return fail(errc::model, "Nonfinite checkpoint values: " + name);
        switch (tensor->type) {
        case GGML_TYPE_BF16: {
            std::vector<ggml_bf16_t> converted(count);
            ggml_fp32_to_bf16_row_ref(values.data(), converted.data(), std::int64_t(count));
            if (layout.amd_bf16_range &&
                !std::ranges::all_of(converted, [](ggml_bf16_t value) { return layaBf16ScaledFitsHalf(value.bits, 0); }))
                return fail(errc::model, "AMD Vulkan BF16 projection weights exceed exact conversion range: " + name);
            ggml_backend_tensor_set(tensor, converted.data(), 0, ggml_nbytes(tensor));
            break;
        }
        case GGML_TYPE_F16: {
            std::vector<ggml_fp16_t> converted(count);
            ggml_fp32_to_fp16_row(values.data(), converted.data(), std::int64_t(count));
            ggml_backend_tensor_set(tensor, converted.data(), 0, ggml_nbytes(tensor));
            break;
        }
        default:
            if (entry->kind == role::bias && layout.bias != GGML_TYPE_F32)
                for (auto& x : values) x = round_through(layout.bias, x);
            ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
        }
    }
    if (auto* table = loaded.tensors.gelu_table) {
        using namespace vulkan_precision;
        const bool half = layout.gelu == GGML_TYPE_F16;
        const auto widen = [half](std::uint16_t bits) {
            return half ? ggml_fp16_to_fp32(ggml_fp16_t(bits)) : ggml_bf16_to_fp32(ggml_bf16_t{bits});
        };
        const std::span<const std::uint16_t> bits = half ? std::span(gelu_fp16_nvidia) : std::span(gelu_bf16_nvidia);
        const auto patches = half ? std::span<const gelu_patch>(gelu_fp16_rocm) : std::span<const gelu_patch>(gelu_bf16_rocm);
        std::vector<float> table_values(bits.size());
        std::ranges::transform(bits, table_values.begin(), widen);
        if (layout.rocm_gelu)
            for (const auto patch : patches) table_values[patch.index] = widen(patch.value);
        ggml_backend_tensor_set(table, table_values.data(), 0, ggml_nbytes(table));
    }
    return loaded;
}
}
