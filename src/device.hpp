#pragma once
#include "laya/checkpoint.hpp"
#include "laya/error.hpp"
#include "laya/mode.hpp"
#include "laya/weights.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <memory>
#include <string>

namespace laya {
template<auto Free>
struct release {
    template<class T> void operator()(T* handle) const noexcept { Free(handle); }
};
using context_handle = std::unique_ptr<ggml_context, release<ggml_free>>;
using backend_handle = std::unique_ptr<ggml_backend, release<ggml_backend_free>>;
using buffer_handle = std::unique_ptr<ggml_backend_buffer, release<ggml_backend_buffer_free>>;
using allocator_handle = std::unique_ptr<ggml_gallocr, release<ggml_gallocr_free>>;

// An initialized compute device. `resolved` is the requested mode plus the
// vendor bits that select device-matched Vulkan mixed-precision numerics.
struct device {
    backend_handle backend;
    mode resolved = 0;
    std::string name, description;
};
[[nodiscard]] result<device> open_device(mode requested);

// How checkpoint tensors are stored on the device, by role.
struct storage {
    ggml_type trunk = GGML_TYPE_F32, projection = GGML_TYPE_F32;
    ggml_type bias = GGML_TYPE_F32;      // biases hold values rounded through this type
    bool f16_checkpoint = false;         // trunk matrices must already be F16 in the checkpoint
    bool amd_bf16_range = false;         // BF16 matrices must convert exactly in AMD Vulkan kernels
    ggml_type gelu = GGML_TYPE_COUNT;    // Vulkan mixed-precision GELU lookup values, if any
    bool rocm_gelu = false;              // apply the ROCm GELU table patches
};

// Device-resident checkpoint tensors.
struct resident_weights {
    context_handle context;
    buffer_handle buffer;
    weights tensors;
};
[[nodiscard]] result<resident_weights> load_weights(const checkpoint& model, ggml_backend_t backend, const storage& layout);
}
