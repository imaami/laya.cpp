#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace laya {
// A mode is one word of feature bits. The process selects it once at startup
// and instantiates the engine for it, so every feature test during inference
// is a compile-time mask compare.
using mode = unsigned;
namespace feature {
enum : mode {
    cpu = 1u << 0, cuda = 1u << 1, vulkan = 1u << 2, coreml = 1u << 3,
    fp16 = 1u << 4, bf16 = 1u << 5,  // mixed precision; neither bit means strict FP32
    flash = 1u << 6,                 // fused attention
    compensated = 1u << 7,           // FP32 products from split FP16 operands
    nvidia = 1u << 8, amd = 1u << 9, // device numerics matched by Vulkan mixed precision
    backend = cpu | cuda | vulkan | coreml, low = fp16 | bf16,
};
}
// Every bit of `bits` is set in `m`.
[[nodiscard]] constexpr bool has(mode m, mode bits) { return (m & bits) == bits; }

// Why a requested mode cannot run, or nullptr. Vendor bits come from the device.
[[nodiscard]] constexpr const char* rejection(mode m) {
    using namespace feature;
    if (has(m, coreml) && (m & (low | flash | compensated)))
        return "Core ML selects precision and kernels from its compiled model; use --coreml with --fp32 only";
    if ((m & (fp16 | vulkan)) == fp16) return "FP16 currently requires Vulkan";
    if ((m & (vulkan | low | flash)) == (vulkan | flash)) return "Vulkan currently supports FP32 without fused attention";
    if ((m & low) && (has(m, cpu) || (m & (cuda | flash)) == cuda)) return "BF16 mode requires a GPU; CUDA requires fused attention";
    if (has(m, compensated) && (m & (low | cpu))) return "Compensated matrix operations require a GPU and FP32 mode";
    return nullptr;
}

// Every supported mode; vendor bits only refine Vulkan mixed precision.
inline constexpr auto modes = [] {
    using namespace feature;
    return std::to_array<mode>({
        cpu, cpu | flash, coreml,
        cuda, cuda | flash, cuda | compensated, cuda | compensated | flash, cuda | bf16 | flash,
        vulkan, vulkan | compensated,
        vulkan | bf16, vulkan | bf16 | flash, vulkan | fp16, vulkan | fp16 | flash,
        vulkan | bf16 | nvidia, vulkan | bf16 | flash | nvidia, vulkan | fp16 | nvidia, vulkan | fp16 | flash | nvidia,
        vulkan | bf16 | amd, vulkan | bf16 | flash | amd, vulkan | fp16 | amd, vulkan | fp16 | flash | amd,
    });
}();
static_assert(std::ranges::none_of(modes, rejection));

// Returns f.template operator()<M>() for the supported mode M == m, or `otherwise`.
// Only modes of the backends in `Built` are instantiated.
template<mode Built, class R, class F>
R dispatch(mode m, R otherwise, F&& f) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        ([&] {
            if constexpr ((modes[I] & Built) != 0)
                if (m == modes[I]) otherwise = f.template operator()<modes[I]>();
        }(), ...);
    }(std::make_index_sequence<modes.size()>{});
    return otherwise;
}
}
