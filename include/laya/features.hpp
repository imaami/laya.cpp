#pragma once
#include "laya/error.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace laya {
// Runtime feature bits. A process selects exactly one combination at startup;
// every code path below dispatch is instantiated for that combination.
enum class feature : std::uint16_t {
    none = 0,
    // Execution backend: exactly one.
    cpu = 1u << 0,
    cuda = 1u << 1,
    vulkan = 1u << 2,
    coreml = 1u << 3,
    // Arithmetic contract: exactly one.
    fp32 = 1u << 4,
    fp16 = 1u << 5,
    bf16 = 1u << 6,
    // Kernel options.
    flash = 1u << 7,        // fused scaled-dot-product attention
    compensated = 1u << 8,  // split-FP16 Tensor Core products with FP32 compensation
    // Vulkan vendor numerics for mixed precision: at most one.
    nvidia = 1u << 9,
    amd = 1u << 10,
};

[[nodiscard]] constexpr feature operator|(feature a, feature b) noexcept {
    return feature(std::to_underlying(a) | std::to_underlying(b));
}
[[nodiscard]] constexpr feature operator&(feature a, feature b) noexcept {
    return feature(std::to_underlying(a) & std::to_underlying(b));
}

// A feature set. Structural, so modes are usable as template arguments.
struct features {
    std::uint16_t bits = 0;

    static constexpr std::uint16_t backends = std::to_underlying(feature::cpu | feature::cuda | feature::vulkan | feature::coreml);
    static constexpr std::uint16_t precisions = std::to_underlying(feature::fp32 | feature::fp16 | feature::bf16);
    static constexpr std::uint16_t vendors = std::to_underlying(feature::nvidia | feature::amd);
    static constexpr std::uint16_t all = (1u << 11) - 1;

    constexpr features() = default;
    constexpr features(feature value) noexcept : bits(std::to_underlying(value)) {}

    [[nodiscard]] constexpr bool has(feature value) const noexcept {
        return (bits & std::to_underlying(value)) == std::to_underlying(value);
    }
    [[nodiscard]] constexpr bool any(feature value) const noexcept { return (bits & std::to_underlying(value)) != 0; }
    [[nodiscard]] constexpr feature backend() const noexcept { return feature(bits & backends); }
    [[nodiscard]] constexpr feature precision() const noexcept { return feature(bits & precisions); }
    [[nodiscard]] constexpr feature vendor() const noexcept { return feature(bits & vendors); }
    [[nodiscard]] constexpr bool low() const noexcept { return any(feature::fp16 | feature::bf16); }
    [[nodiscard]] constexpr features with(feature value) const noexcept { return features(feature(bits | std::to_underlying(value))); }
    [[nodiscard]] constexpr features without(feature value) const noexcept { return features(feature(bits & ~std::to_underlying(value))); }

    // The complete support matrix. Keep these rules and their messages in
    // agreement with first_violation().
    [[nodiscard]] constexpr bool valid() const noexcept {
        if ((bits & ~all) || std::popcount(unsigned(bits & backends)) != 1 || std::popcount(unsigned(bits & precisions)) != 1 ||
            std::popcount(unsigned(bits & vendors)) > 1)
            return false;
        return !first_violation();
    }

    // The first unsupported combination, in the order users are told about it.
    [[nodiscard]] constexpr const char* first_violation() const noexcept {
        if (has(feature::coreml) && (!has(feature::fp32) || any(feature::flash | feature::compensated)))
            return "Core ML selects precision and kernels from its compiled model; use --coreml with --fp32 only";
        if (has(feature::fp16) && !has(feature::vulkan)) return "FP16 currently requires Vulkan";
        if (has(feature::vulkan) && has(feature::fp32) && has(feature::flash))
            return "Vulkan currently supports FP32 without fused attention";
        if (low() && (has(feature::cpu) || has(feature::coreml) || (has(feature::cuda) && !has(feature::flash))))
            return "BF16 mode requires a GPU; CUDA requires fused attention";
        if (has(feature::compensated) && (low() || any(feature::cpu | feature::coreml)))
            return "Compensated matrix operations require a GPU and FP32 mode";
        if (vendor() != feature::none && !(has(feature::vulkan) && low()))
            return "Vendor numerics apply only to mixed-precision Vulkan";
        return nullptr;
    }

    friend constexpr bool operator==(features, features) = default;
};

[[nodiscard]] constexpr features operator|(features a, feature b) noexcept { return a.with(b); }

// Every valid mode, derived from the rules above.
inline constexpr auto all_modes = [] {
    constexpr auto count = [] {
        std::size_t n = 0;
        for (unsigned bits = 0; bits <= features::all; ++bits) n += features(feature(bits)).valid();
        return n;
    }();
    std::array<features, count> modes{};
    std::size_t n = 0;
    for (unsigned bits = 0; bits <= features::all; ++bits)
        if (features(feature(bits)).valid()) modes[n++] = features(feature(bits));
    return modes;
}();
// CPU 2, CUDA 5, Vulkan 2 FP32 + 12 mixed (two formats, fused or explicit
// attention, three vendor contracts), Core ML 1.
static_assert(all_modes.size() == 22);

// Backends compiled into this binary.
inline constexpr feature compiled_backends = feature::cpu
#ifdef LAYA_CUDA
    | feature::cuda
#endif
#ifdef LAYA_VULKAN
    | feature::vulkan
#endif
#ifdef LAYA_COREML
    | feature::coreml
#endif
    ;

// The modes this binary can execute.
inline constexpr auto enabled_modes = [] {
    constexpr auto count = std::ranges::count_if(all_modes, [](features m) { return features(compiled_backends).any(m.backend()); });
    std::array<features, count> modes{};
    std::ranges::copy_if(all_modes, modes.begin(), [](features m) { return features(compiled_backends).any(m.backend()); });
    return modes;
}();

// Mode categories used to select implementations.
template<features M> concept strict_mode = M.has(feature::fp32) && !M.has(feature::compensated) && !M.has(feature::coreml);
template<features M> concept compensated_mode = M.has(feature::compensated);
template<features M> concept mixed_mode = M.low();
template<features M> concept ggml_mode = !M.has(feature::coreml);

// Human-readable mode name, e.g. "vulkan-bf16-flash-amd".
[[nodiscard]] inline std::string describe(features mode) {
    constexpr std::pair<feature, const char*> names[]{
        {feature::cpu, "cpu"}, {feature::cuda, "cuda"}, {feature::vulkan, "vulkan"}, {feature::coreml, "coreml"},
        {feature::fp32, "fp32"}, {feature::fp16, "fp16"}, {feature::bf16, "bf16"}, {feature::flash, "flash"},
        {feature::compensated, "compensated"}, {feature::nvidia, "nvidia"}, {feature::amd, "amd"}};
    std::string text;
    for (auto [bit, name] : names)
        if (mode.has(bit)) text += (text.empty() ? "" : "-") + std::string(name);
    return text.empty() ? "none" : text;
}

// Invokes visitor.template operator()<M>() for the mode equal to `mode`.
// This is the single point where a runtime selection becomes a type.
template<class Visitor>
[[nodiscard]] auto dispatch(features mode, Visitor&& visitor) {
    using value = decltype(visitor.template operator()<enabled_modes[0]>());
    return [&]<std::size_t... I>(std::index_sequence<I...>) -> value {
        value selected = fail(errc::unsupported, "Unsupported execution mode: " + describe(mode));
        (void)((mode == enabled_modes[I] && (selected = visitor.template operator()<enabled_modes[I]>(), true)) || ...);
        return selected;
    }(std::make_index_sequence<enabled_modes.size()>{});
}
}
