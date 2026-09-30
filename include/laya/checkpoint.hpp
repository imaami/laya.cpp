#pragma once
#include "laya/error.hpp"
#include "laya/json.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <string_view>

namespace laya {
// ModernBERT encoder geometries served by Laya.
struct architecture {
    int width, heads, layers, intermediate, vocabulary;
    float local_rope;  // rotary base of sliding-window layers
    static constexpr float global_rope = 160000.f;
    static constexpr int head_width = 64;
    static constexpr int local_window = 64;  // each side of a sliding-window query
    friend constexpr bool operator==(const architecture&, const architecture&) = default;
};
inline constexpr architecture large_encoder{1024, 16, 28, 2624, 50368, 10000.f};
inline constexpr architecture multilingual_encoder{768, 12, 22, 1152, 256000, 160000.f};
// Every third encoder layer attends globally; the rest use a sliding window.
[[nodiscard]] constexpr bool global_layer(int layer) noexcept { return layer % 3 == 0; }

enum class model_variant : std::uint8_t { english, multilingual, typed_decisions };
[[nodiscard]] constexpr std::string_view name(model_variant variant) noexcept {
    return variant == model_variant::english ? "english" : variant == model_variant::multilingual ? "multilingual" : "typed-decisions";
}

// Serving parameters from rl_agent_config.json.
struct serving_config {
    model_variant variant = model_variant::english;
    int max_len = 512, head_max_len = 192;
    int actions = 0;  // act_costs plus the implicit "answer" action
    // Softmax temperature (at least 0.001) by question type (choice, score,
    // noul) and option count (2, 3-5, 6-10, 11+).
    std::array<std::array<double, 4>, 3> temperature{};
    [[nodiscard]] static result<serving_config> parse(const json& document);
};

// A validated checkpoint description, without tensor data.
struct checkpoint {
    std::filesystem::path directory;
    serving_config serving;
    architecture arch;
    [[nodiscard]] static result<checkpoint> load(const std::filesystem::path& directory);
};
}
