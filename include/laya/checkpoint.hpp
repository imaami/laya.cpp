#pragma once
#include "laya/error.hpp"
#include "laya/json.hpp"
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
inline constexpr int max_encoder_layers = 28;
inline constexpr int decision_head_layers = 2;
// Every third encoder layer attends globally; the rest use a sliding window.
[[nodiscard]] constexpr bool global_layer(int layer) noexcept { return layer % 3 == 0; }

enum class model_variant : std::uint8_t { english, multilingual, typed_decisions };
[[nodiscard]] constexpr std::string_view name(model_variant variant) noexcept {
    return variant == model_variant::english ? "english" : variant == model_variant::multilingual ? "multilingual" : "typed-decisions";
}
// The variant a checkpoint identifies in rl_agent_config.json.
[[nodiscard]] result<model_variant> detect_variant(const json& config);

// Serving limits from rl_agent_config.json.
struct serving_config {
    json document;
    int max_len = 512, head_max_len = 192;
    int actions = 0;  // act_costs plus the implicit "answer" action
    [[nodiscard]] static result<serving_config> parse(json document);
};

// A validated ggml checkpoint description, without tensor data.
struct checkpoint {
    std::filesystem::path directory;
    serving_config serving;
    architecture arch;
    [[nodiscard]] static result<checkpoint> load(const std::filesystem::path& directory);
};
}
