#pragma once
#include "laya/error.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace laya {
// Documents keep object insertion order: choice order and answer layout are part of the protocol.
using json = nlohmann::ordered_json;
[[nodiscard]] result<json> parse_json(std::string_view text);  // syntax errors carry nlohmann's message
[[nodiscard]] result<json> read_json(const std::filesystem::path& path);
[[nodiscard]] std::string dump(const json& value);  // invalid UTF-8 becomes U+FFFD
[[nodiscard]] const json& field(const json& object, std::string_view key);  // null when either is missing
// Checked access with the messages nlohmann reports for at() and get<std::string>().
namespace json_access {
[[nodiscard]] result<const json*> at(const json& object, std::string_view key);
[[nodiscard]] result<std::string> string(const json& value);
}

// A mode is one word of feature bits, selected once at startup; every feature
// test inside the engine is a compile-time mask compare.
using mode = unsigned;
namespace feature {
enum : mode {
    cpu = 1u << 0, cuda = 1u << 1, vulkan = 1u << 2, coreml = 1u << 3,
    fp16 = 1u << 4, bf16 = 1u << 5,   // mixed precision; neither bit means strict FP32
    flash = 1u << 6,                  // fused attention
    compensated = 1u << 7,            // FP32 products from split FP16 operands
    nvidia = 1u << 8, amd = 1u << 9,  // device numerics matched by Vulkan mixed precision
    backend = cpu | cuda | vulkan | coreml, low = fp16 | bf16,
};
}
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

// ModernBERT encoder geometries served by Laya.
struct architecture {
    int width, heads, layers, intermediate, vocabulary;
    float local_rope;  // rotary base of sliding-window layers
    static constexpr float global_rope = 160000.f;
    static constexpr int local_window = 64;  // each side of a sliding-window query
    friend constexpr bool operator==(const architecture&, const architecture&) = default;
};
inline constexpr architecture large_encoder{1024, 16, 28, 2624, 50368, 10000.f};
inline constexpr architecture multilingual_encoder{768, 12, 22, 1152, 256000, 160000.f};
[[nodiscard]] constexpr bool global_layer(int layer) noexcept { return layer % 3 == 0; }  // others slide

enum class model_variant : std::uint8_t { english, multilingual, typed_decisions };
[[nodiscard]] constexpr std::string_view name(model_variant variant) noexcept {
    return variant == model_variant::english ? "english" : variant == model_variant::multilingual ? "multilingual" : "typed-decisions";
}
// Serving parameters from rl_agent_config.json.
struct serving_config {
    model_variant variant = model_variant::english;
    int max_len = 512, head_max_len = 192;
    int actions = 0;  // act_costs plus the implicit "answer" action
    // Softmax temperature (at least 0.001) by question type and option count (2, 3-5, 6-10, 11+).
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

// One model batch: a row per question, padded to the longest row.
struct batch {
    int size = 0, length = 0, options = 0;
    std::vector<std::int32_t> ids, lengths, markers, counts, types;
};
// Logits per option marker and action-head outputs per row.
struct raw_result {
    std::vector<float> logits, actions;
    int action_count = 0;
    double compute_ms = 0;
};

using token = std::int32_t;
using tokens = std::vector<token>;
// Byte-pair encoding of a tokenizer.json: the byte-level family of the English
// models or the metaspace family of the multilingual model.
class tokenizer {
public:
    [[nodiscard]] static result<tokenizer> load(const std::filesystem::path& file);
    [[nodiscard]] result<tokens> encode(std::string_view text) const;
    [[nodiscard]] std::optional<token> id(std::string_view text) const;
    tokenizer(tokenizer&&) noexcept;
    tokenizer& operator=(tokenizer&&) noexcept;
    ~tokenizer();

private:
    struct impl;
    explicit tokenizer(std::unique_ptr<impl> state);
    std::unique_ptr<impl> p;
};

inline constexpr std::array<std::string_view, 3> question_types{"choice", "score", "noul"};  // by batch type ID
// Each batch row answers one question of one request.
struct prepared {
    struct row {
        int request;
        std::string id;
        json criteria;  // choice: object, score: array, noul: object
    };
    batch input;
    std::vector<row> rows;
    std::size_t requests = 0;
};
// How preparation treats a question beyond a model token budget.
enum class overflow : bool { reject, truncate };
// Converts typed-decision requests to model batches with the checkpoint's tokenizer and token budgets.
class codec {
public:
    [[nodiscard]] static result<codec> load(const checkpoint& model, overflow policy);
    [[nodiscard]] result<prepared> prepare(const json& requests) const;

private:
    codec(tokenizer&& words, const serving_config& serving, overflow policy);
    template<overflow Policy> result<prepared> build(const json& requests) const;
    tokenizer words;
    std::string mask_text;
    token mask = 0, cls = 0, sep = 0, pad = 0;
    int limit, budget;
    overflow policy;
};
[[nodiscard]] json answers(const prepared& questions, const raw_result& output, const serving_config& serving);
[[nodiscard]] json inputs(const batch& input);                              // as --prepare prints them
[[nodiscard]] json raw_outputs(const batch& input, const raw_result& output);  // as --raw prints them

// Model execution in the mode chosen at load. Callers serialize forward().
class runtime {
public:
    [[nodiscard]] static result<runtime> load(const std::filesystem::path& directory, mode requested);
    [[nodiscard]] result<raw_result> forward(const batch& input);
    [[nodiscard]] const checkpoint& model() const;
    [[nodiscard]] const std::string& backend_name() const;
    [[nodiscard]] const std::string& device_name() const;
    runtime(runtime&&) noexcept;
    runtime& operator=(runtime&&) noexcept;
    ~runtime();

private:
    struct impl;
    explicit runtime(std::unique_ptr<impl> state);
    std::unique_ptr<impl> p;
};

// A served checkpoint. Callers serialize predict() and raw().
class agent {
public:
    [[nodiscard]] static result<agent> load(const std::filesystem::path& directory, mode requested, overflow policy);
    [[nodiscard]] result<json> predict(const json& requests);    // calibrated answers per request
    [[nodiscard]] result<json> raw(const json& requests);        // model inputs and outputs
    [[nodiscard]] result<json> prepare(const json& requests) const;  // model inputs, without running the model
    [[nodiscard]] const checkpoint& model() const { return engine.model(); }
    [[nodiscard]] const std::string& backend_name() const { return engine.backend_name(); }
    [[nodiscard]] const std::string& device_name() const { return engine.device_name(); }

private:
    agent(runtime&& engine, codec&& format) : engine(std::move(engine)), format(std::move(format)) {}
    runtime engine;
    codec format;
};
}
