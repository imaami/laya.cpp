#pragma once
#include <nlohmann/json.hpp>
#include <array>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace laya {
// Laya builds without exceptions: fallible operations return result<T>. Parse, invalid
// and too_large errors are attributable to one request and never to the server.
enum class errc : std::uint8_t { parse, invalid, too_large, unavailable, unsupported, io, model, backend };
struct error { errc code; std::string message; };
template<class T = void> using result = std::expected<T, error>;
inline std::unexpected<error> fail(errc code, std::string message) { return std::unexpected(error{code, std::move(message)}); }
constexpr bool input_error(errc code) { return code <= errc::too_large; }
// Declares `name` as the value of a result expression, or returns its error.
#define LAYA_TRY(name, ...) auto name = (__VA_ARGS__); if (!name) return std::unexpected(std::move(name).error())
#define LAYA_CHECK(...) do { if (auto laya_status_ = (__VA_ARGS__); !laya_status_) return std::unexpected(std::move(laya_status_).error()); } while (false)

// Object order is protocol: it orders choices and answers.
using json = nlohmann::ordered_json;
[[nodiscard]] result<json> parse_json(std::string_view text);  // syntax errors carry nlohmann's message
[[nodiscard]] result<json> read_json(const std::filesystem::path& path);
std::string dump(const json& value);  // invalid UTF-8 becomes U+FFFD
const json& field(const json& object, std::string_view key);  // null when either is missing
// Checked access, failing with the messages of nlohmann's at() and get<std::string>().
[[nodiscard]] result<const json*> json_at(const json& object, std::string_view key);
[[nodiscard]] result<std::string> json_string(const json& value);

// A mode is one word of feature bits, selected once at startup.
using mode = unsigned;
namespace feature {
enum : mode {
    cpu = 1u << 0, cuda = 1u << 1, vulkan = 1u << 2, coreml = 1u << 3,
    fp16 = 1u << 4, bf16 = 1u << 5,   // mixed precision; neither bit means strict FP32
    flash = 1u << 6,                  // fused attention
    compensated = 1u << 7,            // FP32 products from split FP16 operands
    nvidia = 1u << 8, amd = 1u << 9,  // device numerics matched by Vulkan mixed precision
    low = fp16 | bf16,
};
}

// ModernBERT encoder geometry.
struct architecture {
    int width, heads, layers, intermediate, vocabulary;
    float local_rope;  // rotary base of sliding-window layers
    friend constexpr bool operator==(const architecture&, const architecture&) = default;
};
inline constexpr std::array<std::string_view, 3> question_types{"choice", "score", "noul"};  // by batch type ID
// Serving parameters from rl_agent_config.json.
struct serving_config {
    std::string_view variant;  // english, multilingual or typed-decisions
    int max_len, head_max_len;
    int actions;  // act_costs plus the implicit "answer" action
    // Softmax temperature (at least 0.001) by question type and option count (2, 3-5, 6-10, 11+).
    std::array<std::array<double, 4>, 3> temperature;
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
    std::optional<token> id(std::string_view text) const;
    struct impl;
private:
    struct release { void operator()(impl*) const; };
    std::unique_ptr<impl, release> p;
};

// Each batch row answers one question of one request.
struct prepared {
    struct row { int request; std::string id; json criteria; };  // criteria: object (choice, noul) or array (score)
    batch input;
    std::vector<row> rows;
    std::size_t requests = 0;
};
enum class overflow : bool { reject, truncate };  // treatment of questions beyond a token budget
// Converts typed-decision requests to model batches with the checkpoint's tokenizer and token budgets.
struct codec {
    tokenizer words;
    std::string mask_text;
    token mask, cls, sep, pad;
    int limit, budget;
    overflow policy;
    [[nodiscard]] static result<codec> load(const checkpoint& model, overflow policy);
    [[nodiscard]] result<prepared> prepare(const json& requests) const;
    template<overflow Policy> [[nodiscard]] result<prepared> build(const json& requests) const;  // prepare() under Policy
};
json inputs(const batch& input);  // as --prepare prints them

// Model execution in the mode chosen at load. Callers serialize forward().
class runtime {
public:
    [[nodiscard]] static result<runtime> load(const std::filesystem::path& directory, mode requested);
    [[nodiscard]] result<raw_result> forward(const batch& input);
    const checkpoint& model() const;
    const std::string& backend_name() const;
    const std::string& device_name() const;
    struct impl;
private:
    struct release { void operator()(impl*) const; };
    std::unique_ptr<impl, release> p;
};

// A served checkpoint. Callers serialize predict() and raw().
struct agent {
    runtime engine;
    codec format;
    [[nodiscard]] static result<agent> load(const std::filesystem::path& directory, mode requested, overflow policy);
    [[nodiscard]] result<json> predict(const json& requests);        // calibrated answers per request
    [[nodiscard]] result<json> raw(const json& requests);            // model inputs and outputs
    [[nodiscard]] result<json> prepare(const json& requests) const;  // model inputs, without running the model
    const std::string& backend_name() const { return engine.backend_name(); }
    const std::string& device_name() const { return engine.device_name(); }
};
}
