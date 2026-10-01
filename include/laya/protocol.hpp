#pragma once
#include "laya/batch.hpp"
#include "laya/checkpoint.hpp"
#include "laya/tokenizer.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace laya {
// Question types by their batch type ID.
inline constexpr std::array<std::string_view, 3> question_types{"choice", "score", "noul"};

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

// Converts typed-decision requests to model batches with the checkpoint's
// tokenizer and token budgets. Token budget handling is fixed at compile time.
template<text_tokenizer Tokenizer, overflow Overflow>
class codec {
public:
    // Special tokens come from tokenizer/tokenizer_config.json.
    [[nodiscard]] static result<codec> load(Tokenizer tokenizer, const std::filesystem::path& config,
                                            const serving_config& serving);
    [[nodiscard]] result<prepared> prepare(const json& requests) const;

private:
    codec(Tokenizer tokenizer, const serving_config& serving)
        : tokenizer(std::move(tokenizer)), limit(serving.max_len), budget(serving.head_max_len) {}
    Tokenizer tokenizer;
    std::string mask_text;
    token mask = 0, cls = 0, sep = 0, pad = 0;
    int limit, budget;
};

// The native prediction array: calibrated answers per request.
[[nodiscard]] json answers(const prepared& questions, const raw_result& output, const serving_config& serving);
// The model inputs of a prepared batch, as --prepare prints them.
[[nodiscard]] json inputs(const batch& input);
// Model inputs and outputs, as --raw prints them.
[[nodiscard]] json raw_outputs(const batch& input, const raw_result& output);
}
