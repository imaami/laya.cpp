#include "check.hpp"
#include "laya/checkpoint.hpp"
#include "laya/protocol.hpp"
#include <algorithm>
#include <iostream>
#include <variant>

namespace {
using laya::json;
using laya::test::expect;
using laya::test::fail;
using laya::test::require;

json choice_question(std::string instruction, json criteria = json::array({"A", "B"})) {
    return {{"type", "choice"}, {"instructions", std::move(instruction)}, {"criteria", std::move(criteria)}};
}

json request(json state, json questions) {
    return json::array({json{{"state", std::move(state)}, {"questions", std::move(questions)}}});
}

// The model inputs of a request, as --prepare prints them.
template<class Codec>
json prepare(const Codec& codec, const json& input) {
    return laya::inputs(expect(codec.prepare(input)).input);
}

template<class Codec>
void rejects(const Codec& codec, const json& input, const std::string& fragment) {
    auto outcome = codec.prepare(input);
    if (outcome) fail("strict preparation accepted input expected to be rejected: " + fragment);
    const auto& message = outcome.error().message;
    require(laya::input_error(outcome.error().code), "rejection was not an input error: " + message);
    require(message.find(fragment) != std::string::npos,
            "expected error containing '" + fragment + "', got '" + message + "'");
}

std::string repeated(const std::string& phrase, int count) {
    std::string result;
    for (int i = 0; i < count; ++i) {
        if (i) result += ' ';
        result += phrase;
    }
    return result;
}

template<class Tokenizer>
std::size_t encoded_size(const Tokenizer& tokenizer, const std::string& text) {
    return expect(tokenizer.encode(text)).size();
}

template<class Tokenizer>
std::string value_for_option_tokens(const Tokenizer& tokenizer, const std::string& label,
                                    const std::string& word, int target) {
    for (int count = 1; count <= target * 3; ++count) {
        const auto candidate = repeated(word, count);
        if (encoded_size(tokenizer, " " + label + ": " + candidate) == static_cast<size_t>(target)) return candidate;
    }
    fail("could not construct an option with the requested exact token count");
}

template<class Tokenizer>
std::string phrase_with_exact_tokens(const Tokenizer& tokenizer, const std::string& word, int target) {
    for (int count = 1; count <= target * 3; ++count) {
        const auto candidate = repeated(word, count);
        if (encoded_size(tokenizer, " " + candidate) == static_cast<size_t>(target)) return candidate;
    }
    fail("could not construct text with the requested exact token count");
}

int option_span(const json& prepared, size_t index, int sep_id) {
    const auto start = prepared["markers"][index].get<int>();
    int end;
    if (index + 1 < prepared["counts"][0].get<std::size_t>()) {
        end = prepared["markers"][index + 1].get<int>();
    } else {
        const auto& ids = prepared["ids"];
        end = start;
        while (end < prepared["lengths"][0].get<int>() && ids[std::size_t(end)].get<int>() != sep_id) ++end;
    }
    return end - start;
}

int redistribution_count(int head_budget) {
    for (int count = 2; count <= 255; ++count) {
        const int per = std::max(4, (head_budget - 16) / count);
        if (per < 47 && head_budget - count * (per + 2) < 16) return count;
    }
    fail("could not choose an option count that triggers redistribution");
}

int exact_fit_count(int head_budget) {
    for (int count = 2; count <= 255; ++count) {
        const int encoded_tokens = (head_budget - 16) / count - 1;
        if ((head_budget - 16) % count == 0 && encoded_tokens > 0 && encoded_tokens <= 47) return count;
    }
    return 0;
}

template<laya::text_tokenizer Tokenizer>
void check(const laya::checkpoint& model, const Tokenizer& tokenizer) {
    const int max_len = model.serving.max_len;
    const int head_budget = model.serving.head_max_len;
    const auto config_path = model.directory / "tokenizer/tokenizer_config.json";
    // Each codec owns its tokenizer.
    auto reload = [&] { return std::get<Tokenizer>(expect(laya::load_tokenizer(model.directory / "tokenizer/tokenizer.json"))); };
    const auto strict = expect(laya::codec<Tokenizer, laya::overflow::reject>::load(reload(), config_path, model.serving));
    const auto compatibility = expect(laya::codec<Tokenizer, laya::overflow::truncate>::load(reload(), config_path, model.serving));
    const auto tokenizer_config = expect(laya::read_json(config_path));
    const auto& sep_setting = laya::field(tokenizer_config, "sep_token");
    const auto sep_text = expect(laya::json_access::string(sep_setting.is_string() ? sep_setting : laya::field(sep_setting, "content")));
    const auto sep = tokenizer.id(sep_text);
    require(sep.has_value(), "separator token is not in the vocabulary");
    const int sep_id = *sep;

    // Structured state/instructions and multiple questions remain byte-for-byte
    // identical when no shortening is needed.
    const auto ordinary = request(json{{"status", "ready"}, {"count", 3}}, json{
        {"first", choice_question("Choose the best option", json::array({"one", "two"}))},
        {"second", {{"type", "score"}, {"instructions", json{{"goal", "rate"}}},
                    {"criteria", json::array({"low", "high"})}}}
    });
    require(prepare(strict, ordinary) == prepare(compatibility, ordinary),
            "strict preparation changed an input that fits all limits");

    // Exactly filling the option budget is accepted; shortening starts only
    // when the remaining budget falls below 16 tokens.
    const int exact_count = exact_fit_count(head_budget);
    if (exact_count > 0) {
        const int option_tokens = (head_budget - 16) / exact_count - 1; // one mask token per option
        json exact_options = json::object();
        for (int i = 0; i < exact_count; ++i) {
            const auto label = "exact-" + std::to_string(i);
            exact_options[label] = value_for_option_tokens(tokenizer, label, "word", option_tokens);
        }
        const auto exact_fit = request("state", json{{"q", choice_question("Choose", exact_options)}});
        require(prepare(strict, exact_fit) == prepare(compatibility, exact_fit),
                "strict preparation rejected or changed an exact-fit option budget");
    }

    // Each case reaches a distinct legacy shortening point. Verify that strict
    // mode rejects it and that the explicit compatibility mode still prepares it.
    auto too_long_option = phrase_with_exact_tokens(tokenizer, "option", 50);
    auto long_option = request("state", json{{"q", choice_question("Choose", json::array({too_long_option, "other"}))}});
    rejects(strict, long_option, "option token limit");
    const json legacy_option = prepare(compatibility, long_option);
    require(option_span(legacy_option, 0, sep_id) == 49,
            "compatibility mode did not preserve the 48-token option cap plus mask token");

    json many_options = json::object();
    const int option_count = redistribution_count(head_budget);
    const int per_option = std::max(4, (head_budget - 16) / option_count);
    for (int i = 0; i < option_count; ++i) {
        const auto label = "option-" + std::to_string(i);
        many_options[label] = value_for_option_tokens(tokenizer, label, "word", per_option + 1);
    }
    auto redistribution = request("state", json{{"q", choice_question("Choose", many_options)}});
    rejects(strict, redistribution, "option token budget");
    const json legacy_redistribution = prepare(compatibility, redistribution);
    for (int i = 0; i < option_count; ++i) {
        const int actual = option_span(legacy_redistribution, i, sep_id);
        require(actual == per_option,
                "compatibility mode option " + std::to_string(i) + " span was " +
                std::to_string(actual) + ", expected redistributed span " + std::to_string(per_option));
    }

    auto long_heading = request("state", json{{"q", choice_question(repeated("instruction", 500))}});
    rejects(strict, long_heading, "heading token budget");
    const json legacy_heading = prepare(compatibility, long_heading);
    int option_used = 0;
    for (const auto& option : {std::string("A"), std::string("B")})
        option_used += 1 + static_cast<int>(encoded_size(tokenizer, " " + option));
    int remaining = head_budget - option_used;
    if (remaining < 16) {
        const int per = std::max(4, (head_budget - 16) / 2);
        option_used = 2 * per;
        remaining = head_budget - option_used;
    }
    const int expected_heading_tokens = std::max(8, remaining);
    require(legacy_heading["markers"][0].get<int>() == expected_heading_tokens + 2,
            "compatibility mode did not preserve legacy heading truncation");

    auto long_state = request(repeated("context", max_len * 3), json{{"q", choice_question("Choose")}});
    rejects(strict, long_state, "state context limit");
    const json legacy_state = prepare(compatibility, long_state);
    require(legacy_state["lengths"][0].get<int>() == max_len,
            "compatibility mode did not preserve legacy max_len state truncation");

    // A large head layout can exceed max_len before state is appended. This
    // final guard is conditional because current published variants generally
    // keep head_max_len well below max_len.
    if (head_budget > max_len) {
        json options = json::object();
        for (int i = 0; i < 5; ++i) options["option-" + std::to_string(i)] = repeated("word", 30);
        auto final_cap = request("", json{{"q", choice_question(repeated("instruction", head_budget), options)}});
        auto outcome = strict.prepare(final_cap);
        if (outcome) fail("strict preparation accepted an input beyond max_len");
        const auto& message = outcome.error().message;
        require(laya::input_error(outcome.error().code) &&
                (message.find("final sequence limit") != std::string::npos ||
                 message.find("heading token budget") != std::string::npos ||
                 message.find("option token budget") != std::string::npos),
                "unexpected rejection before final sequence limit");
    }
}
}

int main(int argc, char** argv) {
    if (argc != 2) fail("usage: test-request-preparation MODEL_DIR");
    const auto model = expect(laya::checkpoint::load(argv[1]));
    const auto tokenizer = expect(laya::load_tokenizer(model.directory / "tokenizer/tokenizer.json"));
    std::visit([&](const auto& text) { check(model, text); }, tokenizer);
    std::cout << "request preparation truncation cases passed\n";
}
