#include "laya/protocol.hpp"
#include <algorithm>
#include <cmath>

namespace laya {
namespace {
// Python's json.dumps layout: ", " and ": " separators.
std::string dump_python(const json& value, bool ascii = false) {
    if (!value.is_structured()) return value.dump(-1, ' ', ascii, json::error_handler_t::replace);
    std::string text = value.is_object() ? "{" : "[";
    bool first = true;
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (!first) text += ", ";
        first = false;
        if (value.is_object()) text += json(it.key()).dump(-1, ' ', ascii, json::error_handler_t::replace) + ": ";
        text += dump_python(it.value(), ascii);
    }
    return text + (value.is_object() ? "}" : "]");
}

std::string render(const json& value) {
    return value.is_string() ? value.get<std::string>() : dump_python(value);
}

// Request text cannot spell the mask token that marks options.
std::string clean(std::string text, const std::string& mask) {
    for (std::size_t at = 0; (at = text.find(mask, at)) != std::string::npos; ++at) text.replace(at, mask.size(), " ");
    return text;
}

bool empty(const json& value) {
    return value.is_null() || (value.is_string() && value.get_ref<const std::string&>().empty());
}

double rounded(double x) { return std::nearbyint(x * 10000.0) / 10000.0; }

std::unexpected<error> rejected(std::string message) { return fail(errc::invalid, std::move(message)); }

// Question options as text, with criteria normalized for the answer.
result<std::vector<std::string>> options(int type, json& criteria) {
    std::vector<std::string> list;
    switch (type) {
    case 0:  // choice: list or object of named options
        if (criteria.is_array()) {
            auto mapping = json::object();
            for (const auto& value : criteria) {
                LAYA_TRY(name, json_access::string(value));
                mapping[*name] = nullptr;
            }
            criteria = std::move(mapping);
        }
        if (!criteria.is_object()) return rejected("Choice criteria must be a list or object");
        for (const auto& [name, value] : criteria.items()) list.push_back(empty(value) ? name : name + ": " + render(value));
        break;
    case 1:  // score: ordered levels
        if (!criteria.is_array()) return rejected("Score criteria must be an array");
        for (std::size_t i = 0; i < criteria.size(); ++i) list.push_back("level " + std::to_string(i) + ": " + render(criteria[i]));
        break;
    default:  // noul: a statement that holds or not
        if (criteria.is_null()) criteria = json::object();
        if (!criteria.is_object()) return rejected("Boolean criteria must be an object");
        const json &no = field(criteria, "false"), &yes = field(criteria, "true");
        list = {"false: " + (empty(no) ? std::string("no, the statement does not hold") : render(no)),
                "true: " + (empty(yes) ? std::string("yes, the statement holds") : render(yes))};
    }
    if (list.size() < 2 || list.size() > 255) return rejected("Questions require 2 through 255 options");
    return list;
}
}

template<text_tokenizer Tokenizer, overflow Overflow>
result<codec<Tokenizer, Overflow>> codec<Tokenizer, Overflow>::load(Tokenizer tokenizer, const std::filesystem::path& config,
                                                                    const serving_config& serving) {
    LAYA_TRY(settings, read_json(config));
    codec loaded(std::move(tokenizer), serving);
    // A special token is a string or an added-token object with its content.
    auto special = [&](const char* key, std::string& text) -> result<token> {
        const json& value = field(*settings, std::string(key) + "_token");
        const json& content = value.is_string() ? value : field(value, "content");
        if (!content.is_string()) return fail(errc::model, std::string("Missing tokenizer setting: ") + key + "_token");
        text = content.template get<std::string>();
        const auto id = loaded.tokenizer.id(text);
        if (!id) return fail(errc::model, "Tokenizer lacks special token: " + text);
        return *id;
    };
    std::string text;
    LAYA_TRY(mask, special("mask", loaded.mask_text));
    LAYA_TRY(cls, special("cls", text));
    LAYA_TRY(sep, special("sep", text));
    LAYA_TRY(pad, special("pad", text));
    loaded.mask = *mask, loaded.cls = *cls, loaded.sep = *sep, loaded.pad = *pad;
    return loaded;
}

template<text_tokenizer Tokenizer, overflow Overflow>
result<prepared> codec<Tokenizer, Overflow>::prepare(const json& requests) const {
    constexpr bool truncate = Overflow == overflow::truncate;
    if (!requests.is_array() || requests.empty()) return rejected("requests must be a nonempty array");
    auto encode = [&](std::string text) { return tokenizer.encode(clean(std::move(text), mask_text)); };
    prepared out;
    out.requests = requests.size();
    batch& input = out.input;
    std::vector<tokens> sequences, positions;
    int request_index = 0;
    for (const auto& request : requests) {
        LAYA_TRY(state_value, json_access::at(request, "state"));
        LAYA_TRY(state, encode(render(**state_value)));
        LAYA_TRY(questions, json_access::at(request, "questions"));
        if (!(*questions)->is_object() || (*questions)->empty()) return rejected("questions must be a nonempty object");
        for (const auto& [id, definition] : (*questions)->items()) {
            LAYA_TRY(type_value, json_access::at(definition, "type"));
            LAYA_TRY(type, json_access::string(**type_value));
            const auto known = std::ranges::find(question_types, *type);
            if (known == question_types.end()) return rejected("Unsupported question type: " + *type);
            const int kind = int(known - question_types.begin());
            json criteria = field(definition, "criteria");
            LAYA_TRY(texts, options(kind, criteria));
            LAYA_TRY(instruction, json_access::at(definition, "instructions"));
            LAYA_TRY(heading_ids, encode(*type + " question: " +
                                         ((*instruction)->is_string() ? (*instruction)->template get<std::string>()
                                                                      : dump_python(**instruction, true))));
            auto& heading = *heading_ids;
            const int count = int(texts->size());
            std::vector<tokens> encoded;
            encoded.reserve(std::size_t(count));
            int used = 0;
            for (const auto& option : *texts) {
                LAYA_TRY(ids, encode(" " + option));
                if (ids->size() > 48) {
                    if constexpr (!truncate) return rejected("Question '" + id + "' exceeds option token limit (48)");
                    ids->resize(48);
                }
                ids->insert(ids->begin(), mask);
                used += int(ids->size());
                encoded.push_back(std::move(*ids));
            }
            int remaining = budget - used;
            if (remaining < 16) {
                // Options share the budget that leaves 16 heading tokens.
                const int per = std::max(4, (budget - 16) / count);
                if constexpr (!truncate)
                    for (const auto& ids : encoded)
                        if (int(ids.size()) > per)
                            return rejected("Question '" + id + "' exceeds option token budget (" + std::to_string(per) + " tokens per option)");
                used = 0;
                for (auto& ids : encoded) {
                    if (int(ids.size()) > per) ids.resize(std::size_t(per));
                    used += int(ids.size());
                }
                remaining = budget - used;
            }
            const int heading_limit = std::max(8, remaining);
            if (int(heading.size()) > heading_limit) {
                if constexpr (!truncate)
                    return rejected("Question '" + id + "' exceeds heading token budget (" + std::to_string(heading_limit) + " tokens)");
                heading.resize(std::size_t(heading_limit));
            }
            if constexpr (!truncate)
                if (used + int(heading.size()) > budget)
                    return rejected("Question '" + id + "' exceeds question head token budget (" + std::to_string(budget) + " tokens)");
            tokens ids{cls}, markers;
            ids.insert(ids.end(), heading.begin(), heading.end());
            ids.push_back(sep);
            for (const auto& option : encoded) {
                markers.push_back(token(ids.size()));
                ids.insert(ids.end(), option.begin(), option.end());
            }
            ids.push_back(sep);
            const auto room = std::size_t(std::max(0, limit - int(ids.size()) - 1));
            if constexpr (!truncate)
                if (state->size() > room)
                    return rejected("Question '" + id + "' exceeds state context limit (" + std::to_string(room) + " tokens)");
            ids.insert(ids.end(), state->begin(), state->begin() + std::ptrdiff_t(std::min(room, state->size())));
            ids.push_back(sep);
            const auto final_limit = "Question '" + id + "' exceeds final sequence limit (" + std::to_string(limit) + " tokens)";
            if (int(ids.size()) > limit) {
                if constexpr (!truncate) return rejected(final_limit);
                ids.resize(std::size_t(limit));
            }
            if (markers.back() >= limit) return rejected(final_limit);
            input.length = std::max(input.length, int(ids.size()));
            input.options = std::max(input.options, int(markers.size()));
            input.lengths.push_back(token(ids.size()));
            input.counts.push_back(token(markers.size()));
            input.types.push_back(kind);
            sequences.push_back(std::move(ids));
            positions.push_back(std::move(markers));
            out.rows.push_back({request_index, id, std::move(criteria)});
        }
        ++request_index;
    }
    input.size = int(sequences.size());
    input.ids.assign(std::size_t(input.size) * std::size_t(input.length), pad);
    input.markers.resize(std::size_t(input.size) * std::size_t(input.options));
    for (int row = 0; row < input.size; ++row) {
        std::ranges::copy(sequences[row], input.ids.begin() + std::ptrdiff_t(row) * input.length);
        for (int k = 0; k < input.options; ++k)
            input.markers[std::size_t(row) * input.options + k] = row * input.length + (k < input.counts[row] ? positions[row][k] : 0);
    }
    return out;
}

template class codec<byte_level_tokenizer, overflow::reject>;
template class codec<byte_level_tokenizer, overflow::truncate>;
template class codec<metaspace_tokenizer, overflow::reject>;
template class codec<metaspace_tokenizer, overflow::truncate>;

json inputs(const batch& input) {
    return {{"batch", input.size}, {"length", input.length}, {"options", input.options}, {"ids", input.ids},
            {"lengths", input.lengths}, {"markers", input.markers}, {"counts", input.counts}, {"types", input.types}};
}

json raw_outputs(const batch& input, const raw_result& output) {
    return {{"inputs", inputs(input)}, {"logits", output.logits}, {"actions", output.actions},
            {"action_count", output.action_count}, {"compute_ms", output.compute_ms}};
}

json answers(const prepared& questions, const raw_result& output, const serving_config& serving) {
    const batch& input = questions.input;
    auto responses = json::array();
    for (std::size_t i = 0; i < questions.requests; ++i)
        responses.push_back({{"model", "laya-rl-agent"}, {"answers", json::object()}, {"usage", {{"input_tokens", 0}, {"output_tokens", 0}}}});
    for (int row = 0; row < input.size; ++row) {
        const auto& question = questions.rows[std::size_t(row)];
        const int count = input.counts[row], kind = input.types[row];
        const int bucket = count <= 2 ? 0 : count <= 5 ? 1 : count <= 10 ? 2 : 3;
        const auto temperature = float(serving.temperature[kind][bucket]);
        std::vector<float> probs(static_cast<std::size_t>(count));
        for (int k = 0; k < count; ++k) probs[k] = output.logits[std::size_t(row) * input.options + k] / temperature;
        const float maximum = *std::ranges::max_element(probs);
        float total = 0;
        for (auto& v : probs) total += v = std::exp(v - maximum);
        float entropy = 0;
        double score = 0;
        for (int k = 0; k < count; ++k) {
            probs[k] /= total;
            entropy -= probs[k] * std::log(std::max(probs[k], 1e-12f));
            score += k * double(probs[k]);
        }
        const float* action = output.actions.data() + std::size_t(row) * output.action_count;
        const float max_action = *std::max_element(action, action + output.action_count);
        float action_sum = 0;
        for (int k = 0; k < output.action_count; ++k) action_sum += std::exp(action[k] - max_action);
        const auto type = question_types[kind];
        json answer = {{"type", type}, {"confidence", rounded(std::clamp(1.0 - double(entropy) / std::log(double(count)), 0.0, 1.0))},
                       {"action", {{"act_probability", rounded(std::exp(action[0] - max_action) / action_sum)}}}};
        const json& criteria = question.criteria;
        if (kind == 0) {
            const auto selected = std::ranges::max_element(probs) - probs.begin();
            auto probabilities = json::object();
            int k = 0;
            for (const auto& [name, unused] : criteria.items()) {
                probabilities[name] = rounded(probs[k]);
                if (k == selected) answer["choice"] = name;
                ++k;
            }
            answer["probabilities"] = std::move(probabilities);
        } else if (kind == 1) {
            answer["score"] = rounded(score);
            auto probabilities = json::object(), legend = json::object();
            for (int k = 0; k < count; ++k) {
                probabilities[std::to_string(k)] = rounded(probs[k]);
                legend[std::to_string(k)] = criteria[std::size_t(k)];
            }
            answer["probabilities"] = std::move(probabilities);
            answer["legend"] = std::move(legend);
        } else {
            answer["noul"] = rounded(probs[1]);
            answer["confidence"] = rounded(std::max(double(probs[1]), 1.0 - double(probs[1])));
        }
        auto& response = responses[std::size_t(question.request)];
        response["answers"][question.id] = std::move(answer);
        auto& used = response["usage"]["input_tokens"];
        used = used.get<int>() + input.lengths[row];
    }
    return responses;
}
}
