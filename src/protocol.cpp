#include "laya/runtime.hpp"
#include <algorithm>
#include <cmath>

namespace laya {
namespace {
// Python's json.dumps layout: ", " and ": " separators.
std::string dump_python(const json& value, bool ascii = false) {
    if (!value.is_structured()) return value.dump(-1, ' ', ascii, json::error_handler_t::replace);
    std::string text = value.is_object() ? "{" : "[";
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (it != value.begin()) text += ", ";
        if (value.is_object()) text += json(it.key()).dump(-1, ' ', ascii, json::error_handler_t::replace) + ": ";
        text += dump_python(it.value(), ascii);
    }
    return text + (value.is_object() ? "}" : "]");
}
std::string render(const json& value) { return value.is_string() ? value.get<std::string>() : dump_python(value); }
// Request text cannot spell the mask token that marks options.
std::string clean(std::string text, const std::string& mask) {
    for (std::size_t at = 0; (at = text.find(mask, at)) != std::string::npos; ++at) text.replace(at, mask.size(), " ");
    return text;
}
bool empty(const json& value) { return value.is_null() || (value.is_string() && value.get_ref<const std::string&>().empty()); }
double rounded(double x) { return std::nearbyint(x * 10000.0) / 10000.0; }
std::unexpected<error> rejected(std::string message) { return fail(errc::invalid, std::move(message)); }

// Question options as text, with criteria normalized for the answer.
result<std::vector<std::string>> options(int type, json& criteria) {
    std::vector<std::string> list;
    if (type == 0) {  // choice: list or object of named options
        if (criteria.is_array()) {
            auto mapping = json::object();
            for (const auto& value : criteria) {
                LAYA_TRY(name, json_string(value));
                mapping[*name] = nullptr;
            }
            criteria = std::move(mapping);
        }
        if (!criteria.is_object()) return rejected("Choice criteria must be a list or object");
        for (const auto& [name, value] : criteria.items()) list.push_back(empty(value) ? name : name + ": " + render(value));
    } else if (type == 1) {  // score: ordered levels
        if (!criteria.is_array()) return rejected("Score criteria must be an array");
        for (std::size_t i = 0; i < criteria.size(); ++i) list.push_back("level " + std::to_string(i) + ": " + render(criteria[i]));
    } else {  // noul: a statement that holds or not
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

result<codec> codec::load(const checkpoint& model, overflow policy) {
    LAYA_TRY(words, tokenizer::load(model.directory / "tokenizer/tokenizer.json"));
    LAYA_TRY(settings, read_json(model.directory / "tokenizer/tokenizer_config.json"));
    codec loaded{std::move(*words), {}, 0, 0, 0, 0, model.serving.max_len, model.serving.head_max_len, policy};
    // A special token is a string or an added-token object with its content.
    for (auto [key, id] : {std::pair{"mask", &loaded.mask}, {"cls", &loaded.cls}, {"sep", &loaded.sep}, {"pad", &loaded.pad}}) {
        const json& value = field(*settings, std::string(key) + "_token");
        const json& content = value.is_string() ? value : field(value, "content");
        if (!content.is_string()) return fail(errc::model, std::string("Missing tokenizer setting: ") + key + "_token");
        const auto found = loaded.words.id(content.get_ref<const std::string&>());
        if (!found) return fail(errc::model, "Tokenizer lacks special token: " + content.get<std::string>());
        *id = *found;
        if (id == &loaded.mask) loaded.mask_text = content.get<std::string>();
    }
    return loaded;
}

result<prepared> codec::prepare(const json& requests) const {
    return policy == overflow::truncate ? build<overflow::truncate>(requests) : build<overflow::reject>(requests);
}

// Each question becomes one row: [CLS] heading [SEP] options [SEP] state [SEP],
// within the token budgets of the checkpoint; Policy decides what exceeds them.
template<overflow Policy> result<prepared> codec::build(const json& requests) const {
    constexpr bool reject = Policy == overflow::reject;
    if (!requests.is_array() || requests.empty()) return rejected("requests must be a nonempty array");
    auto encode = [&](std::string text) { return words.encode(clean(std::move(text), mask_text)); };
    prepared out;
    out.requests = requests.size();
    batch& input = out.input;
    std::vector<tokens> sequences, positions;
    int request_index = 0;
    for (const auto& request : requests) {
        LAYA_TRY(state_value, json_at(request, "state"));
        LAYA_TRY(state, encode(render(**state_value)));
        LAYA_TRY(questions, json_at(request, "questions"));
        if (!(*questions)->is_object() || (*questions)->empty()) return rejected("questions must be a nonempty object");
        for (const auto& [id, definition] : (*questions)->items()) {
            LAYA_TRY(type_value, json_at(definition, "type"));
            LAYA_TRY(type, json_string(**type_value));
            const auto known = std::ranges::find(question_types, *type);
            if (known == question_types.end()) return rejected("Unsupported question type: " + *type);
            const int kind = int(known - question_types.begin());
            json criteria = field(definition, "criteria");
            LAYA_TRY(texts, options(kind, criteria));
            LAYA_TRY(instruction, json_at(definition, "instructions"));
            LAYA_TRY(heading, encode(*type + " question: " + ((*instruction)->is_string() ? (*instruction)->get<std::string>()
                                                                                          : dump_python(**instruction, true))));
            const int count = int(texts->size());
            const auto exceeds = [&](const char* what, int n, const char* unit = " tokens)") {
                return rejected("Question '" + id + "' exceeds " + what + " (" + std::to_string(n) + unit);
            };
            std::vector<tokens> encoded;
            int used = 0;
            for (const auto& option : *texts) {
                LAYA_TRY(ids, encode(" " + option));
                if (ids->size() > 48) {
                    if constexpr (reject) return exceeds("option token limit", 48, ")");
                    ids->resize(48);
                }
                ids->insert(ids->begin(), mask);
                used += int(ids->size());
                encoded.push_back(std::move(*ids));
            }
            int remaining = budget - used;
            if (remaining < 16) {  // options share the budget that leaves 16 heading tokens
                const int per = std::max(4, (budget - 16) / count);
                used = 0;
                for (auto& ids : encoded) {
                    if (int(ids.size()) > per) {
                        if constexpr (reject) return exceeds("option token budget", per, " tokens per option)");
                        ids.resize(std::size_t(per));
                    }
                    used += int(ids.size());
                }
                remaining = budget - used;
            }
            if (const int heading_limit = std::max(8, remaining); int(heading->size()) > heading_limit) {
                if constexpr (reject) return exceeds("heading token budget", heading_limit);
                heading->resize(std::size_t(heading_limit));
            }
            if (reject && used + int(heading->size()) > budget) return exceeds("question head token budget", budget);
            tokens ids{cls}, markers;
            ids.insert(ids.end(), heading->begin(), heading->end());
            ids.push_back(sep);
            for (const auto& option : encoded) {
                markers.push_back(token(ids.size()));
                ids.insert(ids.end(), option.begin(), option.end());
            }
            ids.push_back(sep);
            const auto room = std::size_t(std::max(0, limit - int(ids.size()) - 1));
            if (reject && state->size() > room) return exceeds("state context limit", int(room));
            ids.insert(ids.end(), state->begin(), state->begin() + std::ptrdiff_t(std::min(room, state->size())));
            ids.push_back(sep);
            if (int(ids.size()) > limit) {
                if constexpr (reject) return exceeds("final sequence limit", limit);
                ids.resize(std::size_t(limit));
            }
            if (markers.back() >= limit) return exceeds("final sequence limit", limit);
            input.length = std::max(input.length, int(ids.size()));
            input.options = std::max(input.options, count);
            input.lengths.push_back(token(ids.size()));
            input.counts.push_back(count);
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

json inputs(const batch& input) {
    return {{"batch", input.size}, {"length", input.length}, {"options", input.options}, {"ids", input.ids},
            {"lengths", input.lengths}, {"markers", input.markers}, {"counts", input.counts}, {"types", input.types}};
}

result<agent> agent::load(const std::filesystem::path& directory, mode requested, overflow policy, padding rows) {
    LAYA_TRY(engine, runtime::load(directory, requested, rows));
    LAYA_TRY(format, codec::load(engine->model(), policy));
    return agent{std::move(*engine), std::move(*format)};
}

result<json> agent::prepare(const json& requests) const {
    LAYA_TRY(questions, format.prepare(requests));
    return inputs(questions->input);
}

result<json> agent::raw(const json& requests) {
    LAYA_TRY(questions, format.prepare(requests));
    LAYA_TRY(output, engine.forward(questions->input));
    return json{{"inputs", inputs(questions->input)}, {"logits", output->logits}, {"actions", output->actions},
                {"action_count", output->action_count}, {"lanes", output->lanes}, {"compute_ms", output->compute_ms}};
}

// Calibrated answers: per-type softmax temperature over each question's option logits.
result<json> agent::predict(const json& requests) {
    LAYA_TRY(questions, format.prepare(requests));
    LAYA_TRY(output, engine.forward(questions->input));
    const batch& input = questions->input;
    const auto& serving = engine.model().serving;
    auto responses = json::array();
    for (std::size_t i = 0; i < questions->requests; ++i)
        responses.push_back({{"model", "laya-rl-agent"}, {"answers", json::object()}, {"usage", {{"input_tokens", 0}, {"output_tokens", 0}}}});
    std::vector<float> probs;
    for (int row = 0; row < input.size; ++row) {
        const auto& question = questions->rows[std::size_t(row)];
        const int count = input.counts[row], kind = input.types[row];
        const auto temperature = float(serving.temperature[kind][count <= 2 ? 0 : count <= 5 ? 1 : count <= 10 ? 2 : 3]);
        probs.assign(output->logits.begin() + std::ptrdiff_t(row) * input.options, output->logits.begin() + std::ptrdiff_t(row) * input.options + count);
        for (auto& v : probs) v /= temperature;
        const float maximum = *std::ranges::max_element(probs);
        float total = 0, entropy = 0;
        double score = 0;
        for (auto& v : probs) total += v = std::exp(v - maximum);
        for (int k = 0; k < count; ++k) {
            probs[k] /= total;
            entropy -= probs[k] * std::log(std::max(probs[k], 1e-12f));
            score += k * double(probs[k]);
        }
        const float* action = output->actions.data() + std::size_t(row) * output->action_count;
        const float max_action = *std::max_element(action, action + output->action_count);
        float action_sum = 0;
        for (int k = 0; k < output->action_count; ++k) action_sum += std::exp(action[k] - max_action);
        json answer = {{"type", question_types[kind]}, {"confidence", rounded(std::clamp(1.0 - double(entropy) / std::log(double(count)), 0.0, 1.0))},
                       {"action", {{"act_probability", rounded(std::exp(action[0] - max_action) / action_sum)}}}};
        auto probabilities = json::object();
        if (kind == 0) {
            const auto selected = std::ranges::max_element(probs) - probs.begin();
            int k = 0;
            for (const auto& [name, unused] : question.criteria.items()) {
                probabilities[name] = rounded(probs[k]);
                if (k++ == selected) answer["choice"] = name;
            }
            answer["probabilities"] = std::move(probabilities);
        } else if (kind == 1) {
            answer["score"] = rounded(score);
            auto legend = json::object();
            for (int k = 0; k < count; ++k) {
                probabilities[std::to_string(k)] = rounded(probs[k]);
                legend[std::to_string(k)] = question.criteria[std::size_t(k)];
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
