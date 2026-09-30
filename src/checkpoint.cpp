#include "laya/checkpoint.hpp"
#include <array>
#include <utility>

namespace laya {
namespace {
// Configuration problems are checkpoint errors regardless of which accessor saw them.
std::unexpected<error> model_failure(error failure) {
    failure.code = errc::model;
    return std::unexpected(std::move(failure));
}

// json.at(key) != expected, where a missing key or a non-object is an error.
result<bool> differs(const json& object, std::string_view key, const json& expected) {
    auto found = json_access::at(object, key);
    if (!found) return std::unexpected(std::move(found.error()));
    return **found != expected;
}

// The encoder must alternate one global layer with two sliding-window layers.
result<void> check_schedule(const json& encoder, int layers) {
    auto types = json_access::at(encoder, "layer_types");
    if (!types) return std::unexpected(std::move(types.error()));
    for (int i = 0; i < layers; ++i) {
        auto type = json_access::element(**types, static_cast<std::size_t>(i));
        if (!type) return std::unexpected(std::move(type.error()));
        if (**type != (global_layer(i) ? "full_attention" : "sliding_attention"))
            return fail(errc::model, "Unsupported attention schedule");
    }
    return {};
}

result<void> check_rotary(const json& encoder, float local_rope) {
    auto parameters = json_access::at(encoder, "rope_parameters");
    if (!parameters) return std::unexpected(std::move(parameters.error()));
    for (auto [kind, base] : {std::pair{"full_attention", double(architecture::global_rope)},
                              std::pair{"sliding_attention", double(local_rope)}}) {
        auto rope = json_access::at(**parameters, kind);
        if (!rope) return std::unexpected(std::move(rope.error()));
        auto type = differs(**rope, "rope_type", "default");
        if (!type) return std::unexpected(std::move(type.error()));
        if (*type) return fail(errc::model, "Unsupported rotary configuration");
        auto theta = differs(**rope, "rope_theta", base);
        if (!theta) return std::unexpected(std::move(theta.error()));
        if (*theta) return fail(errc::model, "Unsupported rotary configuration");
    }
    return {};
}
}

result<model_variant> detect_variant(const json& config) {
    auto model = json_access::string_or(config, "model_name", "");
    if (!model) return std::unexpected(std::move(model.error()));
    if (*model == "laya-typed-decisions") return model_variant::typed_decisions;
    auto encoder = json_access::string_or(config, "encoder", "");
    if (!encoder) return std::unexpected(std::move(encoder.error()));
    return *encoder == "jhu-clsp/mmBERT-base" ? model_variant::multilingual : model_variant::english;
}

result<serving_config> serving_config::parse(json document) {
    serving_config config;
    auto limit = json_access::integer_or(document, "max_len", 512);
    if (!limit) return model_failure(std::move(limit.error()));
    auto budget = json_access::integer_or(document, "head_max_len", 192);
    if (!budget) return model_failure(std::move(budget.error()));
    if ((*limit != 512 && *limit != 1024) || *budget < 1 || *budget >= *limit)
        return fail(errc::model, "Unsupported serving sequence limits");
    auto costs = json_access::at(document, "act_costs");
    if (!costs) return model_failure(std::move(costs.error()));
    config.max_len = *limit;
    config.head_max_len = *budget;
    config.actions = static_cast<int>((*costs)->size()) + 1;
    config.document = std::move(document);
    return config;
}

result<checkpoint> checkpoint::load(const std::filesystem::path& directory) {
    auto config = read_json(directory / "rl_agent_config.json");
    if (!config) return std::unexpected(std::move(config.error()));
    auto encoder = read_json(directory / "encoder/config.json");
    if (!encoder) return std::unexpected(std::move(encoder.error()));

    architecture arch{};
    static constexpr std::array fields{
        std::pair{"hidden_size", &architecture::width}, std::pair{"num_attention_heads", &architecture::heads},
        std::pair{"num_hidden_layers", &architecture::layers}, std::pair{"intermediate_size", &architecture::intermediate},
        std::pair{"vocab_size", &architecture::vocabulary}};
    for (auto [key, field] : fields) {
        auto value = json_access::at(*encoder, key).and_then([](const json* found) { return json_access::integer(*found); });
        if (!value) return model_failure(std::move(value.error()));
        arch.*field = *value;
    }
    bool known = false;
    for (const auto& candidate : {large_encoder, multilingual_encoder}) {
        arch.local_rope = candidate.local_rope;
        if ((known = arch == candidate)) break;
    }
    if (!known) return fail(errc::model, "Unsupported encoder architecture");

    static const json supported{{"model_type", "modernbert"}, {"local_attention", 128}, {"global_attn_every_n_layers", 3},
            {"norm_bias", false}, {"attention_bias", false}, {"mlp_bias", false}, {"hidden_activation", "gelu"}};
    for (const auto& [key, expected] : supported.items())
        if (!encoder->contains(key) || (*encoder)[key] != expected)
            return fail(errc::model, "Unsupported encoder field: " + key);

    auto head_layers = differs(*config, "head_layers", decision_head_layers);
    if (!head_layers) return model_failure(std::move(head_layers.error()));
    auto amp_dtype = *head_layers ? result<bool>(true) : differs(*config, "amp_dtype", "bf16");
    if (!amp_dtype) return model_failure(std::move(amp_dtype.error()));
    if (*amp_dtype) return fail(errc::model, "Expected two head layers and BF16 model configuration");

    auto epsilon = json_access::number_or(*encoder, "norm_eps", 1e-5);
    if (!epsilon) return model_failure(std::move(epsilon.error()));
    if (*epsilon != 1e-5) return fail(errc::model, "Unsupported normalization epsilon");

    auto serving = serving_config::parse(std::move(*config));
    if (!serving) return std::unexpected(std::move(serving.error()));
    if (auto schedule = check_schedule(*encoder, arch.layers); !schedule) return model_failure(std::move(schedule.error()));
    if (auto rotary = check_rotary(*encoder, arch.local_rope); !rotary) return model_failure(std::move(rotary.error()));
    return checkpoint{directory, std::move(*serving), arch};
}
}
