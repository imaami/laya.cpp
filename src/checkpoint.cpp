#include "laya/checkpoint.hpp"
#include <algorithm>
#include <string>

namespace laya {
result<serving_config> serving_config::parse(const json& document) {
    auto integer = [&](std::string_view key, int fallback) {
        const json& value = field(document, key);
        return value.is_null() ? fallback : value.is_number_integer() ? value.get<int>() : 0;
    };
    serving_config config;
    config.variant = field(document, "model_name") == "laya-typed-decisions" ? model_variant::typed_decisions
                   : field(document, "encoder") == "jhu-clsp/mmBERT-base" ? model_variant::multilingual : model_variant::english;
    config.max_len = integer("max_len", 512);
    config.head_max_len = integer("head_max_len", 192);
    if ((config.max_len != 512 && config.max_len != 1024) || config.head_max_len < 1 || config.head_max_len >= config.max_len)
        return fail(errc::model, "Unsupported serving sequence limits");
    const json& costs = field(document, "act_costs");
    if (!costs.is_array()) return fail(errc::model, "Missing act_costs");
    config.actions = static_cast<int>(costs.size()) + 1;
    const json& base = field(document, "temperature");
    const json& by_options = field(document, "temperature_by_options");
    constexpr const char* types[] = {"choice", "score", "noul"};
    constexpr const char* buckets[] = {"2", "3-5", "6-10", "11+"};
    for (int type = 0; type < 3; ++type)
        for (int bucket = 0; bucket < 4; ++bucket) {
            json value = field(by_options, std::string(types[type]) + ":" + buckets[bucket]);
            if (value.is_null()) value = base.is_null() ? json(1) : base.is_array() && base.size() == 3 ? base[type] : json();
            if (!value.is_number()) return fail(errc::model, "Unsupported temperature configuration");
            config.temperature[type][bucket] = std::max(1e-3, value.get<double>());
        }
    return config;
}

result<checkpoint> checkpoint::load(const std::filesystem::path& directory) {
    LAYA_TRY(config, read_json(directory / "rl_agent_config.json"));
    LAYA_TRY(encoder, read_json(directory / "encoder/config.json"));
    auto integer = [&](std::string_view key) {
        const json& value = field(*encoder, key);
        return value.is_number_integer() ? value.get<int>() : 0;
    };
    architecture arch{integer("hidden_size"), integer("num_attention_heads"), integer("num_hidden_layers"),
                      integer("intermediate_size"), integer("vocab_size"), 0};
    arch.local_rope = arch.width == multilingual_encoder.width ? multilingual_encoder.local_rope : large_encoder.local_rope;
    if (arch != large_encoder && arch != multilingual_encoder) return fail(errc::model, "Unsupported encoder architecture");
    static const json supported{{"model_type", "modernbert"}, {"local_attention", 128}, {"global_attn_every_n_layers", 3},
            {"norm_bias", false}, {"attention_bias", false}, {"mlp_bias", false}, {"hidden_activation", "gelu"}};
    for (const auto& [key, expected] : supported.items())
        if (field(*encoder, key) != expected) return fail(errc::model, "Unsupported encoder field: " + key);
    if (field(*config, "head_layers") != 2 || field(*config, "amp_dtype") != "bf16")
        return fail(errc::model, "Expected two head layers and BF16 model configuration");
    if (const json& epsilon = field(*encoder, "norm_eps"); !epsilon.is_null() && epsilon != 1e-5)
        return fail(errc::model, "Unsupported normalization epsilon");
    LAYA_TRY(serving, serving_config::parse(*config));
    const json& types = field(*encoder, "layer_types");
    for (int i = 0; i < arch.layers; ++i)
        if (!types.is_array() || i >= int(types.size()) || types[i] != (global_layer(i) ? "full_attention" : "sliding_attention"))
            return fail(errc::model, "Unsupported attention schedule");
    const json& rope = field(*encoder, "rope_parameters");
    for (auto [kind, base] : {std::pair{"full_attention", double(architecture::global_rope)}, std::pair{"sliding_attention", double(arch.local_rope)}})
        if (field(field(rope, kind), "rope_type") != "default" || field(field(rope, kind), "rope_theta") != base)
            return fail(errc::model, "Unsupported rotary configuration");
    return checkpoint{directory, std::move(*serving), arch};
}
}
