#pragma once
#include "engine.hpp"
#include "laya/protocol.hpp"
#include <utility>
#include <variant>

namespace laya {
// A served checkpoint: the engine of mode M and the request codec of its
// tokenizer and token budget policy. Callers serialize predict().
template<mode M, text_tokenizer Tokenizer, overflow Overflow>
class agent {
public:
    agent(engine<M>&& model, codec<Tokenizer, Overflow>&& requests, const serving_config& serving)
        : model(std::move(model)), requests(std::move(requests)), serving(serving) {}

    // Calibrated answers per request, or model inputs and outputs when Raw.
    template<bool Raw = false>
    [[nodiscard]] result<json> predict(const json& batch) {
        LAYA_TRY(questions, requests.prepare(batch));
        LAYA_TRY(output, model.forward(questions->input));
        if constexpr (Raw) return raw_outputs(questions->input, *output);
        else return answers(*questions, *output, serving);
    }
    // The model inputs the requests produce, without running the model.
    [[nodiscard]] result<json> prepare(const json& batch) const {
        LAYA_TRY(questions, requests.prepare(batch));
        return inputs(questions->input);
    }
    [[nodiscard]] const std::string& backend_name() const { return model.backend_name(); }
    [[nodiscard]] const std::string& device_name() const { return model.device_name(); }

private:
    engine<M> model;
    codec<Tokenizer, Overflow> requests;
    serving_config serving;
};
}

namespace laya {
// Loads the engine of mode M on `opened` and the checkpoint tokenizer, then
// returns f(agent) for the agent of that tokenizer and Overflow.
template<mode M, overflow Overflow, class F>
[[nodiscard]] result<void> with_agent(const checkpoint& model, device&& opened, F&& f) {
    LAYA_TRY(loaded, engine<M>::load(model, std::move(opened)));
    LAYA_TRY(tokenizer, load_tokenizer(model.directory / "tokenizer/tokenizer.json"));
    return std::visit([&]<text_tokenizer Tokenizer>(Tokenizer& text) -> result<void> {
        LAYA_TRY(requests, (codec<Tokenizer, Overflow>::load(std::move(text), model.directory / "tokenizer/tokenizer_config.json",
                                                             model.serving)));
        agent<M, Tokenizer, Overflow> laya(std::move(*loaded), std::move(*requests), model.serving);
        return f(laya);
    }, *tokenizer);
}
}
