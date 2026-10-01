#pragma once
#include "laya/runtime.hpp"
#include <chrono>

namespace laya {
struct http_options {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string variant = "english";
    std::string backend;
    std::string api_key;
    std::size_t max_questions = 8;
    std::size_t max_batch_questions = 0;   // 0 uses max_questions.
    std::size_t max_pending_requests = 32; // Includes active inference.
    unsigned batch_wait_ms = 2;
    bool batching = true;
    std::size_t max_body_bytes = 1024 * 1024;
};

// HTTP admission and batching. Handler threads validate calls and queue them;
// one inference thread pulls batches of whole calls and settles them, so one
// model and tokenizer may safely back the server.
class http_server {
public:
    [[nodiscard]] static result<http_server> create(http_options options);
    [[nodiscard]] int bind(); // port 0 selects an available port; -1 on failure
    bool listen();            // blocks until stop()
    [[nodiscard]] bool running() const;
    void stop();
    // Runs `predict` on every batch until the server stops. `predict` maps a
    // request array to result<json> holding one result per request.
    template<class Predict> void serve(Predict&& predict) {
        while (const json* requests = next()) {
            const auto start = std::chrono::steady_clock::now();
            auto results = predict(*requests);
            settle(std::move(results), std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
    }
    // Binds, prints the endpoint and listens on a background thread until
    // SIGINT, SIGTERM or stop(); join() restores the signal handlers.
    [[nodiscard]] result<void> start();
    [[nodiscard]] result<void> join();
    struct impl;

private:
    struct release { void operator()(impl*) const; };
    std::unique_ptr<impl, release> p;
    // Blocks until a batch is ready; null once the server stops.
    const json* next();
    // Answers each call of the batch with its slice of `results`. Late preprocessing
    // errors must not reject unrelated callers, so each call of the batch runs again alone.
    void settle(result<json> results, double elapsed_ms);
};

// Serves until SIGINT or SIGTERM. Inference runs on the calling thread.
template<class Predict> [[nodiscard]] result<void> serve_http(http_options options, Predict&& predict) {
    LAYA_TRY(server, http_server::create(std::move(options)));
    LAYA_CHECK(server->start());
    server->serve(predict);
    return server->join();
}
}
