#pragma once
#include "laya/error.hpp"
#include "laya/json.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
    struct call;

public:
    // Calls combined into one model batch.
    struct work {
        json requests = json::array();  // the requests of every call, in queue order
        std::uint64_t id = 0;
        std::vector<std::shared_ptr<call>> calls;
    };

    [[nodiscard]] static result<http_server> create(http_options options);
    http_server(http_server&&) noexcept;
    http_server& operator=(http_server&&) noexcept;
    ~http_server();

    [[nodiscard]] int bind(); // port 0 selects an available port; -1 on failure
    bool listen();            // blocks until stop()
    [[nodiscard]] bool running() const;
    void stop();

    // Runs `predict` on every batch until the server stops. `predict` maps a
    // request array to result<json> holding one result per request.
    template<class Predict>
    void serve(Predict&& predict) {
        while (auto batch = next()) {
            const auto calls = batch->calls.size();
            execute(*batch, predict);
            finish(calls);
        }
    }

    // Binds, prints the endpoint and listens on a background thread until
    // SIGINT, SIGTERM or stop(); join() restores the signal handlers.
    [[nodiscard]] result<void> start();
    [[nodiscard]] result<void> join();

private:
    struct impl;
    explicit http_server(std::unique_ptr<impl> state);

    // Blocks until a batch is ready; empty once the server stops.
    std::optional<work> next();
    // Answers each call of `batch` with its slice of `results`.
    void complete(work& batch, result<json> results, double elapsed_ms);
    // One batch per call, each with a new batch identity.
    std::vector<work> split(work&& batch);
    void finish(std::size_t calls);

    template<class Predict>
    void execute(work& batch, Predict& predict) {
        const auto start = std::chrono::steady_clock::now();
        result<json> results = predict(std::as_const(batch.requests));
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        // Late preprocessing errors must not reject unrelated callers.
        if (!results && input_error(results.error().code) && batch.calls.size() > 1) {
            for (auto& single : split(std::move(batch))) execute(single, predict);
            return;
        }
        complete(batch, std::move(results), elapsed);
    }

    std::unique_ptr<impl> p;
};

// Serves until SIGINT or SIGTERM. Inference runs on the calling thread.
template<class Predict>
[[nodiscard]] result<void> serve_http(http_options options, Predict&& predict) {
    LAYA_TRY(server, http_server::create(std::move(options)));
    LAYA_CHECK(server->start());
    server->serve(predict);
    return server->join();
}
}
