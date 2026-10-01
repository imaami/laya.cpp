#include "laya/http.hpp"
#include "httplib.h"
#include <atomic>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>

namespace laya {
namespace {
void send(httplib::Response& response, int status, const json& body) {
    response.status = status;
    response.set_content(dump(body), "application/json");
}
void reject(httplib::Response& response, int status, const std::string& message) {
    send(response, status, {{"error", {{"message", message}, {"status", status}}}});
}
static_assert(std::atomic<bool>::is_always_lock_free);
std::atomic<bool> interrupted{false};
void interrupt(int) { interrupted.store(true, std::memory_order_relaxed); }
using signal_handler = void (*)(int);
}

struct http_server::impl {
    struct call {
        struct reply {
            json results;
            double elapsed;
            std::uint64_t batch;
            std::size_t offset;
        };
        json requests;
        std::size_t questions = 0;
        std::chrono::steady_clock::time_point arrived = std::chrono::steady_clock::now();
        std::promise<result<reply>> answer;
    };
    // Calls combined into one model batch.
    struct work {
        json requests = json::array();  // the requests of every call, in queue order
        std::uint64_t id = 0;
        std::vector<std::shared_ptr<call>> calls;
    };
    http_options options;
    httplib::Server server;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::shared_ptr<call>> queue;
    std::size_t pending = 0;
    bool stopping = false;
    std::uint64_t batch_id = 0;  // inference thread only, as are:
    work current;
    std::deque<work> retry;  // calls of a failed batch, each alone
    std::atomic<std::uint64_t> request_id{0};
    std::string model;
    // start() and join()
    std::thread listener, monitor;
    std::atomic<bool> monitoring{true};
    bool listened = true;
    signal_handler old_int = SIG_DFL, old_term = SIG_DFL;

    explicit impl(http_options value) : options(std::move(value)) {
        model = options.variant == "english" ? "laya" : "laya-" + options.variant;
        // Bound both active connections and queued sockets. Excess sockets close.
        server.new_task_queue = [this] { return new httplib::ThreadPool(options.max_pending_requests + 8, 32); };
        server.set_payload_max_length(options.max_body_bytes);
        server.set_tcp_nodelay(true);
        server.set_read_timeout(10);
        server.set_write_timeout(10);
        server.set_keep_alive_timeout(2);
        server.set_keep_alive_max_count(100);
        server.set_pre_routing_handler([this](const auto& request, auto& response) {
            response.set_header("x-typesafe-request-id", "laya-" + std::to_string(++request_id));
            if (request.path != "/health" && !options.api_key.empty() &&
                request.get_header_value("Authorization") != "Bearer " + options.api_key) {
                response.set_header("WWW-Authenticate", "Bearer");
                reject(response, 401, "Invalid or missing bearer token");
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        server.Get("/health", [this](const auto&, auto& response) {
            std::lock_guard lock(mutex);
            send(response, 200, {{"status", "ok"}, {"model", model}, {"variant", options.variant},
                                 {"backend", options.backend}, {"max_questions", options.max_questions},
                                 {"pending_requests", pending}, {"queued_requests", queue.size()},
                                 {"batching", options.batching}, {"max_batch_questions", options.max_batch_questions},
                                 {"max_pending_requests", options.max_pending_requests}, {"batch_wait_ms", options.batch_wait_ms}});
        });
        server.Get("/v1/models", [this](const auto&, auto& response) {
            send(response, 200, {{"models", json::array({{{"name", model},
                {"description", "Laya " + options.variant + " native typed decisions"},
                {"release_date", "2026-09-20"}}})}});
        });
        server.Post("/v1/systemone", [this](const auto& request, auto& response) { evaluate(request, response, false); });
        server.Post("/predict", [this](const auto& request, auto& response) { evaluate(request, response, true); });
        server.set_error_handler([](const auto& request, auto& response) {
            if (!response.body.empty()) return;
            if (response.status == 404 && (request.path == "/v1/systemone" || request.path == "/predict" ||
                                          request.path == "/health" || request.path == "/v1/models")) {
                response.status = 405;
                response.set_header("Allow", request.path == "/health" || request.path == "/v1/models" ? "GET, HEAD" : "POST");
            }
            reject(response, response.status, httplib::status_message(response.status));
        });
    }

    ~impl() {
        stop();
        if (listener.joinable()) listener.join();
        monitoring.store(false, std::memory_order_relaxed);
        if (monitor.joinable()) monitor.join();
    }

    void stop() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
            for (auto& item : queue) {
                item->answer.set_value(fail(errc::unavailable, "Server is stopping"));
                --pending;
            }
            queue.clear();
        }
        changed.notify_all();
        server.stop();
    }

    result<call::reply> submit(json requests, std::size_t questions) {
        auto item = std::make_shared<call>();
        item->requests = std::move(requests);
        item->questions = questions;
        auto answer = item->answer.get_future();
        {
            std::lock_guard lock(mutex);
            if (stopping) return fail(errc::unavailable, "Server is stopping");
            if (pending >= options.max_pending_requests) return fail(errc::unavailable, "Inference queue is full");
            queue.push_back(std::move(item));
            ++pending;
        }
        changed.notify_one();
        return answer.get();
    }

    result<void> validate(const json& request, std::size_t& questions) const {
        if (!request.is_object()) return fail(errc::invalid, "Request must be an object");
        LAYA_TRY(state, json_at(request, "state"));
        if (!(*state)->is_string() && !(*state)->is_structured())
            return fail(errc::invalid, "state must be a string, object or array");
        if (request.contains("model")) {
            LAYA_TRY(name, json_string(field(request, "model")));
            if (*name != model) return fail(errc::invalid, "Requested model is not loaded; this server serves " + model);
        }
        LAYA_TRY(definitions, json_at(request, "questions"));
        if (!(*definitions)->is_object() || (*definitions)->empty())
            return fail(errc::invalid, "questions must be a nonempty object");
        if ((*definitions)->size() > options.max_questions - questions)
            return fail(errc::too_large, "Request exceeds --max-questions limit");
        questions += (*definitions)->size();
        for (const auto& definition : **definitions) {
            LAYA_TRY(instruction, json_at(definition, "instructions"));
            if (!(*instruction)->is_string() && !(*instruction)->is_structured())
                return fail(errc::invalid, "instructions must be a string, object or array");
            LAYA_TRY(type, json_at(definition, "type"));
            LAYA_TRY(name, json_string(**type));
            if (*name != "choice" && *name != "score" && *name != "noul")
                return fail(errc::invalid, "Unsupported question type: " + *name);
        }
        return {};
    }

    void evaluate(const httplib::Request& request, httplib::Response& response, bool batch_route) {
        auto completed = [&]() -> result<call::reply> {
            LAYA_TRY(value, parse_json(request.body));
            if (!batch_route && !value->is_object()) return fail(errc::invalid, "/v1/systemone expects one request object");
            json requests = value->is_array() ? std::move(*value) : json::array({std::move(*value)});
            if (requests.empty()) return fail(errc::invalid, "Request array must not be empty");
            std::size_t questions = 0;
            for (const auto& item : requests) LAYA_CHECK(validate(item, questions));
            return submit(std::move(requests), questions);
        }();
        if (completed && !batch_route && !completed->results[0].is_object())
            completed = fail(errc::backend, "Predictor returned a non-object result");
        if (!completed) {
            const auto& failure = completed.error();
            switch (failure.code) {
            case errc::unavailable:
                response.set_header("Retry-After", "1");
                return reject(response, 503, failure.message);
            case errc::parse: return reject(response, 400, failure.message);
            case errc::too_large: return reject(response, 413, failure.message);
            case errc::invalid: return reject(response, 422, failure.message);
            default:
                std::cerr << "HTTP inference error: " << failure.message << '\n';
                return reject(response, 500, "Inference failed");
            }
        }
        auto& reply = *completed;
        response.set_header("X-Laya-Batch-Id", std::to_string(reply.batch));
        response.set_header("X-Laya-Batch-Offset", std::to_string(reply.offset));
        if (batch_route) {
            send(response, 200, {{"results", std::move(reply.results)}, {"elapsed_ms", reply.elapsed}, {"backend", options.backend}});
        } else {
            auto result = std::move(reply.results[0]);
            result["model"] = model;
            send(response, 200, result);
        }
    }
};

void http_server::release::operator()(impl* state) const { delete state; }

result<http_server> http_server::create(http_options options) {
    if (options.variant != "english" && options.variant != "multilingual" && options.variant != "typed-decisions")
        return fail(errc::invalid, "Unknown HTTP model variant");
    if (options.port < 0 || options.port > 65535 || options.max_questions == 0 || options.max_body_bytes == 0)
        return fail(errc::invalid, "Invalid HTTP port or request limit");
    if (!options.max_batch_questions) options.max_batch_questions = options.max_questions;
    if (options.max_batch_questions < options.max_questions || options.max_pending_requests == 0 ||
        options.max_pending_requests > 256 || options.batch_wait_ms > 1000)
        return fail(errc::invalid, "Invalid HTTP batching limits");
    http_server server;
    server.p.reset(new impl(std::move(options)));
    return server;
}

int http_server::bind() {
    if (p->options.port == 0) return p->server.bind_to_any_port(p->options.host);
    return p->server.bind_to_port(p->options.host, p->options.port) ? p->options.port : -1;
}
bool http_server::listen() { return p->server.listen_after_bind(); }
bool http_server::running() const { return p->server.is_running(); }
void http_server::stop() { p->stop(); }

const json* http_server::next() {
    auto& s = *p;
    auto& batch = s.current;
    if (!s.retry.empty()) {
        batch = std::move(s.retry.front());
        s.retry.pop_front();
        return &batch.requests;
    }
    batch = {};
    {
        std::unique_lock lock(s.mutex);
        s.changed.wait(lock, [&] { return s.stopping || !s.queue.empty(); });
        if (s.stopping) return nullptr;
        const auto deadline = s.queue.front()->arrived + std::chrono::milliseconds(s.options.batch_wait_ms);
        std::size_t questions = 0;
        for (;;) {
            while (!s.queue.empty() && questions + s.queue.front()->questions <= s.options.max_batch_questions) {
                questions += s.queue.front()->questions;
                batch.calls.push_back(std::move(s.queue.front()));
                s.queue.pop_front();
                if (!s.options.batching) break;
            }
            if (s.stopping || !s.options.batching || questions == s.options.max_batch_questions || !s.queue.empty() ||
                std::chrono::steady_clock::now() >= deadline) break;
            s.changed.wait_until(lock, deadline);
        }
    }
    // Already selected calls finish during shutdown; queued calls get 503.
    batch.id = ++s.batch_id;
    for (const auto& item : batch.calls)
        for (const auto& request : item->requests) batch.requests.push_back(request);
    return &batch.requests;
}

void http_server::settle(result<json> results, double elapsed_ms) {
    auto& s = *p;
    auto& batch = s.current;
    if (!results && input_error(results.error().code) && batch.calls.size() > 1) {
        for (auto& item : batch.calls) s.retry.push_back({item->requests, ++s.batch_id, {std::move(item)}});
        return;
    }
    if (results && (!results->is_array() || results->size() != batch.requests.size()))
        results = fail(errc::backend, "Predictor returned an invalid result count");
    std::size_t offset = 0;
    for (const auto& item : batch.calls) {
        auto slice = json::array();
        for (std::size_t i = 0; results && i < item->requests.size(); ++i) slice.push_back(std::move((*results)[offset + i]));
        item->answer.set_value(results ? result<impl::call::reply>({std::move(slice), elapsed_ms, batch.id, offset}) : std::unexpected(results.error()));
        offset += item->requests.size();
    }
    std::lock_guard lock(s.mutex);
    s.pending -= batch.calls.size();
}

result<void> http_server::start() {
    auto& s = *p;
    const int port = bind();
    if (port < 0) return fail(errc::io, "Cannot bind HTTP listener to " + s.options.host + ":" + std::to_string(s.options.port));
    interrupted.store(false, std::memory_order_relaxed);
    s.old_int = std::signal(SIGINT, interrupt);
    s.old_term = std::signal(SIGTERM, interrupt);
    s.monitor = std::thread([&s] {
        while (s.monitoring.load(std::memory_order_relaxed)) {
            if (interrupted.load(std::memory_order_relaxed) && s.server.is_running()) { s.stop(); return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });
    std::cerr << "Listening: http://" << s.options.host << ':' << port << "/v1/systemone\n";
    s.listener = std::thread([&s] {
        s.listened = s.server.listen_after_bind();
        // Listening ended: release the inference thread.
        s.stop();
    });
    return {};
}

result<void> http_server::join() {
    auto& s = *p;
    if (s.listener.joinable()) s.listener.join();
    s.monitoring.store(false, std::memory_order_relaxed);
    if (s.monitor.joinable()) s.monitor.join();
    std::signal(SIGINT, s.old_int);
    std::signal(SIGTERM, s.old_term);
    if (!s.listened) return fail(errc::io, "HTTP listener failed");
    return {};
}
}
