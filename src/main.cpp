#include "agent.hpp"
#include "laya/http.hpp"
#include "ggml.h"
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>

namespace {
using namespace laya;

// Backends compiled into this binary; only their modes are instantiated.
constexpr mode built = feature::cpu
#if LAYA_CUDA
    | feature::cuda
#endif
#if LAYA_VULKAN
    | feature::vulkan
#endif
#if LAYA_COREML
    | feature::coreml
#endif
    ;

constexpr std::string_view usage =
    "laya-cli [--model DIR] [--variant english|multilingual|typed-decisions] [--input JSON] [--raw|--prepare] [--allow-truncation] [--fp32|--fp16|--bf16] [--cpu|--cuda|--vulkan|--coreml]\n"
    "Requests that exceed model token budgets are rejected by default; --allow-truncation restores legacy truncation.\n"
    "--tensor-core-fp32 --flash-fp32 enables the optimized CUDA path.\n"
    "--bf16 enables mixed BF16 on CUDA or Vulkan; --fp16 currently requires Vulkan.\n--experimental-bf16 is a compatibility alias. See docs/precision.md and docs/vulkan.md for validated hardware and toolchains.\n"
    "--coreml requires a -DLAYA_COREML=ON Apple Silicon build and compiled coreml/ buckets; precision is selected during export. See docs/coreml.md.\n"
    "Reads JSON lines from stdin when --input is absent. Each line is a request or request array.\n"
    "--server listens on HTTP: POST /v1/systemone (JEV schema), POST /predict (batch), GET /health, GET /v1/models.\n"
    "--host ADDRESS (127.0.0.1), --port PORT (8080), --max-questions N (8).\n"
    "--max-batch-questions N (max-questions), --max-pending-requests N (32), --batch-wait-ms N (2), --no-batching.\n"
    "Set LAYA_API_KEY to require a bearer token; /health is unauthenticated.\n";

enum class task : unsigned char { predict, raw, prepare };

struct options {
    std::string model = "models/laya", input_file, variant = "english";
    mode requested = 0;
    task work = task::predict;
    bool allow_truncation = false, server = false, help = false;
    http_options http;
};

result<int> number(std::string_view value, int maximum, int minimum = 1) {
    int n = 0;
    const auto [end, status] = std::from_chars(value.data(), value.data() + value.size(), n);
    if (status != std::errc{} || end != value.data() + value.size() || n < minimum || n > maximum)
        return fail(errc::invalid, "Invalid numeric option: " + std::string(value));
    return n;
}

result<options> parse(int argc, char** argv) {
    options o;
    mode backend = feature::cuda, precision = 0;
    bool flash = false, compensated = false, raw = false, prepare = false, http_option = false;
    // Assigns a bounded numeric HTTP option.
    auto limit = [&](auto& target, const char* value, int maximum, int minimum = 1) -> result<void> {
        LAYA_TRY(n, number(value, maximum, minimum));
        target = static_cast<std::remove_reference_t<decltype(target)>>(*n);
        http_option = true;
        return {};
    };
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const bool valued = i + 1 < argc;
        if (arg == "--model" && valued) o.model = argv[++i];
        else if (arg == "--variant" && valued) o.variant = argv[++i];
        else if (arg == "--input" && valued) o.input_file = argv[++i];
        else if (arg == "--server") o.server = true;
        else if (arg == "--host" && valued) { o.http.host = argv[++i]; http_option = true; }
        else if (arg == "--port" && valued) LAYA_CHECK(limit(o.http.port, argv[++i], 65535));
        else if (arg == "--max-questions" && valued) LAYA_CHECK(limit(o.http.max_questions, argv[++i], 4096));
        else if (arg == "--max-batch-questions" && valued) LAYA_CHECK(limit(o.http.max_batch_questions, argv[++i], 4096));
        else if (arg == "--max-pending-requests" && valued) LAYA_CHECK(limit(o.http.max_pending_requests, argv[++i], 256));
        else if (arg == "--batch-wait-ms" && valued) LAYA_CHECK(limit(o.http.batch_wait_ms, argv[++i], 1000, 0));
        else if (arg == "--no-batching") { o.http.batching = false; http_option = true; }
        else if (arg == "--cpu") backend = feature::cpu;
        else if (arg == "--vulkan") backend = feature::vulkan;
        else if (arg == "--cuda") backend = feature::cuda;
        else if (arg == "--coreml") backend = feature::coreml;
        else if (arg == "--fp32") { precision = 0; flash = false; }
        else if (arg == "--flash-fp32") { precision = 0; flash = true; }
        else if (arg == "--tensor-core-fp32") { precision = 0; compensated = true; }
        else if (arg == "--bf16" || arg == "--experimental-bf16") { precision = feature::bf16; flash = true; }
        else if (arg == "--fp16") { precision = feature::fp16; flash = true; }
        else if (arg == "--no-flash") flash = false;
        else if (arg == "--raw") raw = true;
        else if (arg == "--prepare") prepare = true;
        else if (arg == "--allow-truncation") o.allow_truncation = true;
        else if (arg == "--help") { o.help = true; return o; }
        else return fail(errc::invalid, "Unknown or incomplete option: " + std::string(arg));
    }
    if (o.variant != "english" && o.variant != "multilingual" && o.variant != "typed-decisions")
        return fail(errc::invalid, "Unknown model variant: " + o.variant);
    if (o.server && (!o.input_file.empty() || raw || prepare))
        return fail(errc::invalid, "--server cannot be combined with --input, --raw or --prepare");
    if (http_option && !o.server) return fail(errc::invalid, "HTTP options require --server");
    if (o.variant != "english") o.model = (std::filesystem::path(o.model) / o.variant).string();
    o.requested = backend | precision | (flash ? mode{feature::flash} : mode{}) | (compensated ? mode{feature::compensated} : mode{});
    o.work = prepare ? task::prepare : raw ? task::raw : task::predict;
    return o;
}

// Runs one request or request array read from the CLI.
template<class Run>
result<json> evaluate(json value, Run& run, const std::string& backend, const std::string& device) {
    json requests = value.is_array() ? std::move(value) : json::array({std::move(value)});
    const auto start = std::chrono::steady_clock::now();
    LAYA_TRY(results, run(requests));
    const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return json{{"results", std::move(*results)}, {"elapsed_ms", elapsed}, {"backend", backend}, {"device", device}};
}

// Answers --input, or each JSON line of standard input.
template<class Run>
result<void> interact(const options& o, Run&& run, const std::string& backend, const std::string& device) {
    if (!o.input_file.empty()) {
        std::ifstream file(o.input_file, std::ios::binary);
        if (!file) return fail(errc::io, "Cannot open input file");
        const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        LAYA_TRY(value, parse_json(text));
        LAYA_TRY(output, evaluate(std::move(*value), run, backend, device));
        std::cout << dump(*output) << '\n';
        return {};
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        auto value = parse_json(line);
        auto output = value ? evaluate(std::move(*value), run, backend, device) : std::unexpected(std::move(value).error());
        std::cout << dump(output ? *output : json{{"error", output.error().message}}) << std::endl;
    }
    return {};
}

// Serves the loaded agent until input ends or the HTTP server stops.
template<class Agent>
result<void> session(const options& o, const checkpoint& model, Agent& laya) {
    std::cerr << "Ready: " << laya.backend_name() << " (" << laya.device_name() << ")\n";
    const auto& backend = laya.backend_name();
    const auto& device = laya.device_name();
    if (o.server) {
        http_options http = o.http;
        // A direct --model checkpoint path identifies its actual variant.
        http.variant = name(model.serving.variant);
        http.backend = backend;
        if (const char* key = std::getenv("LAYA_API_KEY")) http.api_key = key;
        return serve_http(std::move(http), [&](const json& requests) { return laya.predict(requests); });
    }
    switch (o.work) {
    case task::prepare: return interact(o, [&](const json& requests) { return laya.prepare(requests); }, backend, device);
    case task::raw: return interact(o, [&](const json& requests) { return laya.template predict<true>(requests); }, backend, device);
    case task::predict: break;
    }
    return interact(o, [&](const json& requests) { return laya.predict(requests); }, backend, device);
}

result<void> run(const options& o) {
    if (has(o.requested, feature::coreml) && !(built & feature::coreml))
        return fail(errc::unsupported, "This build has no Core ML backend; rebuild with -DLAYA_COREML=ON on macOS arm64");
    if (const char* reason = rejection(o.requested)) return fail(errc::unsupported, reason);
    LAYA_TRY(model, checkpoint::load(o.model));
    LAYA_TRY(opened, open_device(o.requested));
    // The only runtime decision: instantiate everything below for one mode.
    return dispatch<built>(opened->resolved, result<void>(fail(errc::unsupported, "Unsupported backend and precision")),
                           [&]<mode M>() -> result<void> {
        auto serve = [&](auto& laya) { return session(o, *model, laya); };
        return o.allow_truncation ? with_agent<M, overflow::truncate>(*model, std::move(*opened), serve)
                                  : with_agent<M, overflow::reject>(*model, std::move(*opened), serve);
    });
}
}

int main(int argc, char** argv) {
    ggml_log_set([](ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_WARN) std::cerr << text;
    }, nullptr);
    auto status = parse(argc, argv).and_then([](const options& o) -> result<void> {
        if (o.help) {
            std::cout << usage;
            return {};
        }
        return run(o);
    });
    if (status) return 0;
    std::cerr << "Error: " << status.error().message << '\n';
    return 1;
}
