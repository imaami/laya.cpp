#include "laya/http.hpp"
#include "ggml.h"
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>

namespace {
using namespace laya;

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

// Parses the command line and runs its task.
result<void> run(int argc, char** argv) {
    std::string model = "models/laya", input_file, variant = "english";
    mode backend = feature::cuda, precision = 0, flash = 0, compensated = 0;
    bool raw = false, prepare = false, allow_truncation = false, server = false, http_option = false;
    http_options http;
    // Assigns a bounded numeric HTTP option.
    auto number = [&](auto& target, std::string_view value, int maximum, int minimum = 1) -> result<void> {
        int n = 0;
        const auto [end, status] = std::from_chars(value.data(), value.data() + value.size(), n);
        if (status != std::errc{} || end != value.data() + value.size() || n < minimum || n > maximum)
            return fail(errc::invalid, "Invalid numeric option: " + std::string(value));
        target = static_cast<std::remove_reference_t<decltype(target)>>(n);
        http_option = true;
        return {};
    };
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const bool valued = i + 1 < argc;
        if (arg == "--model" && valued) model = argv[++i];
        else if (arg == "--variant" && valued) variant = argv[++i];
        else if (arg == "--input" && valued) input_file = argv[++i];
        else if (arg == "--server") server = true;
        else if (arg == "--host" && valued) http.host = argv[++i], http_option = true;
        else if (arg == "--port" && valued) LAYA_CHECK(number(http.port, argv[++i], 65535));
        else if (arg == "--max-questions" && valued) LAYA_CHECK(number(http.max_questions, argv[++i], 4096));
        else if (arg == "--max-batch-questions" && valued) LAYA_CHECK(number(http.max_batch_questions, argv[++i], 4096));
        else if (arg == "--max-pending-requests" && valued) LAYA_CHECK(number(http.max_pending_requests, argv[++i], 256));
        else if (arg == "--batch-wait-ms" && valued) LAYA_CHECK(number(http.batch_wait_ms, argv[++i], 1000, 0));
        else if (arg == "--no-batching") http.batching = false, http_option = true;
        else if (arg == "--cpu") backend = feature::cpu;
        else if (arg == "--vulkan") backend = feature::vulkan;
        else if (arg == "--cuda") backend = feature::cuda;
        else if (arg == "--coreml") backend = feature::coreml;
        else if (arg == "--fp32") precision = 0, flash = 0;
        else if (arg == "--flash-fp32") precision = 0, flash = feature::flash;
        else if (arg == "--tensor-core-fp32") precision = 0, compensated = feature::compensated;
        else if (arg == "--bf16" || arg == "--experimental-bf16") precision = feature::bf16, flash = feature::flash;
        else if (arg == "--fp16") precision = feature::fp16, flash = feature::flash;
        else if (arg == "--no-flash") flash = 0;
        else if (arg == "--raw") raw = true;
        else if (arg == "--prepare") prepare = true;
        else if (arg == "--allow-truncation") allow_truncation = true;
        else if (arg == "--help") return std::cout << usage, result<void>{};
        else return fail(errc::invalid, "Unknown or incomplete option: " + std::string(arg));
    }
    if (variant != "english" && variant != "multilingual" && variant != "typed-decisions")
        return fail(errc::invalid, "Unknown model variant: " + variant);
    if (server && (!input_file.empty() || raw || prepare))
        return fail(errc::invalid, "--server cannot be combined with --input, --raw or --prepare");
    if (http_option && !server) return fail(errc::invalid, "HTTP options require --server");
    if (variant != "english") model = (std::filesystem::path(model) / variant).string();
    LAYA_TRY(laya, agent::load(model, backend | precision | flash | compensated, allow_truncation ? overflow::truncate : overflow::reject));
    std::cerr << "Ready: " << laya->backend_name() << " (" << laya->device_name() << ")\n";
    const auto predict = [&](const json& requests) { return laya->predict(requests); };
    if (server) {
        http.variant = laya->engine.model().serving.variant;  // a direct --model checkpoint path identifies its actual variant
        http.backend = laya->backend_name();
        if (const char* key = std::getenv("LAYA_API_KEY")) http.api_key = key;
        return serve_http(std::move(http), predict);
    }
    // Answers --input, or each JSON line of standard input, with one task.
    const auto interact = [&](auto task) -> result<void> {
        auto answer = [&](result<json> value) -> result<json> {
            if (!value) return value;
            const json requests = value->is_array() ? std::move(*value) : json::array({std::move(*value)});
            const auto start = std::chrono::steady_clock::now();
            LAYA_TRY(results, task(requests));
            const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            return json{{"results", std::move(*results)}, {"elapsed_ms", elapsed}, {"backend", laya->backend_name()}, {"device", laya->device_name()}};
        };
        if (!input_file.empty()) {
            std::ifstream file(input_file, std::ios::binary);
            if (!file) return fail(errc::io, "Cannot open input file");
            LAYA_TRY(output, answer(parse_json(std::string(std::istreambuf_iterator<char>(file), {}))));
            std::cout << dump(*output) << '\n';
        } else for (std::string line; std::getline(std::cin, line);) {
            const auto output = answer(parse_json(line));
            std::cout << dump(output ? *output : json{{"error", output.error().message}}) << std::endl;
        }
        return {};
    };
    if (prepare) return interact([&](const json& requests) { return laya->prepare(requests); });
    if (raw) return interact([&](const json& requests) { return laya->raw(requests); });
    return interact(predict);
}
}

int main(int argc, char** argv) {
    ggml_log_set([](ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_WARN) std::cerr << text;
    }, nullptr);
    const auto status = run(argc, argv);
    if (!status) std::cerr << "Error: " << status.error().message << '\n';
    return status ? 0 : 1;
}
