#include "check.hpp"
#include "laya/runtime.hpp"
#include <cmath>
#include <iostream>

using laya::json;
using laya::test::expect;
using laya::test::fail;
void compare(const json& expected, const json& actual, const std::string& path = "result") {
    if (expected.is_number() && actual.is_number()) {
        const double a = expected.get<double>(), b = actual.get<double>();
        if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b) > 0.000100000001)
            fail(path + ": numeric mismatch");
    } else if (expected.is_object() && actual.is_object()) {
        if (expected.size() != actual.size()) fail(path + ": key count mismatch");
        for (auto it = expected.begin(); it != expected.end(); ++it)
            compare(it.value(), laya::field(actual, it.key()), path + "." + it.key());
    } else if (expected.is_array() && actual.is_array()) {
        if (expected.size() != actual.size()) fail(path + ": row count mismatch");
        for (size_t i = 0; i < expected.size(); ++i) compare(expected[i], actual[i], path + "." + std::to_string(i));
    } else if (expected != actual) fail(path + ": categorical mismatch");
}
int main(int argc, char** argv) {
    if (argc != 2) fail("Expected checkpoint directory");
    auto reference = expect(laya::agent::load(argv[1], laya::feature::cuda, laya::overflow::reject));
    auto candidate = expect(laya::agent::load(argv[1], laya::feature::vulkan, laya::overflow::reject));
    if (!reference.backend_name().starts_with("CUDA") || !candidate.backend_name().starts_with("Vulkan"))
        fail("Wrong execution backend");
    std::string anchor;
    for (int i = 0; i < 128; ++i) anchor += "word ";
    json request = {{"state", anchor}, {"questions", {{"refund", {
        {"type", "noul"}, {"instructions", "Does the customer request a refund?"}}}}}};
    auto first = json::array({request, request});
    first[1]["state"] = "Please refund this payment.";
    auto second = first;
    second[1]["state"] = "The product works correctly. ";
    for (int i = 0; i < 12; ++i) second[1]["state"].get_ref<std::string&>() += "I am happy with it. ";
    const auto shape_a = expect(candidate.prepare(first)), shape_b = expect(candidate.prepare(second));
    if (shape_a["length"] != shape_b["length"] || shape_a["lengths"] == shape_b["lengths"])
        fail("Fixture must reuse a shape with different padding");
    const auto before = expect(candidate.predict(first));
    compare(expect(reference.predict(first)), before);
    const auto changed = expect(candidate.predict(second));
    compare(expect(reference.predict(second)), changed);
    if (expect(candidate.predict(first)) != before || expect(candidate.predict(second)) != changed)
        fail("Replayed graph changed its answer after updating padding");
    std::cout << "CUDA/Vulkan parity and changing-padding graph replays passed\n";
}
