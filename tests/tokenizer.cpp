#include "check.hpp"
#include "laya/tokenizer.hpp"
#include <iostream>
#include <variant>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    using laya::test::expect;
    auto tokenizer = expect(laya::load_tokenizer(argv[1]));
    const auto cases = expect(laya::read_json(argv[2]));
    for (const auto& test : cases) {
        const auto text = expect(laya::json_access::string(laya::field(test, "text")));
        const auto actual = std::visit([&](const auto& encoder) { return expect(encoder.encode(text)); }, tokenizer);
        const laya::json expected = laya::field(test, "ids");
        if (laya::json(actual) != expected) {
            std::cerr << "Token mismatch: " << laya::dump(laya::field(test, "text")) << "\nExpected: "
                      << laya::dump(expected) << "\nActual: " << laya::dump(laya::json(actual)) << '\n';
            return 1;
        }
    }
    std::cout << cases.size() << " tokenizer fixtures passed\n";
}
