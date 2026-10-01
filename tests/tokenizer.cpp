#include "check.hpp"
#include "laya/runtime.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    using laya::test::expect;
    const auto tokenizer = expect(laya::tokenizer::load(argv[1]));
    const auto cases = expect(laya::read_json(argv[2]));
    for (const auto& test : cases) {
        const auto text = expect(laya::json_access::string(laya::field(test, "text")));
        const auto actual = expect(tokenizer.encode(text));
        const laya::json expected = laya::field(test, "ids");
        if (laya::json(actual) != expected) {
            std::cerr << "Token mismatch: " << laya::dump(laya::field(test, "text")) << "\nExpected: "
                      << laya::dump(expected) << "\nActual: " << laya::dump(laya::json(actual)) << '\n';
            return 1;
        }
    }
    std::cout << cases.size() << " tokenizer fixtures passed\n";
}
