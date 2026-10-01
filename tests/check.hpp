#pragma once
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <utility>

// Laya builds without exceptions: a failed check prints its message and exits.
namespace laya::test {
[[noreturn]] inline void fail(std::string_view message) {
    std::cerr << message << '\n';
    std::exit(1);
}

inline void require(bool condition, std::string_view message) {
    if (!condition) fail(message);
}

// The value of a successful result (std::expected); otherwise fails with its error message.
template<class R>
    requires(!std::is_lvalue_reference_v<R>)
auto expect(R&& outcome) {
    if (!outcome) fail(outcome.error().message);
    if constexpr (!std::is_void_v<typename R::value_type>) return std::move(*outcome);
}
}
