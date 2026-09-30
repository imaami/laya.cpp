#pragma once
#include <cstdint>
#include <expected>
#include <string>
#include <utility>

namespace laya {
// Error categories. Transport layers map them to statuses; the CLI prints the
// message. Laya code is built without exceptions, so every fallible operation
// returns result<T>.
enum class errc : std::uint8_t {
    parse,        // malformed JSON text
    invalid,      // request violates the typed-decision protocol
    too_large,    // request exceeds a configured size limit
    unavailable,  // service cannot accept work right now
    unsupported,  // configuration unsupported by this build, device or checkpoint
    io,           // file could not be opened or read
    model,        // checkpoint metadata or tensor payload is invalid
    backend,      // device initialization, allocation or computation failed
};

struct error {
    errc code;
    std::string message;
};

template<class T = void>
using result = std::expected<T, error>;

[[nodiscard]] inline std::unexpected<error> fail(errc code, std::string message) {
    return std::unexpected(error{code, std::move(message)});
}

// Input errors are attributable to one request and never to the server.
[[nodiscard]] constexpr bool input_error(errc code) noexcept {
    return code == errc::parse || code == errc::invalid || code == errc::too_large;
}
}
