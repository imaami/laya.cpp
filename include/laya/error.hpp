#pragma once
#include <cstdint>
#include <expected>
#include <string>
#include <utility>

// Laya is built without exceptions: fallible operations return result<T>.
namespace laya {
// Transport layers map categories to statuses; parse, invalid and too_large are
// input errors, attributable to one request and never to the server.
enum class errc : std::uint8_t { parse, invalid, too_large, unavailable, unsupported, io, model, backend };
struct error {
    errc code;
    std::string message;
};
template<class T = void> using result = std::expected<T, error>;
[[nodiscard]] inline std::unexpected<error> fail(errc code, std::string message) { return std::unexpected(error{code, std::move(message)}); }
[[nodiscard]] constexpr bool input_error(errc code) noexcept { return code <= errc::too_large; }
}

// Declares `name` as the result of an expression and returns its error, if any.
#define LAYA_TRY(name, ...) \
    auto name = (__VA_ARGS__); \
    if (!name) return std::unexpected(std::move(name).error())
// Returns the error of a result<void> expression, if any.
#define LAYA_CHECK(...) \
    do { if (auto laya_status_ = (__VA_ARGS__); !laya_status_) return std::unexpected(std::move(laya_status_).error()); } while (false)
