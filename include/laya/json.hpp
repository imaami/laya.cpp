#pragma once
#include "laya/error.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <string>
#include <string_view>

namespace laya {
// Request, prediction and configuration documents keep object insertion
// order: choice order and answer layout are part of the protocol.
using json = nlohmann::ordered_json;

// Parses one complete JSON document. Syntax errors carry nlohmann's parser
// message, so callers report exactly what the parser saw.
[[nodiscard]] result<json> parse_json(std::string_view text);
// Reads and parses a JSON file; an unreadable file reports "Cannot open PATH".
[[nodiscard]] result<json> read_json(const std::filesystem::path& path);
// Serializes without failing on invalid UTF-8 (replaced by U+FFFD).
[[nodiscard]] std::string dump(const json& value);

// Checked access with the messages nlohmann reports for at() and get<T>().
namespace json_access {
[[nodiscard]] result<const json*> at(const json& object, std::string_view key);
[[nodiscard]] result<const json*> element(const json& array, std::size_t index);
[[nodiscard]] result<std::string> string(const json& value);
[[nodiscard]] result<double> number(const json& value);
[[nodiscard]] result<int> integer(const json& value);
// object.value(key, fallback) for numeric fields; the object must be an object.
[[nodiscard]] result<double> number_or(const json& object, std::string_view key, double fallback);
[[nodiscard]] result<int> integer_or(const json& object, std::string_view key, int fallback);
[[nodiscard]] result<std::string> string_or(const json& object, std::string_view key, std::string_view fallback);
}
}
