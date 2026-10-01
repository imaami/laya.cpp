#include "laya/runtime.hpp"
#include <fstream>
#include <iterator>

namespace laya {
namespace {
// Records the message of the syntax error json::parse rejected a document for.
struct syntax_error {
    std::string message;
    bool null() { return true; }
    bool boolean(bool) { return true; }
    bool number_integer(json::number_integer_t) { return true; }
    bool number_unsigned(json::number_unsigned_t) { return true; }
    bool number_float(json::number_float_t, const json::string_t&) { return true; }
    bool string(json::string_t&) { return true; }
    bool binary(json::binary_t&) { return true; }
    bool start_object(std::size_t) { return true; }
    bool start_array(std::size_t) { return true; }
    bool end_object() { return true; }
    bool end_array() { return true; }
    bool key(json::string_t&) { return true; }
    bool parse_error(std::size_t, const std::string&, const json::exception& failure) {
        message = failure.what();
        return false;
    }
};

std::unexpected<error> type_error(int id, const std::string& text) {
    return fail(errc::invalid, "[json.exception.type_error." + std::to_string(id) + "] " + text);
}
}

result<json> parse_json(std::string_view text) {
    if (json value = json::parse(text, nullptr, false); !value.is_discarded()) return value;
    syntax_error failure;
    json::sax_parse(text, &failure);
    return fail(errc::parse, std::move(failure.message));
}

result<json> read_json(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return fail(errc::io, "Cannot open " + path.string());
    const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    if (file.bad()) return fail(errc::io, "Cannot read " + path.string());
    auto value = parse_json(text);
    if (!value) value.error().code = errc::model;
    return value;
}

std::string dump(const json& value) { return value.dump(-1, ' ', false, json::error_handler_t::replace); }

const json& field(const json& object, std::string_view key) {
    static const json missing;
    if (!object.is_object()) return missing;
    const auto found = object.find(key);
    return found == object.end() ? missing : *found;
}

result<const json*> json_access::at(const json& object, std::string_view key) {
    if (!object.is_object()) return type_error(304, std::string("cannot use at() with ") + object.type_name());
    const auto found = object.find(key);
    if (found == object.end()) return fail(errc::invalid, "[json.exception.out_of_range.403] key '" + std::string(key) + "' not found");
    return &*found;
}

result<std::string> json_access::string(const json& value) {
    if (!value.is_string()) return type_error(302, std::string("type must be string, but is ") + value.type_name());
    return value.get_ref<const std::string&>();
}
}
