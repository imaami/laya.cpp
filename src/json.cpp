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
    bool parse_error(std::size_t, const std::string&, const json::exception& failure) { return message = failure.what(), false; }
};
}

template<class Json> result<Json> parse_json(std::string_view text) {
    if (Json value = Json::parse(text, nullptr, false); !value.is_discarded()) return value;
    syntax_error failure;
    Json::sax_parse(text, &failure);
    return fail(errc::parse, std::move(failure.message));
}

template<class Json> result<Json> read_json(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    const std::string text{std::istreambuf_iterator<char>(file), {}};
    if (!file.is_open() || file.bad()) return fail(errc::io, (file.is_open() ? "Cannot read " : "Cannot open ") + path.string());
    return parse_json<Json>(text).transform_error([](error e) { return e.code = errc::model, e; });
}
template result<json> parse_json(std::string_view);
template result<json> read_json(const std::filesystem::path&);
template result<nlohmann::json> read_json(const std::filesystem::path&);

std::string dump(const json& value) { return value.dump(-1, ' ', false, json::error_handler_t::replace); }

result<const json*> json_at(const json& object, std::string_view key) {
    if (!object.is_object()) return fail(errc::invalid, std::string("[json.exception.type_error.304] cannot use at() with ") + object.type_name());
    const auto found = object.find(key);
    if (found == object.end()) return fail(errc::invalid, "[json.exception.out_of_range.403] key '" + std::string(key) + "' not found");
    return &*found;
}

result<std::string> json_string(const json& value) {
    if (!value.is_string()) return fail(errc::invalid, std::string("[json.exception.type_error.302] type must be string, but is ") + value.type_name());
    return value.get_ref<const std::string&>();
}
}
