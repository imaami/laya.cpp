#include "laya/json.hpp"
#include <fstream>
#include <iterator>

namespace laya {
namespace {
// json::parse, with syntax errors reported as values instead of exceptions.
struct document_parser : nlohmann::detail::json_sax_dom_parser<json> {
    using json_sax_dom_parser::json_sax_dom_parser;
    std::string message;
    template<class Exception> bool parse_error(std::size_t, const std::string&, const Exception& failure) {
        message = failure.what();
        return false;
    }
};

std::string type_error(int id, std::string_view text) {
    return "[json.exception.type_error." + std::to_string(id) + "] " + std::string(text);
}
}

result<json> parse_json(std::string_view text) {
    json value;
    document_parser parser(value, false);
    if (!json::sax_parse(text, &parser)) return fail(errc::parse, std::move(parser.message));
    return value;
}

result<json> read_json(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return fail(errc::io, "Cannot open " + path.string());
    std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    if (file.bad()) return fail(errc::io, "Cannot read " + path.string());
    auto value = parse_json(text);
    if (!value) value.error().code = errc::model;
    return value;
}

std::string dump(const json& value) {
    return value.dump(-1, ' ', false, json::error_handler_t::replace);
}

const json& field(const json& object, std::string_view key) {
    static const json missing;
    if (!object.is_object()) return missing;
    const auto found = object.find(std::string(key));
    return found == object.end() ? missing : *found;
}

namespace json_access {
result<const json*> at(const json& object, std::string_view key) {
    if (!object.is_object()) return fail(errc::invalid, type_error(304, std::string("cannot use at() with ") + object.type_name()));
    const auto found = object.find(std::string(key));
    if (found == object.end())
        return fail(errc::invalid, "[json.exception.out_of_range.403] key '" + std::string(key) + "' not found");
    return &*found;
}

result<std::string> string(const json& value) {
    if (!value.is_string()) return fail(errc::invalid, type_error(302, std::string("type must be string, but is ") + value.type_name()));
    return value.get_ref<const std::string&>();
}

result<int> integer(const json& value) {
    // get<int>() also converts booleans.
    if (!value.is_number() && !value.is_boolean())
        return fail(errc::invalid, type_error(302, std::string("type must be number, but is ") + value.type_name()));
    return value.get<int>();
}
}
}
