#include "laya/json.hpp"
#include <fstream>
#include <iterator>
#include <vector>

namespace laya {
namespace {
// Builds the same document as json::parse, but reports syntax errors as values.
class document_builder {
public:
    explicit document_builder(json& root) : root_(root) {}
    bool null() { return put(nullptr) != nullptr; }
    bool boolean(bool value) { return put(value) != nullptr; }
    bool number_integer(json::number_integer_t value) { return put(value) != nullptr; }
    bool number_unsigned(json::number_unsigned_t value) { return put(value) != nullptr; }
    bool number_float(json::number_float_t value, const json::string_t&) { return put(value) != nullptr; }
    bool string(json::string_t& value) { return put(std::move(value)) != nullptr; }
    bool binary(json::binary_t& value) { return put(json::binary(std::move(value))) != nullptr; }
    bool start_object(std::size_t) { open_.push_back(put(json::object())); return true; }
    bool start_array(std::size_t) { open_.push_back(put(json::array())); return true; }
    bool end_object() { open_.pop_back(); return true; }
    bool end_array() { open_.pop_back(); return true; }
    // Duplicate keys keep their first position and take the last value.
    bool key(json::string_t& name) { member_ = &(*open_.back())[name]; return true; }
    bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception& failure) {
        message_ = failure.what();
        return false;
    }
    std::string& message() { return message_; }

private:
    json* put(json value) {
        if (open_.empty()) {
            root_ = std::move(value);
            return &root_;
        }
        auto* container = open_.back();
        if (container->is_array()) {
            container->push_back(std::move(value));
            return &container->back();
        }
        *member_ = std::move(value);
        return member_;
    }
    json& root_;
    std::vector<json*> open_;
    json* member_ = nullptr;
    std::string message_;
};

std::string type_error(int id, std::string_view text) {
    return "[json.exception.type_error." + std::to_string(id) + "] " + std::string(text);
}
std::string number_error(const json& value) {
    return type_error(302, std::string("type must be number, but is ") + value.type_name());
}
}

result<json> parse_json(std::string_view text) {
    json value;
    document_builder builder(value);
    if (!json::sax_parse(text, &builder)) return fail(errc::parse, std::move(builder.message()));
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

namespace json_access {
result<const json*> at(const json& object, std::string_view key) {
    if (!object.is_object()) return fail(errc::invalid, type_error(304, std::string("cannot use at() with ") + object.type_name()));
    const auto found = object.find(std::string(key));
    if (found == object.end())
        return fail(errc::invalid, "[json.exception.out_of_range.403] key '" + std::string(key) + "' not found");
    return &*found;
}

result<const json*> element(const json& array, std::size_t index) {
    if (!array.is_array()) return fail(errc::invalid, type_error(304, std::string("cannot use at() with ") + array.type_name()));
    if (index >= array.size())
        return fail(errc::invalid, "[json.exception.out_of_range.401] array index " + std::to_string(index) + " is out of range");
    return &array[index];
}

result<std::string> string(const json& value) {
    if (!value.is_string()) return fail(errc::invalid, type_error(302, std::string("type must be string, but is ") + value.type_name()));
    return value.get_ref<const std::string&>();
}

result<double> number(const json& value) {
    if (!value.is_number()) return fail(errc::invalid, number_error(value));
    return value.get<double>();
}

result<int> integer(const json& value) {
    // get<int>() also converts booleans; get<double>() does not.
    if (!value.is_number() && !value.is_boolean()) return fail(errc::invalid, number_error(value));
    return value.get<int>();
}

result<double> number_or(const json& object, std::string_view key, double fallback) {
    if (!object.is_object()) return fail(errc::invalid, type_error(306, std::string("cannot use value() with ") + object.type_name()));
    const auto found = object.find(std::string(key));
    return found == object.end() ? result<double>(fallback) : number(*found);
}

result<int> integer_or(const json& object, std::string_view key, int fallback) {
    if (!object.is_object()) return fail(errc::invalid, type_error(306, std::string("cannot use value() with ") + object.type_name()));
    const auto found = object.find(std::string(key));
    return found == object.end() ? result<int>(fallback) : integer(*found);
}

result<std::string> string_or(const json& object, std::string_view key, std::string_view fallback) {
    if (!object.is_object()) return fail(errc::invalid, type_error(306, std::string("cannot use value() with ") + object.type_name()));
    const auto found = object.find(std::string(key));
    return found == object.end() ? result<std::string>(std::string(fallback)) : string(*found);
}
}
}
