#include "laya/json.hpp"
#include <fstream>
#include <iterator>
#include <utility>
#include <vector>

namespace laya {
namespace {
// Builds the document json::parse would, through nlohmann's public SAX
// interface, and reports syntax errors as values instead of exceptions. Like
// json::parse, a repeated object key keeps its first position and takes its
// last value.
class document_builder {
public:
    explicit document_builder(json& root) noexcept : root_(root) {}

    bool null() { return value(nullptr); }
    bool boolean(bool x) { return value(x); }
    bool number_integer(json::number_integer_t x) { return value(x); }
    bool number_unsigned(json::number_unsigned_t x) { return value(x); }
    bool number_float(json::number_float_t x, const json::string_t&) { return value(x); }
    // The parser hands over its token buffer, so strings move into the document.
    bool string(json::string_t& x) { return value(std::move(x)); }
    bool binary(json::binary_t& x) { return value(std::move(x)); }
    bool start_object(std::size_t) { return open(json::object()); }
    bool start_array(std::size_t) { return open(json::array()); }
    bool end_object() { return close(); }
    bool end_array() { return close(); }
    bool key(json::string_t& name) {
        member_ = &(*open_.back())[std::move(name)];
        return true;
    }
    bool parse_error(std::size_t, const std::string&, const json::exception& failure) {
        message = failure.what();
        return false;
    }

    std::string message;

private:
    template<class Value> json& insert(Value&& x) {
        if (open_.empty()) return root_ = json(std::forward<Value>(x));
        json& container = *open_.back();
        if (container.is_object()) return *member_ = json(std::forward<Value>(x));
        container.push_back(json(std::forward<Value>(x)));
        return container.back();
    }
    template<class Value> bool value(Value&& x) {
        insert(std::forward<Value>(x));
        return true;
    }
    bool open(json container) {
        open_.push_back(&insert(std::move(container)));
        return true;
    }
    bool close() {
        open_.pop_back();
        return true;
    }

    json& root_;
    std::vector<json*> open_;  // containers still being filled, innermost last
    json* member_ = nullptr;   // the object member the next value fills
};

std::string type_error(int id, std::string_view text) {
    return "[json.exception.type_error." + std::to_string(id) + "] " + std::string(text);
}
}

result<json> parse_json(std::string_view text) {
    json value;
    document_builder builder(value);
    if (!json::sax_parse(text, &builder)) return fail(errc::parse, std::move(builder.message));
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
