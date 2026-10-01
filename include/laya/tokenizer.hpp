#pragma once
#include "laya/error.hpp"
#include "laya/json.hpp"
#include <array>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace laya {
using token = std::int32_t;
using tokens = std::vector<token>;

namespace detail {
struct string_hash {
    using is_transparent = void;
    std::size_t operator()(std::string_view text) const noexcept { return std::hash<std::string_view>{}(text); }
};
template<class T>
using string_map = std::unordered_map<std::string, T, string_hash, std::equal_to<>>;
}

// Byte-pair encoding shared by Laya's tokenizer families. The derived family
// (CRTP) provides the normalizer and pre-tokenizer of its tokenizer.json:
//   result<void> normalize(std::string_view text, std::string& out) const;
//   result<void> words(std::string_view normalized, tokens& out) const;  // merge() per word
//   void symbols(std::string_view word, tokens& out) const;              // initial BPE symbols
// Merges are resolved to symbol IDs once, when the tokenizer is loaded.
template<class Derived>
class bpe {
public:
    [[nodiscard]] result<tokens> encode(std::string_view text) const;
    [[nodiscard]] std::optional<token> id(std::string_view text) const;

protected:
    struct added {
        std::string text;
        token id;
        bool lstrip;
    };
    bpe() = default;
    // Loading reads the model vocabulary, then the merges and added tokens.
    [[nodiscard]] result<void> read_vocabulary(const json& model);
    [[nodiscard]] result<void> read_rules(const json& model, const json& added_tokens);
    // Appends the BPE of one pre-tokenized word.
    void merge(std::string_view word, tokens& out) const;

    detail::string_map<token> vocab;
    // Added tokens matched before ([0]) and after ([1]) normalization, longest first.
    std::array<std::vector<added>, 2> specials;

private:
    struct rule {
        int rank;
        token merged;
    };
    [[nodiscard]] result<void> split(std::string_view text, bool normalized, tokens& out) const;
    [[nodiscard]] const Derived& self() const { return static_cast<const Derived&>(*this); }

    std::unordered_map<std::uint64_t, rule> merges;  // symbol pair -> rule
    mutable detail::string_map<tokens> cache;
    mutable std::size_t cache_bytes = 0;
};

// NFC normalization and the GPT-2 byte-level pre-tokenizer (English models).
class byte_level_tokenizer : public bpe<byte_level_tokenizer> {
public:
    [[nodiscard]] static result<byte_level_tokenizer> load(const json& document);
    byte_level_tokenizer(byte_level_tokenizer&&) noexcept;
    byte_level_tokenizer& operator=(byte_level_tokenizer&&) noexcept;
    ~byte_level_tokenizer();

private:
    friend class bpe<byte_level_tokenizer>;
    struct unicode;  // ICU regular expression and normalizer
    byte_level_tokenizer();
    [[nodiscard]] result<void> normalize(std::string_view text, std::string& out) const;
    [[nodiscard]] result<void> words(std::string_view text, tokens& out) const;
    void symbols(std::string_view word, tokens& out) const;

    std::unique_ptr<unicode> icu;
    std::array<token, 256> bytes{};
};

// Metaspace pre-tokenization with byte fallback (the multilingual model).
class metaspace_tokenizer : public bpe<metaspace_tokenizer> {
public:
    [[nodiscard]] static result<metaspace_tokenizer> load(const json& document);

private:
    friend class bpe<metaspace_tokenizer>;
    metaspace_tokenizer() = default;
    [[nodiscard]] result<void> normalize(std::string_view text, std::string& out) const;
    [[nodiscard]] result<void> words(std::string_view text, tokens& out) const;
    void symbols(std::string_view word, tokens& out) const;

    std::array<token, 256> fallback{};
};

template<class T>
concept text_tokenizer = requires(const T& tokenizer, std::string_view text) {
    { tokenizer.encode(text) } -> std::same_as<result<tokens>>;
    { tokenizer.id(text) } -> std::same_as<std::optional<token>>;
};
static_assert(text_tokenizer<byte_level_tokenizer> && text_tokenizer<metaspace_tokenizer>);

// The tokenizer family a tokenizer.json describes.
using any_tokenizer = std::variant<byte_level_tokenizer, metaspace_tokenizer>;
[[nodiscard]] result<any_tokenizer> load_tokenizer(const std::filesystem::path& file);
}
