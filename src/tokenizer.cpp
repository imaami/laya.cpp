#include "laya/runtime.hpp"
#include <unicode/bytestream.h>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/uchar.h>
#include <unicode/utf8.h>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <variant>

namespace laya {
namespace {
constexpr char metaspace_symbol[] = "\xE2\x96\x81";  // U+2581, which replaces spaces
constexpr std::string_view metaspace_mark = metaspace_symbol;

struct string_hash {
    using is_transparent = void;
    std::size_t operator()(std::string_view text) const noexcept { return std::hash<std::string_view>{}(text); }
};
template<class T> using string_map = std::unordered_map<std::string, T, string_hash, std::equal_to<>>;

constexpr std::uint64_t pair_key(token first, token second) { return std::uint64_t(std::uint32_t(first)) << 32 | std::uint32_t(second); }
constexpr token missing = -1;  // a byte without a symbol in the vocabulary

std::string utf8(UChar32 codepoint) {
    char buffer[U8_MAX_LENGTH];
    std::int32_t length = 0;
    [[maybe_unused]] UBool error = false;  // byte-level symbols are valid code points
    U8_APPEND(buffer, length, U8_MAX_LENGTH, codepoint, error);
    return std::string(buffer, std::size_t(length));
}

// The text before an added token with lstrip loses its trailing white space.
std::string_view strip_trailing_space(std::string_view text) {
    auto end = std::int32_t(text.size());
    for (std::int32_t at = end; end > 0; end = at) {
        UChar32 codepoint;
        U8_PREV(text.data(), 0, at, codepoint);
        if (!u_isUWhiteSpace(codepoint)) break;
    }
    return text.substr(0, std::size_t(end));
}

std::unexpected<error> unsupported(std::string message = "Unsupported tokenizer configuration") { return fail(errc::model, std::move(message)); }

// A tokenizer family supplies the normalizer, the pre-tokenizer (which merges
// each word) and the initial BPE symbols of a word.
template<class F, class Model>
concept family = requires(const F& f, const Model& m, std::string_view text, std::string& normal, tokens& out) {
    { f.normalize(text, normal) } -> std::same_as<result<void>>;
    { f.words(m, text, out) } -> std::same_as<result<void>>;
    { f.symbols(m, text, out) } -> std::same_as<bool>;
};

// NFC normalization and the GPT-2 byte-level pre-tokenizer (English models).
struct byte_level {
    std::unique_ptr<icu::RegexPattern> pattern;
    std::unique_ptr<icu::RegexMatcher> matcher;  // reset for each text
    const icu::Normalizer2* nfc = nullptr;
    std::array<token, 256> bytes{};
    result<void> normalize(std::string_view text, std::string& out) const {
        UErrorCode status = U_ZERO_ERROR;
        icu::StringByteSink<std::string> sink(&out, std::int32_t(text.size()));
        nfc->normalizeUTF8(0, icu::StringPiece(text.data(), std::int32_t(text.size())), sink, nullptr, status);
        if (U_FAILURE(status)) return fail(errc::invalid, "NFC normalization failed");
        return {};
    }
    result<void> words(const auto& model, std::string_view text, tokens& out) const {
        UErrorCode status = U_ZERO_ERROR;
        const auto input = icu::UnicodeString::fromUTF8(icu::StringPiece(text.data(), std::int32_t(text.size())));
        matcher->reset(input);  // the matcher reads `input` in place
        for (std::string word; matcher->find(status); word.clear()) LAYA_CHECK(model.merge(*this, matcher->group(status).toUTF8String(word), out));
        if (U_FAILURE(status)) return fail(errc::invalid, "Unicode tokenization failed");
        return {};
    }
    bool symbols(const auto&, std::string_view word, tokens& out) const {
        out.reserve(word.size());
        for (const unsigned char byte : word) out.push_back(bytes[byte]);
        return std::ranges::find(out, missing) == out.end();
    }
};

// Metaspace pre-tokenization with byte fallback (the multilingual model).
struct metaspace {
    std::array<token, 256> fallback{};
    result<void> normalize(std::string_view text, std::string& out) const {
        out.reserve(text.size());
        for (const char c : text) c == ' ' ? out += metaspace_mark : out += c;
        return {};
    }
    // Each word starts at a metaspace mark.
    result<void> words(const auto& model, std::string_view text, tokens& out) const {
        if (text.empty()) return {};
        const std::string value = (text.starts_with(metaspace_mark) ? "" : metaspace_symbol) + std::string(text);
        for (std::size_t begin = 0, end; begin < value.size(); begin = end) {
            end = std::min(value.find(metaspace_mark, begin + metaspace_mark.size()), value.size());
            LAYA_CHECK(model.merge(*this, std::string_view(value).substr(begin, end - begin), out));
        }
        return {};
    }
    bool symbols(const auto& model, std::string_view word, tokens& out) const {
        for (std::int32_t at = 0, size = std::int32_t(word.size()); at < size;) {
            const std::int32_t start = at;
            UChar32 codepoint;
            U8_NEXT(word.data(), at, size, codepoint);
            const auto character = word.substr(std::size_t(start), std::size_t(at - start));
            if (const auto symbol = model.id(character)) out.push_back(*symbol);
            else for (const unsigned char byte : character) out.push_back(fallback[byte]);
        }
        return true;  // every byte has a fallback token
    }
};
}

struct tokenizer::impl {
    struct added {
        std::string text;
        token id;
        bool lstrip;
    };
    struct rule {
        int rank;
        token merged;
    };
    string_map<token> vocab;
    std::unordered_map<std::uint64_t, rule> merges;  // symbol pair -> rule, resolved once at load
    std::array<std::vector<added>, 2> specials;      // matched before and after normalization, longest first
    std::variant<byte_level, metaspace> kind;
    mutable string_map<tokens> cache;
    mutable std::size_t cache_bytes = 0;

    std::optional<token> id(std::string_view text) const {
        const auto found = vocab.find(text);
        return found == vocab.end() ? std::nullopt : std::optional(found->second);
    }

    result<void> read_rules(const json& model, const json& added_tokens) {
        const json& rules = field(model, "merges");
        if (!rules.is_array() || !std::ranges::all_of(rules, [](const json& pair) {
                return pair.is_array() && pair.size() == 2 && pair[0].is_string() && pair[1].is_string();
            }))
            return unsupported("Unsupported BPE merge format");
        if (!added_tokens.is_array()) return unsupported();
        for (const auto& entry : added_tokens) {
            if (field(entry, "single_word") != false || field(entry, "rstrip") != false) return unsupported("Unsupported added-token boundary flags");
            const json &text = field(entry, "content"), &value = field(entry, "id");
            const json &normalized = field(entry, "normalized"), &lstrip = field(entry, "lstrip");
            if (!text.is_string() || text.empty() || !value.is_number_integer() || !normalized.is_boolean() || !lstrip.is_boolean())
                return unsupported();
            specials[normalized.get<bool>()].push_back({text.get<std::string>(), value.get<token>(), lstrip.get<bool>()});
            vocab.insert_or_assign(text.get<std::string>(), value.get<token>());
        }
        for (auto& list : specials) std::ranges::stable_sort(list, std::ranges::greater{}, [](const added& entry) { return entry.text.size(); });
        // A reachable merge joins two vocabulary symbols into another; the first listing of a pair keeps its rank.
        merges.reserve(rules.size());
        for (int rank = 0; const auto& pair : rules) {
            const auto &first = pair[0].get_ref<const std::string&>(), &second = pair[1].get_ref<const std::string&>();
            const auto left = id(first), right = id(second);
            if (left && right) {
                const auto merged = id(first + second);
                if (!merged) return unsupported("Unsupported BPE merge format");
                merges.try_emplace(pair_key(*left, *right), rule{rank, *merged});
            }
            ++rank;
        }
        return {};
    }

    // Appends the BPE of one pre-tokenized word. Only uncached words can lack a symbol.
    template<family<impl> F> result<void> merge(const F& kind, std::string_view word, tokens& out) const {
        if (const auto found = cache.find(word); found != cache.end()) {
            out.insert(out.end(), found->second.begin(), found->second.end());
            return {};
        }
        tokens parts;
        if (!kind.symbols(*this, word, parts)) return fail(errc::invalid, "Text contains a byte the tokenizer cannot encode");
        while (parts.size() > 1) {
            rule best{std::numeric_limits<int>::max(), 0};
            std::size_t at = 0;
            for (std::size_t i = 0; i + 1 < parts.size(); ++i)
                if (const auto found = merges.find(pair_key(parts[i], parts[i + 1])); found != merges.end() && found->second.rank < best.rank)
                    best = found->second, at = i;
            if (best.rank == std::numeric_limits<int>::max()) break;
            parts[at] = best.merged;
            parts.erase(parts.begin() + std::ptrdiff_t(at) + 1);
        }
        out.insert(out.end(), parts.begin(), parts.end());
        if (word.size() > 4096) return {};
        if (cache.size() >= 16384 || cache_bytes > 4 * 1024 * 1024) cache.clear(), cache_bytes = 0;
        cache_bytes += word.size() + parts.size() * sizeof(token);
        cache.emplace(std::string(word), std::move(parts));
        return {};
    }

    // Added tokens split the text; the rest is normalized, pre-tokenized and merged.
    template<family<impl> F> result<void> split(const F& kind, std::string_view text, bool normalized, tokens& out) const {
        for (std::size_t begin = 0; begin < text.size();) {
            std::size_t first = std::string_view::npos;
            const added* selected = nullptr;
            for (const auto& candidate : specials[normalized])
                if (const auto at = text.find(candidate.text, begin); at < first) first = at, selected = &candidate;
            auto prefix = text.substr(begin, (selected ? first : text.size()) - begin);
            if (selected && selected->lstrip) prefix = strip_trailing_space(prefix);
            if (normalized) LAYA_CHECK(kind.words(*this, prefix, out));
            else {
                std::string normal;
                LAYA_CHECK(kind.normalize(prefix, normal));
                LAYA_CHECK(split(kind, normal, true, out));
            }
            if (!selected) break;
            out.push_back(selected->id);
            begin = first + selected->text.size();
        }
        return {};
    }
};

tokenizer::tokenizer(std::unique_ptr<impl> state) : p(std::move(state)) {}
tokenizer::tokenizer(tokenizer&&) noexcept = default;
tokenizer& tokenizer::operator=(tokenizer&&) noexcept = default;
tokenizer::~tokenizer() = default;

result<tokens> tokenizer::encode(std::string_view text) const {
    tokens out;
    LAYA_CHECK(std::visit([&](const auto& kind) { return p->split(kind, text, false, out); }, p->kind));
    return out;
}

std::optional<token> tokenizer::id(std::string_view text) const { return p->id(text); }

result<tokenizer> tokenizer::load(const std::filesystem::path& file) {
    auto document = read_json(file);
    if (!document) return document.error().code == errc::io ? fail(errc::io, "Cannot open tokenizer: " + file.string()) : std::unexpected(document.error());
    const json &model = field(*document, "model"), &normalizer = field(*document, "normalizer");
    const json &pre = field(*document, "pre_tokenizer"), &pattern = field(normalizer, "pattern"), &entries = field(model, "vocab");
    const bool byte_fallback = field(model, "byte_fallback") == true;
    const bool byte_level_family = field(normalizer, "type") == "NFC" && field(pre, "type") == "ByteLevel" &&
        field(pre, "add_prefix_space") == false && field(pre, "use_regex") == true && !byte_fallback;
    const bool metaspace_family = field(pre, "type") == "Metaspace" && normalizer.size() == 3 && field(normalizer, "type") == "Replace" &&
        pattern.size() == 1 && field(pattern, "String") == " " && field(normalizer, "content") == metaspace_symbol &&
        field(pre, "replacement") == metaspace_symbol && field(pre, "prepend_scheme") == "always" && field(pre, "split") == true && byte_fallback;
    if (field(model, "type") != "BPE" || (!byte_level_family && !metaspace_family) || !field(model, "dropout").is_null() ||
        field(model, "ignore_merges") == true || !entries.is_object())
        return unsupported();
    auto state = std::make_unique<impl>();
    state->vocab.reserve(entries.size() + 64);
    for (const auto& [text, value] : entries.items()) {
        if (!value.is_number_integer()) return unsupported();
        state->vocab.emplace(text, value.get<token>());
    }
    const json& added_tokens = field(*document, "added_tokens");
    if (metaspace_family) {
        // A byte without its <0xNN> token in the model vocabulary falls back to its ASCII symbol.
        std::array<std::string, 256> symbols;
        for (int byte = 0; byte < 256; ++byte) {
            constexpr char hex[] = "0123456789ABCDEF";
            symbols[byte] = std::string("<0x") + hex[byte / 16] + hex[byte % 16] + ">";
            if (state->id(symbols[byte])) continue;
            if (byte >= 128 || !state->id(std::string(1, char(byte)))) return unsupported("Missing byte fallback token");
            symbols[byte] = std::string(1, char(byte));
        }
        LAYA_CHECK(state->read_rules(model, added_tokens));
        auto& kind = state->kind.emplace<metaspace>();
        for (int byte = 0; byte < 256; ++byte) kind.fallback[byte] = *state->id(symbols[byte]);
        return tokenizer(std::move(state));
    }
    LAYA_CHECK(state->read_rules(model, added_tokens));
    auto& kind = state->kind.emplace<byte_level>();
    // GPT-2 maps each byte to a printable code point. Vocabularies may lack symbols, such as those of
    // bytes that never occur in UTF-8 (GPT-NeoX lacks C0, C1 and F5-FF); only text that needs one fails.
    for (int byte = 0, next = 256; byte < 256; ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
        kind.bytes[byte] = state->id(utf8(visible ? byte : next++)).value_or(missing);
    }
    UErrorCode status = U_ZERO_ERROR;
    kind.nfc = icu::Normalizer2::getNFCInstance(status);
    kind.pattern.reset(icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"), 0, status));
    if (U_SUCCESS(status)) kind.matcher.reset(kind.pattern->matcher(status));
    if (U_FAILURE(status)) return fail(errc::model, "Cannot initialize Unicode tokenizer");
    return tokenizer(std::move(state));
}
}
