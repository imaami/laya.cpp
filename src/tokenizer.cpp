#include "laya/tokenizer.hpp"
#include <unicode/bytestream.h>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/uchar.h>
#include <unicode/utf8.h>
#include <algorithm>
#include <limits>

namespace laya {
namespace {
constexpr char metaspace_symbol[] = "\xE2\x96\x81";  // U+2581, which replaces spaces
constexpr std::string_view metaspace_mark = metaspace_symbol;

constexpr std::uint64_t pair_key(token first, token second) {
    return std::uint64_t(std::uint32_t(first)) << 32 | std::uint32_t(second);
}

std::string utf8(UChar32 codepoint) {
    char buffer[U8_MAX_LENGTH];
    std::int32_t length = 0;
    [[maybe_unused]] UBool error = false;  // byte-level symbols are valid code points
    U8_APPEND(buffer, length, U8_MAX_LENGTH, codepoint, error);
    return std::string(buffer, std::size_t(length));
}

std::string byte_token(int byte) {
    constexpr char hex[] = "0123456789ABCDEF";
    return std::string("<0x") + hex[byte / 16] + hex[byte % 16] + ">";
}

// The text before an added token with lstrip loses its trailing white space.
std::string_view strip_trailing_space(std::string_view text) {
    auto end = std::int32_t(text.size());
    while (end > 0) {
        std::int32_t at = end;
        UChar32 codepoint;
        U8_PREV(text.data(), 0, at, codepoint);
        if (!u_isUWhiteSpace(codepoint)) break;
        end = at;
    }
    return text.substr(0, std::size_t(end));
}

std::unexpected<error> unsupported(std::string message = "Unsupported tokenizer configuration") {
    return fail(errc::model, std::move(message));
}
}

template<class Derived>
result<void> bpe<Derived>::read_vocabulary(const json& model) {
    const json& entries = field(model, "vocab");
    if (!entries.is_object()) return unsupported();
    vocab.reserve(entries.size() + 64);
    for (const auto& [text, value] : entries.items()) {
        if (!value.is_number_integer()) return unsupported();
        vocab.emplace(text, value.template get<token>());
    }
    return {};
}

template<class Derived>
result<void> bpe<Derived>::read_rules(const json& model, const json& added_tokens) {
    const json& rules = field(model, "merges");
    if (!rules.is_array()) return unsupported("Unsupported BPE merge format");
    for (const auto& pair : rules)
        if (!pair.is_array() || pair.size() != 2 || !pair[0].is_string() || !pair[1].is_string())
            return unsupported("Unsupported BPE merge format");
    if (!added_tokens.is_array()) return unsupported();
    for (const auto& entry : added_tokens) {
        if (field(entry, "single_word") != false || field(entry, "rstrip") != false)
            return unsupported("Unsupported added-token boundary flags");
        const json &text = field(entry, "content"), &id = field(entry, "id");
        const json &normalized = field(entry, "normalized"), &lstrip = field(entry, "lstrip");
        if (!text.is_string() || text.empty() || !id.is_number_integer() || !normalized.is_boolean() || !lstrip.is_boolean())
            return unsupported();
        const auto value = id.template get<token>();
        const auto& content = text.template get_ref<const std::string&>();
        specials[normalized.template get<bool>()].push_back({content, value, lstrip.template get<bool>()});
        vocab.insert_or_assign(content, value);
    }
    for (auto& list : specials)
        std::ranges::stable_sort(list, std::ranges::greater{}, [](const added& entry) { return entry.text.size(); });
    // Merges apply to symbol IDs. A reachable merge joins two vocabulary symbols
    // into another; the first listing of a pair keeps its rank.
    merges.reserve(rules.size());
    int rank = 0;
    for (const auto& pair : rules) {
        const auto& first = pair[0].template get_ref<const std::string&>();
        const auto& second = pair[1].template get_ref<const std::string&>();
        const auto left = vocab.find(first), right = vocab.find(second);
        if (left != vocab.end() && right != vocab.end()) {
            const auto merged = vocab.find(first + second);
            if (merged == vocab.end()) return unsupported("Unsupported BPE merge format");
            merges.try_emplace(pair_key(left->second, right->second), rule{rank, merged->second});
        }
        ++rank;
    }
    return {};
}

template<class Derived>
std::optional<token> bpe<Derived>::id(std::string_view text) const {
    const auto found = vocab.find(text);
    if (found == vocab.end()) return std::nullopt;
    return found->second;
}

template<class Derived>
void bpe<Derived>::merge(std::string_view word, tokens& out) const {
    if (const auto found = cache.find(word); found != cache.end()) {
        out.insert(out.end(), found->second.begin(), found->second.end());
        return;
    }
    tokens parts;
    self().symbols(word, parts);
    while (parts.size() > 1) {
        rule best{std::numeric_limits<int>::max(), 0};
        std::size_t at = 0;
        for (std::size_t i = 0; i + 1 < parts.size(); ++i)
            if (const auto found = merges.find(pair_key(parts[i], parts[i + 1]));
                found != merges.end() && found->second.rank < best.rank) {
                best = found->second;
                at = i;
            }
        if (best.rank == std::numeric_limits<int>::max()) break;
        parts[at] = best.merged;
        parts.erase(parts.begin() + std::ptrdiff_t(at) + 1);
    }
    out.insert(out.end(), parts.begin(), parts.end());
    if (word.size() <= 4096) {
        if (cache.size() >= 16384 || cache_bytes > 4 * 1024 * 1024) {
            cache.clear();
            cache_bytes = 0;
        }
        cache_bytes += word.size() + parts.size() * sizeof(token);
        cache.emplace(std::string(word), std::move(parts));
    }
}

template<class Derived>
result<void> bpe<Derived>::split(std::string_view text, bool normalized, tokens& out) const {
    const auto& candidates = specials[normalized];
    std::size_t begin = 0;
    while (begin < text.size()) {
        std::size_t first = std::string_view::npos;
        const added* selected = nullptr;
        for (const auto& candidate : candidates)
            if (const auto at = text.find(candidate.text, begin); at < first) {
                first = at;
                selected = &candidate;
            }
        auto prefix = text.substr(begin, (selected ? first : text.size()) - begin);
        if (selected && selected->lstrip) prefix = strip_trailing_space(prefix);
        if (normalized) {
            LAYA_CHECK(self().words(prefix, out));
        } else {
            std::string normal;
            LAYA_CHECK(self().normalize(prefix, normal));
            LAYA_CHECK(split(normal, true, out));
        }
        if (!selected) break;
        out.push_back(selected->id);
        begin = first + selected->text.size();
    }
    return {};
}

template<class Derived>
result<tokens> bpe<Derived>::encode(std::string_view text) const {
    tokens out;
    LAYA_CHECK(split(text, false, out));
    return out;
}

template class bpe<byte_level_tokenizer>;
template class bpe<metaspace_tokenizer>;

struct byte_level_tokenizer::unicode {
    std::unique_ptr<icu::RegexPattern> pattern;
    std::unique_ptr<icu::RegexMatcher> matcher;  // reset for each text
    const icu::Normalizer2* nfc = nullptr;
};

byte_level_tokenizer::byte_level_tokenizer() : icu(std::make_unique<unicode>()) {}
byte_level_tokenizer::byte_level_tokenizer(byte_level_tokenizer&&) noexcept = default;
byte_level_tokenizer& byte_level_tokenizer::operator=(byte_level_tokenizer&&) noexcept = default;
byte_level_tokenizer::~byte_level_tokenizer() = default;

result<byte_level_tokenizer> byte_level_tokenizer::load(const json& document) {
    byte_level_tokenizer tokenizer;
    const json& model = field(document, "model");
    LAYA_CHECK(tokenizer.read_vocabulary(model));
    LAYA_CHECK(tokenizer.read_rules(model, field(document, "added_tokens")));
    // GPT-2 maps each byte to a printable code point.
    for (int byte = 0, next = 256; byte < 256; ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
        const auto symbol = tokenizer.id(utf8(visible ? byte : next++));
        if (!symbol) return unsupported();
        tokenizer.bytes[byte] = *symbol;
    }
    UErrorCode status = U_ZERO_ERROR;
    tokenizer.icu->nfc = icu::Normalizer2::getNFCInstance(status);
    tokenizer.icu->pattern.reset(icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"), 0, status));
    if (U_SUCCESS(status)) tokenizer.icu->matcher.reset(tokenizer.icu->pattern->matcher(status));
    if (U_FAILURE(status)) return fail(errc::model, "Cannot initialize Unicode tokenizer");
    return tokenizer;
}

result<void> byte_level_tokenizer::normalize(std::string_view text, std::string& out) const {
    UErrorCode status = U_ZERO_ERROR;
    icu::StringByteSink<std::string> sink(&out, std::int32_t(text.size()));
    icu->nfc->normalizeUTF8(0, icu::StringPiece(text.data(), std::int32_t(text.size())), sink, nullptr, status);
    if (U_FAILURE(status)) return fail(errc::invalid, "NFC normalization failed");
    return {};
}

result<void> byte_level_tokenizer::words(std::string_view text, tokens& out) const {
    UErrorCode status = U_ZERO_ERROR;
    const auto input = icu::UnicodeString::fromUTF8(icu::StringPiece(text.data(), std::int32_t(text.size())));
    auto& matcher = *icu->matcher;
    matcher.reset(input);
    std::string word;
    while (matcher.find(status)) {
        word.clear();
        matcher.group(status).toUTF8String(word);
        merge(word, out);
    }
    if (U_FAILURE(status)) return fail(errc::invalid, "Unicode tokenization failed");
    return {};
}

void byte_level_tokenizer::symbols(std::string_view word, tokens& out) const {
    out.reserve(word.size());
    for (const unsigned char byte : word) out.push_back(bytes[byte]);
}

result<metaspace_tokenizer> metaspace_tokenizer::load(const json& document) {
    metaspace_tokenizer tokenizer;
    const json& model = field(document, "model");
    LAYA_CHECK(tokenizer.read_vocabulary(model));
    // A byte without its <0xNN> token falls back to its ASCII symbol.
    std::array<std::string, 256> symbols;
    for (int byte = 0; byte < 256; ++byte) {
        symbols[byte] = byte_token(byte);
        if (tokenizer.id(symbols[byte])) continue;
        if (byte >= 128 || !tokenizer.id(std::string(1, char(byte)))) return unsupported("Missing byte fallback token");
        symbols[byte] = std::string(1, char(byte));
    }
    LAYA_CHECK(tokenizer.read_rules(model, field(document, "added_tokens")));
    for (int byte = 0; byte < 256; ++byte) tokenizer.fallback[byte] = *tokenizer.id(symbols[byte]);
    return tokenizer;
}

result<void> metaspace_tokenizer::normalize(std::string_view text, std::string& out) const {
    out.reserve(text.size());
    for (const char c : text)
        if (c == ' ') out += metaspace_mark;
        else out += c;
    return {};
}

result<void> metaspace_tokenizer::words(std::string_view text, tokens& out) const {
    if (text.empty()) return {};
    std::string value;
    if (!text.starts_with(metaspace_mark)) value = metaspace_mark;
    value += text;
    // Each word starts at a metaspace mark.
    const std::string_view words = value;
    for (std::size_t begin = 0; begin < words.size();) {
        auto end = words.find(metaspace_mark, begin + metaspace_mark.size());
        if (end == std::string_view::npos) end = words.size();
        merge(words.substr(begin, end - begin), out);
        begin = end;
    }
    return {};
}

void metaspace_tokenizer::symbols(std::string_view word, tokens& out) const {
    for (std::int32_t at = 0, size = std::int32_t(word.size()); at < size;) {
        const std::int32_t start = at;
        UChar32 codepoint;
        U8_NEXT(word.data(), at, size, codepoint);
        const auto character = word.substr(std::size_t(start), std::size_t(at - start));
        if (const auto symbol = id(character)) out.push_back(*symbol);
        else for (const unsigned char byte : character) out.push_back(fallback[byte]);
    }
}

result<any_tokenizer> load_tokenizer(const std::filesystem::path& file) {
    auto document = read_json(file);
    if (!document) {
        if (document.error().code == errc::io) return fail(errc::io, "Cannot open tokenizer: " + file.string());
        return std::unexpected(std::move(document).error());
    }
    const json &model = field(*document, "model"), &normalizer = field(*document, "normalizer");
    const json &pre = field(*document, "pre_tokenizer"), &pattern = field(normalizer, "pattern");
    const bool byte_fallback = field(model, "byte_fallback") == true;
    const bool byte_level = field(normalizer, "type") == "NFC" && field(pre, "type") == "ByteLevel" &&
                            field(pre, "add_prefix_space") == false && field(pre, "use_regex") == true && !byte_fallback;
    const bool metaspace = field(pre, "type") == "Metaspace" && normalizer.size() == 3 && field(normalizer, "type") == "Replace" &&
                           pattern.size() == 1 && field(pattern, "String") == " " && field(normalizer, "content") == metaspace_symbol &&
                           field(pre, "replacement") == metaspace_symbol && field(pre, "prepend_scheme") == "always" &&
                           field(pre, "split") == true && byte_fallback;
    if (field(model, "type") != "BPE" || (!byte_level && !metaspace) || !field(model, "dropout").is_null() ||
        field(model, "ignore_merges") == true)
        return unsupported();
    if (metaspace) {
        LAYA_TRY(tokenizer, metaspace_tokenizer::load(*document));
        return any_tokenizer(std::in_place_type<metaspace_tokenizer>, std::move(*tokenizer));
    }
    LAYA_TRY(tokenizer, byte_level_tokenizer::load(*document));
    return any_tokenizer(std::in_place_type<byte_level_tokenizer>, std::move(*tokenizer));
}
}
