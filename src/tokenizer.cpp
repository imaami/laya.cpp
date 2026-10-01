#include "laya/runtime.hpp"
#include <unicode/bytestream.h>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/uchar.h>
#include <unicode/utf8.h>
#include <algorithm>
#include <limits>
#include <unordered_map>

namespace laya {
namespace {
constexpr char metaspace_mark[] = "\xE2\x96\x81";  // U+2581, which replaces spaces
constexpr token missing = -1;  // a byte without a symbol in the vocabulary
constexpr std::uint64_t key(token first, token second) { return std::uint64_t(std::uint32_t(first)) << 32 | std::uint32_t(second); }
struct string_hash : std::hash<std::string_view> { using is_transparent = void; };
template<class T> using string_map = std::unordered_map<std::string, T, string_hash, std::equal_to<>>;
std::unexpected<error> unsupported(std::string message = "Unsupported tokenizer configuration") { return fail(errc::model, std::move(message)); }
}

// Byte-level BPE (NFC and the GPT-2 pre-tokenizer) or, when Metaspace, metaspace BPE with byte fallback.
struct tokenizer::impl {
    struct added { std::string text; token id; bool lstrip; };
    string_map<token> vocab;
    std::unordered_map<std::uint64_t, std::pair<int, token>> merges;  // symbol pair -> rank and merged symbol
    std::array<std::vector<added>, 2> specials;  // matched before and after normalization, longest first
    std::array<token, 256> bytes;                // byte-level symbols or metaspace byte fallbacks
    bool metaspace;
    std::unique_ptr<icu::RegexPattern> pattern;
    std::unique_ptr<icu::RegexMatcher> matcher;  // reset for each text
    const icu::Normalizer2* nfc;
    mutable string_map<tokens> cache;
    mutable std::size_t cache_bytes = 0;

    std::optional<token> id(std::string_view text) const {
        const auto found = vocab.find(text);
        return found == vocab.end() ? std::nullopt : std::optional(found->second);
    }

    // Appends the BPE of one pre-tokenized word. Only uncached words can lack a symbol.
    template<bool Metaspace> result<void> merge(std::string_view word, tokens& out) const {
        if (const auto found = cache.find(word); found != cache.end()) {
            out.insert(out.end(), found->second.begin(), found->second.end());
            return {};
        }
        tokens parts;
        if constexpr (Metaspace)
            for (std::int32_t at = 0, start = 0, size = std::int32_t(word.size()); (start = at) < size;) {
                [[maybe_unused]] UChar32 codepoint;
                U8_NEXT(word.data(), at, size, codepoint);
                const auto character = word.substr(std::size_t(start), std::size_t(at - start));
                if (const auto symbol = id(character)) parts.push_back(*symbol);
                else for (const unsigned char byte : character) parts.push_back(bytes[byte]);
            }
        else for (const unsigned char byte : word) parts.push_back(bytes[byte]);
        if (std::ranges::contains(parts, missing)) return fail(errc::invalid, "Text contains a byte the tokenizer cannot encode");
        while (parts.size() > 1) {
            std::pair rule{std::numeric_limits<int>::max(), token{}};
            std::size_t at = 0;
            for (std::size_t i = 0; i + 1 < parts.size(); ++i)
                if (const auto found = merges.find(key(parts[i], parts[i + 1])); found != merges.end() && found->second.first < rule.first)
                    rule = found->second, at = i;
            if (rule.first == std::numeric_limits<int>::max()) break;
            parts[at] = rule.second;
            parts.erase(parts.begin() + std::ptrdiff_t(at) + 1);
        }
        out.insert(out.end(), parts.begin(), parts.end());
        if (word.size() > 4096) return {};
        if (cache.size() >= 16384 || cache_bytes > 4 * 1024 * 1024) cache.clear(), cache_bytes = 0;
        cache_bytes += word.size() + parts.size() * sizeof(token);
        cache.emplace(std::string(word), std::move(parts));
        return {};
    }

    // Merges each pre-tokenized word of normalized text.
    template<bool Metaspace> result<void> words(std::string_view text, tokens& out) const {
        if constexpr (Metaspace) {
            if (text.empty()) return {};
            std::string value = text.starts_with(metaspace_mark) ? "" : metaspace_mark;
            value += text;  // each word starts at a metaspace mark
            for (std::size_t begin = 0, end; begin < value.size(); begin = end) {
                end = std::min(value.find(metaspace_mark, begin + sizeof metaspace_mark - 1), value.size());
                LAYA_CHECK(merge<true>(std::string_view(value).substr(begin, end - begin), out));
            }
        } else {
            UErrorCode status = U_ZERO_ERROR;
            const auto input = icu::UnicodeString::fromUTF8(icu::StringPiece(text.data(), std::int32_t(text.size())));
            matcher->reset(input);  // the matcher reads `input` in place
            for (std::string word; matcher->find(status); word.clear()) LAYA_CHECK(merge<false>(matcher->group(status).toUTF8String(word), out));
            if (U_FAILURE(status)) return fail(errc::invalid, "Unicode tokenization failed");
        }
        return {};
    }

    // Added tokens split the text; the rest is normalized, pre-tokenized and merged.
    template<bool Metaspace> result<void> split(std::string_view text, bool normalized, tokens& out) const {
        for (std::size_t begin = 0; begin < text.size();) {
            std::size_t first = std::string_view::npos;
            const added* selected = nullptr;
            for (const auto& candidate : specials[normalized])
                if (const auto at = text.find(candidate.text, begin); at < first) first = at, selected = &candidate;
            auto prefix = text.substr(begin, (selected ? first : text.size()) - begin);
            for (std::int32_t at = std::int32_t(prefix.size()); selected && selected->lstrip && at > 0;) {
                UChar32 codepoint;  // the text before an lstrip token loses its trailing white space
                U8_PREV(prefix.data(), 0, at, codepoint);
                if (!u_isUWhiteSpace(codepoint)) break;
                prefix = prefix.substr(0, std::size_t(at));
            }
            std::string normal;
            if (normalized) LAYA_CHECK(words<Metaspace>(prefix, out));
            else if constexpr (Metaspace) {
                for (const char c : prefix) c == ' ' ? normal += metaspace_mark : normal += c;
                LAYA_CHECK(split<true>(normal, true, out));
            } else {
                UErrorCode status = U_ZERO_ERROR;
                icu::StringByteSink<std::string> sink(&normal, std::int32_t(prefix.size()));
                nfc->normalizeUTF8(0, icu::StringPiece(prefix.data(), std::int32_t(prefix.size())), sink, nullptr, status);
                if (U_FAILURE(status)) return fail(errc::invalid, "NFC normalization failed");
                LAYA_CHECK(split<false>(normal, true, out));
            }
            if (!selected) break;
            out.push_back(selected->id);
            begin = first + selected->text.size();
        }
        return {};
    }
};

void tokenizer::release::operator()(impl* state) const { delete state; }

result<tokens> tokenizer::encode(std::string_view text) const {
    tokens out;
    LAYA_CHECK(p->metaspace ? p->split<true>(text, false, out) : p->split<false>(text, false, out));
    return out;
}

std::optional<token> tokenizer::id(std::string_view text) const { return p->id(text); }

result<tokenizer> tokenizer::load(const std::filesystem::path& file) {
    auto document = read_json<nlohmann::json>(file);  // vocabulary order is irrelevant
    if (!document) return document.error().code == errc::io ? fail(errc::io, "Cannot open tokenizer: " + file.string()) : std::unexpected(document.error());
    using json = nlohmann::json;
    const json &model = field(*document, "model"), &normalizer = field(*document, "normalizer"), &pre = field(*document, "pre_tokenizer");
    const json &pattern = field(normalizer, "pattern"), &entries = field(model, "vocab"), &rules = field(model, "merges"), &added_tokens = field(*document, "added_tokens");
    const bool byte_fallback = field(model, "byte_fallback") == true;
    const bool byte_level = field(normalizer, "type") == "NFC" && field(pre, "type") == "ByteLevel" &&
        field(pre, "add_prefix_space") == false && field(pre, "use_regex") == true && !byte_fallback;
    const bool metaspace = field(pre, "type") == "Metaspace" && normalizer.size() == 3 && field(normalizer, "type") == "Replace" &&
        pattern.size() == 1 && field(pattern, "String") == " " && field(normalizer, "content") == metaspace_mark &&
        field(pre, "replacement") == metaspace_mark && field(pre, "prepend_scheme") == "always" && field(pre, "split") == true && byte_fallback;
    if (field(model, "type") != "BPE" || (!byte_level && !metaspace) || !field(model, "dropout").is_null() ||
        field(model, "ignore_merges") == true || !entries.is_object())
        return unsupported();
    tokenizer result;
    result.p.reset(new impl{});
    impl& t = *result.p;
    t.metaspace = metaspace;
    t.vocab.reserve(entries.size() + 64);
    for (const auto& [text, value] : entries.items()) {
        if (!value.is_number_integer()) return unsupported();
        t.vocab.emplace(text, value.get<token>());
    }
    // Metaspace bytes fall back to <0xNN> tokens of the model vocabulary, or to ASCII symbols. The GPT-2 byte-level
    // symbols are printable code points, which vocabularies may lack for bytes UTF-8 never has (GPT-NeoX lacks
    // C0, C1 and F5-FF); only text that needs a missing symbol fails.
    std::array<std::string, 256> symbols;
    for (int byte = 0, next = 256; byte < 256; ++byte) {
        constexpr char hex[] = "0123456789ABCDEF";
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
        if (!metaspace) icu::UnicodeString(UChar32(visible ? byte : next++)).toUTF8String(symbols[byte]);
        else if (!t.id(symbols[byte] = std::string("<0x") + hex[byte / 16] + hex[byte % 16] + ">")) {
            if (byte >= 128 || !t.id(symbols[byte] = std::string(1, char(byte)))) return unsupported("Missing byte fallback token");
        }
    }
    if (!rules.is_array() || !std::ranges::all_of(rules, [](const json& pair) {
            return pair.is_array() && pair.size() == 2 && pair[0].is_string() && pair[1].is_string();
        }))
        return unsupported("Unsupported BPE merge format");
    if (!added_tokens.is_array()) return unsupported();
    for (const auto& entry : added_tokens) {
        if (field(entry, "single_word") != false || field(entry, "rstrip") != false) return unsupported("Unsupported added-token boundary flags");
        const json &text = field(entry, "content"), &value = field(entry, "id"), &normalized = field(entry, "normalized"), &lstrip = field(entry, "lstrip");
        if (!text.is_string() || text.get_ref<const std::string&>().empty() || !value.is_number_integer() || !normalized.is_boolean() || !lstrip.is_boolean())
            return unsupported();
        t.specials[normalized.get<bool>()].push_back({text.get<std::string>(), value.get<token>(), lstrip.get<bool>()});
        t.vocab.insert_or_assign(text.get<std::string>(), value.get<token>());
    }
    for (auto& list : t.specials) std::ranges::stable_sort(list, std::ranges::greater{}, [](const impl::added& entry) { return entry.text.size(); });
    // A reachable merge joins two vocabulary symbols into another; the first listing of a pair keeps its rank.
    t.merges.reserve(rules.size());
    for (std::size_t rank = 0; rank < rules.size(); ++rank) {
        const auto &first = rules[rank][0].get_ref<const std::string&>(), &second = rules[rank][1].get_ref<const std::string&>();
        const auto left = t.id(first), right = t.id(second), merged = t.id(first + second);
        if (left && right && !merged) return unsupported("Unsupported BPE merge format");
        if (left && right) t.merges.try_emplace(key(*left, *right), int(rank), *merged);
    }
    for (int byte = 0; byte < 256; ++byte) t.bytes[byte] = t.id(symbols[byte]).value_or(missing);
    if (metaspace) return result;
    UErrorCode status = U_ZERO_ERROR;
    t.nfc = icu::Normalizer2::getNFCInstance(status);
    t.pattern.reset(icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"), 0, status));
    if (U_SUCCESS(status)) t.matcher.reset(t.pattern->matcher(status));
    if (U_FAILURE(status)) return fail(errc::model, "Cannot initialize Unicode tokenizer");
    return result;
}
}
