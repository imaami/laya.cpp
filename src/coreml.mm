#include "laya/coreml.hpp"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <system_error>
#include <tuple>
#include <vector>

namespace laya {
namespace {

std::string ns_string(NSString* value) {
    if (!value) return {};
    const char* text = value.UTF8String;
    return text ? std::string(text) : std::string();
}

std::string ns_error(NSError* error) {
    return error ? ns_string(error.localizedDescription) : "unknown Core ML error";
}

std::unexpected<error> incompatible(const std::string& detail) {
    return fail(errc::model, "Incompatible Core ML model: " + detail);
}

result<void> require_multi_array(NSDictionary<NSString*, MLFeatureDescription*>* descriptions, NSString* name,
                                 MLMultiArrayDataType type, const char* role) {
    MLFeatureDescription* description = [descriptions objectForKey:name];
    if (!description) return incompatible(std::string("missing ") + role + " '" + ns_string(name) + "'");
    if (description.type != MLFeatureTypeMultiArray)
        return incompatible(std::string(role) + " '" + ns_string(name) + "' is not a multi-array");
    MLMultiArrayConstraint* constraint = description.multiArrayConstraint;
    if (constraint && constraint.dataType != type)
        return incompatible(std::string(role) + " '" + ns_string(name) + "' has an unexpected element type");
    return {};
}

result<std::size_t> checked_size(int first, int second, const char* name) {
    if (first < 1 || second < 1 ||
        static_cast<std::size_t>(first) > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(second))
        return fail(errc::backend, std::string("Invalid Core ML ") + name + " dimensions");
    return static_cast<std::size_t>(first) * static_cast<std::size_t>(second);
}

result<MLMultiArray*> make_input(NSArray<NSNumber*>* shape, const std::int32_t* source, std::size_t count, const char* name) {
    NSError* error = nil;
    MLMultiArray* value = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeInt32 error:&error];
    if (!value) return fail(errc::backend, "Cannot allocate Core ML input '" + std::string(name) + "': " + ns_error(error));
    if (value.count != static_cast<NSInteger>(count)) {
        [value release];
        return fail(errc::backend, "Core ML input '" + std::string(name) + "' has an unexpected element count");
    }
    std::memcpy(value.dataPointer, source, count * sizeof(std::int32_t));
    return [value autorelease];
}

result<void> copy_output(MLMultiArray* value, const char* name, int rows, int columns, std::vector<float>& destination) {
    if (!value) return incompatible(std::string("missing output '") + name + "'");
    if (value.shape.count != 2 || value.shape[0].intValue != rows || value.shape[1].intValue != columns)
        return incompatible(std::string("output '") + name + "' does not have shape [batch, columns]");
    if (value.dataType != MLMultiArrayDataTypeFloat32)
        return incompatible(std::string("output '") + name + "' is not Float32");
    const NSInteger row_stride = value.strides[0].integerValue;
    const NSInteger column_stride = value.strides[1].integerValue;
    const auto* source = static_cast<const float*>(value.dataPointer);
    LAYA_TRY(size, checked_size(rows, columns, "output"));
    destination.resize(*size);
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < columns; ++column)
            destination[static_cast<std::size_t>(row) * columns + column] = source[row * row_stride + column * column_stride];
    return {};
}

result<int> manifest_dimension(const json& bucket, const char* key) {
    const json& value = field(bucket, key);
    if (!value.is_number_integer()) return incompatible(std::string("bucket is missing integer '") + key + "'");
    const auto n = value.get<std::int64_t>();
    if (n < 1 || n > std::numeric_limits<int>::max()) return incompatible(std::string("bucket has invalid '") + key + "'");
    return static_cast<int>(n);
}

}  // namespace

struct coreml_runtime::impl {
    struct bucket {
        int batch = 0;
        int length = 0;
        int options = 0;
        std::filesystem::path compiled;
        std::string name;
    };

    int vocabulary = 0;
    int action_count = 0;
    int max_len = 0;
    std::vector<bucket> buckets;
    std::map<std::string, MLModel*> models;

    ~impl() {
        for (auto& entry : models) [entry.second release];
    }

    result<void> validate_batch(const batch& input) const {
        if (input.size < 1 || input.length < 1) return fail(errc::backend, "Invalid Core ML model batch");
        LAYA_TRY(cells, checked_size(input.size, input.length, "batch"));
        if (input.length > max_len || input.options < 2 || input.options > 255 || input.ids.size() != *cells)
            return fail(errc::backend, "Invalid Core ML model batch");
        LAYA_TRY(markers, checked_size(input.size, input.options, "batch"));
        if (input.lengths.size() != static_cast<std::size_t>(input.size) || input.types.size() != static_cast<std::size_t>(input.size) ||
            input.counts.size() != static_cast<std::size_t>(input.size) || input.markers.size() != *markers)
            return fail(errc::backend, "Invalid Core ML batch metadata sizes");
        for (int row = 0; row < input.size; ++row) {
            if (input.lengths[row] < 1 || input.lengths[row] > input.length || input.types[row] < 0 || input.types[row] > 2 ||
                input.counts[row] < 2 || input.counts[row] > input.options)
                return fail(errc::backend, "Invalid Core ML batch row metadata");
            for (int column = 0; column < input.counts[row]; ++column) {
                const auto marker = static_cast<std::int64_t>(input.markers[static_cast<std::size_t>(row) * input.options + column]) -
                                    static_cast<std::int64_t>(row) * input.length;
                if (marker < 0 || marker >= input.lengths[row])
                    return fail(errc::backend, "Core ML option marker outside its sequence");
            }
        }
        for (const auto id : input.ids)
            if (id < 0 || id >= vocabulary) return fail(errc::backend, "Core ML token ID outside the vocabulary");
        return {};
    }

    result<const bucket*> bucket_for(const batch& input) const {
        const bucket* selected = nullptr;
        for (const auto& candidate : buckets) {
            if (candidate.batch < input.size || candidate.length < input.length || candidate.options < input.options) continue;
            if (!selected || std::tie(candidate.batch, candidate.length, candidate.options) <
                                 std::tie(selected->batch, selected->length, selected->options))
                selected = &candidate;
        }
        if (selected) return selected;
        std::string available;
        for (const auto& candidate : buckets) {
            if (!available.empty()) available += ", ";
            available += candidate.name;
        }
        return fail(errc::unsupported, "Core ML bucket unavailable for batch=" + std::to_string(input.size) +
                                       ", length=" + std::to_string(input.length) + ", options=" + std::to_string(input.options) +
                                       "; need a bucket whose batch/length/options are at least this large. Available: " + available);
    }

    result<batch> pad_for_bucket(const batch& input, const bucket& selected) const {
        batch padded;
        padded.size = selected.batch;
        padded.length = selected.length;
        padded.options = selected.options;
        LAYA_TRY(cells, checked_size(padded.size, padded.length, "padded batch"));
        LAYA_TRY(markers, checked_size(padded.size, padded.options, "padded batch"));
        padded.ids.assign(*cells, 0);
        padded.markers.assign(*markers, 0);
        padded.lengths.assign(padded.size, std::min(3, padded.length));
        padded.counts.assign(padded.size, 2);
        padded.types.assign(padded.size, 0);
        std::copy(input.lengths.begin(), input.lengths.end(), padded.lengths.begin());
        std::copy(input.counts.begin(), input.counts.end(), padded.counts.begin());
        std::copy(input.types.begin(), input.types.end(), padded.types.begin());
        for (int row = 0; row < padded.size; ++row) {
            const auto offset = static_cast<std::int64_t>(row) * padded.length;
            for (int column = 0; column < padded.options; ++column) {
                const auto local = std::min(column + 1, padded.length - 1);
                padded.markers[static_cast<std::size_t>(row) * padded.options + column] = static_cast<std::int32_t>(offset + local);
            }
        }
        for (int row = 0; row < input.size; ++row) {
            std::copy_n(input.ids.begin() + static_cast<std::size_t>(row) * input.length, input.length,
                        padded.ids.begin() + static_cast<std::size_t>(row) * padded.length);
            for (int column = 0; column < input.counts[row]; ++column) {
                const auto old_marker = input.markers[static_cast<std::size_t>(row) * input.options + column];
                const auto local_marker = static_cast<std::int64_t>(old_marker) - static_cast<std::int64_t>(row) * input.length;
                const auto new_marker = static_cast<std::int64_t>(row) * padded.length + local_marker;
                if (local_marker < 0 || local_marker >= input.length || new_marker > std::numeric_limits<std::int32_t>::max())
                    return fail(errc::backend, "Cannot pad Core ML marker index safely");
                padded.markers[static_cast<std::size_t>(row) * padded.options + column] = static_cast<std::int32_t>(new_marker);
            }
        }
        return padded;
    }

    result<MLModel*> model_for(const bucket& selected) {
        const auto& path = selected.compiled;
        std::error_code status;
        if (!std::filesystem::is_directory(path, status))
            return fail(errc::io, "Core ML compiled bucket is missing: " + path.string());
        const auto key = path.string();
        if (const auto found = models.find(key); found != models.end()) return found->second;

        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:key.c_str()]];
        MLModelConfiguration* configuration = [[MLModelConfiguration alloc] init];
        configuration.computeUnits = MLComputeUnitsAll;
        NSError* error = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:url configuration:configuration error:&error];
        [configuration release];
        if (!model) return fail(errc::backend, "Cannot load Core ML model " + key + ": " + ns_error(error));

        const auto inputs = model.modelDescription.inputDescriptionsByName;
        for (NSString* name in @[ @"ids", @"lengths", @"markers", @"counts", @"types" ])
            LAYA_CHECK(require_multi_array(inputs, name, MLMultiArrayDataTypeInt32, "input"));
        const auto outputs = model.modelDescription.outputDescriptionsByName;
        LAYA_CHECK(require_multi_array(outputs, @"logits", MLMultiArrayDataTypeFloat32, "output"));
        LAYA_CHECK(require_multi_array(outputs, @"actions", MLMultiArrayDataTypeFloat32, "output"));

        [model retain];
        return models.emplace(key, model).first->second;
    }

    result<raw_result> forward(const batch& input) {
        LAYA_CHECK(validate_batch(input));
        LAYA_TRY(selected, bucket_for(input));
        LAYA_TRY(padded, pad_for_bucket(input, **selected));
        @autoreleasepool {
            LAYA_TRY(model, model_for(**selected));
            NSError* error = nil;
            LAYA_TRY(ids, make_input(@[ @(padded->size), @(padded->length) ], padded->ids.data(), padded->ids.size(), "ids"));
            LAYA_TRY(lengths, make_input(@[ @(padded->size) ], padded->lengths.data(), padded->lengths.size(), "lengths"));
            LAYA_TRY(markers, make_input(@[ @(padded->size), @(padded->options) ], padded->markers.data(), padded->markers.size(), "markers"));
            LAYA_TRY(counts, make_input(@[ @(padded->size) ], padded->counts.data(), padded->counts.size(), "counts"));
            LAYA_TRY(types, make_input(@[ @(padded->size) ], padded->types.data(), padded->types.size(), "types"));
            NSDictionary* features = @{
                @"ids": [MLFeatureValue featureValueWithMultiArray:*ids],
                @"lengths": [MLFeatureValue featureValueWithMultiArray:*lengths],
                @"markers": [MLFeatureValue featureValueWithMultiArray:*markers],
                @"counts": [MLFeatureValue featureValueWithMultiArray:*counts],
                @"types": [MLFeatureValue featureValueWithMultiArray:*types],
            };
            MLDictionaryFeatureProvider* provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
            if (!provider) return fail(errc::backend, "Cannot construct Core ML feature provider: " + ns_error(error));

            const auto started = std::chrono::steady_clock::now();
            id<MLFeatureProvider> prediction = [*model predictionFromFeatures:provider error:&error];
            [provider release];
            if (!prediction) return fail(errc::backend, "Core ML prediction failed: " + ns_error(error));

            raw_result output;
            output.action_count = action_count;
            std::vector<float> padded_logits;
            LAYA_CHECK(copy_output([[prediction featureValueForName:@"logits"] multiArrayValue], "logits", padded->size,
                                   padded->options, padded_logits));
            LAYA_TRY(logits, checked_size(input.size, input.options, "result"));
            output.logits.resize(*logits);
            for (int row = 0; row < input.size; ++row)
                std::copy_n(padded_logits.begin() + static_cast<std::size_t>(row) * padded->options, input.options,
                            output.logits.begin() + static_cast<std::size_t>(row) * input.options);
            std::vector<float> padded_actions;
            LAYA_CHECK(copy_output([[prediction featureValueForName:@"actions"] multiArrayValue], "actions", padded->size,
                                   action_count, padded_actions));
            LAYA_TRY(actions, checked_size(input.size, action_count, "result"));
            output.actions.assign(padded_actions.begin(), padded_actions.begin() + *actions);
            output.compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            return output;
        }
    }
};

result<coreml_runtime> coreml_runtime::load(const checkpoint& model) {
    auto state = std::make_unique<impl>();
    state->vocabulary = model.arch.vocabulary;
    state->action_count = model.serving.actions;
    state->max_len = model.serving.max_len;
    const auto manifest_path = model.directory / "coreml/manifest.json";
    std::error_code status;
    if (!std::filesystem::is_regular_file(manifest_path, status))
        return fail(errc::io, "Core ML manifest missing: " + manifest_path.string());
    LAYA_TRY(manifest, read_json(manifest_path));
    if (!manifest->is_object() || field(*manifest, "schema_version") != 1)
        return fail(errc::model, "Unsupported Core ML manifest schema; expected schema_version=1: " + manifest_path.string());
    const json& buckets = field(*manifest, "buckets");
    if (!buckets.is_array() || buckets.empty())
        return fail(errc::model, "Incompatible Core ML manifest: missing nonempty buckets array: " + manifest_path.string());
    for (const auto& entry : buckets) {
        if (!entry.is_object()) return incompatible("bucket entry is not an object");
        impl::bucket parsed;
        LAYA_TRY(rows, manifest_dimension(entry, "batch"));
        LAYA_TRY(length, manifest_dimension(entry, "length"));
        LAYA_TRY(options, manifest_dimension(entry, "options"));
        parsed.batch = *rows;
        parsed.length = *length;
        parsed.options = *options;
        if (parsed.options < 2 || parsed.options > 255) return incompatible("bucket options must be between 2 and 255");
        if (parsed.length > state->max_len) return incompatible("bucket length exceeds checkpoint max_len");
        const json &name = field(entry, "name"), &compiled = field(entry, "compiled");
        if (!name.is_string() || !compiled.is_string()) return incompatible("bucket is missing name or compiled artifact path");
        const auto expected_name = "b" + std::to_string(parsed.batch) + "-l" + std::to_string(parsed.length) + "-o" + std::to_string(parsed.options);
        const auto expected_compiled = expected_name + "/Laya.mlmodelc";
        if (name.get_ref<const std::string&>() != expected_name || compiled.get_ref<const std::string&>() != expected_compiled)
            return incompatible("bucket metadata does not match the required b{B}-l{L}-o{O}/Laya.mlmodelc layout");
        LAYA_CHECK(checked_size(parsed.batch, parsed.length, "bucket"));
        LAYA_CHECK(checked_size(parsed.batch, parsed.options, "bucket"));
        parsed.name = expected_name;
        parsed.compiled = model.directory / "coreml" / expected_compiled;
        state->buckets.push_back(std::move(parsed));
    }
    return coreml_runtime(std::move(state));
}

coreml_runtime::coreml_runtime(std::unique_ptr<impl> state) : p(std::move(state)) {}
coreml_runtime::coreml_runtime(coreml_runtime&&) noexcept = default;
coreml_runtime& coreml_runtime::operator=(coreml_runtime&&) noexcept = default;
coreml_runtime::~coreml_runtime() = default;
result<raw_result> coreml_runtime::forward(const batch& input) { return p->forward(input); }

}  // namespace laya
