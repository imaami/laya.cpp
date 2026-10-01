#pragma once
#include "laya/runtime.hpp"
#include <memory>

namespace laya {
// Internal bridge implemented in Objective-C++. The exported model owns all
// precision and accelerator decisions; this interface stays at the
// batch/raw_result boundary.
class coreml_runtime {
public:
    // Reads coreml/manifest.json; compiled buckets load on first use.
    [[nodiscard]] static result<coreml_runtime> load(const checkpoint& model);
    coreml_runtime(coreml_runtime&&) noexcept;
    coreml_runtime& operator=(coreml_runtime&&) noexcept;
    ~coreml_runtime();

    [[nodiscard]] result<raw_result> forward(const batch& input);

private:
    struct impl;
    explicit coreml_runtime(std::unique_ptr<impl> state);
    std::unique_ptr<impl> p;
};
}  // namespace laya
