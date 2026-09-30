#pragma once
#include <cstdint>
#include <vector>

namespace laya {
// One model batch: a row per question, padded to the longest row.
struct batch {
    int size = 0, length = 0, options = 0;
    std::vector<std::int32_t> ids, lengths, markers, counts, types;
};

// Logits per option marker and action-head outputs per row.
struct raw_result {
    std::vector<float> logits, actions;
    int action_count = 0;
    double compute_ms = 0;
};
}
