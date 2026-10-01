#pragma once
#include "laya/runtime.hpp"
#include <span>
#include <vector>

namespace laya {
struct decision {
    std::vector<double> probabilities;
    unsigned choice;
    double expected_score;
    double confidence;
};
// Logits must contain only valid options; temperature is clamped to 0.001.
[[nodiscard]] result<decision> calibrate(std::span<const float> logits, double temperature = 1.0);
}
