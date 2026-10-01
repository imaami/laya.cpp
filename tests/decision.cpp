#include "laya/decision.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

void check(bool ok) {
    if (!ok) {
        std::cerr << "decision check failed\n";
        std::exit(1);
    }
}
int main() {
    auto uniform = laya::calibrate(std::array{0.f, 0.f, 0.f});
    check(uniform && std::abs(uniform->expected_score - 1) < 1e-12);
    check(uniform->confidence < 1e-12);
    auto peaked = laya::calibrate(std::array{-1000.f, 1000.f});
    check(peaked && peaked->choice == 1 && peaked->probabilities[1] == 1);
    auto soft = laya::calibrate(std::array{0.f, 2.f}, 2);
    check(soft && std::abs(soft->probabilities[1] - 0.7310585786300049) < 1e-12);
    auto single = laya::calibrate(std::array{5.f});
    check(single && single->confidence == 1);
    auto infinite = laya::calibrate(std::array{std::numeric_limits<float>::infinity()});
    check(!infinite && infinite.error().code == laya::errc::invalid);
}
