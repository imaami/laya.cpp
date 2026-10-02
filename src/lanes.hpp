#pragma once
#include "laya/runtime.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

// Lanes: the rows of a batch that run together, each lane padded only to its
// own longest row. A lane computes exactly what a batch of just its rows
// computes; the planner is constexpr so its contract is checked at compile time.
namespace laya {
struct lane {
    int length = 0, options = 0;  // longest row and most options
    std::vector<int> rows;        // batch rows, in request order
    constexpr int size() const { return int(rows.size()); }
    friend constexpr bool operator==(const lane&, const lane&) = default;
};

// Every row of a batch in one lane, padded as the batch pads it.
constexpr lane whole(const batch& input) {
    lane all{input.length, input.options, std::vector<int>(std::size_t(input.size))};
    std::ranges::iota(all.rows, 0);
    return all;
}

// Modeled device work. Per token and layer, projections take 4W^2 + 3WI
// products and attention 2W per attended position; explicit attention runs
// about 6 times fewer products per second than projections (RX 9070 XT
// Vulkan profile). A token of a lane of length L thus costs kappa + L
// position units, and each further lane costs the launches and idle compute
// units of lane_overhead short tokens.
constexpr std::int64_t kappa(const architecture& arch) { return (4 * std::int64_t(arch.width) + 3 * std::int64_t(arch.intermediate)) / 12; }
inline constexpr std::int64_t lane_overhead = 512;
constexpr std::int64_t lane_cost(std::int64_t rows, std::int64_t length, std::int64_t k) {
    return rows * length * (k + length) + lane_overhead * k;
}

// The lanes of least total cost, the fewest among equal costs, ordered by
// length. Lanes span consecutive row lengths: a row costs least in the
// shortest lane that holds it, so no other partition can cost less.
constexpr std::vector<lane> group_lanes(const batch& input, const architecture& arch) {
    const std::int64_t k = kappa(arch);
    std::vector<int> order(std::size_t(input.size));
    std::ranges::iota(order, 0);
    std::ranges::sort(order, {}, [&](int row) { return std::pair{input.lengths[std::size_t(row)], row}; });
    // Distinct lengths; bounds[j] rows of order are no longer than lengths[j - 1].
    std::vector<int> lengths, bounds{0};
    for (std::size_t i = 0; i < order.size(); ++i)
        if (i + 1 == order.size() || input.lengths[std::size_t(order[i])] != input.lengths[std::size_t(order[i + 1])])
            lengths.push_back(input.lengths[std::size_t(order[i])]), bounds.push_back(int(i) + 1);
    // best[j]: the cheapest lanes over the first j lengths; the last starts after length `from`.
    struct plan { std::int64_t cost = 0; int count = 0, from = 0; };
    std::vector<plan> best(bounds.size());
    for (std::size_t j = 1; j < best.size(); ++j) {
        best[j].cost = std::numeric_limits<std::int64_t>::max();
        for (std::size_t i = 0; i < j; ++i) {
            const plan option{best[i].cost + lane_cost(bounds[j] - bounds[i], lengths[j - 1], k), best[i].count + 1, int(i)};
            if (std::pair{option.cost, option.count} < std::pair{best[j].cost, best[j].count}) best[j] = option;
        }
    }
    std::vector<lane> lanes(std::size_t(best.back().count));
    for (std::size_t j = best.size() - 1, n = lanes.size(); j > 0; j = std::size_t(best[j].from)) {
        lane& g = lanes[--n];
        g.length = lengths[j - 1];
        g.rows.assign(order.begin() + bounds[std::size_t(best[j].from)], order.begin() + bounds[j]);
        std::ranges::sort(g.rows);
        for (const int row : g.rows) g.options = std::max(g.options, input.counts[std::size_t(row)]);
    }
    return lanes;
}

// Option marker k of a batch row, relative to row r of a lane of the given length.
constexpr std::int32_t lane_marker(const batch& input, int row, int k, int r, int length) {
    return std::int32_t(r * length + input.markers[std::size_t(row) * std::size_t(input.options) + std::size_t(k)] - row * input.length);
}
// The rows of one lane as a batch of their own.
constexpr batch subset(const batch& input, const lane& g) {
    batch part;
    part.size = g.size(), part.length = g.length, part.options = g.options;
    part.ids.resize(std::size_t(g.size()) * std::size_t(g.length));
    part.markers.resize(std::size_t(g.size()) * std::size_t(g.options));
    for (int r = 0; r < g.size(); ++r) {
        const int row = g.rows[std::size_t(r)];
        std::copy_n(input.ids.begin() + std::ptrdiff_t(row) * input.length, g.length, part.ids.begin() + std::ptrdiff_t(r) * g.length);
        for (int k = 0; k < g.options; ++k) part.markers[std::size_t(r) * std::size_t(g.options) + std::size_t(k)] = lane_marker(input, row, k, r, g.length);
        part.lengths.push_back(input.lengths[std::size_t(row)]);
        part.counts.push_back(input.counts[std::size_t(row)]);
        part.types.push_back(input.types[std::size_t(row)]);
    }
    return part;
}
}
