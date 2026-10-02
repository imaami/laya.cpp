#include "check.hpp"
#include "lanes.hpp"
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

// Lane planning, and with a checkpoint, lane execution: a call's lanes compute
// exactly what each lane's rows compute as a batch of their own.
using namespace laya;
using laya::test::expect;
using laya::test::fail;
using laya::test::require;

namespace {
constexpr architecture large{1024, 16, 28, 2624, 50368, 10000}, multilingual{768, 12, 22, 1152, 256000, 160000};

// A batch padded as the codec pads it: rows of the given lengths and option
// counts, with markers inside each row and pad ID 0 after it.
constexpr batch make(const std::vector<int>& lengths, const std::vector<int>& counts, int vocabulary = 50368) {
    batch b;
    b.size = int(lengths.size());
    b.length = *std::ranges::max_element(lengths);
    b.options = *std::ranges::max_element(counts);
    for (int row = 0; row < b.size; ++row) {
        for (int t = 0; t < b.length; ++t) b.ids.push_back(t < lengths[std::size_t(row)] ? 1 + (row * 7919 + t * 104729) % (vocabulary - 1) : 0);
        for (int k = 0; k < b.options; ++k)
            b.markers.push_back(row * b.length + (std::min(k, counts[std::size_t(row)] - 1) * 5 + 1) % lengths[std::size_t(row)]);
        b.lengths.push_back(lengths[std::size_t(row)]);
        b.counts.push_back(counts[std::size_t(row)]);
        b.types.push_back(row % 3);
    }
    return b;
}
constexpr std::int64_t plan_cost(const std::vector<lane>& lanes, const architecture& arch) {
    std::int64_t total = 0;
    for (const lane& g : lanes) total += lane_cost(g.size(), g.length, kappa(arch));
    return total;
}

// Rows of equal length share one lane, which is the whole batch.
static_assert([] {
    const batch b = make({40, 40, 40}, {2, 5, 3});
    return group_lanes(b, large) == std::vector{whole(b)};
}());
// A long row no longer pads the short ones; lanes are ordered by length and
// hold their rows in request order.
static_assert([] {
    const batch b = make({60, 787, 55, 70}, {3, 2, 4, 2});
    return group_lanes(b, large) == std::vector{lane{70, 4, {0, 2, 3}}, lane{787, 2, {1}}};
}());
// A lane's options are its rows' most, so its markers and logits narrow too.
static_assert([] {
    const batch b = make({20, 21, 600, 610}, {2, 3, 12, 2});
    const auto lanes = group_lanes(b, multilingual);
    return lanes.size() == 2 && lanes[0].options == 3 && lanes[1].options == 12;
}());
// A lane as a batch: rows cut to the lane length, markers counted within the lane.
static_assert([] {
    const batch b = make({60, 787, 55, 70}, {3, 2, 4, 2});
    lane g{70, 4, {0, 2, 3}};  // not const: GCC 13 rejects constant-evaluated const aggregates with vectors
    const batch part = subset(b, g);
    bool same = part.size == 3 && part.length == 70 && part.options == 4 && part.lengths == std::vector{60, 55, 70} &&
                part.counts == std::vector{3, 4, 2} && part.types == std::vector{0, 2, 0};
    for (int r = 0; r < 3; ++r) {
        const int row = g.rows[std::size_t(r)];
        for (int t = 0; t < 70; ++t) same = same && part.ids[std::size_t(r * 70 + t)] == b.ids[std::size_t(row * b.length + t)];
        for (int k = 0; k < 4; ++k)
            same = same && part.markers[std::size_t(r * 4 + k)] - r * 70 == b.markers[std::size_t(row * b.options + k)] - row * b.length;
    }
    return same;
}());

// The least (cost, lane count) over every partition of the rows.
std::pair<std::int64_t, int> exhaustive(const batch& b, const architecture& arch) {
    std::pair<std::int64_t, int> least{std::numeric_limits<std::int64_t>::max(), 0};
    std::vector<int> label(std::size_t(b.size), 0);  // restricted growth strings enumerate set partitions
    while (true) {
        const int count = *std::ranges::max_element(label) + 1;
        std::int64_t total = 0;
        for (int part = 0; part < count; ++part) {
            int rows = 0, length = 0;
            for (int row = 0; row < b.size; ++row)
                if (label[std::size_t(row)] == part) ++rows, length = std::max(length, b.lengths[std::size_t(row)]);
            total += lane_cost(rows, length, kappa(arch));
        }
        least = std::min(least, std::pair{total, count});
        int i = b.size - 1;
        for (; i > 0; --i) {
            const int bound = *std::max_element(label.begin(), label.begin() + i) + 1;
            if (label[std::size_t(i)] < bound) break;
            label[std::size_t(i)] = 0;
        }
        if (i == 0) return least;
        ++label[std::size_t(i)];
    }
}

void properties(const batch& b, const architecture& arch, bool small) {
    const auto lanes = group_lanes(b, arch);
    std::vector<int> seen(std::size_t(b.size), 0);
    int previous = 0;
    for (const lane& g : lanes) {
        require(!g.rows.empty() && std::ranges::is_sorted(g.rows), "Lane rows out of request order");
        int length = 0, options = 0, shortest = std::numeric_limits<int>::max();
        for (const int row : g.rows) {
            ++seen[std::size_t(row)];
            length = std::max(length, b.lengths[std::size_t(row)]), shortest = std::min(shortest, b.lengths[std::size_t(row)]);
            options = std::max(options, b.counts[std::size_t(row)]);
        }
        require(g.length == length && g.options == options, "Lane is not padded to its own rows");
        require(shortest > previous, "Lanes do not span ascending length ranges");
        previous = length;
        // Planned alone, a lane's rows are one lane: grouped execution equals per-lane batches.
        lane alone{g.length, g.options, std::vector<int>(std::size_t(g.size()))};
        std::ranges::iota(alone.rows, 0);
        require(group_lanes(subset(b, g), arch) == std::vector{alone}, "A lane's rows plan differently alone");
    }
    require(std::ranges::all_of(seen, [](int n) { return n == 1; }), "Rows are not partitioned");
    if (small) {
        const auto least = exhaustive(b, arch);
        require(plan_cost(lanes, arch) == least.first && int(lanes.size()) == least.second, "Plan is not the least costly");
    }
}

// Bitwise equality of float results.
bool same(float a, float b) { return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b); }

// A grouped call computes each lane exactly as that lane's rows alone, keeps
// its answers when replayed, and agrees with padding to the longest row.
void execution(const std::filesystem::path& directory, mode requested) {
    auto loaded = runtime::load(directory, requested);
    if (!loaded && (loaded.error().code == errc::unsupported || loaded.error().message == "Cannot initialize requested backend")) {
        std::cout << "Skipped: " << loaded.error().message << '\n';
        std::exit(77);
    }
    runtime grouped = expect(std::move(loaded));
    runtime longest = expect(runtime::load(directory, requested, padding::longest));
    const auto& arch = grouped.model().arch;
    const batch b = make({14, 9, 180, 12, 15, 230, 11, 13, 40, 8}, {2, 4, 3, 2, 6, 2, 5, 2, 3, 2}, arch.vocabulary);
    const auto lanes = group_lanes(b, arch);
    require(lanes.size() >= 2, "Fixture must form several lanes");
    auto first = grouped.forward(b);
    if (!first && first.error().code == errc::unsupported) {  // a device without an operation the checkpoint needs
        std::cout << "Skipped: " << first.error().message << '\n';
        std::exit(77);
    }
    const auto output = expect(std::move(first));
    const int actions = output.action_count;
    require(int(output.lanes.size()) == b.size, "Missing lane per row");
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        const lane& g = lanes[i];
        const auto alone = expect(grouped.forward(subset(b, g)));
        for (int r = 0; r < g.size(); ++r) {
            const int row = g.rows[std::size_t(r)];
            require(output.lanes[std::size_t(row)] == int(i), "Row reported in the wrong lane");
            for (int k = 0; k < b.options; ++k) {
                const float logit = output.logits[std::size_t(row * b.options + k)];
                require(k < g.options ? same(logit, alone.logits[std::size_t(r * g.options + k)]) : logit == -1e4f, "Lane logits differ");
            }
            for (int a = 0; a < actions; ++a)
                require(same(output.actions[std::size_t(row * actions + a)], alone.actions[std::size_t(r * actions + a)]), "Lane actions differ");
        }
    }
    const auto replay = expect(grouped.forward(b));
    require(std::ranges::equal(replay.logits, output.logits, same) && std::ranges::equal(replay.actions, output.actions, same),
            "Replayed lanes changed their answer");
    const auto padded = expect(longest.forward(b));
    require(std::ranges::all_of(padded.lanes, [](int i) { return i == 0; }), "Padding to the longest row formed lanes");
    // Padding changes only masked positions, but reduction lengths and kernel
    // shapes follow it, so the difference is reported rather than required.
    const auto finite = [](float value) { return std::isfinite(value); };
    require(std::ranges::all_of(padded.logits, finite) && std::ranges::all_of(padded.actions, finite),
            "Padding to the longest row produced a non-finite output");
    double deviation = 0;  // relative to magnitudes of at least 1
    for (const auto& [x, y] : {std::pair{&padded.logits, &output.logits}, std::pair{&padded.actions, &output.actions}})
        for (std::size_t i = 0; i < x->size(); ++i)
            deviation = std::max(deviation, std::abs(double((*x)[i]) - double((*y)[i])) / std::max(1.0, std::abs(double((*y)[i]))));
    std::cout << lanes.size() << " lanes match their rows alone; largest relative deviation from padding to the longest row "
              << deviation << '\n';
}
}

int main(int argc, char** argv) {
    std::mt19937 random(20260930);
    const auto draw = [&](int low, int high) { return std::uniform_int_distribution<int>(low, high)(random); };
    for (int trial = 0; trial < 400; ++trial) {
        const bool small = trial % 2 == 0;
        const int rows = small ? draw(1, 7) : draw(1, 160);
        std::vector<int> lengths, counts;
        for (int row = 0; row < rows; ++row) {
            lengths.push_back(draw(0, 9) == 0 ? draw(120, 1024) : draw(4, 120));
            counts.push_back(draw(2, 12));
        }
        const batch b = make(lengths, counts);
        properties(b, large, small);
        properties(b, multilingual, small);
    }
    std::cout << "Lane plans partition rows, pad each lane to its own rows and cost least\n";
    if (argc == 1) return 0;
    if (argc != 3 && argc != 4) fail("Usage: test-lanes [CHECKPOINT cpu|cuda|vulkan [fp32|flash-fp32|tensor-core-fp32|fp16|bf16]]");
    const std::string_view backend = argv[2], precision = argc == 4 ? argv[3] : "fp32";
    mode requested = backend == "cpu" ? feature::cpu : backend == "cuda" ? feature::cuda : backend == "vulkan" ? feature::vulkan : 0;
    if (!requested) fail("Unknown backend");
    if (precision == "flash-fp32") requested |= feature::flash;
    else if (precision == "tensor-core-fp32") requested |= feature::compensated;
    else if (precision == "fp16") requested |= feature::fp16 | feature::flash;
    else if (precision == "bf16") requested |= feature::bf16 | feature::flash;
    else if (precision != "fp32") fail("Unknown precision");
    execution(argv[1], requested);
}
