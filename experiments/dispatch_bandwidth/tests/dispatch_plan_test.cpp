// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "dispatch_plan.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using namespace deepep::dispatch_bench;

namespace {

uint32_t cases = 0U;

void Require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Deliberately enumerate every source's [token, top-k] route stream rather
// than using the production planner's modular arithmetic for expert queues.
std::vector<CopyTask> Oracle(const Config& config, uint32_t rank)
{
    const uint32_t local_experts = config.experts / config.world;
    std::vector<std::vector<CopyTask>> queues(local_experts * config.world);
    for (uint32_t source = 0U; source < config.world; ++source) {
        if (config.cross_only && source / config.group_size == rank / config.group_size) {
            continue;
        }
        for (uint32_t token = 0U; token < config.batch; ++token) {
            for (uint32_t slot = 0U; slot < config.topk; ++slot) {
                const uint64_t route = (static_cast<uint64_t>(source) * config.batch + token) * config.topk + slot;
                if (route % config.world != rank) {
                    continue;
                }
                const uint32_t local_expert = static_cast<uint32_t>(route / config.world % local_experts);
                if (config.first_expert && local_expert != 0U) {
                    continue;
                }
                queues[local_expert * config.world + source].push_back({source, token, slot, 0U});
            }
        }
    }
    std::vector<CopyTask> expected;
    for (uint32_t expert = 0U; expert < local_experts; ++expert) {
        for (uint32_t ordinal = 0U; ordinal < config.world; ++ordinal) {
            const uint32_t source = (rank + ordinal) % config.world;
            for (CopyTask row : queues[expert * config.world + source]) {
                row.destination = static_cast<uint32_t>(expected.size());
                expected.push_back(row);
            }
        }
    }
    return expected;
}

uint32_t Classify(uint32_t source, uint32_t rank, uint32_t group_size)
{
    if (source == rank) {
        return 0U;
    }
    return source / group_size == rank / group_size ? 1U : 2U;
}

void Verify(const Config& config, uint32_t rank)
{
    const Plan plan = BuildPlan(config, rank);
    const auto expected = Oracle(config, rank);
    Require(plan.total_rows == expected.size() && plan.tasks.size() == expected.size(), "route row count");
    Require(config.first_expert || config.cross_only ||
                plan.total_rows == static_cast<uint64_t>(config.batch) * config.topk,
            "full Dispatch row conservation");
    Require(plan.scale_elements == config.hidden / 32U, "scale element count");
    const uint32_t padded_hidden = ((config.hidden + 255U) / 256U) * 256U;
    const uint32_t padded_scale = ((plan.scale_elements + 31U) / 32U) * 32U;
    Require(plan.row_bytes == padded_hidden + padded_scale, "packed source row stride");
    Require(plan.workers.size() == config.cores, "worker count");

    std::vector<uint8_t> seen_destination(expected.size(), 0U), seen_task(expected.size(), 0U);
    std::array<uint32_t, kTrafficKinds> workers{}, minimum{}, maximum{};
    std::array<uint64_t, kTrafficKinds> rows{};
    minimum.fill(std::numeric_limits<uint32_t>::max());
    uint64_t next_task = 0U;
    uint32_t previous_traffic = 0U;
    for (const Worker& worker : plan.workers) {
        Require(worker.traffic < kTrafficKinds, "worker traffic class");
        Require(worker.traffic >= previous_traffic, "workers must be grouped by traffic");
        previous_traffic = worker.traffic;
        Require(worker.reserved == 0U, "worker ABI reserved word");
        Require(worker.begin == next_task && static_cast<uint64_t>(worker.begin) + worker.count <= plan.tasks.size(),
                "worker task interval");
        next_task += worker.count;
        ++workers[worker.traffic];
        rows[worker.traffic] += worker.count;
        minimum[worker.traffic] = std::min(minimum[worker.traffic], worker.count);
        maximum[worker.traffic] = std::max(maximum[worker.traffic], worker.count);
        for (uint32_t index = worker.begin; index < worker.begin + worker.count; ++index) {
            Require(seen_task[index]++ == 0U, "task assigned to multiple workers");
            const CopyTask& task = plan.tasks[index];
            Require(task.peer < config.world && task.token < config.batch && task.topk_slot < config.topk,
                    "source task bounds");
            Require(Classify(task.peer, rank, config.group_size) == worker.traffic, "worker mixes traffic kinds");
            Require(task.destination < expected.size(), "destination bounds");
            Require(seen_destination[task.destination]++ == 0U, "destination written more than once");
            const CopyTask& oracle = expected[task.destination];
            Require(task.peer == oracle.peer && task.token == oracle.token && task.topk_slot == oracle.topk_slot,
                    "source identity or destination row changed");
        }
    }
    Require(next_task == plan.tasks.size(), "unassigned task suffix");
    Require(std::all_of(seen_task.begin(), seen_task.end(), [](uint8_t seen) { return seen == 1U; }), "task omitted");
    Require(std::all_of(seen_destination.begin(), seen_destination.end(), [](uint8_t seen) { return seen == 1U; }),
            "destination omitted");
    for (uint32_t traffic = 0U; traffic < kTrafficKinds; ++traffic) {
        Require(workers[traffic] == config.worker_counts[traffic], "traffic worker allocation");
        Require(rows[traffic] == plan.traffic_rows[traffic], "traffic row accounting");
        if (workers[traffic] == 0U) {
            Require(rows[traffic] == 0U, "rows assigned to a traffic class without workers");
        } else {
            Require(maximum[traffic] - minimum[traffic] <= 1U, "unbalanced traffic queue");
        }
    }
    if (!config.first_expert && config.batch * config.topk % config.world == 0U) {
        const uint64_t per_peer = static_cast<uint64_t>(config.batch) * config.topk / config.world;
        Require(plan.traffic_rows[0] == (config.cross_only ? 0U : per_peer) &&
                    plan.traffic_rows[1] == (config.cross_only ? 0U : per_peer * (config.group_size - 1U)) &&
                    plan.traffic_rows[2] == per_peer * (config.world - config.group_size),
                "balanced full Dispatch traffic proportions");
    }
    if (config.cross_only) {
        Require(rows[0] == 0U && rows[1] == 0U && rows[2] == plan.total_rows, "cross-only total row conservation");
        Require(workers[0] == 0U && workers[1] == 0U && workers[2] == config.cores, "cross-only worker purity");
    }
    ++cases;
}

template <typename Mutate>
void Invalid(Mutate mutate)
{
    Config config;
    mutate(config);
    try {
        BuildPlan(config, 0U);
    } catch (const std::invalid_argument&) {
        return;
    } catch (const std::overflow_error&) {
        return;
    }
    throw std::runtime_error("invalid configuration was accepted");
}

void Validation()
{
    for (uint32_t value : {0U, 1U, 2U, 3U, 17U, UINT32_MAX}) {
        Invalid([value](Config& config) { config.world = value; });
    }
    for (uint32_t value : {0U, 1U, 3U, 16U, UINT32_MAX}) {
        Invalid([value](Config& config) { config.group_size = value; });
    }
    for (uint32_t value : {0U, 2U, 33U, UINT32_MAX}) {
        Invalid([value](Config& config) { config.cores = value; });
    }
    Invalid([](Config& config) { config.worker_counts = {0U, 7U, 21U}; });
    Invalid([](Config& config) { config.worker_counts = {2U, 0U, 26U}; });
    Invalid([](Config& config) { config.worker_counts = {2U, 26U, 0U}; });
    Invalid([](Config& config) { config.worker_counts = {2U, 5U, 20U}; });
    Invalid([](Config& config) { config.worker_counts = {UINT32_MAX, UINT32_MAX, 30U}; });
    Invalid([](Config& config) { config.cross_only = true; });
    constexpr std::array<std::array<uint32_t, 3>, 5> invalid_cross_workers{
        {{1U, 0U, 27U}, {0U, 1U, 27U}, {0U, 0U, 27U}, {0U, 0U, 0U}, {0U, 0U, UINT32_MAX}}};
    for (const auto& workers : invalid_cross_workers) {
        Invalid([workers](Config& config) {
            config.cross_only = true;
            config.worker_counts = workers;
        });
    }
    for (uint32_t value : {0U, 33U, UINT32_MAX}) {
        Invalid([value](Config& config) {
            config.cross_only = true;
            config.cores = value;
            config.worker_counts = {0U, 0U, value};
        });
    }
    for (uint32_t value : {0U, 65537U, UINT32_MAX}) {
        Invalid([value](Config& config) { config.batch = value; });
    }
    for (uint32_t value : {0U, 1023U, 1056U, 8193U, UINT32_MAX}) {
        Invalid([value](Config& config) { config.hidden = value; });
    }
    for (uint32_t value : {0U, 255U, 257U, 4224U, UINT32_MAX}) {
        Invalid([value](Config& config) { config.intermediate = value; });
    }
    for (uint32_t value : {0U, 383U, 2064U, 16385U, UINT32_MAX}) {
        Invalid([value](Config& config) { config.experts = value; });
    }
    for (uint32_t value : {0U, 33U, UINT32_MAX}) {
        Invalid([value](Config& config) { config.topk = value; });
    }
    Invalid([](Config& config) {
        config.experts = 16U;
        config.topk = 17U;
    });
    for (uint32_t value : {0U, 7U, UINT32_MAX}) {
        Invalid([value](Config& config) { config.ring_slots = value; });
    }
    for (uint32_t rank : {16U, UINT32_MAX}) {
        bool rejected = false;
        try {
            BuildPlan(Config{}, rank);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        Require(rejected, "invalid rank accepted");
    }
}

void Defaults()
{
    const Config defaults;
    Require(!defaults.first_expert && !defaults.cross_only && defaults.world == 16U && defaults.group_size == 4U &&
                defaults.cores == 28U && defaults.batch == 4096U && defaults.hidden == 3584U &&
                defaults.intermediate == 1280U && defaults.experts == 384U && defaults.topk == 8U &&
                defaults.ring_slots == 6U && defaults.worker_counts == std::array<uint32_t, 3>{2U, 5U, 21U},
            "documented defaults changed");
    for (uint32_t batch : {2048U, 4096U, 8192U}) {
        for (uint32_t rank = 0U; rank < 16U; ++rank) {
            Config config;
            config.batch = batch;
            Verify(config, rank);
            config.first_expert = true;
            Verify(config, rank);
            const Plan first = BuildPlan(config, rank);
            Require(first.total_rows == (batch == 2048U ? 683U : (batch == 4096U ? 1366U : 2731U)),
                    "first-expert reference row count");
            std::array<uint32_t, 16> source_rows{};
            for (const CopyTask& task : first.tasks) {
                ++source_rows[task.peer];
            }
            for (uint32_t source = 0U; source < 16U; ++source) {
                const uint32_t expected = batch == 4096U ? 85U + static_cast<uint32_t>(source % 3U == 0U)
                                                         : batch / 48U + static_cast<uint32_t>(source % 3U != 2U);
                Require(source_rows[source] == expected, "first-expert source quota");
            }
        }
    }
}

void VerifyCrossSubset(const Config& config, uint32_t rank)
{
    Config all = config;
    all.cross_only = false;
    all.cores = 28U;
    all.worker_counts = {2U, 5U, 21U};
    const Plan full = BuildPlan(all, rank);
    const Plan selected = BuildPlan(config, rank);
    std::size_t index = 0U;
    for (const CopyTask& task : full.tasks) {
        if (Classify(task.peer, rank, config.group_size) != static_cast<uint32_t>(Traffic::Cross)) {
            continue;
        }
        Require(index < selected.tasks.size(), "cross-only omitted a full-plan source row");
        const CopyTask& cross = selected.tasks[index];
        Require(cross.peer == task.peer && cross.token == task.token && cross.topk_slot == task.topk_slot,
                "cross-only source stream differs from full-plan cross traffic");
        Require(cross.destination == index, "cross-only destinations are not compact");
        ++index;
    }
    Require(index == selected.tasks.size(), "cross-only added a source row");
    Require(selected.total_rows == full.traffic_rows[2], "cross-only row count differs from full-plan cross traffic");
}

void CrossDefaultsAndSweep()
{
    constexpr std::array<uint32_t, 4> hidden_sizes{3712U, 4096U, 6144U, 8192U};
    constexpr std::array<uint32_t, 4> row_strides{3968U, 4224U, 6336U, 8448U};
    constexpr std::array<uint32_t, 10> core_counts{2U, 4U, 6U, 8U, 10U, 12U, 16U, 20U, 24U, 28U};
    Config config;
    config.cross_only = true;
    config.worker_counts = {0U, 0U, config.cores};
    // Exercise every production BS/hidden/rank combination at 28 workers.
    for (uint32_t batch : {2048U, 4096U, 8192U}) {
        config.batch = batch;
        for (std::size_t shape = 0U; shape < hidden_sizes.size(); ++shape) {
            config.hidden = hidden_sizes[shape];
            for (uint32_t rank = 0U; rank < config.world; ++rank) {
                Verify(config, rank);
                const Plan full = BuildPlan(config, rank);
                Require(full.total_rows == static_cast<uint64_t>(batch) * 6U, "default cross-only row count");
                Require(full.row_bytes == row_strides[shape], "sweep hidden packed source stride");
                if (shape == 0U) {
                    VerifyCrossSubset(config, rank);
                    config.first_expert = true;
                    Verify(config, rank);
                    VerifyCrossSubset(config, rank);
                    const uint32_t group = rank / config.group_size;
                    const bool extra_row = batch == 4096U ? (group == 1U || group == 2U) : group == 2U;
                    Require(BuildPlan(config, rank).total_rows == batch / 4U + static_cast<uint32_t>(extra_row),
                            "cross-only first-expert row count");
                    config.first_expert = false;
                }
            }
        }
    }
    // Cover the whole core/hidden/rank sweep at a smaller, non-divisible BS
    // instead of repeating the full large-BS Cartesian product.
    config.batch = 97U;
    for (uint32_t cores : core_counts) {
        config.cores = cores;
        config.worker_counts = {0U, 0U, cores};
        for (uint32_t hidden : hidden_sizes) {
            config.hidden = hidden;
            for (uint32_t rank = 0U; rank < config.world; ++rank) {
                Verify(config, rank);
            }
        }
    }
    for (uint32_t cores : {1U, 32U}) {
        config.cores = cores;
        config.worker_counts = {0U, 0U, cores};
        Verify(config, 0U);
        Verify(config, 15U);
    }
}

void BoundaryAndVariedShapes()
{
    Config tiny;
    tiny.world = 4U;
    tiny.group_size = 2U;
    tiny.cores = 3U;
    tiny.worker_counts = {1U, 1U, 1U};
    tiny.batch = 1U;
    tiny.hidden = 1024U;
    tiny.intermediate = 256U;
    tiny.experts = 4U;
    tiny.topk = 1U;
    tiny.ring_slots = 1U;
    for (uint32_t rank = 0U; rank < tiny.world; ++rank) {
        Verify(tiny, rank);
        const Plan plan = BuildPlan(tiny, rank);
        Require(plan.workers[1].count == 0U && plan.workers[2].count == 0U, "empty traffic classes must be allowed");
        Config cross = tiny;
        cross.cross_only = true;
        cross.cores = 1U;
        cross.worker_counts = {0U, 0U, 1U};
        Verify(cross, rank);
        Require(BuildPlan(cross, rank).total_rows == 0U, "empty cross-only plans must be allowed");
    }
    Config maximum;
    maximum.batch = 65536U;
    maximum.hidden = 8192U;
    maximum.intermediate = 4096U;
    maximum.experts = maximum.world * 128U;
    maximum.topk = 32U;
    maximum.cores = 32U;
    maximum.worker_counts = {2U, 6U, 24U};
    Verify(maximum, 15U);
    Require(static_cast<uint64_t>(maximum.batch) * maximum.topk * maximum.hidden > UINT32_MAX,
            "maximum case must exercise 64-bit allocation sizes");
    maximum.cross_only = true;
    maximum.worker_counts = {0U, 0U, maximum.cores};
    Verify(maximum, 15U);

    std::mt19937 random(20261015U);
    const std::array<uint32_t, 5> worlds{4U, 6U, 8U, 12U, 16U};
    for (uint32_t iteration = 0U; iteration < 100U; ++iteration) {
        Config config;
        config.world = worlds[random() % worlds.size()];
        config.group_size = 2U;
        config.cores = 7U;
        config.worker_counts = {2U, 2U, 3U};
        config.batch = 1U + random() % 193U;
        config.hidden = 1024U + 64U * (random() % 16U);
        config.intermediate = 256U + 128U * (random() % 8U);
        config.experts = config.world * (1U + random() % 11U);
        config.topk = 1U + random() % std::min(32U, config.experts);
        config.ring_slots = 1U + random() % 6U;
        config.first_expert = iteration % 2U == 0U;
        const uint32_t rank = random() % config.world;
        Verify(config, rank);
        config.cross_only = true;
        config.cores = 1U + random() % 32U;
        config.worker_counts = {0U, 0U, config.cores};
        Verify(config, rank);
        VerifyCrossSubset(config, rank);
    }
}

}  // namespace

int main()
{
    try {
        Validation();
        Defaults();
        CrossDefaultsAndSweep();
        BoundaryAndVariedShapes();
        std::cout << "dispatch_plan: " << cases << " independent-oracle cases passed\n";
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "dispatch_plan: " << failure.what() << '\n';
        return 1;
    }
}
