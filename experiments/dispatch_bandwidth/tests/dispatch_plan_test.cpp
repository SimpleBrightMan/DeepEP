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
    Require(config.first_expert || plan.total_rows == static_cast<uint64_t>(config.batch) * config.topk,
            "full Dispatch row conservation");
    Require(plan.scale_elements == config.hidden / 32U, "scale element count");
    const uint32_t padded_hidden = ((config.hidden + 255U) / 256U) * 256U;
    const uint32_t padded_scale = ((plan.scale_elements + 31U) / 32U) * 32U;
    Require(plan.row_bytes == padded_hidden + padded_scale, "packed source row stride");
    Require(plan.workers.size() == config.cores, "worker count");
    Require(plan.phase_count == 1U && plan.phases.empty(), "concurrent phase metadata changed");

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
        Require(maximum[traffic] - minimum[traffic] <= 1U, "unbalanced traffic queue");
    }
    if (!config.first_expert && config.batch * config.topk % config.world == 0U) {
        const uint64_t per_peer = static_cast<uint64_t>(config.batch) * config.topk / config.world;
        Require(plan.traffic_rows[0] == per_peer && plan.traffic_rows[1] == per_peer * (config.group_size - 1U) &&
                    plan.traffic_rows[2] == per_peer * (config.world - config.group_size),
                "balanced full Dispatch traffic proportions");
    }
    ++cases;
}

void VerifyGroupRoundRobin(const Config& config, uint32_t rank)
{
    const Plan plan = BuildPlan(config, rank);
    const auto expected = Oracle(config, rank);
    const uint32_t groups = config.world / config.group_size;
    Require(config.group_round_robin, "group-round-robin verification mode");
    Require(plan.total_rows == expected.size() && plan.tasks.size() == expected.size(), "group route row count");
    Require(plan.phase_count == groups && plan.phases.size() == groups, "group phase count");
    Require(plan.workers.size() == config.cores, "group worker count");
    Require(plan.scale_elements == config.hidden / 32U, "group scale element count");
    Require(plan.row_bytes == ((config.hidden + 255U) / 256U) * 256U + ((plan.scale_elements + 31U) / 32U) * 32U,
            "group packed source stride");
    std::vector<uint32_t> group_to_phase(groups);
    uint32_t peer_group = rank / config.group_size;
    for (uint32_t phase = 0U; phase < groups; ++phase) {
        Require(plan.phases[phase].peer_group == peer_group, "GET source group phase direction");
        group_to_phase[peer_group] = phase;
        peer_group = peer_group == 0U ? groups - 1U : peer_group - 1U;
    }
    std::vector<std::vector<CopyTask>> phase_rows(groups);
    std::vector<uint32_t> phase_offsets(expected.size());
    std::vector<std::array<uint64_t, kTrafficKinds>> expected_traffic(groups), actual_traffic(groups);
    for (const CopyTask& task : expected) {
        const uint32_t phase = group_to_phase[task.peer / config.group_size];
        phase_offsets[task.destination] = static_cast<uint32_t>(phase_rows[phase].size());
        phase_rows[phase].push_back(task);
        ++expected_traffic[phase][Classify(task.peer, rank, config.group_size)];
    }
    std::vector<std::vector<uint32_t>> counts(groups, std::vector<uint32_t>(config.cores, 0U));
    std::vector<std::vector<uint32_t>> starts(groups, std::vector<uint32_t>(config.cores, UINT32_MAX));
    std::vector<uint8_t> seen(expected.size(), 0U);
    uint64_t next_task = 0U;
    uint32_t minimum = UINT32_MAX, maximum = 0U;
    for (uint32_t logical = 0U; logical < config.cores; ++logical) {
        const Worker& worker = plan.workers[logical];
        Require(worker.traffic == static_cast<uint32_t>(Traffic::All) && worker.reserved == 0U,
                "group worker must use the mixed-traffic tag");
        Require(worker.begin == next_task && static_cast<uint64_t>(worker.begin) + worker.count <= plan.tasks.size(),
                "group worker interval");
        next_task += worker.count;
        minimum = std::min(minimum, worker.count);
        maximum = std::max(maximum, worker.count);
        uint32_t previous_phase = 0U;
        for (uint32_t row = 0U; row < worker.count; ++row) {
            const CopyTask& task = plan.tasks[worker.begin + row];
            Require(task.peer < config.world && task.destination < expected.size(), "group task bounds");
            Require(seen[task.destination]++ == 0U, "group destination duplicated");
            const CopyTask& reference = expected[task.destination];
            Require(
                task.peer == reference.peer && task.token == reference.token && task.topk_slot == reference.topk_slot,
                "group source identity or destination changed");
            const uint32_t phase = group_to_phase[task.peer / config.group_size];
            Require(phase >= previous_phase, "worker GET source groups are not phase ordered");
            previous_phase = phase;
            if (counts[phase][logical] == 0U) {
                starts[phase][logical] = phase_offsets[task.destination];
            }
            Require(phase_offsets[task.destination] == starts[phase][logical] + counts[phase][logical],
                    "worker phase chunk is not contiguous");
            ++counts[phase][logical];
            ++actual_traffic[phase][Classify(task.peer, rank, config.group_size)];
        }
    }
    Require(next_task == plan.tasks.size(), "group unassigned task suffix");
    Require(std::all_of(seen.begin(), seen.end(), [](uint8_t value) { return value == 1U; }), "group route omitted");
    Require(maximum - minimum <= 1U, "whole-worker round workload imbalance");
    std::array<uint64_t, kTrafficKinds> totals{};
    uint32_t extra_cursor = 0U;
    for (uint32_t phase = 0U; phase < groups; ++phase) {
        Require(actual_traffic[phase] == expected_traffic[phase] &&
                    plan.phases[phase].traffic_rows == expected_traffic[phase],
                "group phase traffic row accounting");
        const auto range = std::minmax_element(counts[phase].begin(), counts[phase].end());
        Require(*range.second - *range.first <= 1U, "group phase workload imbalance");
        const uint32_t phase_start = extra_cursor;
        std::vector<uint32_t> reference_counts(config.cores,
                                               static_cast<uint32_t>(phase_rows[phase].size() / config.cores));
        // Independently distribute leftover rows one at a time around the ring.
        for (uint32_t extra = static_cast<uint32_t>(phase_rows[phase].size() % config.cores); extra != 0U; --extra) {
            ++reference_counts[extra_cursor];
            extra_cursor = (extra_cursor + 1U) % config.cores;
        }
        Require(counts[phase] == reference_counts, "phase remainder work did not rotate fairly");
        uint32_t next_offset = 0U;
        for (uint32_t chunk = 0U; chunk < config.cores; ++chunk) {
            const uint32_t worker = (phase_start + chunk) % config.cores;
            if (counts[phase][worker] != 0U) {
                Require(starts[phase][worker] == next_offset, "rotated phase chunk ownership changed");
            }
            next_offset += counts[phase][worker];
        }
        Require(next_offset == phase_rows[phase].size(), "phase chunks do not cover the source group");
        for (uint32_t traffic = 0U; traffic < kTrafficKinds; ++traffic) {
            totals[traffic] += expected_traffic[phase][traffic];
        }
    }
    Require(plan.traffic_rows == totals, "group total traffic accounting");
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
    for (uint32_t cores : {0U, 33U, UINT32_MAX}) {
        Invalid([cores](Config& config) {
            config.group_round_robin = true;
            config.cores = cores;
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
    for (uint32_t value : {0U, 1U, 7U, UINT32_MAX}) {
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
    Require(!defaults.first_expert && !defaults.group_round_robin && defaults.world == 16U &&
                defaults.group_size == 4U && defaults.cores == 28U && defaults.batch == 4096U &&
                defaults.hidden == 3584U && defaults.intermediate == 1280U && defaults.experts == 384U &&
                defaults.topk == 8U && defaults.ring_slots == 6U &&
                defaults.worker_counts == std::array<uint32_t, 3>{2U, 5U, 21U},
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

void GroupRoundRobin()
{
    Config config;
    config.group_round_robin = true;
    // These fields have no ownership meaning under the phase schedule.
    config.worker_counts = {0U, 0U, 0U};
    for (uint32_t batch : {2048U, 4096U, 8192U}) {
        config.batch = batch;
        for (uint32_t rank = 0U; rank < config.world; ++rank) {
            VerifyGroupRoundRobin(config, rank);
            config.first_expert = true;
            VerifyGroupRoundRobin(config, rank);
            config.first_expert = false;
        }
    }
    config.batch = 4096U;
    const Plan reference = BuildPlan(config, 0U);
    // Each phase has 8192 rows: base 292, with 16 extras rotating across
    // all 28 workers instead of repeatedly favoring the first 16 workers.
    constexpr std::array<uint32_t, 4> extra_starts{0U, 16U, 4U, 20U};
    for (uint32_t logical = 0U; logical < config.cores; ++logical) {
        const Worker& worker = reference.workers[logical];
        Require(worker.count == 1170U + static_cast<uint32_t>(logical < 8U), "whole-round reference quota");
        std::array<uint32_t, 4> phase_counts{};
        for (uint32_t row = 0U; row < worker.count; ++row) {
            const uint32_t group = reference.tasks[worker.begin + row].peer / config.group_size;
            const uint32_t phase = group == 0U ? 0U : 4U - group;
            ++phase_counts[phase];
        }
        for (uint32_t phase = 0U; phase < 4U; ++phase) {
            Require(
                phase_counts[phase] == 292U + static_cast<uint32_t>((logical + 28U - extra_starts[phase]) % 28U < 16U),
                "per-phase reference remainder rotation");
        }
    }
    for (uint32_t hidden : {3712U, 4096U, 6144U, 8192U}) {
        config.hidden = hidden;
        config.batch = 97U;
        for (uint32_t rank = 0U; rank < config.world; ++rank) {
            VerifyGroupRoundRobin(config, rank);
        }
    }
    for (uint32_t cores : {1U, 3U, 5U, 28U, 32U}) {
        config.cores = cores;
        VerifyGroupRoundRobin(config, 0U);
        VerifyGroupRoundRobin(config, 15U);
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
    tiny.ring_slots = 2U;
    for (uint32_t rank = 0U; rank < tiny.world; ++rank) {
        Verify(tiny, rank);
        const Plan plan = BuildPlan(tiny, rank);
        Require(plan.workers[1].count == 0U && plan.workers[2].count == 0U, "empty traffic classes must be allowed");
        Config grouped = tiny;
        grouped.group_round_robin = true;
        grouped.cores = 2U;
        grouped.worker_counts = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
        VerifyGroupRoundRobin(grouped, rank);
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
        config.ring_slots = 2U + random() % 5U;
        config.first_expert = iteration % 2U == 0U;
        const uint32_t rank = random() % config.world;
        Verify(config, rank);
        config.group_round_robin = true;
        config.cores = config.group_size * (1U + random() % 8U);
        VerifyGroupRoundRobin(config, rank);
    }
}

}  // namespace

int main()
{
    try {
        Validation();
        Defaults();
        GroupRoundRobin();
        BoundaryAndVariedShapes();
        std::cout << "dispatch_plan: " << cases << " independent-oracle cases passed\n";
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "dispatch_plan: " << failure.what() << '\n';
        return 1;
    }
}
