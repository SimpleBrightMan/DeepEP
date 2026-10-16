// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once

#include "dispatch_types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace deepep::dispatch_bench {

struct Config {
    uint32_t world = 16U;
    uint32_t group_size = 4U;
    uint32_t cores = 28U;
    uint32_t batch = 4096U;
    uint32_t hidden = 3584U;
    uint32_t intermediate = 1280U;
    uint32_t experts = 384U;
    uint32_t topk = 8U;
    uint32_t ring_slots = 6U;
    std::array<uint32_t, kTrafficKinds> worker_counts{2U, 5U, 21U};
    bool first_expert = false;
    bool group_round_robin = false;
};

struct PhasePlan {
    uint32_t peer_group;
    std::array<uint64_t, kTrafficKinds> traffic_rows;
};

struct Plan {
    // Concurrent tasks are grouped by traffic. Group-round-robin tasks are
    // grouped by worker, with source-group chunks in phase order per worker.
    // destination retains the expert/peer ordered output row before regrouping.
    std::vector<CopyTask> tasks;
    std::vector<Worker> workers;
    std::array<uint64_t, kTrafficKinds> traffic_rows{};
    uint32_t total_rows = 0U;
    uint32_t row_bytes = 0U;
    uint32_t scale_elements = 0U;
    uint32_t phase_count = 1U;
    // Only the group-round-robin schedule has explicit source-group phases.
    std::vector<PhasePlan> phases;
};

namespace detail {

inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return ((value + alignment - 1U) / alignment) * alignment;
}

inline void Require(bool condition, const char* message)
{
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

inline void CheckBytes(uint64_t elements, uint64_t bytes_per_element)
{
    if (elements > std::numeric_limits<std::size_t>::max() / bytes_per_element) {
        throw std::overflow_error("Dispatch buffer exceeds host address capacity");
    }
}

inline void Validate(const Config& config, uint32_t rank)
{
    Require(config.world >= 4U && config.world <= kMaxRanks, "world must be in [4,16]");
    Require(rank < config.world, "rank must be less than world");
    Require(config.group_size >= 2U && config.group_size < config.world && config.world % config.group_size == 0U,
            "group_size must divide world and satisfy 1 < group_size < world");
    Require(config.cores >= kTrafficKinds && config.cores <= 32U, "cores must be in [3,32]");
    if (config.group_round_robin) {
        Require(config.world == 16U && config.group_size == 4U && config.cores == kIntraWorkers + kCrossWorkers,
                "group round robin requires EP16, groups of four and 28 cores");
    } else {
        uint64_t workers = 0U;
        for (uint32_t count : config.worker_counts) {
            Require(count > 0U, "each traffic class must have at least one worker");
            workers += count;
        }
        Require(workers == config.cores, "worker_counts must sum to cores");
    }
    Require(config.batch > 0U && config.batch <= 65536U, "batch must be in [1,65536]");
    Require(config.hidden >= 1024U && config.hidden <= 8192U && config.hidden % 64U == 0U,
            "hidden must be a multiple of 64 in [1024,8192]");
    Require(config.intermediate >= 256U && config.intermediate <= 4096U && config.intermediate % 128U == 0U,
            "intermediate must be a multiple of 128 in [256,4096]");
    Require(config.experts > 0U && config.experts <= 16384U && config.experts % config.world == 0U &&
                config.experts / config.world <= 128U,
            "experts must be divisible by world with at most 128 experts per rank");
    Require(config.topk > 0U && config.topk <= 32U && config.topk <= config.experts,
            "topk must be in [1,min(32,experts)]");
    Require(config.ring_slots >= 2U && config.ring_slots <= 6U, "ring_slots must be in [2,6]");
}

}  // namespace detail

inline Plan BuildPlan(const Config& config, uint32_t rank)
{
    detail::Validate(config, rank);
    const uint64_t source_items = static_cast<uint64_t>(config.batch) * config.topk;
    // Across all source ranks, the balanced route stream contains exactly
    // world * source_items entries. Every receiver therefore owns source_items
    // rows, even when an individual source's peer counts differ by one.
    const uint64_t full_rows = source_items;
    if (full_rows > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("Task indices exceed the uint32 kernel ABI");
    }

    Plan plan;
    plan.scale_elements = config.hidden / 32U;
    const uint64_t row_bytes = detail::AlignUp(config.hidden, 256U) + detail::AlignUp(plan.scale_elements, 32U);
    if (row_bytes > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("Packed source stride exceeds the uint32 kernel ABI");
    }
    plan.row_bytes = static_cast<uint32_t>(row_bytes);
    detail::CheckBytes(config.batch, row_bytes);
    detail::CheckBytes(full_rows, sizeof(CopyTask));
    detail::CheckBytes(full_rows, config.hidden);
    detail::CheckBytes(full_rows, plan.scale_elements);
    detail::CheckBytes(full_rows, kMetaWords * sizeof(uint32_t));

    std::array<std::vector<CopyTask>, kTrafficKinds> queues;
    const uint32_t local_experts = config.experts / config.world;
    const uint32_t selected_experts = config.first_expert ? 1U : local_experts;
    uint64_t destination = 0U;
    for (uint32_t expert = 0U; expert < selected_experts; ++expert) {
        // benchmark.py routes destination rank first, then local expert:
        // rank = route % world, local_expert = (route / world) % local_experts.
        const uint64_t residue = static_cast<uint64_t>(expert) * config.world + rank;
        for (uint32_t ordinal = 0U; ordinal < config.world; ++ordinal) {
            const uint32_t peer = (rank + ordinal) % config.world;
            const uint64_t source_start = static_cast<uint64_t>(peer) * source_items;
            const uint64_t first = (residue + config.experts - source_start % config.experts) % config.experts;
            const uint32_t traffic = peer == rank ? static_cast<uint32_t>(Traffic::Local)
                                                  : (peer / config.group_size == rank / config.group_size
                                                         ? static_cast<uint32_t>(Traffic::Group)
                                                         : static_cast<uint32_t>(Traffic::Cross));
            for (uint64_t index = first; index < source_items; index += config.experts) {
                if (destination >= full_rows) {
                    throw std::overflow_error("Generated rows exceed the balanced route capacity");
                }
                queues[traffic].push_back({peer, static_cast<uint32_t>(index / config.topk),
                                           static_cast<uint32_t>(index % config.topk),
                                           static_cast<uint32_t>(destination++)});
            }
        }
    }
    if (!config.first_expert && destination != full_rows) {
        throw std::logic_error("Balanced route conservation failed");
    }
    plan.total_rows = static_cast<uint32_t>(destination);
    plan.tasks.reserve(plan.total_rows);
    if (config.group_round_robin) {
        const uint32_t group_count = config.world / config.group_size;
        std::vector<std::vector<CopyTask>> group_rows(group_count), worker_rows(config.cores);
        for (uint32_t traffic = 0U; traffic < kTrafficKinds; ++traffic) {
            plan.traffic_rows[traffic] = queues[traffic].size();
            for (const CopyTask& task : queues[traffic]) {
                group_rows[task.peer / config.group_size].push_back(task);
            }
        }
        for (auto& rows : group_rows) {
            std::sort(rows.begin(), rows.end(),
                      [](const CopyTask& left, const CopyTask& right) { return left.destination < right.destination; });
        }
        plan.phase_count = group_count;
        plan.phases.reserve(group_count);
        uint32_t extra_cursor = 0U;
        for (uint32_t phase = 0U; phase < group_count; ++phase) {
            const uint32_t peer_group = (rank / config.group_size + group_count - phase) % group_count;
            PhasePlan phase_plan{peer_group, {}};
            const auto& rows = group_rows[peer_group];
            for (const CopyTask& task : rows) {
                const uint32_t traffic = task.peer == rank ? static_cast<uint32_t>(Traffic::Local)
                                                           : (peer_group == rank / config.group_size
                                                                  ? static_cast<uint32_t>(Traffic::Group)
                                                                  : static_cast<uint32_t>(Traffic::Cross));
                ++phase_plan.traffic_rows[traffic];
            }
            const uint32_t workers = phase == 0U ? kIntraWorkers : kCrossWorkers;
            const uint32_t first_worker = phase == 0U ? 0U : kIntraWorkers;
            const uint32_t cursor = phase == 0U ? 0U : extra_cursor;
            const uint32_t base = static_cast<uint32_t>(rows.size() / workers);
            const uint32_t extra = static_cast<uint32_t>(rows.size() % workers);
            for (uint32_t logical_worker = 0U; logical_worker < workers; ++logical_worker) {
                const uint32_t quota_owner = (logical_worker + workers - cursor) % workers;
                const uint32_t offset = quota_owner * base + std::min(quota_owner, extra);
                const uint32_t count = base + static_cast<uint32_t>(quota_owner < extra);
                auto& owned = worker_rows[first_worker + logical_worker];
                owned.insert(owned.end(), rows.begin() + offset, rows.begin() + offset + count);
            }
            if (phase != 0U) {
                extra_cursor = (extra_cursor + extra) % kCrossWorkers;
            }
            plan.phases.push_back(phase_plan);
        }
        plan.workers.reserve(config.cores);
        for (uint32_t logical_worker = 0U; logical_worker < config.cores; ++logical_worker) {
            const auto& rows = worker_rows[logical_worker];
            const uint32_t begin = static_cast<uint32_t>(plan.tasks.size());
            plan.tasks.insert(plan.tasks.end(), rows.begin(), rows.end());
            const Traffic traffic = logical_worker < kIntraWorkers ? Traffic::Intra : Traffic::Cross;
            plan.workers.push_back({static_cast<uint32_t>(traffic), begin, static_cast<uint32_t>(rows.size()), 0U});
        }
        if (plan.tasks.size() != plan.total_rows) {
            throw std::logic_error("Group-round-robin route conservation failed");
        }
        return plan;
    }
    plan.workers.reserve(config.cores);
    for (uint32_t traffic = 0U; traffic < kTrafficKinds; ++traffic) {
        const uint64_t rows = queues[traffic].size();
        plan.traffic_rows[traffic] = rows;
        const uint32_t begin = static_cast<uint32_t>(plan.tasks.size());
        plan.tasks.insert(plan.tasks.end(), queues[traffic].begin(), queues[traffic].end());
        const uint32_t workers = config.worker_counts[traffic];
        const uint32_t base = static_cast<uint32_t>(rows / workers);
        const uint32_t extra = static_cast<uint32_t>(rows % workers);
        for (uint32_t worker = 0U; worker < workers; ++worker) {
            const uint32_t offset = worker * base + (worker < extra ? worker : extra);
            plan.workers.push_back({traffic, begin + offset, base + static_cast<uint32_t>(worker < extra), 0U});
        }
    }
    return plan;
}

}  // namespace deepep::dispatch_bench
