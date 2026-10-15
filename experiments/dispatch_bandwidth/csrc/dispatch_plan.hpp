// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once

#include "dispatch_types.hpp"

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
};

struct Plan {
    // Tasks are grouped by traffic, then divided into worker-owned intervals.
    // destination retains the expert/peer ordered output row before regrouping.
    std::vector<CopyTask> tasks;
    std::vector<Worker> workers;
    std::array<uint64_t, kTrafficKinds> traffic_rows{};
    uint32_t total_rows = 0U;
    uint32_t row_bytes = 0U;
    uint32_t scale_elements = 0U;
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
    uint64_t workers = 0U;
    for (uint32_t count : config.worker_counts) {
        Require(count > 0U, "each traffic class must have at least one worker");
        workers += count;
    }
    Require(workers == config.cores, "worker_counts must sum to cores");
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
