// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

// Host-only routing planner. Keep it independent of torch/CANN so the exact
// descriptors consumed by the kernel can be exercised in CPU regression tests.
#include "abi.hpp"
#include "pack_plan.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace MegaMoeUrma {
struct ExpertPlan {
    Config config{};
    std::vector<Fragment> fragments;
    std::vector<RowSegment> segments;
    std::vector<ComputeBatch> computeBatches;
    std::vector<ComputeBatchTask> computeTasks;
    std::vector<Chunk> sendChunks;
    std::vector<Chunk> recvChunks;
    std::vector<PackRound> packRounds;
    std::vector<uint32_t> chunkGroupIndices;
    // Original expanded-row order. UrmaPlan binds this once through the NPU
    // inverse map to cache token-major descriptors consumed by Reduce.
    std::vector<uint64_t> reduceDependencies;
};

// Replenish the outstanding GMM1 group window before EACH GMM2 expert, not
// just at startup. Count actual M-groups, including a short final group, and
// retain at least the legacy one-expert lookahead. Zero exactly reproduces
// the previous G1(0), G1(1), G2(0), ... order, including empty expert batches.
// The window is a schedule distance; paired engines advance independently.
inline std::vector<ComputeBatchTask> BuildComputeTasks(const std::vector<ComputeBatch>& batches, uint32_t aheadMGroups)
{
    std::vector<ComputeBatchTask> tasks;
    tasks.reserve(batches.size() * 2U);
    uint32_t nextGmm1 = 0U;
    uint64_t pendingGroups = 0U;
    for (uint32_t gmm2 = 0U; gmm2 < batches.size(); ++gmm2) {
        const uint64_t minimumGmm1 = std::min<uint64_t>(batches.size(), static_cast<uint64_t>(gmm2) + 2U);
        while (nextGmm1 < batches.size() && (nextGmm1 < minimumGmm1 || pendingGroups < aheadMGroups)) {
            tasks.push_back({ComputeBatchStage::Gmm1, nextGmm1});
            pendingGroups += batches[nextGmm1].fragmentCount;
            ++nextGmm1;
        }
        tasks.push_back({ComputeBatchStage::Gmm2, gmm2});
        pendingGroups -= batches[gmm2].fragmentCount;
    }
    return tasks;
}

inline ExpertPlan BuildExpertPlan(const int64_t* counts, const Config& input, uint64_t maxReceiveRows)
{
    if (counts == nullptr || input.rank >= kWorld || input.localExperts != 24 || input.bs == 0 || input.bs > 8192 ||
        input.topk != 8 || input.hidden != 3584 || input.intermediate != 1280 || input.cvRows < 128 ||
        input.cvRows > 4096 || input.cvRows % 128 != 0 || input.maxPeerChunks == 0 ||
        input.maxPeerChunks > kMaxPeerChunks || input.gmm1AheadMGroups > 65536U || maxReceiveRows == 0) {
        throw std::invalid_argument("Invalid URMA expert-plan dimensions");
    }
    ExpertPlan plan;
    plan.config = input;
    auto& config = plan.config;
    config.gmm1Tiles = config.gmm2Tiles = 0;
    constexpr uint32_t kExperts = kWorld * 24U;
    constexpr uint32_t kRowsPerGroup = 256;
    const uint64_t expandedRows = static_cast<uint64_t>(config.bs) * config.topk;
    constexpr uint64_t kUnassignedDependency = std::numeric_limits<uint64_t>::max();
    plan.reduceDependencies.assign(expandedRows, kUnassignedDependency);
    using RankMatrix = std::array<std::array<uint64_t, kWorld>, kWorld>;
    RankMatrix peerRows{}, sendBase{}, receiveBase{};
    for (uint32_t source = 0; source < kWorld; ++source) {
        uint64_t row = 0;
        for (uint32_t destination = 0; destination < kWorld; ++destination) {
            sendBase[source][destination] = row;
            for (uint32_t expert = 0; expert < config.localExperts; ++expert) {
                const int64_t count = counts[source * kExperts + destination * config.localExperts + expert];
                if (count < 0 || static_cast<uint64_t>(count) > expandedRows) {
                    throw std::invalid_argument("Invalid URMA per-source expert count");
                }
                peerRows[source][destination] += static_cast<uint64_t>(count);
            }
            row += peerRows[source][destination];
            if ((peerRows[source][destination] + config.cvRows - 1U) / config.cvRows > config.maxPeerChunks) {
                throw std::invalid_argument("URMA peer stream exceeds dispatch-ready slot capacity");
            }
        }
        if (row != expandedRows) {
            throw std::invalid_argument("URMA route count sum differs from BS*topk");
        }
    }
    for (uint32_t destination = 0; destination < kWorld; ++destination) {
        uint64_t row = 0;
        for (uint32_t source = 0; source < kWorld; ++source) {
            receiveBase[destination][source] = row;
            row += peerRows[source][destination];
        }
        if (row > maxReceiveRows) {
            throw std::invalid_argument("URMA destination receive rows exceed max_recv_rows");
        }
        if (destination == config.rank) config.receiveRows = row;
    }
    config.selfSendRow = sendBase[config.rank][config.rank];
    config.selfRecvRow = receiveBase[config.rank][config.rank];
    config.selfRows = peerRows[config.rank][config.rank];

    constexpr uint32_t kNoChunk = std::numeric_limits<uint32_t>::max();
    // A ready slot uniquely identifies its original contiguous receive chunk.
    std::vector<uint32_t> readyToChunk(static_cast<uint64_t>(kWorld) * config.maxPeerChunks, kNoChunk);
    for (uint32_t peer = 0; peer < kWorld; ++peer) {
        const uint64_t outgoing = peerRows[config.rank][peer];
        for (uint64_t begin = 0, cv = 0; begin < outgoing; begin += config.cvRows, ++cv) {
            plan.sendChunks.push_back({peer, static_cast<uint32_t>(cv), sendBase[config.rank][peer] + begin,
                                       receiveBase[peer][config.rank] + begin,
                                       static_cast<uint32_t>(std::min<uint64_t>(config.cvRows, outgoing - begin)), 0, 0,
                                       0});
            if (peer != config.rank) {
                const uint64_t end = sendBase[config.rank][peer] + std::min<uint64_t>(begin + config.cvRows, outgoing);
                const uint64_t ready = static_cast<uint64_t>(peer) * config.maxPeerChunks + cv;
                std::fill(plan.reduceDependencies.begin() + sendBase[config.rank][peer] + begin,
                          plan.reduceDependencies.begin() + end, ready);
            }
        }
        const uint64_t incoming = peerRows[peer][config.rank];
        for (uint64_t begin = 0, cv = 0; begin < incoming; begin += config.cvRows, ++cv) {
            const auto index = static_cast<uint32_t>(plan.recvChunks.size());
            readyToChunk[peer * config.maxPeerChunks + cv] = index;
            plan.recvChunks.push_back({peer, static_cast<uint32_t>(cv), receiveBase[config.rank][peer] + begin,
                                       sendBase[peer][config.rank] + begin,
                                       static_cast<uint32_t>(std::min<uint64_t>(config.cvRows, incoming - begin)), 0, 0,
                                       0});
        }
    }

    std::vector<std::vector<uint32_t>> chunkGroups(plan.recvChunks.size());
    std::array<uint64_t, kWorld> peerExpertBegin{};
    uint64_t compactRow = 0;
    for (uint32_t expert = 0; expert < config.localExperts; ++expert) {
        std::array<uint64_t, kWorld + 1U> sourceBegin{};
        for (uint32_t source = 0; source < kWorld; ++source) {
            sourceBegin[source + 1U] =
                sourceBegin[source] + counts[source * kExperts + config.rank * config.localExperts + expert];
        }
        const uint64_t expertRows = sourceBegin[kWorld];
        const auto firstGroup = static_cast<uint32_t>(plan.fragments.size());
        for (uint64_t begin = 0; begin < expertRows; begin += kRowsPerGroup) {
            const uint64_t end = std::min<uint64_t>(begin + kRowsPerGroup, expertRows);
            const auto groupIndex = static_cast<uint32_t>(plan.fragments.size());
            Fragment group{expert,
                           static_cast<uint32_t>(plan.segments.size()),
                           compactRow + begin,
                           static_cast<uint32_t>(end - begin),
                           0,
                           config.gmm1Tiles,
                           config.gmm2Tiles};
            for (uint32_t source = 0; source < kWorld; ++source) {
                uint64_t lo = std::max(begin, sourceBegin[source]);
                const uint64_t hi = std::min(end, sourceBegin[source + 1U]);
                while (lo < hi) {
                    const uint64_t peerRow = peerExpertBegin[source] + lo - sourceBegin[source];
                    const auto cv = static_cast<uint32_t>(peerRow / config.cvRows);
                    const auto count =
                        static_cast<uint32_t>(std::min<uint64_t>(hi - lo, config.cvRows - peerRow % config.cvRows));
                    const uint32_t ready = source * config.maxPeerChunks + cv;
                    plan.segments.push_back({source, cv, receiveBase[config.rank][source] + peerRow,
                                             sendBase[source][config.rank] + peerRow, static_cast<uint32_t>(lo - begin),
                                             count, ready, 0});
                    if (source == config.rank) {
                        const uint64_t first = sendBase[source][config.rank] + peerRow;
                        for (uint64_t row = first; row < first + count; ++row) {
                            if (plan.reduceDependencies[row] != kUnassignedDependency) {
                                throw std::logic_error("URMA self Reduce dependency assigned more than once");
                            }
                            plan.reduceDependencies[row] = kReduceSelfDependency | groupIndex;
                        }
                    }
                    const uint32_t chunkIndex = readyToChunk[ready];
                    if (chunkIndex == kNoChunk) {
                        throw std::logic_error("URMA compute segment has no transport chunk");
                    }
                    auto& dependencies = chunkGroups[chunkIndex];
                    if (dependencies.empty() || dependencies.back() != groupIndex) {
                        dependencies.push_back(groupIndex);
                    }
                    lo += count;
                }
            }
            group.segmentCount = static_cast<uint32_t>(plan.segments.size()) - group.segmentBegin;
            plan.fragments.push_back(group);
            config.gmm1Tiles += (config.intermediate + 255U) / 256U;
            config.gmm2Tiles += (config.hidden + 255U) / 256U;
        }
        plan.computeBatches.push_back({firstGroup, static_cast<uint32_t>(plan.fragments.size()) - firstGroup});
        compactRow += expertRows;
        for (uint32_t source = 0; source < kWorld; ++source) {
            peerExpertBegin[source] += sourceBegin[source + 1U] - sourceBegin[source];
        }
    }
    for (uint32_t index = 0; index < plan.recvChunks.size(); ++index) {
        auto& chunk = plan.recvChunks[index];
        chunk.fragmentBegin = static_cast<uint32_t>(plan.chunkGroupIndices.size());
        chunk.fragmentCount = static_cast<uint32_t>(chunkGroups[index].size());
        plan.chunkGroupIndices.insert(plan.chunkGroupIndices.end(), chunkGroups[index].begin(),
                                      chunkGroups[index].end());
    }
    if (compactRow != config.receiveRows) {
        throw std::logic_error("URMA expert-major rows do not cover the receive layout");
    }
    if (std::find(plan.reduceDependencies.begin(), plan.reduceDependencies.end(), kUnassignedDependency) !=
        plan.reduceDependencies.end()) {
        throw std::logic_error("URMA Reduce dependencies do not cover all expanded rows");
    }
    config.fragmentCount = static_cast<uint32_t>(plan.fragments.size());
    plan.computeTasks = BuildComputeTasks(plan.computeBatches, config.gmm1AheadMGroups);
    config.computeTaskCount = static_cast<uint32_t>(plan.computeTasks.size());
    config.sendChunkCount = static_cast<uint32_t>(plan.sendChunks.size());
    config.recvChunkCount = static_cast<uint32_t>(plan.recvChunks.size());
    plan.packRounds = BuildPackRounds(plan.sendChunks, config.maxPeerChunks);
    config.packRoundCount = static_cast<uint32_t>(plan.packRounds.size());
    return plan;
}
}  // namespace MegaMoeUrma
