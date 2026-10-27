// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

// Host-only inverse routing for full-H GMM2 reorder. Device owners consume
// one direct range per receive CV; they never search the compute fragments.
#include "abi.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace MegaMoeUrma {
struct PeerReorderPlan {
    // Indexed by recvChunks, including self CVs. The remote peer owner handles
    // its CVs; self CV c belongs to communication worker c % (world - 1).
    std::vector<PeerReorderRange> ranges;
    std::vector<PeerReorderSpan> spans;
};

inline PeerReorderPlan BuildPeerReorderPlan(const Config& config, const std::vector<Fragment>& fragments,
                                            const std::vector<RowSegment>& segments,
                                            const std::vector<Chunk>& recvChunks)
{
    if ((config.world != 4U && config.world != 8U && config.world != 16U) || config.rank >= config.world ||
        config.maxPeerChunks == 0U || config.maxPeerChunks > kMaxPeerChunks || config.cvRows == 0U ||
        fragments.size() > UINT32_MAX || segments.size() > UINT32_MAX || recvChunks.size() > UINT32_MAX) {
        throw std::invalid_argument("Invalid URMA peer-reorder dimensions");
    }
    constexpr uint32_t kNoChunk = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> readyToChunk(static_cast<uint64_t>(config.world) * config.maxPeerChunks, kNoChunk);
    std::vector<uint32_t> chunkRows(recvChunks.size(), 0U);
    PeerReorderPlan plan;
    plan.ranges.resize(recvChunks.size());
    for (uint32_t index = 0U; index < recvChunks.size(); ++index) {
        const auto& chunk = recvChunks[index];
        if (chunk.peer >= config.world || chunk.cv >= config.maxPeerChunks || chunk.rows == 0U ||
            chunk.rows > config.cvRows || chunk.localRow > config.receiveRows ||
            chunk.rows > config.receiveRows - chunk.localRow) {
            throw std::invalid_argument("Invalid URMA peer-reorder receive chunk");
        }
        auto& slot = readyToChunk[chunk.peer * config.maxPeerChunks + chunk.cv];
        if (slot != kNoChunk) throw std::invalid_argument("Duplicate URMA peer-reorder peer/CV");
        slot = index;
    }

    // Retain the inverse lookup once per segment. Counting, prefixing, and
    // scattering are linear in the plan size, including skewed one-expert
    // routes where a peer/CV can intersect many M256 groups.
    std::vector<uint32_t> segmentChunks(segments.size(), kNoChunk);
    uint64_t compactRows = 0U;
    uint32_t nextSegment = 0U;
    for (const auto& fragment : fragments) {
        if (fragment.rows == 0U || fragment.rows > 256U || fragment.rowBegin != compactRows ||
            fragment.rowBegin > config.receiveRows || fragment.rows > config.receiveRows - fragment.rowBegin ||
            fragment.segmentBegin != nextSegment || fragment.segmentCount > segments.size() - nextSegment) {
            throw std::invalid_argument("Invalid URMA peer-reorder fragment");
        }
        uint32_t groupRows = 0U;
        for (uint32_t offset = 0U; offset < fragment.segmentCount; ++offset) {
            const uint32_t index = fragment.segmentBegin + offset;
            const auto& segment = segments[index];
            if (segment.sourceRank >= config.world || segment.cv >= config.maxPeerChunks || segment.rows == 0U ||
                segment.tileRow != groupRows || segment.rows > fragment.rows - groupRows) {
                throw std::invalid_argument("Invalid URMA peer-reorder row segment");
            }
            const auto chunkIndex = readyToChunk[segment.sourceRank * config.maxPeerChunks + segment.cv];
            if (chunkIndex == kNoChunk) throw std::invalid_argument("URMA peer-reorder segment has no receive CV");
            const auto& chunk = recvChunks[chunkIndex];
            const auto covered = chunkRows[chunkIndex];
            if (segment.receiveRow != chunk.localRow + covered || segment.returnRow < chunk.remoteRow ||
                segment.returnRow - chunk.remoteRow != covered || segment.rows > chunk.rows - covered) {
                throw std::invalid_argument("URMA peer-reorder rows do not partition their receive CV");
            }
            segmentChunks[index] = chunkIndex;
            ++plan.ranges[chunkIndex].spanCount;
            chunkRows[chunkIndex] += segment.rows;
            groupRows += segment.rows;
        }
        if (groupRows != fragment.rows) throw std::invalid_argument("Incomplete URMA peer-reorder M-group");
        compactRows += fragment.rows;
        nextSegment += fragment.segmentCount;
    }
    if (compactRows != config.receiveRows || nextSegment != segments.size()) {
        throw std::invalid_argument("URMA peer-reorder does not cover the receive layout");
    }

    uint32_t nextSpan = 0U;
    for (uint32_t index = 0U; index < recvChunks.size(); ++index) {
        if (chunkRows[index] != recvChunks[index].rows) {
            throw std::invalid_argument("Incomplete URMA peer-reorder receive CV");
        }
        auto& range = plan.ranges[index];
        range.spanBegin = nextSpan;
        nextSpan += range.spanCount;
    }
    plan.spans.resize(nextSpan);
    std::vector<uint32_t> cursors(recvChunks.size(), 0U);
    for (uint32_t group = 0U; group < fragments.size(); ++group) {
        const auto& fragment = fragments[group];
        for (uint32_t offset = 0U; offset < fragment.segmentCount; ++offset) {
            const uint32_t index = fragment.segmentBegin + offset;
            const auto& segment = segments[index];
            const uint32_t chunkIndex = segmentChunks[index];
            const uint32_t spanIndex = plan.ranges[chunkIndex].spanBegin + cursors[chunkIndex]++;
            plan.spans[spanIndex] = {fragment.rowBegin + segment.tileRow,
                                     segment.sourceRank == config.rank ? segment.returnRow : segment.receiveRow, group,
                                     segment.rows};
        }
    }
    return plan;
}
}  // namespace MegaMoeUrma
