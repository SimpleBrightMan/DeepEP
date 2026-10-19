// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "mega_moe_peer_rows.hpp"

namespace MegaMoeImpl {

constexpr uint32_t COMM_SPLIT_PEERS = 16U;
constexpr uint32_t COMM_SPLIT_GROUP_SIZE = 4U;

// A source ordinal and unchanged expert-local destination rows.
// Keeping both coordinates makes Dispatch and the reverse Combine share one plan.
struct CommPeerSegment {
    uint32_t peer = 0U;
    uint32_t sourceBegin = 0U;
    uint32_t rowBegin = 0U;
    uint32_t count = 0U;
    uint32_t stride = 1U;
};

struct CommOwnedSegments {
    CommPeerSegment segments[COMM_SPLIT_PEERS];
    uint32_t count = 0U;
    uint32_t rows = 0U;
};

// Reconstruct the existing source-major Decode / peer-balanced Prefill layout.
// This changes task ownership and issue order, never the expert/GMM row layout.
template <typename Counts>
__aicore__ inline void BuildCommPeerSegments(const Counts& counts, uint32_t rank, uint32_t ranksPerNode,
                                             uint32_t sliceBegin, uint32_t sliceEnd, bool peerBalanced,
                                             CommPeerSegment* peers)
{
    for (uint32_t peer = 0U; peer < COMM_SPLIT_PEERS; ++peer) {
        peers[peer] = {peer, 0U, 0U, 0U};
    }
    if (!peerBalanced) {
        uint32_t prefix = 0U;
        for (uint32_t peer = 0U; peer < COMM_SPLIT_PEERS; ++peer) {
            uint32_t end = prefix + counts.Get(peer);
            uint32_t begin = prefix > sliceBegin ? prefix : sliceBegin;
            uint32_t stop = end < sliceEnd ? end : sliceEnd;
            if (begin < stop) {
                peers[peer] = {peer, begin - prefix, begin, stop - begin};
            }
            prefix = end;
        }
        return;
    }
    auto limits = GetPeerRowLimits(counts, COMM_SPLIT_PEERS);
    auto beginCut = GetPeerRowCut(counts, COMM_SPLIT_PEERS, limits, sliceBegin);
    auto endCut = GetPeerRowCut(counts, COMM_SPLIT_PEERS, limits, sliceEnd);
    uint32_t destination = sliceBegin;
    for (uint32_t ordinal = 0U; ordinal < COMM_SPLIT_PEERS; ++ordinal) {
        uint32_t peer = GetDispatchPeer(ordinal, rank, COMM_SPLIT_PEERS, ranksPerNode);
        uint32_t count = counts.Get(peer);
        uint32_t begin = TakePeerRowPrefix(count, beginCut);
        uint32_t end = TakePeerRowPrefix(count, endCut);
        peers[peer] = {peer, begin, destination, end - begin};
        destination += end - begin;
    }
}

// Each cohort balances the concatenated rows of one physical group. Cross
// cohorts visit own-1, own-2, own-3 by default. Paired groups use own XOR 1/2/3:
// (0,1)+(2,3), (0,2)+(1,3), (0,3)+(1,2). A rotating prefix carries remainders
// across groups/slices/waves. There is no communication or synchronization here.
__aicore__ inline CommOwnedSegments SelectCommOwnedSegments(const CommPeerSegment* peers, uint32_t rank,
                                                            uint32_t physicalCore, uint32_t totalCores,
                                                            uint32_t intraCores, uint64_t rotation,
                                                            bool pairedGroups = false, bool alltoall = false)
{
    CommOwnedSegments owned{};
    const bool intra = physicalCore < intraCores;
    const uint32_t workers = intra ? intraCores : totalCores - intraCores;
    if (workers == 0U || physicalCore >= totalCores) {
        return owned;
    }
    const uint32_t worker = intra ? physicalCore : physicalCore - intraCores;
    const uint32_t ownGroup = rank / COMM_SPLIT_GROUP_SIZE;
    if (alltoall && !intra) {
        // Match case10's peer-major, cyclic token ownership. Each worker visits
        // the same +1/+2/+3 group order and rank-relative peer order. Carry the
        // actual peer row counts instead of assuming equal n; retain the slice
        // rotation so short-slice remainders do not always select the same cores.
        uint64_t prefix = rotation;
        for (uint32_t offset = 1U; offset < COMM_SPLIT_PEERS / COMM_SPLIT_GROUP_SIZE; ++offset) {
            const uint32_t group = (ownGroup + offset) % (COMM_SPLIT_PEERS / COMM_SPLIT_GROUP_SIZE);
            for (uint32_t lane = 0U; lane < COMM_SPLIT_GROUP_SIZE; ++lane) {
                const uint32_t peer = group * COMM_SPLIT_GROUP_SIZE + (rank + lane) % COMM_SPLIT_GROUP_SIZE;
                const auto& segment = peers[peer];
                const uint32_t firstRow = (worker + workers - static_cast<uint32_t>(prefix % workers)) % workers;
                if (firstRow < segment.count) {
                    const uint32_t count = 1U + (segment.count - firstRow - 1U) / workers;
                    owned.segments[owned.count++] = {peer, segment.sourceBegin + firstRow, segment.rowBegin + firstRow,
                                                     count, workers};
                    owned.rows += count;
                }
                prefix += segment.count;
            }
        }
        return owned;
    }
    const uint32_t first = intra ? 0U : 1U;
    const uint32_t end = intra ? 1U : COMM_SPLIT_PEERS / COMM_SPLIT_GROUP_SIZE;
    for (uint32_t phase = first; phase < end; ++phase) {
        const uint32_t group = intra ? ownGroup : (pairedGroups ? ownGroup ^ phase : (ownGroup + 4U - phase) % 4U);
        uint32_t rows = 0U;
        for (uint32_t lane = 0U; lane < COMM_SPLIT_GROUP_SIZE; ++lane) {
            rows += peers[group * COMM_SPLIT_GROUP_SIZE + lane].count;
        }
        const uint32_t startWorker = static_cast<uint32_t>(rotation % workers);
        const uint32_t logical = (worker + workers - startWorker) % workers;
        const uint32_t base = rows / workers;
        const uint32_t extra = rows % workers;
        const uint32_t begin = logical * base + (logical < extra ? logical : extra);
        const uint32_t stop = begin + base + static_cast<uint32_t>(logical < extra);
        uint32_t prefix = 0U;
        for (uint32_t lane = 0U; lane < COMM_SPLIT_GROUP_SIZE; ++lane) {
            const auto& segment = peers[group * COMM_SPLIT_GROUP_SIZE + lane];
            const uint32_t left = prefix > begin ? prefix : begin;
            const uint32_t right = prefix + segment.count < stop ? prefix + segment.count : stop;
            if (left < right) {
                owned.segments[owned.count++] = {segment.peer, segment.sourceBegin + left - prefix,
                                                 segment.rowBegin + left - prefix, right - left};
                owned.rows += right - left;
            }
            prefix += segment.count;
        }
        rotation += rows;
    }
    return owned;
}

}  // namespace MegaMoeImpl
