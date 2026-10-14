// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#ifndef MEGA_MOE_DOMAIN_ROWS_H
#define MEGA_MOE_DOMAIN_ROWS_H

#include "mega_moe_peer_rows.hpp"

namespace MegaMoeImpl {

constexpr uint32_t DOMAIN_ROW_WORLD_SIZE = 16U;
constexpr uint32_t DOMAIN_ROW_DOMAIN_SIZE = 4U;
constexpr uint32_t DOMAIN_ROW_DOMAIN_COUNT = DOMAIN_ROW_WORLD_SIZE / DOMAIN_ROW_DOMAIN_SIZE;

MEGAMOE_PEER_ROWS_INLINE bool UseDomainRowCombine(uint32_t worldSize, uint32_t workers, bool prefill,
                                                  uint32_t ranksPerNode)
{
    return worldSize == DOMAIN_ROW_WORLD_SIZE && workers == 28U && prefill && ranksPerNode == DOMAIN_ROW_WORLD_SIZE;
}

struct DomainDispatchPeerOrder {
    uint32_t rank;
    uint32_t worldSize;
    uint32_t ranksPerNode;

    MEGAMOE_PEER_ROWS_INLINE uint32_t Get(uint32_t ordinal) const
    {
        return GetDispatchPeer(ordinal, rank, worldSize, ranksPerNode);
    }
};

// Values only: a plan may outlive the UB count/prefix views used to build it.
// expertRowBegin is the original expert-relative GM row, never a domain queue index.
struct DomainRowSegment {
    uint32_t peer;
    uint32_t sourceBegin;
    uint32_t expertRowBegin;
    uint32_t rowCount;
};

struct DomainRowPlan {
    uint32_t segmentCount = 0U;
    uint32_t totalRows = 0U;
    DomainRowSegment segments[DOMAIN_ROW_WORLD_SIZE];
};

struct DomainOwnedRows {
    uint32_t begin;
    uint32_t end;
};

// Partition each domain independently. Canonical domain prefixes make remainder
// owners form one circular sequence, so total row counts also differ by at most one.
MEGAMOE_PEER_ROWS_INLINE DomainOwnedRows GetDomainOwnedRows(uint32_t rows, uint32_t worker, uint32_t workers,
                                                            uint64_t globalRowPrefix, uint32_t domainPrefix)
{
    const uint32_t first =
        static_cast<uint32_t>((globalRowPrefix % workers + static_cast<uint64_t>(domainPrefix) % workers) % workers);
    const uint32_t logical = worker >= first ? worker - first : worker + (workers - first);
    const uint32_t base = rows / workers;
    const uint32_t extra = rows % workers;
    const uint32_t begin = logical * base + (logical < extra ? logical : extra);
    return {begin, begin + base + static_cast<uint32_t>(logical < extra)};
}

// Reconstruct exactly the Prefill Dispatch peer quotas, then change only the
// communication owner. workers/worker already include the caller's wave rotation.
// visitStart changes submission order; it must not change quotas or ownership.
template <typename Counts, typename PeerOrder>
MEGAMOE_PEER_ROWS_INLINE bool BuildDomainRowPlan(const Counts& counts, const PeerOrder& peers, uint32_t worldSize,
                                                 uint32_t sliceBegin, uint32_t sliceEnd, uint32_t worker,
                                                 uint32_t workers, uint64_t globalRowPrefix, uint32_t visitStart,
                                                 DomainRowPlan& plan)
{
    plan.segmentCount = 0U;
    plan.totalRows = 0U;
    if (worldSize != DOMAIN_ROW_WORLD_SIZE || workers == 0U || worker >= workers || sliceEnd < sliceBegin) {
        return false;
    }
    const PeerRowLimits limits = GetPeerRowLimits(counts, worldSize);
    if (limits.total > static_cast<uint64_t>(UINT32_MAX) || sliceEnd > limits.total) {
        return false;
    }
    if (sliceBegin == sliceEnd) {
        return true;
    }

    DomainRowSegment original[DOMAIN_ROW_WORLD_SIZE];
    uint32_t domainRows[DOMAIN_ROW_DOMAIN_COUNT]{};
    const bool regular = CanUseRegularPeerSlice(limits, worldSize, sliceBegin, sliceEnd);
    const RegularPeerSlice regularSlice = CreateRegularPeerSlice(worldSize, sliceBegin, sliceEnd);
    PeerRowCut beginCut{}, endCut{};
    if (!regular) {
        beginCut = GetPeerRowCut(counts, worldSize, limits, sliceBegin);
        endCut = GetPeerRowCut(counts, worldSize, limits, sliceEnd);
    }
    uint32_t destination = sliceBegin;
    uint32_t seenPeers = 0U;
    for (uint32_t ordinal = 0U; ordinal < worldSize; ++ordinal) {
        const uint32_t peer = peers.Get(ordinal);
        if (peer >= worldSize || (seenPeers & (1U << peer)) != 0U) {
            return false;
        }
        seenPeers |= 1U << peer;
        const uint32_t count = counts.Get(peer);
        const uint32_t sourceBegin = regular ? regularSlice.SourceBegin(ordinal) : TakePeerRowPrefix(count, beginCut);
        const uint32_t sourceEnd = regular ? regularSlice.SourceEnd(ordinal) : TakePeerRowPrefix(count, endCut);
        if (sourceEnd < sourceBegin || sourceEnd > count || sourceEnd - sourceBegin > sliceEnd - destination) {
            return false;
        }
        const uint32_t rows = sourceEnd - sourceBegin;
        original[ordinal] = {peer, sourceBegin, destination, rows};
        domainRows[peer / DOMAIN_ROW_DOMAIN_SIZE] += rows;
        destination += rows;
    }
    if (destination != sliceEnd) {
        return false;
    }

    DomainOwnedRows owned[DOMAIN_ROW_DOMAIN_COUNT];
    uint32_t prefix = 0U;
    for (uint32_t domain = 0U; domain < DOMAIN_ROW_DOMAIN_COUNT; ++domain) {
        owned[domain] = GetDomainOwnedRows(domainRows[domain], worker, workers, globalRowPrefix, prefix);
        prefix += domainRows[domain];
    }
    visitStart %= DOMAIN_ROW_DOMAIN_COUNT;
    for (uint32_t step = 0U; step < DOMAIN_ROW_DOMAIN_COUNT; ++step) {
        const uint32_t domain = (visitStart + step) % DOMAIN_ROW_DOMAIN_COUNT;
        uint32_t queueBegin = 0U;
        for (uint32_t ordinal = 0U; ordinal < worldSize; ++ordinal) {
            const DomainRowSegment& segment = original[ordinal];
            if (segment.peer / DOMAIN_ROW_DOMAIN_SIZE != domain) {
                continue;
            }
            const uint32_t queueEnd = queueBegin + segment.rowCount;
            const uint32_t begin = queueBegin > owned[domain].begin ? queueBegin : owned[domain].begin;
            const uint32_t end = queueEnd < owned[domain].end ? queueEnd : owned[domain].end;
            if (begin < end) {
                const uint32_t offset = begin - queueBegin;
                plan.segments[plan.segmentCount++] = {segment.peer, segment.sourceBegin + offset,
                                                      segment.expertRowBegin + offset, end - begin};
                plan.totalRows += end - begin;
            }
            queueBegin = queueEnd;
        }
    }
    return true;
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_DOMAIN_ROWS_H
