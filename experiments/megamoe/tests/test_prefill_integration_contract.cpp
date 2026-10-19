// Copyright (c) 2026, SimpleBright_Man
#include <algorithm>
#include <array>
#undef NDEBUG
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#define __aicore__
#include "prefill_policy.hpp"
#include "kernels/mega_moe/common/mega_moe_comm_split.hpp"

using deepep::megamoe::FillPrefillMetadata;
using deepep::megamoe::PrefillProfileInputs;
using deepep::megamoe::UsePrefillProfile;
using namespace MegaMoeImpl;

static PrefillProfileInputs GoodProfile()
{
    return {true, true, true, false, 16U, 28U, 56U, 4096U, 3584U, 0U, 24U, 256U * 1024U, 6U, 3072U, 3712U};
}

static void CheckPolicy()
{
    auto p = GoodProfile();
    assert(UsePrefillProfile(p));
    for (uint32_t bs : {2048U, 4096U, 8192U}) {
        p = GoodProfile();
        p.bs = bs;
        assert(UsePrefillProfile(p));
    }
    for (uint32_t bs : {0U, 1U, 128U, 256U}) {
        p = GoodProfile();
        p.bs = bs;
        assert(!UsePrefillProfile(p));
    }
    p = GoodProfile();
    p.bs = 257U;
    assert(UsePrefillProfile(p));
    p = GoodProfile();
    p.requested = false;
    assert(!UsePrefillProfile(p));
    p = GoodProfile();
    p.isA8W4 = false;
    assert(!UsePrefillProfile(p));
    p = GoodProfile();
    p.combineNoQuant = false;
    assert(!UsePrefillProfile(p));
    p = GoodProfile();
    p.topkWeightsPrefetch = true;
    assert(!UsePrefillProfile(p));
    for (uint32_t value : {0U, 1U, 4U, 8U, 32U}) {
        p = GoodProfile();
        p.world = value;
        assert(!UsePrefillProfile(p));
    }
    for (uint32_t value : {0U, 1U, 24U, 32U}) {
        p = GoodProfile();
        p.aic = value;
        assert(!UsePrefillProfile(p));
    }
    for (uint32_t value : {0U, 28U, 55U, 64U}) {
        p = GoodProfile();
        p.aiv = value;
        assert(!UsePrefillProfile(p));
    }
    for (uint32_t value : {1U, 2U, 8U}) {
        p = GoodProfile();
        p.sharedExperts = value;
        assert(!UsePrefillProfile(p));
    }
    p = GoodProfile();
    p.localExperts = 0U;
    assert(!UsePrefillProfile(p));
    p = GoodProfile();
    p.hidden = 0U;
    assert(!UsePrefillProfile(p));
    p = GoodProfile();
    p.hidden = 7680U;
    assert(UsePrefillProfile(p));
    p.hidden = 7681U;
    assert(!UsePrefillProfile(p));
    p.hidden = 8192U;
    assert(!UsePrefillProfile(p));
    p = GoodProfile();
    p.hidden = std::numeric_limits<uint32_t>::max();
    assert(!UsePrefillProfile(p));
    for (uint32_t value : {0U, 1U, 2U, 4U, 5U, 7U, 8U}) {
        p = GoodProfile();
        p.dispatchSlots = value;
        assert(!UsePrefillProfile(p));
    }
    p = GoodProfile();
    p.dispatchRouteItems = 0U;
    assert(!UsePrefillProfile(p));
    p = GoodProfile();
    p.dispatchCopyBytes = 0U;
    assert(!UsePrefillProfile(p));
    p = GoodProfile();
    p.dispatchCopyBytes = std::numeric_limits<uint32_t>::max();
    assert(!UsePrefillProfile(p));
    // Combine metadata is [200 KiB,248 KiB), beyond the row-ring endpoint.
    for (uint32_t ub : {0U, 128U * 1024U, 184U * 1024U, 224U * 1024U, 248U * 1024U - 1U}) {
        p = GoodProfile();
        p.ubBytes = ub;
        assert(!UsePrefillProfile(p));
    }
    p = GoodProfile();
    p.ubBytes = 248U * 1024U;
    assert(UsePrefillProfile(p));
    // Route-index batches must include the same alignment as the actual UB view.
    p = GoodProfile();
    p.dispatchRouteItems = 60001U;
    const uint64_t cumsumBytes = 16U * 24U * 4U;
    const uint64_t expertBytes = 24U * 4U;
    const uint64_t routeBytes = (uint64_t{60001U} * 4U + 31U) / 32U * 32U;
    const uint64_t end = cumsumBytes + expertBytes + routeBytes + 6U * (3712U + 32U);
    p.ubBytes = static_cast<uint32_t>(end - 1U);
    assert(!UsePrefillProfile(p));
    p.ubBytes = static_cast<uint32_t>(end);
    assert(UsePrefillProfile(p));
}

static void CheckMetadata()
{
    for (uint32_t bs : {1U, 257U, 2048U, 4096U, 8192U}) {
        constexpr uint32_t world = 16U, topk = 8U;
        const size_t words = static_cast<size_t>(world) * bs * topk * 8U;
        std::vector<int32_t> values(words + 2U, -987654321);
        FillPrefillMetadata(values.data() + 1U, world, bs, topk);
        assert(values.front() == -987654321 && values.back() == -987654321);
        for (size_t row = 0; row < words / 8U; ++row) {
            const auto* record = values.data() + 1U + row * 8U;
            assert(record[0] == static_cast<int32_t>(row / (bs * topk)));
            assert(record[1] == static_cast<int32_t>((row / topk) % bs));
            assert(record[2] == static_cast<int32_t>(row % topk));
            for (size_t field = 3U; field < 8U; ++field) assert(record[field] == 0);
        }
    }
    // Empty inputs never write; top-k32 retains exact flat-route slot identity.
    std::array<int32_t, 256> small;
    small.fill(-1);
    FillPrefillMetadata(small.data(), 0U, 1U, 8U);
    assert(small.front() == -1);
    FillPrefillMetadata(small.data(), 1U, 1U, 32U);
    assert(small[31U * 8U + 2U] == 31);
}

struct Counts {
    std::array<uint32_t, 16> rows{};
    uint32_t Get(uint32_t peer) const
    {
        return rows[peer];
    }
};

static void CheckSlice(const Counts& counts, uint32_t rank, uint32_t first, uint32_t stop, uint64_t rotation)
{
    CommPeerSegment peers[16];
    BuildCommPeerSegments(counts, rank, 0U, first, stop, true, peers);
    // Independent fair-round oracle determines the source ordinals in this slice.
    std::vector<uint32_t> expectedSource;
    for (uint32_t row = 0U; expectedSource.size() < stop; ++row) {
        for (uint32_t ordinal = 0U; ordinal < 16U; ++ordinal) {
            const uint32_t peer = (rank + ordinal) % 16U;
            if (row < counts.Get(peer)) expectedSource.push_back(peer * 1000000U + row);
        }
    }
    std::vector<uint32_t> expected(expectedSource.begin() + first, expectedSource.begin() + stop);
    std::vector<uint32_t> expectedRows;
    for (uint32_t ordinal = 0U; ordinal < 16U; ++ordinal) {
        const uint32_t peer = (rank + ordinal) % 16U;
        for (const auto identity : expected) {
            if (identity / 1000000U == peer) expectedRows.push_back(identity);
        }
    }
    std::sort(expected.begin(), expected.end());
    std::vector<uint32_t> observed;
    std::vector<uint32_t> hits(stop - first, 0U);
    std::vector<uint32_t> ready((stop + 255U) / 256U, 0U);
    std::array<uint32_t, 28> coreRows{};
    for (uint32_t core = 0U; core < 28U; ++core) {
        const auto owned = SelectCommOwnedSegments(peers, rank, core, 28U, 4U, rotation, true);
        uint32_t previousPhase = 0U;
        for (uint32_t i = 0U; i < owned.count; ++i) {
            const auto& segment = owned.segments[i];
            const uint32_t phase = (rank / 4U) ^ (segment.peer / 4U);
            assert((core < 4U) == (phase == 0U));
            assert(phase >= previousPhase);
            previousPhase = phase;
            assert(segment.rowBegin >= first && segment.rowBegin + segment.count <= stop);
            for (uint32_t j = 0U; j < segment.count; ++j) {
                const uint32_t destination = segment.rowBegin + j;
                assert(++hits[destination - first] == 1U);
                ++ready[destination / 256U];
                observed.push_back(segment.peer * 1000000U + segment.sourceBegin + j);
                // Reverse Combine indexes this exact output row's Dispatch metadata.
                assert(expectedRows[destination - first] == segment.peer * 1000000U + segment.sourceBegin + j);
                assert(segment.sourceBegin + j < counts.Get(segment.peer));
            }
            coreRows[core] += segment.count;
        }
        assert(coreRows[core] == owned.rows);
    }
    std::sort(observed.begin(), observed.end());
    assert(observed == expected);
    assert(std::all_of(hits.begin(), hits.end(), [](uint32_t n) { return n == 1U; }));
    assert(std::accumulate(coreRows.begin(), coreRows.end(), 0U) == stop - first);
    for (uint32_t tile = first / 256U; tile < ready.size(); ++tile) {
        const uint32_t begin = std::max(first, tile * 256U);
        const uint32_t end = std::min(stop, (tile + 1U) * 256U);
        assert(ready[tile] == end - begin);
    }
}

static void CheckMapping()
{
    // Per-source/per-expert counts for repository balanced routes (384 experts,K8).
    // Sweep every slice for all local experts, including expert and M256 tails.
    for (uint32_t bs : {2048U, 4096U, 8192U}) {
        for (uint32_t rank = 0U; rank < 16U; ++rank) {
            uint32_t usedGroups = 0U, wave = 0U;
            uint64_t globalRow = 0U;
            for (uint32_t localExpert = 0U; localExpert < 24U; ++localExpert) {
                Counts counts;
                for (uint32_t source = 0U; source < 16U; ++source) {
                    const uint32_t length = bs * 8U;
                    const uint32_t first = (localExpert * 16U + rank + 384U - (source * length) % 384U) % 384U;
                    counts.rows[source] = (length - first + 383U) / 384U;
                }
                const uint32_t total = std::accumulate(counts.rows.begin(), counts.rows.end(), 0U);
                for (uint32_t first = 0U; first < total;) {
                    if (usedGroups == 12U) {
                        usedGroups = 0U;
                        ++wave;
                    }
                    const uint32_t count = std::min(total - first, (12U - usedGroups) * 256U);
                    CheckSlice(counts, rank, first, first + count, globalRow + wave);
                    first += count;
                    globalRow += count;
                    usedGroups += (count + 255U) / 256U;
                }
            }
            assert(globalRow == uint64_t{bs} * 8U);
        }
    }
    Counts skewed{{0U, 1U, 3U, 257U, 9U, 0U, 500U, 12U, 1U, 1024U, 0U, 31U, 256U, 5U, 17U, 2U}};
    const auto total = std::accumulate(skewed.rows.begin(), skewed.rows.end(), 0U);
    for (uint32_t rank = 0U; rank < 16U; ++rank) {
        for (uint32_t first = 0U; first < total; first += 257U) {
            CheckSlice(skewed, rank, first, std::min(first + 257U, total), 10000000000ULL + first);
        }
    }
}

int main()
{
    CheckPolicy();
    CheckMetadata();
    std::cout << "profile allocation boundaries and metadata lifetime-independent identities passed\n";
    CheckMapping();
    std::cout << "BS2048/4096/8192: EP16 XOR 4/24 ownership, M256 ready counts and inverse rows passed\n";
}
