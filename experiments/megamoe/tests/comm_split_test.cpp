// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __gm__
#include "common/mega_moe_comm_split.hpp"

using namespace MegaMoeImpl;

struct Counts {
    std::array<uint32_t, 16> values{};
    uint32_t Get(uint32_t peer) const
    {
        return values.at(peer);
    }
};

static void Require(bool condition)
{
    if (!condition) throw std::runtime_error("communication split invariant failed");
}

namespace AscendC {
enum class HardEvent { MTE3_S };
}
constexpr int32_t SYNC_EVENT_ID5 = 5;
constexpr int32_t INT_CACHELINE = 16;
struct MoeSyncWorkspaceLayout {
    uint32_t dispatchFlagSlotCountPerExpert;
};
struct Params {
    struct {
        uint8_t* flagDispatchToGmm1Ptr;
    } workspaceInfo;
};
static uint32_t readyFences = 0U;
template <AscendC::HardEvent Event, int32_t Id>
static void SyncFuncStatic()
{
    static_assert(Event == AscendC::HardEvent::MTE3_S && Id == SYNC_EVENT_ID5);
    ++readyFences;
}
static void AtomicAdd(int32_t* address, int32_t value)
{
    Require(value > 0);
    *address += value;
}

// INSERT_PRODUCTION_READY

using Identity = std::pair<uint32_t, uint32_t>;

// Enumerate source queues one item at a time, independently of the device cut
// formulas. Group the selected items by the original peer order for this slice.
static std::vector<Identity> Reference(const Counts& counts, uint32_t rank, uint32_t ranksPerNode, uint32_t begin,
                                       uint32_t end, bool balanced)
{
    std::vector<Identity> stream;
    if (!balanced) {
        for (uint32_t peer = 0; peer < 16; ++peer)
            for (uint32_t item = 0; item < counts.Get(peer); ++item) stream.emplace_back(peer, item);
        return {stream.begin() + begin, stream.begin() + end};
    }
    std::array<uint32_t, 16> order{};
    for (uint32_t ordinal = 0; ordinal < 16; ++ordinal) {
        if (ranksPerNode == 0 || ranksPerNode == 16) {
            order[ordinal] = (rank + ordinal) % 16;
        } else {
            const uint32_t nodes = 16 / ranksPerNode;
            uint32_t nodeOffset = ordinal % nodes;
            if (nodeOffset < 2) nodeOffset = 1 - nodeOffset;
            order[ordinal] = ((rank / ranksPerNode + nodeOffset) % nodes) * ranksPerNode +
                             (rank % ranksPerNode + ordinal / nodes) % ranksPerNode;
        }
    }
    const uint32_t largest = *std::max_element(counts.values.begin(), counts.values.end());
    for (uint32_t item = 0; item < largest; ++item)
        for (uint32_t peer : order)
            if (item < counts.Get(peer)) stream.emplace_back(peer, item);
    std::vector<Identity> result;
    for (uint32_t peer : order)
        for (uint32_t i = begin; i < end; ++i)
            if (stream[i].first == peer) result.push_back(stream[i]);
    return result;
}

static void Check(const Counts& counts, uint32_t rank, uint32_t ranksPerNode, uint32_t begin, uint32_t end,
                  bool balanced, uint64_t rotation, bool paired = false, uint32_t pairedIntra = 12U,
                  bool alltoall = false)
{
    const auto reference = Reference(counts, rank, ranksPerNode, begin, end, balanced);
    CommPeerSegment peers[16];
    BuildCommPeerSegments(counts, rank, ranksPerNode, begin, end, balanced, peers);
    // Explicit matchings provide an oracle independent of the device XOR.
    constexpr uint32_t partners[4][4] = {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}};
    std::array<std::vector<uint32_t>, 4> groupedRows;
    if (paired || alltoall) {
        for (uint32_t peer = 0; peer < 16; ++peer)
            for (uint32_t index = 0; index < reference.size(); ++index)
                if (reference[index].first == peer) groupedRows[peer / 4].push_back(begin + index);
    }
    // pairedIntra=0 exercises every legal cohort split with the same route
    // enumeration, avoiding redundant source-queue construction per split.
    const uint32_t firstSplit = (paired || alltoall) && pairedIntra ? pairedIntra : 1U;
    const uint32_t lastSplit = (paired || alltoall) && pairedIntra ? pairedIntra + 1U : 28U;
    for (uint32_t intra = firstSplit; intra < lastSplit; ++intra) {
        std::vector<uint32_t> visits(end - begin, 0U);
        const uint32_t readyWords = (end / 256U + 1U) * INT_CACHELINE;
        std::vector<int32_t> readyCounts(readyWords * 2U, 0);
        MoeSyncWorkspaceLayout syncLayout{readyWords};
        Params params{{reinterpret_cast<uint8_t*>(readyCounts.data())}};
        std::array<std::array<uint32_t, 28>, 4> phaseRows{};
        std::array<std::vector<uint32_t>, 28> expectedOwned;
        if (paired || alltoall) {
            for (bool local : {true, false}) {
                const uint32_t workers = local ? intra : 28U - intra;
                uint64_t prefix = rotation;
                if (alltoall && !local) {
                    // Independently enumerate every selected row in peer order
                    // and hand it to the next worker. No device stride/cut
                    // formula is used to build the expected assignment.
                    uint32_t owner = static_cast<uint32_t>(rotation % workers);
                    for (uint32_t offset = 1U; offset < 4U; ++offset) {
                        const uint32_t group = (rank / 4U + offset) % 4U;
                        for (uint32_t lane = 0U; lane < 4U; ++lane) {
                            const uint32_t peer = group * 4U + (rank % 4U + lane) % 4U;
                            for (uint32_t index = 0U; index < reference.size(); ++index) {
                                if (reference[index].first != peer) continue;
                                expectedOwned[intra + owner].push_back(begin + index);
                                owner = (owner + 1U) % workers;
                            }
                        }
                    }
                    continue;
                }
                for (uint32_t phase = local ? 0U : 1U; phase < (local ? 1U : 4U); ++phase) {
                    const uint32_t group = partners[rank / 4][phase];
                    Require(partners[group][phase] == rank / 4);
                    const auto& groupRows = groupedRows[group];
                    uint32_t cursor = 0;
                    for (uint32_t logical = 0; logical < workers; ++logical) {
                        const uint32_t count = groupRows.size() / workers + (logical < groupRows.size() % workers);
                        const uint32_t owner = (local ? 0U : intra) + (logical + prefix) % workers;
                        for (uint32_t j = 0; j < count; ++j) expectedOwned[owner].push_back(groupRows[cursor++]);
                    }
                    Require(cursor == groupRows.size());
                    prefix += groupRows.size();
                }
            }
        }
        for (uint32_t core = 0; core < 28; ++core) {
            const auto owned = SelectCommOwnedSegments(peers, rank, core, 28, intra, rotation, paired, alltoall);
            if (alltoall && core < intra) {
                const auto original = SelectCommOwnedSegments(peers, rank, core, 28, intra, rotation, paired);
                Require(original.count == owned.count && original.rows == owned.rows);
                for (uint32_t i = 0U; i < owned.count; ++i) {
                    Require(original.segments[i].peer == owned.segments[i].peer);
                    Require(original.segments[i].sourceBegin == owned.segments[i].sourceBegin);
                    Require(original.segments[i].rowBegin == owned.segments[i].rowBegin);
                    Require(original.segments[i].count == owned.segments[i].count);
                    Require(original.segments[i].stride == owned.segments[i].stride);
                }
            }
            uint32_t counted = 0;
            uint32_t lastPhase = 0;
            std::vector<uint32_t> actualOwned;
            for (uint32_t i = 0; i < owned.count; ++i) {
                const auto& segment = owned.segments[i];
                Require(segment.count > 0);
                Require(segment.stride == (alltoall && core >= intra ? 28U - intra : 1U));
                Require((segment.peer / 4 == rank / 4) == (core < intra));
                uint32_t phase = 0;
                if (alltoall) {
                    phase = (segment.peer / 4U + 4U - rank / 4U) % 4U;
                } else if (paired) {
                    while (partners[rank / 4][phase] != segment.peer / 4) ++phase;
                } else {
                    phase = (rank / 4 + 4 - segment.peer / 4) % 4;
                }
                Require(phase >= lastPhase);
                lastPhase = phase;
                phaseRows[phase][core] += segment.count;
                counted += segment.count;
                PublishGmm1TileReady(
                    syncLayout, params, 1U, 256, static_cast<int32_t>(segment.rowBegin),
                    static_cast<int32_t>(segment.rowBegin + (segment.count - 1U) * segment.stride + 1U),
                    segment.stride);
                for (uint32_t j = 0; j < segment.count; ++j) {
                    const uint32_t row = segment.rowBegin + j * segment.stride;
                    Require(row >= begin && row < end);
                    const uint32_t index = row - begin;
                    actualOwned.push_back(row);
                    ++visits[index];
                    // This identity is also the inverse Combine return target:
                    // changing D/C ownership must not change source/top-k rows.
                    Require(reference[index] == Identity(segment.peer, segment.sourceBegin + j * segment.stride));
                }
            }
            Require(counted == owned.rows);
            if (paired || alltoall) Require(actualOwned == expectedOwned[core]);
        }
        for (uint32_t n : visits) Require(n == 1U);
        for (uint32_t phase = 0; phase < 4; ++phase) {
            auto first = phaseRows[phase].begin() + (phase == 0 ? 0 : intra);
            auto last = phaseRows[phase].begin() + (phase == 0 ? intra : 28);
            Require(*std::max_element(first, last) - *std::min_element(first, last) <= 1U);
        }
        // Equal visits imply each GMM1 tile receives exactly its row count,
        // including slices that start/end inside a tile and tiny/empty groups.
        for (uint32_t tile = begin / 256; tile <= end / 256; ++tile) {
            uint32_t ready = 0, expected = 0;
            for (uint32_t row = begin; row < end; ++row) {
                if (row / 256 == tile) {
                    ready += visits[row - begin];
                    ++expected;
                }
            }
            Require(ready == expected);
            Require(readyCounts[readyWords + tile * INT_CACHELINE] == static_cast<int32_t>(expected));
        }
        Require(std::all_of(readyCounts.begin(), readyCounts.begin() + readyWords,
                            [](int32_t count) { return count == 0; }));
    }
}

int main()
{
    uint32_t seed = 731U, cases = 0U, pairedCases = 0U;
    for (uint32_t scenario = 0; scenario < 12; ++scenario) {
        Counts counts;
        for (uint32_t peer = 0; peer < 16; ++peer) {
            seed = seed * 1664525U + 1013904223U;
            counts.values[peer] = scenario < 3    ? (42U << scenario) + (peer % 3 == 0)
                                  : scenario == 3 ? 0U
                                  : scenario == 4 ? (peer == 11 ? 93U : 0U)
                                                  : seed % (scenario * 19U);
        }
        uint32_t total = 0;
        for (auto count : counts.values) total += count;
        for (uint32_t rank = 0; rank < 16; ++rank) {
            for (uint32_t ranksPerNode : {0U, 4U, 8U, 16U}) {
                for (bool balanced : {false, true}) {
                    for (uint32_t part = 0; part < 5; ++part) {
                        uint32_t begin = total * part / 5;
                        uint32_t end = part == 4 ? total : total * (part + 1) / 5;
                        Check(counts, rank, ranksPerNode, begin, end, balanced,
                              static_cast<uint64_t>(cases) * 8192U + (cases % 59));
                        Check(counts, rank, ranksPerNode, begin, end, balanced,
                              static_cast<uint64_t>(cases) * 8192U + (cases % 59), true,
                              rank % 4 == 0 && ranksPerNode == 16 && balanced ? 0U : 12U);
                        Check(counts, rank, ranksPerNode, begin, end, balanced,
                              static_cast<uint64_t>(cases) * 8192U + (cases % 59), false,
                              rank % 4 == 0 && ranksPerNode == 16 && balanced ? 0U : 12U, true);
                        ++cases;
                        ++pairedCases;
                    }
                }
            }
        }
    }
    std::cout << "comm_split: " << cases << " independent route/ready/inverse cases, all 27 splits passed\n";
    uint32_t balancedCases = 0U;
    for (uint32_t bs : {2048U, 4096U, 8192U}) {
        for (uint32_t rank = 0; rank < 16; ++rank) {
            uint32_t usedGroups = 0U, wave = 0U;
            uint64_t globalRow = 0U;
            for (uint32_t expert = 0; expert < 24; ++expert) {
                Counts counts;
                uint32_t total = 0U;
                for (uint32_t source = 0; source < 16; ++source) {
                    const uint32_t sourceLength = bs * 8;
                    const uint32_t first = (expert * 16 + rank + 384 - (source * sourceLength) % 384) % 384;
                    counts.values[source] = (sourceLength - first + 383) / 384;
                    total += counts.values[source];
                }
                for (uint32_t begin = 0; begin < total;) {
                    if (usedGroups == 12U) {
                        usedGroups = 0;
                        ++wave;
                    }
                    const uint32_t count = std::min(total - begin, (12U - usedGroups) * 256U);
                    Check(counts, rank, 16U, begin, begin + count, true, globalRow + wave, true, 0U);
                    Check(counts, rank, 16U, begin, begin + count, true, globalRow + wave, false, 0U, true);
                    begin += count;
                    globalRow += count;
                    usedGroups += (count + 255U) / 256U;
                    ++balancedCases;
                }
            }
            Require(globalRow == static_cast<uint64_t>(bs) * 8U);
        }
    }
    std::cout << "paired-xor: " << pairedCases << " varied cases and " << balancedCases
              << " BS2048/4096/8192 EP16 slices, exact order/owners/ready/inverse passed\n";
    std::cout << "paired-xor: all 27 splits checked for each balanced EP16 slice\n";
    std::cout << "alltoall: rank-relative peer order, strided source/output rows, unchanged INTRA, exact ready counts; "
                 "all 27 splits and BS2048/4096/8192 passed\n";
}
