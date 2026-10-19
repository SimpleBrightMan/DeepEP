// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#define __aicore__
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
                  bool balanced, uint64_t rotation, bool paired = false)
{
    const auto reference = Reference(counts, rank, ranksPerNode, begin, end, balanced);
    CommPeerSegment peers[16];
    BuildCommPeerSegments(counts, rank, ranksPerNode, begin, end, balanced, peers);
    // Explicit matchings provide an oracle independent of the device XOR.
    constexpr uint32_t partners[4][4] = {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}};
    for (uint32_t intra = paired ? 12U : 1U; intra < (paired ? 13U : 28U); ++intra) {
        std::vector<uint32_t> visits(end - begin, 0U);
        std::array<std::array<uint32_t, 28>, 4> phaseRows{};
        std::array<std::vector<uint32_t>, 28> expectedOwned;
        if (paired) {
            for (bool local : {true, false}) {
                const uint32_t workers = local ? intra : 28U - intra;
                uint64_t prefix = rotation;
                for (uint32_t phase = local ? 0U : 1U; phase < (local ? 1U : 4U); ++phase) {
                    const uint32_t group = partners[rank / 4][phase];
                    Require(partners[group][phase] == rank / 4);
                    std::vector<uint32_t> groupRows;
                    for (uint32_t peer = group * 4; peer < (group + 1) * 4; ++peer)
                        for (uint32_t index = 0; index < reference.size(); ++index)
                            if (reference[index].first == peer) groupRows.push_back(begin + index);
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
            const auto owned = SelectCommOwnedSegments(peers, rank, core, 28, intra, rotation, paired);
            uint32_t counted = 0;
            uint32_t lastPhase = 0;
            std::vector<uint32_t> actualOwned;
            for (uint32_t i = 0; i < owned.count; ++i) {
                const auto& segment = owned.segments[i];
                Require(segment.count > 0);
                Require((segment.peer / 4 == rank / 4) == (core < intra));
                uint32_t phase = 0;
                if (paired) {
                    while (partners[rank / 4][phase] != segment.peer / 4) ++phase;
                } else {
                    phase = (rank / 4 + 4 - segment.peer / 4) % 4;
                }
                Require(phase >= lastPhase);
                lastPhase = phase;
                phaseRows[phase][core] += segment.count;
                counted += segment.count;
                for (uint32_t j = 0; j < segment.count; ++j) {
                    Require(segment.rowBegin + j >= begin && segment.rowBegin + j < end);
                    const uint32_t index = segment.rowBegin + j - begin;
                    actualOwned.push_back(segment.rowBegin + j);
                    ++visits[index];
                    // This identity is also the inverse Combine return target:
                    // changing D/C ownership must not change source/top-k rows.
                    Require(reference[index] == Identity(segment.peer, segment.sourceBegin + j));
                }
            }
            Require(counted == owned.rows);
            if (paired) Require(actualOwned == expectedOwned[core]);
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
        }
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
                              static_cast<uint64_t>(cases) * 8192U + (cases % 59), true);
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
                    Check(counts, rank, 16U, begin, begin + count, true, globalRow + wave, true);
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
}
