// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "common/mega_moe_domain_rows.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace MegaMoeImpl;

namespace {

using CountsArray = std::array<uint32_t, 16>;

struct Counts {
    CountsArray values;
    uint32_t Get(uint32_t peer) const
    {
        return values[peer];
    }
};

struct Identity {
    uint32_t peer;
    uint32_t ordinal;
};

void Require(bool value, const char* message)
{
    if (!value) {
        throw std::runtime_error(message);
    }
}

uint32_t ReferencePeer(uint32_t rank, uint32_t ordinal, uint32_t ranksPerNode)
{
    if (ranksPerNode == 0U || ranksPerNode == 16U) {
        return (rank + ordinal) % 16U;
    }
    const uint32_t nodes = 16U / ranksPerNode;
    uint32_t offset = ordinal % nodes;
    if (offset == 0U) {
        offset = 1U;
    } else if (offset == 1U) {
        offset = 0U;
    }
    return ((rank / ranksPerNode + offset) % nodes) * ranksPerNode +
           (rank % ranksPerNode + ordinal / nodes) % ranksPerNode;
}

// Independent oracle: consume one item from every nonempty peer queue each
// round. Dispatch then groups the selected slice items in the original peer order.
std::vector<Identity> ReferenceRows(const Counts& counts, uint32_t rank, uint32_t ranksPerNode, uint32_t begin,
                                    uint32_t end)
{
    CountsArray consumed{}, before{}, after{};
    uint32_t position = 0U;
    while (position < end) {
        bool progressed = false;
        for (uint32_t ordinal = 0U; ordinal < 16U && position < end; ++ordinal) {
            const uint32_t peer = ReferencePeer(rank, ordinal, ranksPerNode);
            if (consumed[peer] == counts.Get(peer)) {
                continue;
            }
            if (position < begin) {
                ++before[peer];
            }
            ++consumed[peer];
            ++after[peer];
            ++position;
            progressed = true;
        }
        Require(progressed, "oracle slice exceeds input");
    }
    std::vector<Identity> result;
    for (uint32_t ordinal = 0U; ordinal < 16U; ++ordinal) {
        const uint32_t peer = ReferencePeer(rank, ordinal, ranksPerNode);
        for (uint32_t source = before[peer]; source < after[peer]; ++source) {
            result.push_back({peer, source});
        }
    }
    Require(result.size() == end - begin, "oracle length");
    return result;
}

uint64_t cases = 0U;

void Verify(const Counts& counts, uint32_t rank, uint32_t ranksPerNode, uint32_t begin, uint32_t end, uint32_t workers,
            uint64_t globalPrefix, uint64_t wave)
{
    const auto expected = ReferenceRows(counts, rank, ranksPerNode, begin, end);
    DomainDispatchPeerOrder order{rank, 16U, ranksPerNode};
    for (uint32_t ordinal = 0U; ordinal < 16U; ++ordinal) {
        Require(order.Get(ordinal) == ReferencePeer(rank, ordinal, ranksPerNode), "peer order");
    }
    std::vector<uint32_t> seen(expected.size(), 0U), totals(workers, 0U);
    std::array<std::vector<uint32_t>, 4> domainTotals;
    for (auto& totalsPerDomain : domainTotals) {
        totalsPerDomain.resize(workers, 0U);
    }
    for (uint32_t physical = 0U; physical < workers; ++physical) {
        const uint32_t shift = static_cast<uint32_t>(wave % workers);
        const uint32_t worker = physical >= shift ? physical - shift : physical + workers - shift;
        std::vector<uint32_t> canonical;
        for (uint32_t visit = 0U; visit < 4U; ++visit) {
            DomainRowPlan plan{};
            Require(BuildDomainRowPlan(counts, order, 16U, begin, end, worker, workers, globalPrefix, visit, plan),
                    "valid plan rejected");
            Require(plan.segmentCount <= 16U, "segment capacity");
            std::vector<uint32_t> owned;
            uint32_t rows = 0U;
            for (uint32_t segmentIndex = 0U; segmentIndex < plan.segmentCount; ++segmentIndex) {
                const auto& segment = plan.segments[segmentIndex];
                Require(segment.rowCount > 0U, "empty segment");
                Require(segment.expertRowBegin >= begin && segment.expertRowBegin + segment.rowCount <= end,
                        "destination bounds");
                Require(segment.sourceBegin + segment.rowCount <= counts.Get(segment.peer), "source bounds");
                for (uint32_t row = 0U; row < segment.rowCount; ++row) {
                    const uint32_t relative = segment.expertRowBegin + row - begin;
                    Require(expected[relative].peer == segment.peer, "peer identity changed");
                    Require(expected[relative].ordinal == segment.sourceBegin + row, "source ordinal changed");
                    owned.push_back(relative);
                    if (visit == 0U) {
                        ++seen[relative];
                        ++domainTotals[segment.peer / 4U][physical];
                    }
                }
                rows += segment.rowCount;
            }
            Require(rows == plan.totalRows, "plan row count");
            std::sort(owned.begin(), owned.end());
            Require(std::adjacent_find(owned.begin(), owned.end()) == owned.end(), "duplicate owner row");
            if (visit == 0U) {
                canonical = owned;
                totals[physical] = rows;
            } else {
                Require(owned == canonical, "visit order changed ownership");
            }
        }
    }
    Require(std::all_of(seen.begin(), seen.end(), [](uint32_t n) { return n == 1U; }), "row lost or sent twice");
    Require(*std::max_element(totals.begin(), totals.end()) - *std::min_element(totals.begin(), totals.end()) <= 1U,
            "total load imbalance");
    for (const auto& totalsPerDomain : domainTotals) {
        Require(*std::max_element(totalsPerDomain.begin(), totalsPerDomain.end()) -
                        *std::min_element(totalsPerDomain.begin(), totalsPerDomain.end()) <=
                    1U,
                "domain load imbalance");
    }
    ++cases;
}

void InvalidAndPolicy()
{
    Require(UseDomainRowCombine(16U, 28U, true, 16U), "target policy disabled");
    Require(!UseDomainRowCombine(16U, 32U, true, 16U), "32-core policy");
    Require(!UseDomainRowCombine(4U, 28U, true, 4U), "EP4 policy");
    Require(!UseDomainRowCombine(16U, 28U, false, 16U), "decode policy");
    Require(!UseDomainRowCombine(16U, 28U, true, 0U), "unspecified topology policy");
    Require(!UseDomainRowCombine(16U, 28U, true, 4U), "different peer policy");
    Counts counts{};
    counts.values.fill(64U);
    DomainDispatchPeerOrder order{0U, 16U, 16U};
    DomainRowPlan plan{};
    Require(!BuildDomainRowPlan(counts, order, 8U, 0U, 1U, 0U, 28U, 0U, 0U, plan), "invalid world accepted");
    Require(!BuildDomainRowPlan(counts, order, 16U, 2U, 1U, 0U, 28U, 0U, 0U, plan), "reversed slice");
    Require(!BuildDomainRowPlan(counts, order, 16U, 0U, 1025U, 0U, 28U, 0U, 0U, plan), "oversized slice");
    Require(!BuildDomainRowPlan(counts, order, 16U, 0U, 1U, 0U, 0U, 0U, 0U, plan), "zero workers");
    Require(!BuildDomainRowPlan(counts, order, 16U, 0U, 1U, 28U, 28U, 0U, 0U, plan), "invalid worker");
    struct DuplicateOrder {
        uint32_t Get(uint32_t) const
        {
            return 0U;
        }
    } duplicate;
    Require(!BuildDomainRowPlan(counts, duplicate, 16U, 0U, 16U, 0U, 28U, 0U, 0U, plan), "duplicate peer");
}

}  // namespace

int main()
{
    try {
        InvalidAndPolicy();
        Counts balanced{};
        balanced.values.fill(64U);
        for (uint32_t rank = 0U; rank < 16U; ++rank) {
            for (uint32_t topology : {0U, 4U, 8U, 16U}) {
                for (uint64_t wave = 0U; wave < 56U; ++wave) {
                    Verify(balanced, rank, topology, 768U, 1024U, 28U, 23U * 1024U + 768U, wave);
                }
                for (uint32_t boundary : {0U, 1U, 15U, 16U, 27U, 28U, 255U, 256U, 257U, 511U, 512U, 1023U, 1024U}) {
                    Verify(balanced, rank, topology, 0U, boundary, 28U, 0U, 0U);
                    Verify(balanced, rank, topology, boundary, 1024U, 32U, 1024U + boundary, 100000003U);
                }
            }
        }
        std::mt19937 random(20261014U);
        for (uint32_t iteration = 0U; iteration < 1000U; ++iteration) {
            Counts skew{};
            uint32_t total = 0U;
            for (uint32_t peer = 0U; peer < 16U; ++peer) {
                skew.values[peer] = iteration % 7U == 0U ? (peer == iteration % 16U ? 1024U : 0U) : random() % 65U;
                total += skew.values[peer];
            }
            const uint32_t begin = random() % (total + 1U);
            const uint32_t end = begin + random() % (total - begin + 1U);
            Verify(skew, iteration % 16U, iteration % 2U == 0U ? 16U : 4U, begin, end, iteration % 3U == 0U ? 1U : 28U,
                   UINT64_MAX - 31U, random());
        }
        Counts empty{};
        Verify(empty, 0U, 16U, 0U, 0U, 28U, 0U, 0U);
        std::cout << "domain_rows: " << cases << " independent-oracle cases passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "domain_rows: " << error.what() << '\n';
        return 1;
    }
}
