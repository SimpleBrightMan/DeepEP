// Production Dispatch ownership/planning bodies, with transfer leaves recorded on CPU.
// Legacy functions below are frozen from fa7fbc8; no Git history is needed to run this test.
#define __aicore__
#include "common/mega_moe_peer_rows.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
using namespace MegaMoeImpl;

constexpr int AIC = 0, AIV = 1, g_coreType = TEST_CORE;
uint32_t subBlock = 1;
uint32_t GetSubBlockIdx() { return subBlock; }
using GM_ADDR = uint8_t*;
template <typename T> struct LocalTensor {
    const std::vector<T>* data;
    T GetValue(uint32_t index) const { return data->at(index); }
};
struct MoeStageCommonConfig { uint32_t worldSize, rankId, moeExpertPerRank; };
struct BlockJobContext { uint32_t jobIndex, totalJobs; };
struct TokenDispatchConfig { uint64_t maxOutputSize; uint32_t ranksPerNode; };
struct MoeSyncWorkspaceLayout {};
struct Params {};
template <typename T> struct TokenDispatchScratch {
    LocalTensor<int32_t> cumsumInfoTensor;
    PeerRowLimitsCache peerRowLimitsCache{};
};
struct DispatchRouteBatch {};
struct DispatchCopyProgress {};
using Event = std::array<uint64_t, 6>;
std::vector<Event> events;
#include "dispatch_plan_types.inc"

template <typename A, typename S, bool Prefetch>
int32_t AppendPeerRouteSegment(const TokenDispatchConfig&, const MoeStageCommonConfig&, const Params&, GM_ADDR*,
                               TokenDispatchScratch<A>&, uint32_t e, const PeerTransfer& t,
                               DispatchRouteBatch&, DispatchCopyProgress&) {
    events.push_back({0, e, t.sourceRank, t.sourceOrdinalBegin, t.destinationRow, t.rowCount});
    return t.rowCount;
}
template <typename A, typename S, bool Prefetch>
int32_t DispatchRankTokens(const TokenDispatchConfig& c, const MoeStageCommonConfig& common, const Params& p, GM_ADDR* w,
                          TokenDispatchScratch<A>& scratch, uint32_t e, int32_t expertBase, uint32_t peer,
                          int32_t destination, int32_t begin, int32_t end,
                          DispatchRouteBatch& batch, DispatchCopyProgress& progress) {
    const PeerTransfer t{peer, static_cast<uint32_t>(begin), static_cast<uint32_t>(end - begin),
                         static_cast<uint32_t>(expertBase + destination)};
    return AppendPeerRouteSegment<A, S, Prefetch>(c, common, p, w, scratch, e, t, batch, progress);
}
template <typename A, typename S, bool P>
void ConsumeRouteBatch(const TokenDispatchConfig&, const Params&, GM_ADDR*, TokenDispatchScratch<A>&,
                       DispatchRouteBatch&, DispatchCopyProgress&) {
    events.push_back({1, 0, 0, 0, 0, 0});
}
template <typename A, typename S, bool P>
void FlushDispatchRouteBatch(const TokenDispatchConfig& c, const Params& p, GM_ADDR* w, TokenDispatchScratch<A>& scratch,
                             DispatchRouteBatch& batch, DispatchCopyProgress& progress) {
    ConsumeRouteBatch<A, S, P>(c, p, w, scratch, batch, progress);
}
template <typename A, typename S, bool P>
void DrainDispatchCopyBuffers(const TokenDispatchConfig&, const Params&, TokenDispatchScratch<A>&,
                              const DispatchCopyProgress&) {
    events.push_back({2, 0, 0, 0, 0, 0});
}
void PublishGmm1TileReady(const MoeSyncWorkspaceLayout&, const Params&, uint32_t e, int32_t tileM,
                         int32_t begin, int32_t end) {
    assert(begin <= end);
    events.push_back({3, e, static_cast<uint32_t>(tileM), static_cast<uint32_t>(begin),
                      static_cast<uint32_t>(end), 0});
}
#include "dispatch_plan_methods.inc"

template <typename ActivationType>
__aicore__ inline uint32_t LegacyFindDispatchSourceRank(const MoeStageCommonConfig& common,
                                                  const TokenDispatchScratch<ActivationType>& scratch,
                                                  uint32_t localExpertId, uint32_t globalRowIdx)
{
    uint32_t expertRankBegin = localExpertId * common.worldSize;
    uint32_t searchBegin = expertRankBegin;
    uint32_t searchEnd = expertRankBegin + common.worldSize;
    while (searchBegin < searchEnd) {
        uint32_t middle = searchBegin + (searchEnd - searchBegin) / 2U;
        uint32_t rankGlobalRowEnd = static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(middle));
        if (rankGlobalRowEnd <= globalRowIdx) {
            searchBegin = middle + 1U;
        } else {
            searchEnd = middle;
        }
    }
    return searchBegin - expertRankBegin;
}

template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void LegacyDispatchOwnedExpertRows(const TokenDispatchConfig& context, const MoeStageCommonConfig& common,
                                               const MoeSyncWorkspaceLayout& syncLayout, const Params& params,
                                               GM_ADDR* winRankAddr, TokenDispatchScratch<ActivationType>& scratch,
                                               uint32_t localExpertId, const ExpertDispatchCoreRange& coreRange)
{
    constexpr int32_t gmm1TileRowCount = static_cast<int32_t>(PipelineTileM);
    uint32_t coreGlobalRowBegin = coreRange.expertGlobalRowBegin + coreRange.localExpertRowBegin;
    uint32_t coreGlobalRowEnd = coreRange.expertGlobalRowBegin + coreRange.localExpertRowEnd;
    // prefix 单调递增，二分找到与本核首行相交的第一个 source rank，避免从 rank 0 线性扫描。
    uint32_t sourceRankIdx = LegacyFindDispatchSourceRank(common, scratch, localExpertId, coreGlobalRowBegin);
    int32_t dispatchedRowCount = 0;
    DispatchRouteBatch batch{};
    DispatchCopyProgress progress{};

    while (sourceRankIdx < common.worldSize) {
        uint32_t prefixIndex = localExpertId * common.worldSize + sourceRankIdx;
        uint32_t rankGlobalRowEnd = static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(prefixIndex));
        uint32_t rankGlobalRowBegin =
            prefixIndex == 0U ? 0U : static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(prefixIndex - 1U));
        if (rankGlobalRowBegin >= coreGlobalRowEnd) {
            break;
        }

        uint32_t overlapGlobalRowBegin =
            coreGlobalRowBegin > rankGlobalRowBegin ? coreGlobalRowBegin : rankGlobalRowBegin;
        uint32_t overlapGlobalRowEnd = coreGlobalRowEnd < rankGlobalRowEnd ? coreGlobalRowEnd : rankGlobalRowEnd;
        if (overlapGlobalRowBegin < overlapGlobalRowEnd) {
            int32_t rankDispatchedRowCount = DispatchRankTokens<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                context, common, params, winRankAddr, scratch, localExpertId,
                static_cast<int32_t>(coreRange.expertGlobalRowBegin), sourceRankIdx,
                static_cast<int32_t>(overlapGlobalRowBegin - coreRange.expertGlobalRowBegin),
                static_cast<int32_t>(overlapGlobalRowBegin - rankGlobalRowBegin),
                static_cast<int32_t>(overlapGlobalRowEnd - rankGlobalRowBegin), batch, progress);
            dispatchedRowCount += rankDispatchedRowCount;
        }
        ++sourceRankIdx;
    }
    FlushDispatchRouteBatch<ActivationType, QuantScaleType, TopkWeightsPrefetch>(context, params, winRankAddr, scratch,
                                                                                 batch, progress);
    DrainDispatchCopyBuffers<ActivationType, QuantScaleType, TopkWeightsPrefetch>(context, params, scratch, progress);
    // 所有 source-rank 子段完成后统一发布本核贡献；GMM1 按 tile 累计到完整行数后即可启动。
    PublishGmm1TileReady(syncLayout, params, localExpertId, gmm1TileRowCount,
                         static_cast<int32_t>(coreRange.localExpertRowBegin),
                         static_cast<int32_t>(coreRange.localExpertRowBegin) + dispatchedRowCount);
}

template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void LegacyDispatchPeerBalancedRows(const TokenDispatchConfig& context, const MoeStageCommonConfig& common,
                                                const MoeSyncWorkspaceLayout& syncLayout, const Params& params,
                                                GM_ADDR* winRankAddr, TokenDispatchScratch<ActivationType>& scratch,
                                                uint32_t expertIdx, const ExpertDispatchCoreRange& coreRange,
                                                uint32_t sliceBegin, uint32_t sliceEnd)
{
    DispatchRouteBatch batch{};
    DispatchCopyProgress progress{};
    DispatchPeerCounts counts{scratch.cumsumInfoTensor, expertIdx * common.worldSize};
#if MEGAMOE_PREFILL_PEER_FAST_PATH
    const PeerRowLimits& limits =
        GetCachedPeerRowLimits(scratch.peerRowLimitsCache, counts, expertIdx, common.worldSize);
    bool regular = CanUseRegularPeerSlice(limits, common.worldSize, sliceBegin, sliceEnd);
#else
    PeerRowLimits limits = GetPeerRowLimits(counts, common.worldSize);
    bool regular = false;
#endif
    PeerRowCut beginCut{}, endCut{};
    RegularPeerSlice regularSlice{};
    uint32_t firstOrdinal = 0U;
    if (regular) {
        regularSlice = CreateRegularPeerSlice(common.worldSize, sliceBegin, sliceEnd);
        firstOrdinal =
            FindRegularPeerOrdinal(regularSlice, common.worldSize, coreRange.localExpertRowBegin - sliceBegin);
    } else {
        // 真实 peer 已耗尽：保留原配额算法及余数消耗顺序，不能套用均匀配额假设。
        beginCut = GetPeerRowCut(counts, common.worldSize, limits, sliceBegin);
        endCut = GetPeerRowCut(counts, common.worldSize, limits, sliceEnd);
    }
    uint32_t destination = sliceBegin + (regular ? regularSlice.Prefix(firstOrdinal) : 0U);
    for (uint32_t ordinal = firstOrdinal; ordinal < common.worldSize; ++ordinal) {
        uint32_t peer = GetDispatchPeer(ordinal, common.rankId, common.worldSize, context.ranksPerNode);
        uint32_t sourceBegin, sourceEnd;
        if (regular) {
            // 直接恢复本 peer 的源区间，不再读取无关 peer 的 count 或从 ordinal=0 扫描。
            sourceBegin = regularSlice.SourceBegin(ordinal);
            sourceEnd = regularSlice.SourceEnd(ordinal);
        } else {
            uint32_t count = counts.Get(peer);
            sourceBegin = TakePeerRowPrefix(count, beginCut);
            sourceEnd = TakePeerRowPrefix(count, endCut);
        }
        uint32_t destinationEnd = destination + sourceEnd - sourceBegin;
        uint32_t overlapBegin =
            destination > coreRange.localExpertRowBegin ? destination : coreRange.localExpertRowBegin;
        uint32_t overlapEnd =
            destinationEnd < coreRange.localExpertRowEnd ? destinationEnd : coreRange.localExpertRowEnd;
        if (overlapBegin < overlapEnd) {
            uint32_t first = sourceBegin + overlapBegin - destination;
            DispatchRankTokens<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                context, common, params, winRankAddr, scratch, expertIdx,
                static_cast<int32_t>(coreRange.expertGlobalRowBegin), peer, static_cast<int32_t>(overlapBegin),
                static_cast<int32_t>(first), static_cast<int32_t>(first + overlapEnd - overlapBegin), batch, progress);
        }
        destination = destinationEnd;
        if (destination >= coreRange.localExpertRowEnd) {
            break;
        }
    }
    FlushDispatchRouteBatch<ActivationType, QuantScaleType, TopkWeightsPrefetch>(context, params, winRankAddr, scratch,
                                                                                 batch, progress);
    DrainDispatchCopyBuffers<ActivationType, QuantScaleType, TopkWeightsPrefetch>(context, params, scratch, progress);
    // 每个 owner 仍独占连续目标行；本核所有 peer 段落地后按目标行发布一次贡献。
    PublishGmm1TileReady(syncLayout, params, expertIdx, static_cast<int32_t>(PipelineTileM),
                         static_cast<int32_t>(coreRange.localExpertRowBegin),
                         static_cast<int32_t>(coreRange.localExpertRowEnd));
}

template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch,
          bool PeerBalanced = false>
__aicore__ inline void LegacyDispatchTokenRange(const TokenDispatchConfig& context, const MoeStageCommonConfig& common,
                                          const BlockJobContext& blockJob, const MoeSyncWorkspaceLayout& syncLayout,
                                          const Params& params, GM_ADDR* winRankAddr,
                                          TokenDispatchScratch<ActivationType>& scratch, const ExpertTokenRange& range)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || blockJob.totalJobs == 0U) {
        return;
    }

    // maxOutputSize 是 Dispatch workspace 的硬边界，先统一裁剪再做分核。
    uint32_t dispatchGlobalRowBegin = static_cast<uint32_t>(
        range.begin.globalTokenIndex < context.maxOutputSize ? range.begin.globalTokenIndex : context.maxOutputSize);
    uint32_t dispatchGlobalRowEnd = static_cast<uint32_t>(
        range.end.globalTokenIndex < context.maxOutputSize ? range.end.globalTokenIndex : context.maxOutputSize);
    if (dispatchGlobalRowEnd <= dispatchGlobalRowBegin) {
        return;
    }

    uint32_t dispatchRowCount = dispatchGlobalRowEnd - dispatchGlobalRowBegin;
    // 对整个输入范围一次连续均分；按全局起点轮转首 owner，避免余数长期集中在低编号 AIV1。
    WorkRange ownedRange =
        GetRotatedBalancedTokenRange(dispatchRowCount, blockJob.jobIndex, blockJob.totalJobs, dispatchGlobalRowBegin);
    if (ownedRange.count == 0U) {
        return;
    }
    uint32_t coreGlobalRowBegin = dispatchGlobalRowBegin + ownedRange.start;
    uint32_t coreGlobalRowEnd = coreGlobalRowBegin + ownedRange.count;

    // end 位于专家内部时需要包含该专家；恰好位于专家边界时 end.expertIdx 已指向下一专家。
    uint32_t lastExpertExclusive = range.end.expertIdx + (range.end.tokenIndexInExpert == 0U ? 0U : 1U);
    if (lastExpertExclusive > common.moeExpertPerRank) {
        lastExpertExclusive = common.moeExpertPerRank;
    }
    for (uint32_t expertIdx = range.begin.expertIdx; expertIdx < lastExpertExclusive; ++expertIdx) {
        uint32_t expertGlobalRowBegin =
            expertIdx == 0U
                ? 0U
                : static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(expertIdx * common.worldSize - 1U));
        uint32_t expertGlobalRowEnd =
            static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue((expertIdx + 1U) * common.worldSize - 1U));
        if (expertGlobalRowEnd <= coreGlobalRowBegin) {
            continue;
        }
        if (expertGlobalRowBegin >= coreGlobalRowEnd) {
            break;
        }
        uint32_t overlapGlobalRowBegin =
            coreGlobalRowBegin > expertGlobalRowBegin ? coreGlobalRowBegin : expertGlobalRowBegin;
        uint32_t overlapGlobalRowEnd = coreGlobalRowEnd < expertGlobalRowEnd ? coreGlobalRowEnd : expertGlobalRowEnd;
        if (overlapGlobalRowBegin < overlapGlobalRowEnd) {
            // 将本核的全局连续区间投影为当前专家内的局部 row 区间。
            ExpertDispatchCoreRange expertRange{expertGlobalRowBegin, overlapGlobalRowBegin - expertGlobalRowBegin,
                                                overlapGlobalRowEnd - expertGlobalRowBegin};
            if constexpr (PeerBalanced) {
                uint32_t sliceBegin =
                    dispatchGlobalRowBegin > expertGlobalRowBegin ? dispatchGlobalRowBegin - expertGlobalRowBegin : 0U;
                uint32_t sliceEnd = dispatchGlobalRowEnd < expertGlobalRowEnd
                                        ? dispatchGlobalRowEnd - expertGlobalRowBegin
                                        : expertGlobalRowEnd - expertGlobalRowBegin;
                LegacyDispatchPeerBalancedRows<ActivationType, QuantScaleType, PipelineTileM, TopkWeightsPrefetch>(
                    context, common, syncLayout, params, winRankAddr, scratch, expertIdx, expertRange, sliceBegin,
                    sliceEnd);
            } else {
                LegacyDispatchOwnedExpertRows<ActivationType, QuantScaleType, PipelineTileM, TopkWeightsPrefetch>(
                    context, common, syncLayout, params, winRankAddr, scratch, expertIdx, expertRange);
            }
        }
    }
}



ExpertTokenPosition Locate(const std::vector<std::vector<uint32_t>>& counts, uint32_t row) {
    uint32_t base = 0;
    for (uint32_t e = 0; e < counts.size(); ++e) {
        uint32_t total = 0;
        for (auto n : counts[e]) total += n;
        if (row < base + total) return {e, row - base, row};
        base += total;
    }
    assert(row == base);
    return {static_cast<uint32_t>(counts.size()), 0, row};
}

template <bool Balanced>
void CheckRange(const std::vector<std::vector<uint32_t>>& counts, uint32_t rank, uint32_t ranksPerNode,
                uint32_t begin, uint32_t end, uint32_t capacity, uint32_t workers, bool single) {
    const uint32_t peers = counts.front().size();
    std::vector<int32_t> prefix;
    uint32_t total = 0;
    for (const auto& expert : counts) for (auto n : expert) { total += n; prefix.push_back(total); }
    const auto first = Locate(counts, begin);
    const auto last = Locate(counts, end);
    const ExpertTokenRange range{first, last};
    const TokenDispatchConfig config{capacity, ranksPerNode};
    const MoeStageCommonConfig common{peers, rank, static_cast<uint32_t>(counts.size())};
    const Params params;
    const MoeSyncWorkspaceLayout sync;
    std::vector<uint32_t> written(total, 0);
    std::vector<uint32_t> sourceBase(counts.size() * peers, 0);
    uint32_t sources = 0;
    for (uint32_t e = 0; e < counts.size(); ++e) for (uint32_t p = 0; p < peers; ++p) {
        sourceBase[e * peers + p] = sources;
        sources += counts[e][p];
    }
    std::vector<uint32_t> sourceSeen(sources, 0);

    // Also cover an out-of-range worker, which must receive no work.
    for (uint32_t worker = 0; worker <= workers; ++worker) {
        const BlockJobContext job{worker, workers};
        TokenDispatchScratch<int> scratch{{&prefix}}, oldScratch{{&prefix}};
        events.clear();
        LegacyDispatchTokenRange<int, int, 256, false, Balanced>(
            config, common, job, sync, params, nullptr, oldScratch, range);
        const auto expected = events;
        events.clear();
        if (single && begin < end) {
            uint32_t expertEnd = begin - first.tokenIndexInExpert;
            for (auto n : counts.at(first.expertIdx)) expertEnd += n;
            assert(end <= expertEnd);
            const ExpertSlice slice{first.expertIdx, first.tokenIndexInExpert, end - begin, begin};
            DispatchExpertSlice<int, int, 256, false, Balanced>(
                config, common, job, sync, params, nullptr, scratch, slice);
        } else {
            DispatchTokenRange<int, int, 256, false, Balanced>(
                config, common, job, sync, params, nullptr, scratch, range);
        }
        assert(events == expected);
        for (const auto& event : events) {
            if (event[0] != 0) continue;
            const auto e = event[1], peer = event[2], ordinal = event[3], dst = event[4], rows = event[5];
            assert(ordinal + rows <= counts.at(e).at(peer));
            for (uint32_t i = 0; i < rows; ++i) {
                assert(dst + i < capacity);
                ++written.at(dst + i);
                ++sourceSeen.at(sourceBase.at(e * peers + peer) + ordinal + i);
            }
        }
    }
    for (uint32_t row = 0; row < total; ++row) {
        const bool needed = g_coreType == AIV && subBlock == 1 && workers != 0 &&
                            row >= begin && row < end && row < capacity;
        assert(written[row] == static_cast<uint32_t>(needed));
    }
    for (auto n : sourceSeen) {
        assert(n <= 1); // A source route record cannot be copied twice within one range.
        if (g_coreType == AIV && subBlock == 1 && workers && begin == 0 && end == total && capacity >= total) assert(n == 1);
    }
}

void Check(const std::vector<std::vector<uint32_t>>& counts, uint32_t rank, uint32_t perNode, uint32_t workers) {
    uint32_t total = 0;
    for (const auto& expert : counts) for (auto n : expert) total += n;
    for (bool single : {false, true}) {
        // A prefix ending in the first nonempty expert is suitable for the direct slice entry.
        uint32_t end = total;
        if (single) {
            const auto first = Locate(counts, 0);
            if (first.expertIdx < counts.size()) {
                end = 0;
                for (auto n : counts[first.expertIdx]) end += n;
            }
        }
        CheckRange<false>(counts, rank, perNode, 0, end, total, workers, single);
        CheckRange<true>(counts, rank, perNode, 0, end, total, workers, single);
        CheckRange<true>(counts, rank, perNode, 0, end, end / 2, workers, single);
    }
}
int main() {
    const std::vector<std::vector<uint32_t>> ep64(6, std::vector<uint32_t>(64, 64));
    CheckRange<true>(ep64, 0, 8, 3072, 4096, 24576, 32, true);
    CheckRange<true>(ep64, 0, 8, 4096, 4608, 24576, 32, true);
    CheckRange<true>(ep64, 63, 8, 3072, 4608, 24576, 32, false);
    CheckRange<false>(ep64, 0, 0, 3072, 4096, 3500, 32, true);
    Check(ep64, 0, 8, 32);
    Check({{0, 0}, {0, 0}}, 0, 0, 32);
    Check({{0, 1, 0, 0}, {7, 0, 100, 1}, {0, 0, 0, 0}}, 3, 2, 32);
    Check({{1, 3}, {0, 9}}, 0, 0, 0);
    subBlock = 0;
    Check(ep64, 0, 8, 32);
    subBlock = 1;
    uint32_t seed = 20261016;
    for (uint32_t i = 0; i < 120; ++i) {
        const uint32_t peers = std::array<uint32_t, 6>{2, 8, 16, 32, 64, 128}[i % 6];
        std::vector<std::vector<uint32_t>> counts(1 + i % 4, std::vector<uint32_t>(peers));
        for (auto& expert : counts) for (auto& n : expert) {
            seed = seed * 1664525U + 1013904223U;
            n = seed % 5 == 0 ? 0 : seed % 31;
        }
        Check(counts, i % peers, i % 2 ? 0 : (peers <= 8 ? peers : 8), 1 + i % 32);
    }
}
