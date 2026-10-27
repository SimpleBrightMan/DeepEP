// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "peer_reorder.hpp"

namespace MegaMoeUrma {

constexpr uint32_t kPeerReorderBitmapWords = (kMaxPeerReorderSpans + 31U) / 32U;
constexpr uint32_t kPeerReorderBitmapBytes = kPeerReorderBitmapWords * sizeof(uint32_t);
constexpr uint32_t kReorderCvSubmitted = UINT32_MAX;
constexpr uint32_t kCombinePutRows = 128U;

__aicore__ inline PeerReorderRange LoadPeerReorderRange(const Arguments& args, uint32_t chunkIndex)
{
    const auto* gm = reinterpret_cast<const __gm__ PeerReorderRange*>(args.peerReorderRanges) + chunkIndex;
    return {gm->spanBegin, gm->spanCount};
}

// Each peer owns a contiguous slice of the host span table. Self owners have
// separate bitmaps and only touch their statically assigned CVs in that slice.
__aicore__ inline PeerReorderRange PeerReorderSpanRange(const Arguments& args, uint32_t begin, uint32_t end)
{
    if (begin == end) return {0U, 0U};
    const auto first = LoadPeerReorderRange(args, begin);
    const auto last = LoadPeerReorderRange(args, end - 1U);
    return {first.spanBegin, last.spanBegin + last.spanCount - first.spanBegin};
}

__aicore__ inline void ClearReorderBitmap(AscendC::LocalTensor<uint32_t> copied, uint32_t spans)
{
    for (uint32_t word = 0U; word < (spans + 31U) / 32U; ++word) copied.SetValue(word, 0U);
}

// The host plan partitions a CV in destination-row order. Bits are consumed
// only after ReorderReadyPeerSpans has drained its MTE3 writes. A count of
// ready rows is insufficient: an unready hole must never enter a payload PUT.
__aicore__ inline uint32_t ReorderedPeerPrefixRows(
    const Arguments& args, const PeerReorderRange& range, uint32_t peerSpanBegin,
    AscendC::LocalTensor<uint32_t> copied)
{
    uint32_t rows = 0U;
    for (uint32_t j = 0U; j < range.spanCount; ++j) {
        const uint32_t spanIndex = range.spanBegin + j;
        const uint32_t bitIndex = spanIndex - peerSpanBegin;
        if ((copied.GetValue(bitIndex / 32U) & (1U << (bitIndex % 32U))) == 0U) break;
        const auto* span = reinterpret_cast<const __gm__ PeerReorderSpan*>(args.peerReorderSpans) + spanIndex;
        rows += span->rows;
    }
    return rows;
}

// Single-writer UB bits skip spans already copied in earlier scans. Unready
// producer groups never block another span/CV. All groups have M<=256, so a
// full-width span needs ceil(H/256) completed N tiles regardless of its rows.
// Return the number of NEW spans whose local GM writes have completed.
__aicore__ inline uint32_t ReorderReadyPeerSpans(
    const Arguments& args, const PeerReorderRange& range, const Chunk& chunk, uint32_t peerSpanBegin,
    AscendC::LocalTensor<uint32_t> copied, Timer& timer, bool& waiting, uint64_t waitStart,
    const TraceMeta& waitMeta)
{
    using namespace AscendC;
    const int32_t required = static_cast<int32_t>(ComputeTileCount(kComputeTileM, args.config.hidden));
    PeerReorderPipeline pipeline;
    uint32_t copiedSpans = 0U;
    uint32_t copiedRows = 0U;
    uint64_t start = 0U;
    uint64_t onlyFragment = kTraceUnknown;
    for (uint32_t j = 0U; j < range.spanCount; ++j) {
        const uint32_t spanIndex = range.spanBegin + j;
        const uint32_t bitIndex = spanIndex - peerSpanBegin;
        const uint32_t mask = 1U << (bitIndex % 32U);
        const uint32_t word = copied.GetValue(bitIndex / 32U);
        if ((word & mask) != 0U) continue;
        const auto* gm = reinterpret_cast<const __gm__ PeerReorderSpan*>(args.peerReorderSpans) + spanIndex;
        const uint32_t group = gm->fragmentIndex;
        auto* done = reinterpret_cast<__gm__ int32_t*>(args.gmm2Done + static_cast<uint64_t>(group) * kFlagBytes);
        if (ReadGmByPassDCache(done) < required) continue;
        if (copiedSpans == 0U) {
            if (waiting) {
                timer.TraceEnd(TraceEvent::WaitCombineReady, waitStart, waitMeta);
                waiting = false;
            }
            start = timer.TraceBegin();
            pipeline.Init(args, chunk);
            onlyFragment = group;
        } else if (onlyFragment != group) {
            onlyFragment = kTraceUnknown;
        }
        const PeerReorderSpan span{gm->sourceRow, gm->destinationRow, group, gm->rows};
        pipeline.Copy(span);
        copied.SetValue(bitIndex / 32U, word | mask);
        copiedRows += span.rows;
        ++copiedSpans;
    }
    if (copiedSpans != 0U) {
        // One drain per ready batch, never per span. No PUT or self ready can
        // be published until these stores and all previous batches completed.
        pipeline.Finish();
        TraceMeta meta;
        meta.peer = chunk.peer;
        meta.cv = chunk.cv;
        meta.fragment = onlyFragment;
        // A batch may contain nonadjacent rows; do not claim a contiguous row.
        meta.rows = copiedRows;
        meta.columns = args.config.hidden;
        meta.bytes = static_cast<uint64_t>(copiedRows) * args.config.hidden * sizeof(bfloat16_t);
        timer.TraceEnd(TraceEvent::Gmm2Reorder, start, meta);
    }
    return copiedSpans;
}
}  // namespace MegaMoeUrma
