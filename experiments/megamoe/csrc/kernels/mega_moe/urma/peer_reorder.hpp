// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"
#include "compute_contract.hpp"
#include "timer.hpp"

namespace MegaMoeUrma {

// Full rows, not N tiles. Two 64KiB slots overlap MTE2/MTE3 across spans.
// All producer groups must be complete before entry. Source GM remains
// immutable throughout the invocation; output rows are written exactly once.
__aicore__ inline void ReorderPeerChunk(const Arguments& args, uint32_t chunkIndex, const Chunk& chunk, Timer& timer)
{
    using namespace AscendC;
    constexpr uint32_t slotBytes = 64U * 1024U;
    static_assert(2U * slotBytes <= ActivationUbContract::inputEnd);
    const uint32_t rowBytes = args.config.hidden * sizeof(bfloat16_t);
    const uint32_t rowsPerSlot = slotBytes / rowBytes;
    const auto* rangeGm = reinterpret_cast<const __gm__ PeerReorderRange*>(args.peerReorderRanges) + chunkIndex;
    const PeerReorderRange range{rangeGm->spanBegin, rangeGm->spanCount};
    GlobalTensor<uint8_t> input, output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.gmm2Contiguous));
    input.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
        chunk.peer == args.config.rank ? args.symmetric + args.config.returnYOffset : args.gmm2Output));
    const uint64_t start = timer.TraceBegin();
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    uint32_t iteration = 0U;
    for (uint32_t j = 0U; j < range.spanCount; ++j) {
        const auto* gm = reinterpret_cast<const __gm__ PeerReorderSpan*>(args.peerReorderSpans) + range.spanBegin + j;
        const PeerReorderSpan span{gm->sourceRow, gm->destinationRow, gm->fragmentIndex, gm->rows};
        for (uint32_t row = 0U; row < span.rows; row += rowsPerSlot) {
            const uint32_t rows = span.rows - row < rowsPerSlot ? span.rows - row : rowsPerSlot;
            const uint32_t slot = iteration++ % 2U;
            const auto event = static_cast<TEventID>(slot);
            LocalTensor<uint8_t> buffer(TPosition::VECCALC, slot * slotBytes, slotBytes);
            WaitFlag<HardEvent::MTE3_MTE2>(event);
            DataCopyPad(buffer, input[(span.sourceRow + row) * rowBytes], {1U, rows * rowBytes, 0U, 0U, 0U},
                        {false, 0U, 0U, 0U});
            SetFlag<HardEvent::MTE2_MTE3>(event);
            WaitFlag<HardEvent::MTE2_MTE3>(event);
            DataCopyPad(output[(span.destinationRow + row) * rowBytes], buffer, {1U, rows * rowBytes, 0U, 0U, 0U});
            SetFlag<HardEvent::MTE3_MTE2>(event);
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    // These MTE2 waits consume the last MTE3 credits after every store.
    // Fence their consumption before re-seeding IDs on the next invocation;
    // this also guarantees output GM is complete before its URMA WQE.
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    TraceMeta meta;
    meta.peer = chunk.peer;
    meta.cv = chunk.cv;
    meta.row = chunk.localRow;
    meta.rows = chunk.rows;
    meta.columns = args.config.hidden;
    meta.bytes = static_cast<uint64_t>(chunk.rows) * rowBytes;
    timer.TraceEnd(TraceEvent::Gmm2Reorder, start, meta);
}
}  // namespace MegaMoeUrma
