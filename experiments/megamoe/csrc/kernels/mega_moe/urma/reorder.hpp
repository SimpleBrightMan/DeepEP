// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "reorder_plan.hpp"
#include "compute_contract.hpp"
#include "timer.hpp"

namespace MegaMoeUrma {
static_assert(kReorderUbBytes == ActivationUbContract::inputEnd);

// One paired AIV1 owns a complete N tile. Two M128 slots overlap read and write
// without an S wait between rows/segments. Each call closes its event credits
// and drains stores before activation/gather reuse the same low 128KiB UB.
__aicore__ inline void ReorderGmm2Tile(const Arguments& args, const Fragment& fragment, uint32_t fragmentIndex,
                                       uint32_t tile, Timer& timer)
{
    using namespace AscendC;
    const uint32_t h = args.config.hidden;
    const uint32_t n = tile * kReorderColumns;  // M-group never exceeds M256.
    const uint32_t columns = h - n < kReorderColumns ? h - n : kReorderColumns;
    GlobalTensor<bfloat16_t> input, remote, self;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(args.gmm2Contiguous));
    remote.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(args.gmm2Output));
    self.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(args.symmetric + args.config.returnYOffset));
    const uint64_t start = timer.TraceBegin();
    // At most two slices, each with its own slot used exactly once. There
    // are no seeded free-slot events to leak across activation/Pack tasks.
    for (uint32_t m = 0U; m < fragment.rows; m += kReorderRows) {
        const uint32_t rows = fragment.rows - m < kReorderRows ? fragment.rows - m : kReorderRows;
        const uint32_t slot = m / kReorderRows;
        const auto event = static_cast<TEventID>(slot);
        LocalTensor<bfloat16_t> scratch(TPosition::VECCALC, slot * kReorderSlotBytes, kReorderRows * kReorderColumns);
        DataCopyPad(scratch, input[(fragment.rowBegin + m) * h + n],
                    {static_cast<uint16_t>(rows), columns * 2U, (h - columns) * 2U, 0U, 0U}, {false, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_MTE3>(event);
        WaitFlag<HardEvent::MTE2_MTE3>(event);
        for (uint32_t index = 0U; index < fragment.segmentCount; ++index) {
            const auto* gm = reinterpret_cast<const __gm__ RowSegment*>(args.segments) + fragment.segmentBegin + index;
            const RowSegment segment{gm->sourceRank, gm->cv,   gm->receiveRow,      gm->returnRow,
                                     gm->tileRow,    gm->rows, gm->inputReadyIndex, gm->reserved};
            const auto span = PlanReorderSpan(segment, args.config.rank, m, rows);
            if (span.rows == 0U) continue;
            // H=3584 and N256 imply full-width N tiles; each UB row is 512B.
            const uint64_t output = span.outputRow * h + n;
            const DataCopyExtParams copy{static_cast<uint16_t>(span.rows), columns * 2U, 0U, (h - columns) * 2U, 0U};
            if (span.self)
                DataCopyPad(self[output], scratch[span.scratchRow * columns], copy);
            else
                DataCopyPad(remote[output], scratch[span.scratchRow * columns], copy);
        }
    }
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    // Preserve the existing consumer contract: one count means one N tile
    // has reached its FINAL source-major or self-return layout, not scratch.
    auto* done = reinterpret_cast<__gm__ int32_t*>(args.gmm2Done + fragmentIndex * kFlagBytes);
    AtomicAdd(done, 1);
    TraceMeta meta;
    meta.fragment = fragmentIndex;
    meta.expert = fragment.expert;
    meta.row = fragment.rowBegin;
    meta.rows = fragment.rows;
    meta.tileM = 0U;
    meta.tileN = n;
    meta.columns = columns;
    meta.bytes = static_cast<uint64_t>(fragment.rows) * columns * 2U;
    timer.TraceEnd(TraceEvent::Gmm2Reorder, start, meta);
}

}  // namespace MegaMoeUrma
