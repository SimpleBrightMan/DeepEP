// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "compute_contract.hpp"
#include "kernel_operator.h"

namespace MegaMoeUrma {

constexpr uint32_t kPackPipelineSlots = 6U;
constexpr uint32_t kPackTokenCapacity = 8192U;
constexpr uint32_t kPackScaleCapacity = kPackTokenCapacity / 32U;
constexpr uint32_t kPackSlotBytes = kPackTokenCapacity + kPackScaleCapacity;
constexpr uint32_t kPackOrderRows = 1024U;
constexpr uint32_t kPackOrderOffset = kPackPipelineSlots * kPackSlotBytes;
constexpr uint32_t kPackUbBytes = kPackOrderOffset + kPackOrderRows * sizeof(int64_t);
// Activation and scale gather use the low 128KiB. Pack may run between their
// tasks on AIV1; keep its data above that region, and close all event credits
// before returning. AIV0 releases this region before opening its W4 context.
constexpr uint32_t kPackUbBase = ActivationUbContract::inputEnd;
static_assert(kPackSlotBytes % 32U == 0U && kPackOrderOffset % 32U == 0U);
static_assert(kPackUbBase + kPackUbBytes <= kRequiredUbBytes, "Pack and activation must fit the minimum UB");

// Pack one statically assigned contiguous expanded-row segment. The caller
// publishes its row count only after this function certifies BOTH GM stores.
// No whole-pipeline barrier: only these six MTE slots are completed here.
__aicore__ inline void ExpandRows(const Arguments& args, uint64_t begin, uint64_t count)
{
    using namespace AscendC;
    const uint32_t h = args.config.hidden;  // Host currently requires H=3584.
    const uint32_t scales = h / 32U;
    GlobalTensor<uint8_t> x, xScale, expandedX, expandedScale, selfX, selfScale;
    GlobalTensor<int64_t> order;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.x));
    xScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.xScale));
    expandedX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.expandedX));
    expandedScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.expandedScale));
    selfX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.symmetric + args.config.receiveXOffset));
    selfScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.symmetric + args.config.receiveScaleOffset));
    order.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.order));
    LocalTensor<int64_t> routeIndices(TPosition::VECCALC, kPackUbBase + kPackOrderOffset, kPackOrderRows);
    for (uint32_t slot = 0U; slot < kPackPipelineSlots; ++slot) {
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slot));
    }
    for (uint64_t batch = 0U; batch < count; batch += kPackOrderRows) {
        const uint32_t batchRows =
            static_cast<uint32_t>(count - batch < kPackOrderRows ? count - batch : kPackOrderRows);
        // This dependency protects the preceding batch's scalar UB reads.
        // The MTE2_S wait occurs once per index batch, never once per token.
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        DataCopyPad(routeIndices, order[begin + batch],
                    {1U, static_cast<uint32_t>(batchRows * sizeof(int64_t)), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t index = 0U; index < batchRows; ++index) {
            const uint64_t localRow = batch + index;
            const uint64_t row = begin + localRow;
            const uint32_t slot = static_cast<uint32_t>(localRow % kPackPipelineSlots);
            const auto event = static_cast<TEventID>(slot);
            LocalTensor<uint8_t> token(TPosition::VECCALC, kPackUbBase + slot * kPackSlotBytes, kPackTokenCapacity);
            LocalTensor<uint8_t> scale(TPosition::VECCALC, kPackUbBase + slot * kPackSlotBytes + kPackTokenCapacity,
                                       kPackScaleCapacity);
            const uint64_t tokenRow = static_cast<uint64_t>(routeIndices.GetValue(index)) / args.config.topk;
            WaitFlag<HardEvent::MTE3_MTE2>(event);
            DataCopyPad(token, x[tokenRow * h], {1U, h, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            DataCopyPad(scale, xScale[tokenRow * scales], {1U, scales, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            SetFlag<HardEvent::MTE2_MTE3>(event);
            WaitFlag<HardEvent::MTE2_MTE3>(event);
            if (IsSelfExpandedRow(row, args.config)) {
                const uint64_t receiveRow = SelfReceiveRow(row, args.config);
                DataCopyPad(selfX[receiveRow * h], token, {1U, h, 0U, 0U, 0U});
                DataCopyPad(selfScale[receiveRow * scales], scale, {1U, scales, 0U, 0U, 0U});
            } else {
                DataCopyPad(expandedX[row * h], token, {1U, h, 0U, 0U, 0U});
                DataCopyPad(expandedScale[row * scales], scale, {1U, scales, 0U, 0U, 0U});
            }
            SetFlag<HardEvent::MTE3_MTE2>(event);
        }
    }
    for (uint32_t slot = 0U; slot < kPackPipelineSlots; ++slot) {
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slot));
    }
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

// Retain the original full-expand entry point for standalone callers.
__aicore__ inline void Expand(const Arguments& args, uint32_t worker)
{
    const uint64_t rows = static_cast<uint64_t>(args.config.bs) * args.config.topk;
    ExpandRows(args, LocalRowBegin(rows, worker), LocalOwnedRows(rows, worker));
}

}  // namespace MegaMoeUrma
