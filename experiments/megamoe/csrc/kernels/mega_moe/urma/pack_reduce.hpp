// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "compute_contract.hpp"
#include "kernel_operator.h"

namespace MegaMoeUrma {

// Phase-private UB: Expand/Reduce run before/after the compute workers and
// return only after all local DMA/vector work has completed.
__aicore__ inline void Expand(const Arguments& args, uint32_t worker)
{
    using namespace AscendC;
    const uint32_t h = args.config.hidden;
    const uint32_t scales = h / 32U;
    const uint64_t rows = static_cast<uint64_t>(args.config.bs) * args.config.topk;
    GlobalTensor<uint8_t> x, xScale, expandedX, expandedScale, selfX, selfScale;
    GlobalTensor<int64_t> order;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.x));
    xScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.xScale));
    expandedX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.expandedX));
    expandedScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.expandedScale));
    selfX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.symmetric + args.config.receiveXOffset));
    selfScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.symmetric + args.config.receiveScaleOffset));
    order.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.order));
    LocalTensor<uint8_t> token(TPosition::VECCALC, 0, 8192);
    LocalTensor<uint8_t> scale(TPosition::VECCALC, 8192, 256);
    for (uint64_t row = worker; row < rows; row += kLocalWorkers) {
        const uint64_t tokenRow = static_cast<uint64_t>(order.GetValue(row)) / args.config.topk;
        DataCopyPad(token, x[tokenRow * h], {1U, h, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        DataCopyPad(scale, xScale[tokenRow * scales], {1U, scales, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        if (IsSelfExpandedRow(row, args.config)) {
            const uint64_t receiveRow = SelfReceiveRow(row, args.config);
            DataCopyPad(selfX[receiveRow * h], token, {1U, h, 0U, 0U, 0U});
            DataCopyPad(selfScale[receiveRow * scales], scale, {1U, scales, 0U, 0U, 0U});
        } else {
            DataCopyPad(expandedX[row * h], token, {1U, h, 0U, 0U, 0U});
            DataCopyPad(expandedScale[row * scales], scale, {1U, scales, 0U, 0U, 0U});
        }
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Reduce(const Arguments& args, uint32_t worker)
{
    using namespace AscendC;
    const uint32_t h = args.config.hidden;
    const uint32_t topk = args.config.topk;
    GlobalTensor<bfloat16_t> returned, output;
    GlobalTensor<int64_t> inverse;
    GlobalTensor<float> weights;
    returned.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(args.symmetric + args.config.returnYOffset));
    returned.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(args.output));
    inverse.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.reduceIndex));
    weights.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(args.weights));
    LocalTensor<bfloat16_t> input(TPosition::VECCALC, 0, 8192);
    LocalTensor<float> term(TPosition::VECCALC, 32768, 8192);
    LocalTensor<float> accumulator(TPosition::VECCALC, 65536, 8192);
    for (uint32_t token = worker; token < args.config.bs; token += kLocalWorkers) {
        for (uint32_t slot = 0; slot < topk; ++slot) {
            const uint64_t route = static_cast<uint64_t>(token) * topk + slot;
            const uint64_t row = static_cast<uint64_t>(inverse.GetValue(route));
            const float weight = weights.GetValue(route);
            DataCopy(input, returned[row * h], h);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(term, input, RoundMode::CAST_NONE, h);
            PipeBarrier<PIPE_V>();
            if (slot == 0U) {
                Muls(accumulator, term, weight, h);
            } else {
                Muls(term, term, weight, h);
                PipeBarrier<PIPE_V>();
                Add(accumulator, accumulator, term, h);
            }
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        Cast(input, accumulator, RoundMode::CAST_RINT, h);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(output[static_cast<uint64_t>(token) * h], input, h);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    PipeBarrier<PIPE_ALL>();
}
}  // namespace MegaMoeUrma
