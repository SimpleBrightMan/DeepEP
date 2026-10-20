// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "compute_contract.hpp"
#include "kernel_operator.h"

namespace MegaMoeUrma {

// The host contract is H3584/K8. Pack/Reduce own UB only outside the fused
// compute phase, so these buffers can reuse the complete phase-private space.
struct ReduceUbContract {
    static constexpr uint32_t hidden = 3584U;
    static constexpr uint32_t topk = 8U;
    static constexpr uint32_t inputs = 6U;
    static constexpr uint32_t outputs = 2U;
    static constexpr uint32_t batchTokens = 128U;
    static constexpr uint32_t bf16RowBytes = hidden * sizeof(uint16_t);
    static constexpr uint32_t fp32RowBytes = hidden * sizeof(float);
    static constexpr uint32_t inputBf16Offset = 0U;
    static constexpr uint32_t inputFp32Offset = inputs * bf16RowBytes;
    static constexpr uint32_t accumulatorOffset = inputFp32Offset + inputs * fp32RowBytes;
    static constexpr uint32_t outputOffset = accumulatorOffset + fp32RowBytes;
    static constexpr uint32_t inverseOffset = outputOffset + outputs * bf16RowBytes;
    static constexpr uint32_t weightsOffset = inverseOffset + batchTokens * topk * sizeof(int64_t);
    static constexpr uint32_t end = weightsOffset + batchTokens * topk * sizeof(float);
};
static_assert(ReduceUbContract::end == 166U * 1024U);
static_assert(ReduceUbContract::end <= kRequiredUbBytes);
static_assert(ReduceUbContract::inputFp32Offset % 32U == 0U && ReduceUbContract::accumulatorOffset % 32U == 0U &&
              ReduceUbContract::outputOffset % 32U == 0U && ReduceUbContract::inverseOffset % 32U == 0U &&
              ReduceUbContract::weightsOffset % 32U == 0U);

template <typename Element>
__aicore__ inline void IssueReduceInput(AscendC::LocalTensor<Element>& inputs,
                                        const AscendC::GlobalTensor<Element>& returned,
                                        AscendC::LocalTensor<int64_t>& inverseUb, uint32_t route)
{
    using namespace AscendC;
    using Ub = ReduceUbContract;
    const uint32_t slot = route % Ub::inputs;
    const TEventID event = static_cast<TEventID>(slot);
    const uint64_t row = static_cast<uint64_t>(inverseUb.GetValue(route));
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopy(inputs[slot * Ub::hidden], returned[row * Ub::hidden], Ub::hidden);
    SetFlag<HardEvent::MTE2_V>(event);
}

__aicore__ inline void Reduce(const Arguments& args, uint32_t worker)
{
    using namespace AscendC;
    using Ub = ReduceUbContract;
    const uint64_t owned = LocalOwnedRows(args.config.bs, worker);
    if (owned == 0U) return;
    const uint64_t firstToken = LocalRowBegin(args.config.bs, worker);

    GlobalTensor<bfloat16_t> returned, output;
    GlobalTensor<int64_t> inverse;
    GlobalTensor<float> weights;
    returned.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(args.symmetric + args.config.returnYOffset));
    returned.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(args.output));
    inverse.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.reduceIndex));
    weights.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(args.weights));

    LocalTensor<bfloat16_t> inputs(TPosition::VECCALC, Ub::inputBf16Offset, Ub::inputs * Ub::hidden);
    LocalTensor<float> terms(TPosition::VECCALC, Ub::inputFp32Offset, Ub::inputs * Ub::hidden);
    LocalTensor<float> accumulator(TPosition::VECCALC, Ub::accumulatorOffset, Ub::hidden);
    LocalTensor<bfloat16_t> outputs(TPosition::VECCALC, Ub::outputOffset, Ub::outputs * Ub::hidden);
    LocalTensor<int64_t> inverseUb(TPosition::VECCALC, Ub::inverseOffset, Ub::batchTokens * Ub::topk);
    LocalTensor<float> weightsUb(TPosition::VECCALC, Ub::weightsOffset, Ub::batchTokens * Ub::topk);

    for (uint32_t slot = 0U; slot < Ub::inputs; ++slot) {
        SetFlag<HardEvent::V_MTE2>(static_cast<TEventID>(slot));
    }
    for (uint32_t slot = 0U; slot < Ub::outputs; ++slot) {
        SetFlag<HardEvent::MTE3_V>(static_cast<TEventID>(slot));
    }

    for (uint64_t batch = 0U; batch < owned; batch += Ub::batchTokens) {
        const uint32_t tokens =
            static_cast<uint32_t>(owned - batch < Ub::batchTokens ? owned - batch : Ub::batchTokens);
        const uint32_t routes = tokens * Ub::topk;
        const uint64_t firstRoute = (firstToken + batch) * Ub::topk;
        // Protect metadata still being consumed by Scalar in the previous
        // batch, then publish both DMA results with a single Scalar wait.
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        DataCopy(inverseUb, inverse[firstRoute], routes);
        DataCopy(weightsUb, weights[firstRoute], routes);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        const uint32_t initial = routes < Ub::inputs ? routes : Ub::inputs;
        for (uint32_t route = 0U; route < initial; ++route) {
            IssueReduceInput(inputs, returned, inverseUb, route);
        }

        for (uint32_t route = 0U; route < routes; ++route) {
            const uint32_t slot = route % Ub::inputs;
            const TEventID event = static_cast<TEventID>(slot);
            LocalTensor<bfloat16_t> input = inputs[slot * Ub::hidden];
            LocalTensor<float> term = terms[slot * Ub::hidden];
            const float weight = weightsUb.GetValue(route);
            WaitFlag<HardEvent::MTE2_V>(event);
            Cast(term, input, RoundMode::CAST_NONE, Ub::hidden);
            PipeBarrier<PIPE_V>();
            if (route % Ub::topk == 0U) {
                Muls(accumulator, term, weight, Ub::hidden);
            } else {
                Muls(term, term, weight, Ub::hidden);
                PipeBarrier<PIPE_V>();
                Add(accumulator, accumulator, term, Ub::hidden);
            }
            // Each slot is released only after its Cast/multiply/add have
            // consumed it. Other slots' loads can overlap this Vector work.
            SetFlag<HardEvent::V_MTE2>(event);

            if (route % Ub::topk == Ub::topk - 1U) {
                const uint64_t localToken = batch + route / Ub::topk;
                const TEventID outputEvent = static_cast<TEventID>(localToken % Ub::outputs);
                LocalTensor<bfloat16_t> result = outputs[(localToken % Ub::outputs) * Ub::hidden];
                WaitFlag<HardEvent::MTE3_V>(outputEvent);
                // Preserve the existing non-fused FP32 accumulation order and
                // the final BF16 round-to-nearest conversion.
                PipeBarrier<PIPE_V>();
                Cast(result, accumulator, RoundMode::CAST_RINT, Ub::hidden);
                SetFlag<HardEvent::V_MTE3>(outputEvent);
                WaitFlag<HardEvent::V_MTE3>(outputEvent);
                DataCopy(output[(firstToken + localToken) * Ub::hidden], result, Ub::hidden);
                SetFlag<HardEvent::MTE3_V>(outputEvent);
            }
            if (route + Ub::inputs < routes) {
                IssueReduceInput(inputs, returned, inverseUb, route + Ub::inputs);
            }
        }
    }
    // Consume every initial/recycled event and complete all output writes
    // before the phase-private UB can be reused or the kernel can return.
    for (uint32_t slot = 0U; slot < Ub::inputs; ++slot) {
        WaitFlag<HardEvent::V_MTE2>(static_cast<TEventID>(slot));
    }
    for (uint32_t slot = 0U; slot < Ub::outputs; ++slot) {
        WaitFlag<HardEvent::MTE3_V>(static_cast<TEventID>(slot));
    }
    PipeBarrier<PIPE_ALL>();
}

}  // namespace MegaMoeUrma
