// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "compute_contract.hpp"
#include "kernel_operator.h"

namespace MegaMoeUrma {

// The host contract is H3584/K8. A reducer enters only after its own prologue
// or activation role drains, so it can reuse that engine's complete UB.
struct ReduceUbContract {
    static constexpr uint32_t hidden = 3584U;
    static constexpr uint32_t topk = 8U;
    static constexpr uint32_t inputs = 6U;
    static constexpr uint32_t outputs = 2U;
    static constexpr uint32_t readyGateEvent = 6U;
    static constexpr uint32_t batchTokens = 128U;
    static constexpr uint32_t bf16RowBytes = hidden * sizeof(uint16_t);
    static constexpr uint32_t fp32RowBytes = hidden * sizeof(float);
    static constexpr uint32_t inputBf16Offset = 0U;
    static constexpr uint32_t inputFp32Offset = inputs * bf16RowBytes;
    static constexpr uint32_t accumulatorOffset = inputFp32Offset + inputs * fp32RowBytes;
    static constexpr uint32_t outputOffset = accumulatorOffset + fp32RowBytes;
    static constexpr uint32_t inverseOffset = outputOffset + outputs * bf16RowBytes;
    static constexpr uint32_t weightsOffset = inverseOffset + batchTokens * topk * sizeof(int64_t);
    static constexpr uint32_t dependenciesOffset = weightsOffset + batchTokens * topk * sizeof(float);
    static constexpr uint32_t end = dependenciesOffset + batchTokens * topk * sizeof(uint64_t);
};
static_assert(ReduceUbContract::end == 174U * 1024U);
static_assert(ReduceUbContract::end <= kRequiredUbBytes);
static_assert(ReduceUbContract::inputFp32Offset % 32U == 0U && ReduceUbContract::accumulatorOffset % 32U == 0U &&
              ReduceUbContract::outputOffset % 32U == 0U && ReduceUbContract::inverseOffset % 32U == 0U &&
              ReduceUbContract::weightsOffset % 32U == 0U && ReduceUbContract::dependenciesOffset % 32U == 0U);

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

__aicore__ inline bool AllReduceInputsReady(const Arguments& args)
{
    auto* flag =
        reinterpret_cast<__gm__ int32_t*>(args.localFlags + static_cast<uint64_t>(kLocalFlagIncomingDone) * kFlagBytes);
    return AscendC::ReadGmByPassDCache(flag) == static_cast<int32_t>(kWorld);
}

// Return true only after observing the aggregate completion counter. Otherwise
// check all eight dependencies before this token's first payload DMA. Probe the
// aggregate again during a long wait, without reading it on every failed poll.
__aicore__ inline bool WaitReduceToken(const Arguments& args, AscendC::LocalTensor<uint64_t>& dependencies,
                                       uint32_t firstRoute)
{
    using namespace AscendC;
    constexpr uint64_t pollBackoffCycles = 200U;
    constexpr uint32_t expectedSelfTiles = (ReduceUbContract::hidden + kComputeTileN - 1U) / kComputeTileN;
    uint32_t failedPolls = 0U;
    for (uint32_t slot = 0U; slot < ReduceUbContract::topk; ++slot) {
        const uint64_t dependency = dependencies.GetValue(firstRoute + slot);
        const uint64_t index = dependency & ~kReduceSelfDependency;
        if ((dependency & kReduceSelfDependency) != 0U) {
            auto* flag = reinterpret_cast<__gm__ int32_t*>(args.gmm2Done + index * kFlagBytes);
            while (ReadGmByPassDCache(flag) < static_cast<int32_t>(expectedSelfTiles)) {
                if ((failedPolls++ & 7U) == 0U && AllReduceInputsReady(args)) return true;
                const uint64_t start = GetSystemCycle();
                while (GetSystemCycle() - start < pollBackoffCycles) {
                }
            }
        } else {
            auto* flag = reinterpret_cast<__gm__ uint64_t*>(args.symmetric + args.config.combineReadyOffset +
                                                            index * kFlagBytes);
            while (ReadGmByPassDCache(flag) != args.config.epoch) {
                if ((failedPolls++ & 7U) == 0U && AllReduceInputsReady(args)) return true;
                const uint64_t start = GetSystemCycle();
                while (GetSystemCycle() - start < pollBackoffCycles) {
                }
            }
        }
    }
    return false;
}

__aicore__ inline void GateReduceInputs()
{
    using namespace AscendC;
    // Payload reads follow Scalar's observation of dependency flags or the
    // aggregate completion counter. Reuse this gate only after MTE2 consumed
    // its previous signal; the credit is
    // returned BEFORE payload loads, so this is not a token DMA drain.
    const auto gate = static_cast<TEventID>(ReduceUbContract::readyGateEvent);
    WaitFlag<HardEvent::MTE2_S>(gate);
    SetFlag<HardEvent::S_MTE2>(gate);
    WaitFlag<HardEvent::S_MTE2>(gate);
    SetFlag<HardEvent::MTE2_S>(gate);
}

template <typename TraceSink>
__aicore__ inline void Reduce(const Arguments& args, uint32_t worker, TraceSink& timer)
{
    using namespace AscendC;
    using Ub = ReduceUbContract;
    const uint64_t owned = ReduceOwnedRows(args.config.bs, worker);
    if (owned == 0U) return;
    // One interval per reducer includes metadata, ready waits, arithmetic and
    // the existing output drain. Never write trace records inside the pipeline.
    const uint64_t reduceStart = timer.TraceBegin();
    const uint64_t firstToken = ReduceRowBegin(args.config.bs, worker);

    GlobalTensor<bfloat16_t> returned, output;
    GlobalTensor<int64_t> inverse;
    GlobalTensor<float> weights;
    GlobalTensor<uint64_t> dependencies;
    returned.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(args.symmetric + args.config.returnYOffset));
    returned.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(args.output));
    inverse.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.reduceIndex));
    weights.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(args.weights));
    dependencies.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(args.reduceDependencies));

    LocalTensor<bfloat16_t> inputs(TPosition::VECCALC, Ub::inputBf16Offset, Ub::inputs * Ub::hidden);
    LocalTensor<float> terms(TPosition::VECCALC, Ub::inputFp32Offset, Ub::inputs * Ub::hidden);
    LocalTensor<float> accumulator(TPosition::VECCALC, Ub::accumulatorOffset, Ub::hidden);
    LocalTensor<bfloat16_t> outputs(TPosition::VECCALC, Ub::outputOffset, Ub::outputs * Ub::hidden);
    LocalTensor<int64_t> inverseUb(TPosition::VECCALC, Ub::inverseOffset, Ub::batchTokens * Ub::topk);
    LocalTensor<float> weightsUb(TPosition::VECCALC, Ub::weightsOffset, Ub::batchTokens * Ub::topk);
    LocalTensor<uint64_t> dependenciesUb(TPosition::VECCALC, Ub::dependenciesOffset, Ub::batchTokens * Ub::topk);

    for (uint32_t slot = 0U; slot < Ub::inputs; ++slot) {
        SetFlag<HardEvent::V_MTE2>(static_cast<TEventID>(slot));
    }
    for (uint32_t slot = 0U; slot < Ub::outputs; ++slot) {
        SetFlag<HardEvent::MTE3_V>(static_cast<TEventID>(slot));
    }
    SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(Ub::readyGateEvent));
    bool allInputsReady = AllReduceInputsReady(args);
    if (allInputsReady) GateReduceInputs();

    for (uint64_t batch = 0U; batch < owned; batch += Ub::batchTokens) {
        const uint32_t tokens =
            static_cast<uint32_t>(owned - batch < Ub::batchTokens ? owned - batch : Ub::batchTokens);
        const uint32_t routes = tokens * Ub::topk;
        const uint64_t firstRoute = (firstToken + batch) * Ub::topk;
        // Protect metadata still being consumed by Scalar in the previous
        // batch, then publish all metadata DMA results with one Scalar wait.
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        DataCopy(inverseUb, inverse[firstRoute], routes);
        DataCopy(weightsUb, weights[firstRoute], routes);
        if (!allInputsReady) DataCopy(dependenciesUb, dependencies[firstRoute], routes);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        uint32_t nextInput = 0U;
        for (uint32_t token = 0U; token < tokens; ++token) {
            const uint32_t tokenRoute = token * Ub::topk;
            const uint64_t localToken = batch + token;
            if (!allInputsReady) {
                if (localToken != 0U && (localToken & 7U) == 0U) allInputsReady = AllReduceInputsReady(args);
                if (!allInputsReady) allInputsReady = WaitReduceToken(args, dependenciesUb, tokenRoute);
                GateReduceInputs();
            }
            // The fast refill already prefetched the next token's first six
            // routes. Slow tokens and the slow->fast transition seed them here.
            const uint32_t initialEnd = tokenRoute + Ub::inputs;
            for (; nextInput < initialEnd; ++nextInput) {
                IssueReduceInput(inputs, returned, inverseUb, nextInput);
            }
            for (uint32_t topk = 0U; topk < Ub::topk; ++topk) {
                const uint32_t route = tokenRoute + topk;
                const uint32_t slot = route % Ub::inputs;
                const TEventID event = static_cast<TEventID>(slot);
                LocalTensor<bfloat16_t> input = inputs[slot * Ub::hidden];
                LocalTensor<float> term = terms[slot * Ub::hidden];
                const float weight = weightsUb.GetValue(route);
                WaitFlag<HardEvent::MTE2_V>(event);
                Cast(term, input, RoundMode::CAST_NONE, Ub::hidden);
                PipeBarrier<PIPE_V>();
                if (topk == 0U) {
                    Muls(accumulator, term, weight, Ub::hidden);
                } else {
                    Muls(term, term, weight, Ub::hidden);
                    PipeBarrier<PIPE_V>();
                    Add(accumulator, accumulator, term, Ub::hidden);
                }
                // Once all inputs are ready, retain six in-flight routes
                // across token boundaries, as in the original local pipeline.
                // The slow path never prefetches an unready next token.
                SetFlag<HardEvent::V_MTE2>(event);
                const uint32_t readyEnd = allInputsReady ? routes : tokenRoute + Ub::topk;
                if (route + Ub::inputs < readyEnd) {
                    IssueReduceInput(inputs, returned, inverseUb, route + Ub::inputs);
                    ++nextInput;
                }
            }
            const TEventID outputEvent = static_cast<TEventID>(localToken % Ub::outputs);
            LocalTensor<bfloat16_t> result = outputs[(localToken % Ub::outputs) * Ub::hidden];
            WaitFlag<HardEvent::MTE3_V>(outputEvent);
            // Preserve non-fused FP32 accumulation and final BF16 rounding.
            PipeBarrier<PIPE_V>();
            Cast(result, accumulator, RoundMode::CAST_RINT, Ub::hidden);
            SetFlag<HardEvent::V_MTE3>(outputEvent);
            WaitFlag<HardEvent::V_MTE3>(outputEvent);
            DataCopy(output[(firstToken + localToken) * Ub::hidden], result, Ub::hidden);
            SetFlag<HardEvent::MTE3_V>(outputEvent);
        }
    }
    // Consume every initial/recycled event and complete all output writes
    // before the phase-private UB can be reused or the kernel can return.
    WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(Ub::readyGateEvent));
    for (uint32_t slot = 0U; slot < Ub::inputs; ++slot) {
        WaitFlag<HardEvent::V_MTE2>(static_cast<TEventID>(slot));
    }
    for (uint32_t slot = 0U; slot < Ub::outputs; ++slot) {
        WaitFlag<HardEvent::MTE3_V>(static_cast<TEventID>(slot));
    }
    PipeBarrier<PIPE_ALL>();
    TraceMeta meta;
    meta.row = firstToken;
    meta.rows = owned;
    meta.bytes = owned * Ub::bf16RowBytes;
    timer.TraceEnd(TraceEvent::Reduce, reduceStart, meta);
}

}  // namespace MegaMoeUrma
