// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

/*!
 * \file mega_moe_tp_unpermute_rs.hpp
 * \brief TP MegaMoE Phase 2 P1 提前部分 Unpermute + Phase 3 融合 Unpermute+Send -> Wait+LocalReduce。
 */

#ifndef MEGA_MOE_TP_UNPERMUTE_RS_HPP
#define MEGA_MOE_TP_UNPERMUTE_RS_HPP

#include <c_api/asc_simd.h>
#include "../common/mega_moe_tp_layout.hpp"
#include "../common/mega_moe_utils.hpp"
#include "../common/aiv1_async_events.hpp"
#include "../common/ascend_timer_v2_device.hpp"
#include "mega_moe_unpermute_fused.hpp"

namespace MegaMoeImpl {

using namespace AscendC;

#if defined(__DAV_C310__)
// Phase 2 P1 部分 Unpermute：将已完成的 count (>= 2) 行 BF16 按各自权重在寄存器内累加为精确 FP32，
// 以原生 (sum0, sum1) 交错寄存器布局连续写出到 dstFp32（零额外解交错开销，与单趟 WeightedUnpermuteVf 100% 同构）。
__simd_vf__ inline void WeightedPartialToFp32Vf(__ubuf__ float* dstFp32, __ubuf__ bfloat16_t* inputs,
                                                __ubuf__ float* weights, uint32_t count, uint32_t channels)
{
    const auto mask32 = asc_create_mask_b32(PAT_ALL);
    vector_bfloat16_t zero;
    asc_duplicate_scalar(zero, 0);
    auto outFp32 = dstFp32;
    for (uint32_t channel = 0U; channel < channels; channel += 128U) {
        vector_float sum0, sum1, firstScale;
        auto input = inputs + channel;
        auto weight = weights;
        vector_bfloat16_t firstValue;
        asc_loadalign_postupdate(firstValue, input, channels);
        asc_loadalign_brc_postupdate(firstScale, weight, 1);
        asc_intlv(reinterpret_cast<vector_bfloat16_t&>(sum0), reinterpret_cast<vector_bfloat16_t&>(sum1), zero,
                  firstValue);
        asc_mul(sum0, sum0, firstScale, mask32);
        asc_mul(sum1, sum1, firstScale, mask32);
        for (uint32_t item = 1U; item < count; ++item) {
            vector_bfloat16_t value;
            vector_float low, high, scale;
            asc_loadalign_postupdate(value, input, channels);
            asc_loadalign_brc_postupdate(scale, weight, 1);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(low), reinterpret_cast<vector_bfloat16_t&>(high), zero,
                      value);
            asc_mul(low, low, scale, mask32);
            asc_mul(high, high, scale, mask32);
            asc_add(sum0, sum0, low, mask32);
            asc_add(sum1, sum1, high, mask32);
        }
        asc_storealign_postupdate(outFp32, sum0, 64U, mask32);
        asc_storealign_postupdate(outFp32, sum1, 64U, mask32);
    }
}

// Phase 3 归并收尾：从 dstAndSrcFp32 载入 Phase 2 预累加的 FP32 部分和 (sum0, sum1)，
// 继续累加剩余 remCount (>= 1) 行 BF16，单次 RINT 舍入后原地写回 dstAndSrcFp32 起始处的 BF16 槽（每 128 列读 512B 写 256B，零覆盖风险）。
__simd_vf__ inline void WeightedFinishFromFp32Vf(__ubuf__ bfloat16_t* dstOutAndSrcFp32, __ubuf__ bfloat16_t* remInputs,
                                                 __ubuf__ float* remWeights, uint32_t remCount, uint32_t channels)
{
    const auto mask16 = asc_create_mask_b16(PAT_ALL);
    const auto mask32 = asc_create_mask_b32(PAT_ALL);
    vector_bfloat16_t zero;
    asc_duplicate_scalar(zero, 0);
    auto inFp32 = reinterpret_cast<__ubuf__ float*>(dstOutAndSrcFp32);
    auto output = dstOutAndSrcFp32;
    for (uint32_t channel = 0U; channel < channels; channel += 128U) {
        vector_float sum0, sum1;
        asc_loadalign_postupdate(sum0, inFp32, 64U);
        asc_loadalign_postupdate(sum1, inFp32, 64U);
        auto input = remInputs + channel;
        auto weight = remWeights;
        for (uint32_t item = 0U; item < remCount; ++item) {
            vector_bfloat16_t value;
            vector_float low, high, scale;
            asc_loadalign_postupdate(value, input, channels);
            asc_loadalign_brc_postupdate(scale, weight, 1);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(low), reinterpret_cast<vector_bfloat16_t&>(high), zero,
                      value);
            asc_mul(low, low, scale, mask32);
            asc_mul(high, high, scale, mask32);
            asc_add(sum0, sum0, low, mask32);
            asc_add(sum1, sum1, high, mask32);
        }
        vector_bfloat16_t out0, out1, packed, unused;
        asc_float2bfloat16_rn(out0, sum0, mask32, ASC_POSITION_EVEN);
        asc_float2bfloat16_rn(out1, sum1, mask32, ASC_POSITION_EVEN);
        asc_deintlv(packed, unused, out0, out1);
        asc_storealign_postupdate(output, packed, 128U, mask16);
    }
}

// 将 [numRows, channels] BF16 按行精确展开为 FP32 求和，并在首行原地输出一次 RINT 舍入后的 BF16（要求 channels % 128 == 0）
__simd_vf__ inline void SumBf16RowsToBf16Vf(__ubuf__ bfloat16_t* rows, uint32_t numRows, uint32_t channels)
{
    const auto mask16 = asc_create_mask_b16(PAT_ALL);
    const auto mask32 = asc_create_mask_b32(PAT_ALL);
    vector_bfloat16_t zero;
    asc_duplicate_scalar(zero, 0);
    auto output = rows;
    for (uint32_t channel = 0U; channel < channels; channel += 128U) {
        vector_float sum0, sum1;
        auto input = rows + channel;
        vector_bfloat16_t firstValue;
        asc_loadalign_postupdate(firstValue, input, channels);
        asc_intlv(reinterpret_cast<vector_bfloat16_t&>(sum0), reinterpret_cast<vector_bfloat16_t&>(sum1), zero,
                  firstValue);
        for (uint32_t item = 1U; item < numRows; ++item) {
            vector_bfloat16_t value;
            vector_float low, high;
            asc_loadalign_postupdate(value, input, channels);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(low), reinterpret_cast<vector_bfloat16_t&>(high), zero,
                      value);
            asc_add(sum0, sum0, low, mask32);
            asc_add(sum1, sum1, high, mask32);
        }
        vector_bfloat16_t out0, out1, packed, unused;
        asc_float2bfloat16_rn(out0, sum0, mask32, ASC_POSITION_EVEN);
        asc_float2bfloat16_rn(out1, sum1, mask32, ASC_POSITION_EVEN);
        asc_deintlv(packed, unused, out0, out1);
        asc_storealign_postupdate(output, packed, 128U, mask16);
    }
}
#endif

// 通用向量累加回退（兼容任意 64 对齐列宽，如 H % 128 == 64）
__aicore__ inline void WeightedSumColumnsFallback(LocalTensor<bfloat16_t>& packedBf16, LocalTensor<float>& accFp32,
                                                  LocalTensor<float>& tmpFp32, const LocalTensor<float>& weights,
                                                  uint32_t weightOffset, uint32_t count, uint32_t cols, bool weighted)
{
    Cast(accFp32, packedBf16, RoundMode::CAST_NONE, cols);
    PipeBarrier<PIPE_V>();
    if (weighted) {
        Muls(accFp32, accFp32, weights.GetValue(weightOffset), cols);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t k = 1U; k < count; ++k) {
        Cast(tmpFp32, packedBf16[k * cols], RoundMode::CAST_NONE, cols);
        PipeBarrier<PIPE_V>();
        if (weighted) {
            Muls(tmpFp32, tmpFp32, weights.GetValue(weightOffset + k), cols);
            PipeBarrier<PIPE_V>();
        }
        Add(accFp32, accFp32, tmpFp32, cols);
        PipeBarrier<PIPE_V>();
    }
    Cast(packedBf16, accFp32, RoundMode::CAST_RINT, cols);
}

/*
 * Phase 2 P1：利用 Permute 结束后 AIV1 的空闲窗口（~500us），以 8-token 微步可抢占粒度预累加已完成专家的 GMM2 行。
 *   - UB 严格驻留在 [68 KiB, 250 KiB)（> ActivationBytes 66.03 KiB），不触碰 Routed SwiGLU 常驻 UB；
 *   - 硬件事件仅使用 EVENT_ID2/3 (双槽流水) 与 EVENT_ID7 (DispatchDone)，与 Routed SwiGLU (EVENT_ID0/6) 完全正交；
 *   - 优先处理 3 个远端目标卡的 chunk（使 Phase 3 跨卡 HCCS 发送零阻碍起飞），最后处理本卡 chunk；
 *   - 仅当整块 chunk 完成落盘后才发布 flagEarlyChunkDonePtr = splitPermRow + 1；一旦 Phase 2 全部 GMM2 结束立即让路。
 */
struct TpEarlyUnpermCursor {
    uint32_t workIdx = 0U;
    uint32_t totalWorkChunks = 0U;
    uint32_t numChunksPerRank = 0U;
    uint32_t tokInChunk = 0U;
    uint32_t splitPermRow = 0U;
    bool splitReady = false;
    bool permDrained = false;
    bool done = true;

    __aicore__ inline void Init(const MegaMoeTpTilingData& tiling, uint32_t blockIdx)
    {
#if defined(__DAV_C310__)
        const bool supported = (tiling.bs * tiling.tpWorldSize >= 512U) && (tiling.numExperts >= 2U) &&
                               (tiling.h > 0U) && (tiling.h <= 4096U) && (tiling.h % 128U == 0U) &&
                               (tiling.topK >= 2U) && (tiling.topK <= 16U) &&
                               (tiling.rsTokensPerChunk > 0U) && (tiling.rsTokensPerChunk <= 128U) &&
                               (static_cast<uint64_t>(tiling.topK) * tiling.h * sizeof(uint16_t) <= 64U * 1024U);
        if (supported) {
            numChunksPerRank = Ops::Base::CeilDiv(tiling.bs, tiling.rsTokensPerChunk);
            totalWorkChunks = tiling.tpWorldSize * numChunksPerRank;
            workIdx = blockIdx;
            tokInChunk = 0U;
            splitPermRow = 0U;
            splitReady = false;
            permDrained = false;
            done = (workIdx >= totalWorkChunks);
            return;
        }
#else
        (void)tiling;
        (void)blockIdx;
#endif
        permDrained = false;
        done = true;
    }
};

__aicore__ inline bool StepTpEarlySplitUnpermute(const MegaMoeTpTilingData& tiling, uint32_t rankId,
                                                 const TpWorkspaceInfo& workspace, TpEarlyUnpermCursor& cursor,
                                                 AscendTimerDevice& timer)
{
#if defined(__DAV_C310__)
    if constexpr (g_coreType == AIC) {
        return false;
    }
    if (GetSubBlockIdx() != 1U || cursor.done) {
        return false;
    }

    if (!cursor.splitReady) {
        auto* splitFlags = reinterpret_cast<__gm__ int32_t*>(workspace.flagGmm2SplitDonePtr);
        int32_t encoded = 0;
        for (uint32_t c = 0U; c < tiling.aicNum; ++c) {
            const int32_t v = ReadGmByPassDCache(splitFlags + static_cast<uint64_t>(c) * INT_CACHELINE);
            if (v <= 0) {
                return false;
            }
            encoded = v;
        }
        cursor.splitPermRow = static_cast<uint32_t>(encoded - 1);
        cursor.splitReady = true;
        SyncFuncStatic<HardEvent::S_MTE2, static_cast<TEventID>(Aiv1AsyncEvents::DispatchDone)>();
    }

    if (cursor.workIdx >= cursor.totalWorkChunks) {
        cursor.done = true;
        return false;
    }

    const uint32_t bs = tiling.bs;
    const uint32_t tpWorldSize = tiling.tpWorldSize;
    const uint32_t topK = tiling.topK;
    const uint32_t h = tiling.h;
    const uint32_t chunkTokens = tiling.rsTokensPerChunk;
    const uint32_t maxChunksPerRank = Ops::Base::CeilDiv(tiling.numMaxTokensPerRank, chunkTokens);
    const bool preQuantWeight = tiling.routeWeightBeforeQuant != 0U;

    // 优先处理远端目标卡 chunk (stage = 0 .. tpWorldSize-2)，最后处理本卡 chunk (stage = tpWorldSize-1)
    const uint32_t stage = cursor.workIdx / cursor.numChunksPerRank;
    const uint32_t c = cursor.workIdx % cursor.numChunksPerRank;
    const uint32_t dstRank = (stage + 1U < tpWorldSize) ? ((rankId + 1U + stage) % tpWorldSize) : rankId;
    const uint32_t chunkBegin = c * chunkTokens;
    const uint32_t chunkEnd = Min(bs, chunkBegin + chunkTokens);
    const uint32_t chunkLen = chunkEnd - chunkBegin;
    const uint64_t globalChunkTokBase = static_cast<uint64_t>(dstRank) * bs + chunkBegin;

    // AIV1 P1 UB 布局（严格位于 [68 KiB, 240 KiB)，与 Routed SwiGLU [0, 66.03 KiB) 零重叠）：
    //   [68, 70 KiB)    p1PermIdxUb        (32 tokens * 16 topK * 4B = 2 KiB)
    //   [70, 72 KiB)    p1WeightsUb        (32 tokens * 16 topK * 4B = 2 KiB)
    //   [72, 74 KiB)    p1CompactWeightsUb (32 tokens * 16 stride * 4B = 2 KiB)
    //   [74, 76 KiB)    p1StateUb          (最多 128 ints = 512B，跨微步保留直到整块 chunk 完成一次写出)
    //   [76, 140 KiB)   p1PackSlot0        (64 KiB: BF16 gather + 原地写出或输入)
    //   [140, 204 KiB)  p1PackSlot1        (64 KiB)
    //   [204, 220 KiB)  p1OutFp32Slot0     (16 KiB: 4096 floats)
    //   [220, 236 KiB)  p1OutFp32Slot1     (16 KiB: 4096 floats)
    //   [236, 238 KiB)  p1ReadyRowsUb      (32 tokens * 16 stride * 4B = 2 KiB)
    constexpr uint32_t kP1StepTokens = 32U;
    constexpr uint32_t kP1WeightStride = 16U;
    constexpr uint32_t kUbPermIdx = 68U * 1024U;
    constexpr uint32_t kUbWeights = 70U * 1024U;
    constexpr uint32_t kUbCompWeights = 72U * 1024U;
    constexpr uint32_t kUbState = 74U * 1024U;
    constexpr uint32_t kUbPack0 = 76U * 1024U;
    constexpr uint32_t kUbPackSlotBytes = 64U * 1024U;
    constexpr uint32_t kUbOutFp32_0 = 204U * 1024U;
    constexpr uint32_t kUbOutFp32Bytes = 16U * 1024U;
    constexpr uint32_t kUbReadyRows = 236U * 1024U;

    timer.SetWave(AscendTimer::NO_WAVE);
    AscendTimerSpanScope earlySpan(timer.Enabled() ? &timer : nullptr, AscendTimer::AIV_UNPERMUTE_SEND);

    const uint32_t subOff = cursor.tokInChunk;
    const uint32_t subCnt = Min(kP1StepTokens, chunkLen - subOff);
    const uint64_t subGlobalTokBase = globalChunkTokBase + subOff;
    const uint32_t metaBytes = subCnt * topK * static_cast<uint32_t>(sizeof(int32_t));

    LocalTensor<int32_t> p1PermIdxUb(TPosition::VECCALC, kUbPermIdx, kP1StepTokens * kP1WeightStride);
    LocalTensor<float> p1WeightsUb(TPosition::VECCALC, kUbWeights, kP1StepTokens * kP1WeightStride);
    LocalTensor<float> p1CompWeightsUb(TPosition::VECCALC, kUbCompWeights, kP1StepTokens * kP1WeightStride);
    LocalTensor<int32_t> p1StateUb(TPosition::VECCALC, kUbState, 256U);
    LocalTensor<int32_t> p1ReadyRowsUb(TPosition::VECCALC, kUbReadyRows, kP1StepTokens * kP1WeightStride);

    GlobalTensor<int32_t> tokenToPermRowGm, earlyStateGm;
    GlobalTensor<float> agWeightsGm, earlyFp32Gm;
    GlobalTensor<bfloat16_t> gmm2OutGm, earlyBf16Gm;
    tokenToPermRowGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.tokenToPermRowPtr));
    agWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace.agTopkWeightsPtr));
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspace.gmm2OutPtr));
    gmm2OutGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    earlyStateGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.earlyTokenStatePtr));
    earlyFp32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace.earlyPartialFp32Ptr));
    earlyBf16Gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspace.earlyPartialFp32Ptr));

    DataCopyPad(p1PermIdxUb, tokenToPermRowGm[subGlobalTokBase * topK], {1U, metaBytes, 0U, 0U, 0U},
                {false, 0U, 0U, 0U});
    if (!preQuantWeight) {
        DataCopyPad(p1WeightsUb, agWeightsGm[subGlobalTokBase * topK], {1U, metaBytes, 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
    }
    SyncFuncStatic<HardEvent::MTE2_S, static_cast<TEventID>(Aiv1AsyncEvents::DispatchDone)>();

    uint32_t readyCounts[kP1StepTokens];
    const uint32_t splitRow = cursor.splitPermRow;

    for (uint32_t b = 0U; b < subCnt; ++b) {
        uint32_t m = 0U;
        for (uint32_t k = 0U; k < topK; ++k) {
            const int32_t pr = p1PermIdxUb.GetValue(b * topK + k);
            if (pr >= 0 && static_cast<uint32_t>(pr) < splitRow) {
                p1ReadyRowsUb.SetValue(b * kP1WeightStride + m, pr);
                p1CompWeightsUb.SetValue(b * kP1WeightStride + m,
                                         preQuantWeight ? 1.0F : p1WeightsUb.GetValue(b * topK + k));
                ++m;
            }
        }
        readyCounts[b] = m;
        // 仅当至少合并 2 行时才写预累加结果（确保 Phase 3 读量 <= 原读量且减少随机寻址）
        p1StateUb.SetValue(subOff + b, m >= 2U ? static_cast<int32_t>(m) : 0);
    }
    SyncFuncStatic<HardEvent::S_V, static_cast<TEventID>(Aiv1AsyncEvents::DispatchDone)>();

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    uint32_t activeSeq = 0U;
    for (uint32_t b = 0U; b < subCnt; ++b) {
        const uint32_t m = readyCounts[b];
        if (m < 2U) {
            continue;
        }
        const uint32_t slot = activeSeq++ % 2U;
        const TEventID ev = slot == 0U ? EVENT_ID2 : EVENT_ID3;
        LocalTensor<bfloat16_t> packUb(TPosition::VECCALC, kUbPack0 + slot * kUbPackSlotBytes, topK * h);
        LocalTensor<float> outFp32Ub(TPosition::VECCALC, kUbOutFp32_0 + slot * kUbOutFp32Bytes, h);

        WaitFlag<HardEvent::MTE3_MTE2>(ev);
        for (uint32_t r = 0U; r < m; ++r) {
            const int32_t pr = p1ReadyRowsUb.GetValue(b * kP1WeightStride + r);
            DataCopy(packUb[r * h], gmm2OutGm[static_cast<uint64_t>(pr) * h], h);
        }
        SetFlag<HardEvent::MTE2_V>(ev);
        WaitFlag<HardEvent::MTE2_V>(ev);

        const uint64_t gTok = subGlobalTokBase + b;
        __ubuf__ float* wPtr = reinterpret_cast<__ubuf__ float*>(p1CompWeightsUb.GetPhyAddr()) + b * kP1WeightStride;
        if (m == topK) {
            WeightedUnpermuteVf(reinterpret_cast<__ubuf__ bfloat16_t*>(packUb.GetPhyAddr()), wPtr, topK, h);
            SetFlag<HardEvent::V_MTE3>(ev);
            WaitFlag<HardEvent::V_MTE3>(ev);
            // 每个 gTok 在 earlyPartialFp32Ptr 中拥有独立的 2*h 个 bfloat16 空间，前 h 个存最终 BF16
            DataCopy(earlyBf16Gm[gTok * 2ULL * h], packUb, h);
        } else {
            WeightedPartialToFp32Vf(reinterpret_cast<__ubuf__ float*>(outFp32Ub.GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ bfloat16_t*>(packUb.GetPhyAddr()), wPtr, m, h);
            SetFlag<HardEvent::V_MTE3>(ev);
            WaitFlag<HardEvent::V_MTE3>(ev);
            DataCopy(earlyFp32Gm[gTok * h], outFp32Ub, h);
        }
        SetFlag<HardEvent::MTE3_MTE2>(ev);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    SyncFuncStatic<HardEvent::V_S, static_cast<TEventID>(Aiv1AsyncEvents::DispatchDone)>();
    SyncFuncStatic<HardEvent::S_MTE2, static_cast<TEventID>(Aiv1AsyncEvents::DispatchDone)>();

    cursor.tokInChunk += subCnt;
    if (cursor.tokInChunk >= chunkLen) {
        SyncFuncStatic<HardEvent::S_MTE3, static_cast<TEventID>(Aiv1AsyncEvents::DispatchDone)>();
        DataCopyPad(earlyStateGm[globalChunkTokBase], p1StateUb,
                    {1U, chunkLen * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
        SetFlag<HardEvent::MTE3_S>(static_cast<TEventID>(Aiv1AsyncEvents::DispatchDone));
        WaitFlag<HardEvent::MTE3_S>(static_cast<TEventID>(Aiv1AsyncEvents::DispatchDone));
        asc_sync_data_barrier(DSB_DDR);

        auto* earlyFlagPtr = reinterpret_cast<__gm__ int32_t*>(workspace.flagEarlyChunkDonePtr) +
                             (static_cast<uint64_t>(dstRank) * maxChunksPerRank + c) * INT_CACHELINE;
        WriteGmByPassDCache(earlyFlagPtr, static_cast<int32_t>(splitRow) + 1);

        cursor.tokInChunk = 0U;
        cursor.workIdx += tiling.aicNum;
        if (cursor.workIdx >= cursor.totalWorkChunks) {
            cursor.done = true;
        }
    }
    return true;
#else
    (void)tiling;
    (void)rankId;
    (void)workspace;
    (void)cursor;
    (void)timer;
    return false;
#endif
}

/*
 * Phase 3：64 AIV 执行完整 Unpermute + ReduceScatter + Shared 归并。
 *   1. 等待本卡 32 个 AIC 完成全部 Routed GMM2；
 *   2. AIV_UNPERMUTE_SEND：
 *      - 若目标 chunk 已由 Phase 2 P1 预累加（flagEarlyChunkDonePtr > 0），直接复用 FP32 部分和或已完成 BF16，
 *        仅读取 >= splitPermRow 的剩余行做单次 RINT 归并，大幅削减随机 HBM Gather；
 *      - 保持 3 槽 64KiB MTE3_MTE2 节流流水与单 token 8KiB 即时远端发送，避免瞬时 BIU 洪峰饿死同核 AIC_SHARED_GMM2；
 *   3. AIV_RECV_REDUCE：等待全部 TP 分片与共享专家计数就绪后，FP32 归并写出 BF16 yOut。
 */
__aicore__ inline void ProcessTpUnpermuteAndReduceScatter(const MegaMoeTpTilingData& tiling, uint32_t rankId,
                                                          GM_ADDR yOutGm, const TpPeermemInfo& localPeermem,
                                                          const TpWorkspaceInfo& workspace, int32_t epoch,
                                                          bool hasRoutedWork, AscendTimerDevice& timer)
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    if (hasRoutedWork) {
        WaitForAllCoreFlags(workspace.flagRoutedGmm2DonePtr, tiling.aicNum);
    }
    SyncFuncStatic<HardEvent::S_MTE2, EVENT_ID2>();
    SyncFuncStatic<HardEvent::S_V, EVENT_ID2>();

    const uint32_t aivIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t aivNum = tiling.aivNum;
    const uint32_t bs = tiling.bs;
    const uint32_t maxTokens = tiling.numMaxTokensPerRank;
    const uint32_t tpWorldSize = tiling.tpWorldSize;
    const uint32_t topK = tiling.topK;
    const uint32_t h = tiling.h;
    const uint32_t chunkTokens = tiling.rsTokensPerChunk;
    const uint32_t numChunks = Ops::Base::CeilDiv(bs, chunkTokens);
    const uint32_t maxChunksPerRank = Ops::Base::CeilDiv(maxTokens, chunkTokens);
    const bool preQuantWeight = tiling.routeWeightBeforeQuant != 0U;

    // ==================== 阶段 3A：融合 Unpermute + MTE3 Send ====================
    {
        timer.SetWave(AscendTimer::NO_WAVE);
        AscendTimerSpanScope sendSpan(timer.Enabled() ? &timer : nullptr, AscendTimer::AIV_UNPERMUTE_SEND);

        // 256 KiB UB 分区（Phase 3 全核独占完整 UB）：
        //   [0, 8 KiB)       permIdxUb     (64 * 32 * 4B = 8 KiB)
        //   [8, 16 KiB)      weightsUb     (64 * 32 * 4B = 8 KiB)
        //   [16, 24 KiB)     remWeightsUb  (P1 快路径专用: 64 * 32 * 4B = 8 KiB; 慢路径复用为 outRowUb [16, 32 KiB))
        //   [24, 25 KiB)     earlyStateUb  (P1 快路径专用: 256 ints = 1 KiB)
        //   [32, 48 KiB)     fbAccUb       (非 C310 回退专用, 16 KiB)
        //   [48, 64 KiB)     fbTmpUb       (非 C310 回退专用, 16 KiB)
        //   [64, 256 KiB)    packUb 3 槽环 (3 * 64 KiB = 192 KiB，MTE2 || V || MTE3 三级节流流水)
        constexpr uint32_t kSubBatchMaxTokens = 64U;
        constexpr uint32_t kMetaStride = 32U;
        constexpr uint32_t kMetaMaxItems = kSubBatchMaxTokens * kMetaStride;  // 2048 items = 8 KiB
        LocalTensor<int32_t> permIdxUb(TPosition::VECCALC, 0U, kMetaMaxItems);
        LocalTensor<float> weightsUb(TPosition::VECCALC, 8U * 1024U, kMetaMaxItems);
        LocalTensor<float> remWeightsUb(TPosition::VECCALC, 16U * 1024U, kMetaMaxItems);
        LocalTensor<int32_t> earlyStateUb(TPosition::VECCALC, 24U * 1024U, 256U);
        LocalTensor<bfloat16_t> outRowUb(TPosition::VECCALC, 16U * 1024U, 8192U);
        LocalTensor<float> fbAccUb(TPosition::VECCALC, 32U * 1024U, 4096U);
        LocalTensor<float> fbTmpUb(TPosition::VECCALC, 48U * 1024U, 4096U);

        constexpr uint32_t kPackBaseAddr = 64U * 1024U;
        constexpr uint32_t kPackSlotBytes = 96U * 1024U;
        uint32_t colChunk = (kPackSlotBytes / (topK * static_cast<uint32_t>(sizeof(bfloat16_t)))) / 128U * 128U;
        if (colChunk == 0U) {
            colChunk = 64U;
        }
        if (colChunk >= h) {
            colChunk = h;
        }

        if (preQuantWeight || !hasRoutedWork) {
            Duplicate<float>(weightsUb, hasRoutedWork ? 1.0F : 0.0F, static_cast<int32_t>(kMetaMaxItems));
            SyncFuncStatic<HardEvent::V_S, EVENT_ID0>();
        }

        GlobalTensor<int32_t> tokenToPermRowGm, earlyStateGm;
        GlobalTensor<float> agWeightsGm;
        GlobalTensor<bfloat16_t> gmm2OutGm, earlyBf16Gm;
        tokenToPermRowGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.tokenToPermRowPtr));
        agWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace.agTopkWeightsPtr));
        gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspace.gmm2OutPtr));
        gmm2OutGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        earlyStateGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.earlyTokenStatePtr));
        earlyBf16Gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspace.earlyPartialFp32Ptr));
        earlyBf16Gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        for (uint32_t step = 0U; step < tpWorldSize; ++step) {
            uint32_t dstRank = rankId;
            if (step + 1U < tpWorldSize) {
                const uint32_t peerSlot = (aivIdx + step) % (tpWorldSize - 1U);
                dstRank = (rankId + 1U + peerSlot) % tpWorldSize;
            }
            GlobalTensor<bfloat16_t> remRecvGm;
            remRecvGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t*>(g_winRankAddr_[dstRank] + localPeermem.layout.rsRecvBufOffset));

            for (uint32_t c = aivIdx; c < numChunks; c += aivNum) {
                const uint32_t chunkBegin = c * chunkTokens;
                const uint32_t chunkEnd = Min(bs, chunkBegin + chunkTokens);
                const uint32_t chunkLen = chunkEnd - chunkBegin;

                if (!hasRoutedWork) {
                    LocalTensor<bfloat16_t> zeroRowUb(TPosition::VECCALC, kPackBaseAddr, h);
                    Duplicate<bfloat16_t>(zeroRowUb, static_cast<bfloat16_t>(0), static_cast<int32_t>(h));
                    SyncFuncStatic<HardEvent::V_MTE3, EVENT_ID0>();
                    for (uint32_t t = 0U; t < chunkLen; ++t) {
                        const uint64_t dstRow = static_cast<uint64_t>(rankId) * maxTokens + chunkBegin + t;
                        DataCopy(remRecvGm[dstRow * h], zeroRowUb, h);
                    }
                    SyncFuncStatic<HardEvent::MTE3_S, EVENT_ID0>();
                } else if (colChunk == h && (h % 128U == 0U)) {
                    // 快路径：3 槽 64 KiB MTE3_MTE2 节流环 + 融合消费 Phase 2 P1 预累加部分和
                    const uint32_t slotElems = topK * h;
                    const uint32_t slotBytes =
                        static_cast<uint32_t>(TpCeilAlign(static_cast<int64_t>(slotElems) * sizeof(bfloat16_t), ALIGN_256));
                    constexpr uint32_t kPackTotalBytes = 192U * 1024U;
                    const uint32_t numSlots = Min(3U, kPackTotalBytes / slotBytes);

                    const int32_t earlyEncoded = ReadGmByPassDCache(
                        reinterpret_cast<__gm__ int32_t*>(workspace.flagEarlyChunkDonePtr) +
                        (static_cast<uint64_t>(dstRank) * maxChunksPerRank + c) * INT_CACHELINE);
#if defined(__DAV_C310__)
                    const bool useEarlyChunk = (earlyEncoded > 0);
#else
                    const bool useEarlyChunk = false;
#endif
                    const uint32_t splitPermRow = useEarlyChunk ? static_cast<uint32_t>(earlyEncoded - 1) : 0U;
                    if (useEarlyChunk) {
                        SyncFuncStatic<HardEvent::S_MTE2, EVENT_ID3>();
                    }

                    for (uint32_t s = 0U; s < numSlots; ++s) {
                        SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(s));
                    }
                    uint32_t seq = 0U;

                    for (uint32_t subOff = 0U; subOff < chunkLen; subOff += kSubBatchMaxTokens) {
                        const uint32_t subCnt = Min(kSubBatchMaxTokens, chunkLen - subOff);
                        const uint64_t globalTokBase =
                            static_cast<uint64_t>(dstRank) * bs + chunkBegin + subOff;
                        const uint32_t metaBytes = subCnt * topK * static_cast<uint32_t>(sizeof(int32_t));

                        DataCopyPad(permIdxUb, tokenToPermRowGm[globalTokBase * topK], {1U, metaBytes, 0U, 0U, 0U},
                                    {false, 0U, 0U, 0U});
                        if (!preQuantWeight) {
                            DataCopyPad(weightsUb, agWeightsGm[globalTokBase * topK], {1U, metaBytes, 0U, 0U, 0U},
                                        {false, 0U, 0U, 0U});
                            SyncFuncStatic<HardEvent::MTE2_V, EVENT_ID3>();
                        }
                        if (useEarlyChunk) {
                            DataCopyPad(earlyStateUb, earlyStateGm[globalTokBase],
                                        {1U, subCnt * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U},
                                        {false, 0U, 0U, 0U});
                        }
                        SyncFuncStatic<HardEvent::MTE2_S, EVENT_ID3>();

#if defined(__DAV_C310__)
                        bool hasPartialTokens = false;
                        if (useEarlyChunk) {
                            for (uint32_t localT = 0U; localT < subCnt; ++localT) {
                                const int32_t st = earlyStateUb.GetValue(localT);
                                if (st >= 2 && static_cast<uint32_t>(st) < topK) {
                                    hasPartialTokens = true;
                                    uint32_t rem = 0U;
                                    for (uint32_t k = 0U; k < topK; ++k) {
                                        const int32_t pr = permIdxUb.GetValue(localT * topK + k);
                                        if (pr >= 0 && static_cast<uint32_t>(pr) >= splitPermRow) {
                                            remWeightsUb.SetValue(
                                                localT * kMetaStride + rem,
                                                preQuantWeight ? 1.0F : weightsUb.GetValue(localT * topK + k));
                                            ++rem;
                                        }
                                    }
                                }
                            }
                            if (hasPartialTokens) {
                                SyncFuncStatic<HardEvent::S_V, EVENT_ID3>();
                            }
                        }
#endif

                        for (uint32_t localT = 0U; localT < subCnt; ++localT) {
                            const uint32_t slot = seq++ % numSlots;
                            const TEventID ev = static_cast<TEventID>(slot);
                            LocalTensor<bfloat16_t> packUb(
                                TPosition::VECCALC, kPackBaseAddr + slot * slotBytes, slotElems);
                            const uint64_t gTok = globalTokBase + localT;
                            const uint64_t dstRow =
                                static_cast<uint64_t>(rankId) * maxTokens + chunkBegin + subOff + localT;
                            const int32_t st = useEarlyChunk ? earlyStateUb.GetValue(localT) : 0;

                            WaitFlag<HardEvent::MTE3_MTE2>(ev);
#if defined(__DAV_C310__)
                            if (useEarlyChunk && static_cast<uint32_t>(st) == topK) {
                                // 全部 topK 行已在 Phase 2 归并为 BF16，仅需单次连续 8KiB 搬运
                                DataCopy(packUb, earlyBf16Gm[gTok * 2ULL * h], h);
                                SetFlag<HardEvent::MTE2_MTE3>(ev);
                                WaitFlag<HardEvent::MTE2_MTE3>(ev);
                                DataCopy(remRecvGm[dstRow * h], packUb, h);
                                SetFlag<HardEvent::MTE3_MTE2>(ev);
                                continue;
                            }
                            if (useEarlyChunk && st >= 2 && static_cast<uint32_t>(st) < topK) {
                                // 载入 16KiB FP32 部分和 (占 2*h 个 bfloat16 元素) + 剩余 >= splitPermRow 的 BF16 行
                                DataCopy(packUb, earlyBf16Gm[gTok * 2ULL * h], 2U * h);
                                uint32_t rem = 0U;
                                for (uint32_t k = 0U; k < topK; ++k) {
                                    const int32_t permRow = permIdxUb.GetValue(localT * topK + k);
                                    if (permRow >= 0 && static_cast<uint32_t>(permRow) >= splitPermRow) {
                                        DataCopy(packUb[(2U + rem) * h],
                                                 gmm2OutGm[static_cast<uint64_t>(permRow) * h], h);
                                        ++rem;
                                    }
                                }
                                SetFlag<HardEvent::MTE2_V>(ev);
                                WaitFlag<HardEvent::MTE2_V>(ev);
                                WeightedFinishFromFp32Vf(
                                    reinterpret_cast<__ubuf__ bfloat16_t*>(packUb.GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ bfloat16_t*>(packUb[2U * h].GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ float*>(remWeightsUb.GetPhyAddr()) +
                                        localT * kMetaStride,
                                    rem, h);
                                SetFlag<HardEvent::V_MTE3>(ev);
                                WaitFlag<HardEvent::V_MTE3>(ev);
                                DataCopy(remRecvGm[dstRow * h], packUb, h);
                                SetFlag<HardEvent::MTE3_MTE2>(ev);
                                continue;
                            }
#endif
                            for (uint32_t k = 0U; k < topK; ++k) {
                                const int32_t permRow = permIdxUb.GetValue(localT * topK + k);
                                DataCopy(packUb[k * h], gmm2OutGm[static_cast<uint64_t>(permRow) * h], h);
                            }
                            SetFlag<HardEvent::MTE2_V>(ev);
                            WaitFlag<HardEvent::MTE2_V>(ev);

#if defined(__DAV_C310__)
                            WeightedUnpermuteVf(
                                reinterpret_cast<__ubuf__ bfloat16_t*>(packUb.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float*>(weightsUb.GetPhyAddr()) + localT * topK, topK, h);
#else
                            WeightedSumColumnsFallback(packUb, fbAccUb, fbTmpUb, weightsUb, localT * topK, topK, h,
                                                       !preQuantWeight);
#endif
                            SetFlag<HardEvent::V_MTE3>(ev);
                            WaitFlag<HardEvent::V_MTE3>(ev);

                            DataCopy(remRecvGm[dstRow * h], packUb, h);
                            SetFlag<HardEvent::MTE3_MTE2>(ev);
                        }
                        SyncFuncStatic<HardEvent::S_MTE2, EVENT_ID3>();
                        if (!preQuantWeight) {
                            SyncFuncStatic<HardEvent::V_MTE2, EVENT_ID3>();
                        }
#if defined(__DAV_C310__)
                        if (hasPartialTokens) {
                            SyncFuncStatic<HardEvent::V_S, EVENT_ID3>();
                        }
#endif
                    }
                    for (uint32_t s = 0U; s < numSlots; ++s) {
                        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(s));
                    }
                    SyncFuncStatic<HardEvent::MTE3_S, EVENT_ID0>();
                } else {
                    // 通用分列路径（兼容 topK > 12 或 H > 4096 或 H % 128 != 0）：组装完整 outRowUb 后整行推送
                    for (uint32_t subOff = 0U; subOff < chunkLen; subOff += kSubBatchMaxTokens) {
                        const uint32_t subCnt = Min(kSubBatchMaxTokens, chunkLen - subOff);
                        const uint64_t globalTokBase =
                            static_cast<uint64_t>(dstRank) * bs + chunkBegin + subOff;
                        const uint32_t metaBytes = subCnt * topK * static_cast<uint32_t>(sizeof(int32_t));
                        DataCopyPad(permIdxUb, tokenToPermRowGm[globalTokBase * topK], {1U, metaBytes, 0U, 0U, 0U},
                                    {false, 0U, 0U, 0U});
                        if (!preQuantWeight) {
                            DataCopyPad(weightsUb, agWeightsGm[globalTokBase * topK], {1U, metaBytes, 0U, 0U, 0U},
                                        {false, 0U, 0U, 0U});
                        }
                        SyncFuncStatic<HardEvent::MTE2_S, EVENT_ID2>();
                        SyncFuncStatic<HardEvent::MTE2_V, EVENT_ID2>();

                        for (uint32_t localT = 0U; localT < subCnt; ++localT) {
                            for (uint32_t ch = 0U; ch < h; ch += colChunk) {
                                const uint32_t cols = Min(colChunk, h - ch);
                                LocalTensor<bfloat16_t> packUb(TPosition::VECCALC, kPackBaseAddr, topK * cols);
                                for (uint32_t k = 0U; k < topK; ++k) {
                                    const int32_t permRow = permIdxUb.GetValue(localT * topK + k);
                                    DataCopy(packUb[k * cols], gmm2OutGm[static_cast<uint64_t>(permRow) * h + ch],
                                             cols);
                                }
                                SyncFuncStatic<HardEvent::MTE2_V, EVENT_ID0>();
#if defined(__DAV_C310__)
                                if (cols % 128U == 0U) {
                                    WeightedUnpermuteVf(
                                        reinterpret_cast<__ubuf__ bfloat16_t*>(packUb.GetPhyAddr()),
                                        reinterpret_cast<__ubuf__ float*>(weightsUb.GetPhyAddr()) + localT * topK, topK,
                                        cols);
                                } else
#endif
                                {
                                    WeightedSumColumnsFallback(packUb, fbAccUb, fbTmpUb, weightsUb, localT * topK, topK,
                                                               cols, !preQuantWeight);
                                }
                                SyncFuncStatic<HardEvent::V_MTE3, EVENT_ID0>();
                                DataCopy(outRowUb[ch], packUb, cols);
                                SyncFuncStatic<HardEvent::MTE3_MTE2, EVENT_ID0>();
                            }
                            const uint64_t dstRow =
                                static_cast<uint64_t>(rankId) * maxTokens + chunkBegin + subOff + localT;
                            DataCopy(remRecvGm[dstRow * h], outRowUb, h);
                            SyncFuncStatic<HardEvent::MTE3_MTE2, EVENT_ID0>();
                        }
                    }
                    SyncFuncStatic<HardEvent::MTE3_S, EVENT_ID0>();
                }

                // 单写者发布目标卡对应 (rankId, c) 的 epoch 就绪标志
                asc_sync_data_barrier(DSB_DDR);
                auto* remoteFlag = reinterpret_cast<__gm__ int32_t*>(g_winRankAddr_[dstRank] +
                                                                     localPeermem.layout.rsChunkReadyOffset) +
                                   (static_cast<uint64_t>(rankId) * maxChunksPerRank + c) * INT_CACHELINE;
                WriteGmByPassDCache(remoteFlag, epoch);
                asc_sync_data_barrier(DSB_DDR);
            }
        }
    }

    // ==================== 阶段 3B：Wait + Local-Reduce（含共享专家归并） ====================
    {
        timer.SetWave(AscendTimer::NO_WAVE);
        AscendTimerSpanScope reduceSpan(timer.Enabled() ? &timer : nullptr, AscendTimer::AIV_RECV_REDUCE);

        const uint32_t sharedNum = tiling.sharedExpertNum;
        const uint32_t numTerms = tpWorldSize + sharedNum;
        const uint32_t sharedMGroups = Ops::Base::CeilDiv(bs, L1_TILE_M_256);
        const int32_t expectedSharedNTiles = static_cast<int32_t>(Ops::Base::CeilDiv(h, L1_TILE_N));

        GlobalTensor<bfloat16_t> localRecvGm, sharedOutGm, yGm;
        localRecvGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(localPeermem.rsRecvBufPtr));
        localRecvGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        if (sharedNum > 0U) {
            sharedOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspace.sharedGmm2OutPtr));
            sharedOutGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(yOutGm));

        constexpr uint32_t kReduceSlotBytes = 96U * 1024U;
        uint32_t redColChunk = (kReduceSlotBytes / (numTerms * static_cast<uint32_t>(sizeof(bfloat16_t)))) / 128U * 128U;
        if (redColChunk == 0U) {
            redColChunk = 64U;
        }
        if (redColChunk >= h) {
            redColChunk = h;
        }

        LocalTensor<float> redAccUb(TPosition::VECCALC, 0U, 4096U);
        LocalTensor<float> redTmpUb(TPosition::VECCALC, 16U * 1024U, 4096U);
        LocalTensor<float> dummyWeightsUb;
        constexpr uint32_t kReducePackBase = 32U * 1024U;

        for (uint32_t c = aivIdx; c < numChunks; c += aivNum) {
            const uint32_t tokBegin = c * chunkTokens;
            const uint32_t tokEnd = Min(bs, tokBegin + chunkTokens);

            // 等待本 chunk 的全部 TP 分片及覆盖的共享专家 256-row group 就绪
            {
                AscendTimerWaitScope wait(timer.Enabled() ? &timer : nullptr, AscendTimer::AIV_RECV_REDUCE_WAIT);
                auto* localReadyBase = reinterpret_cast<__gm__ int32_t*>(localPeermem.rsChunkReadyPtr);
                for (uint32_t srcRank = 0U; srcRank < tpWorldSize; ++srcRank) {
                    WaitUntilGmFlagEquals(
                        localReadyBase + (static_cast<uint64_t>(srcRank) * maxChunksPerRank + c) * INT_CACHELINE,
                        epoch);
                }
                if (sharedNum > 0U) {
                    const uint32_t firstGroup = tokBegin / L1_TILE_M_256;
                    const uint32_t lastGroup = (tokEnd - 1U) / L1_TILE_M_256;
                    auto* sharedCntBase = reinterpret_cast<__gm__ int32_t*>(workspace.sharedGmm2TileCounterPtr);
                    for (uint32_t s = 0U; s < sharedNum; ++s) {
                        for (uint32_t g = firstGroup; g <= lastGroup; ++g) {
                            WaitUntilGmFlagEquals(
                                sharedCntBase + (static_cast<uint64_t>(s) * sharedMGroups + g) * INT_CACHELINE,
                                expectedSharedNTiles);
                        }
                    }
                }
            }
            SyncFuncStatic<HardEvent::S_MTE2, EVENT_ID2>();

            // 双缓冲 FP32 Local-Reduce + 写出 yOut
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            uint32_t seq = 0U;
            for (uint32_t tok = tokBegin; tok < tokEnd; ++tok) {
                for (uint32_t ch = 0U; ch < h; ch += redColChunk) {
                    const uint32_t cols = Min(redColChunk, h - ch);
                    const uint32_t slot = seq++ % 2U;
                    const TEventID ev = slot == 0U ? EVENT_ID0 : EVENT_ID1;
                    LocalTensor<bfloat16_t> rowsUb(
                        TPosition::VECCALC, kReducePackBase + slot * kReduceSlotBytes, numTerms * cols);

                    WaitFlag<HardEvent::MTE3_MTE2>(ev);
                    for (uint32_t srcRank = 0U; srcRank < tpWorldSize; ++srcRank) {
                        const uint64_t srcRow = static_cast<uint64_t>(srcRank) * maxTokens + tok;
                        DataCopy(rowsUb[srcRank * cols], localRecvGm[srcRow * h + ch], cols);
                    }
                    for (uint32_t s = 0U; s < sharedNum; ++s) {
                        const uint64_t shRow = static_cast<uint64_t>(s) * bs + tok;
                        DataCopy(rowsUb[(tpWorldSize + s) * cols], sharedOutGm[shRow * h + ch], cols);
                    }
                    SetFlag<HardEvent::MTE2_V>(ev);
                    WaitFlag<HardEvent::MTE2_V>(ev);

#if defined(__DAV_C310__)
                    if (cols % 128U == 0U) {
                        asc_vf_call<SumBf16RowsToBf16Vf>(
                            reinterpret_cast<__ubuf__ bfloat16_t*>(rowsUb.GetPhyAddr()), numTerms, cols);
                    } else
#endif
                    {
                        WeightedSumColumnsFallback(rowsUb, redAccUb, redTmpUb, dummyWeightsUb, 0U, numTerms, cols,
                                                   false);
                    }
                    SetFlag<HardEvent::V_MTE3>(ev);
                    WaitFlag<HardEvent::V_MTE3>(ev);

                    DataCopy(yGm[static_cast<uint64_t>(tok) * h + ch], rowsUb, cols);
                    SetFlag<HardEvent::MTE3_MTE2>(ev);
                }
            }
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            SyncFuncStatic<HardEvent::MTE3_S, EVENT_ID0>();
        }
    }
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_TP_UNPERMUTE_RS_HPP
