// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

/*!
 * \file mega_moe_tp_comm_route.hpp
 * \brief TP MegaMoE Phase 0 输入准备、Phase 1 AllGather + 全局确定性路由表构建、Phase 2 Wave 级 Permute。
 */

#ifndef MEGA_MOE_TP_COMM_ROUTE_HPP
#define MEGA_MOE_TP_COMM_ROUTE_HPP

#include <c_api/asc_simd.h>
#if defined(__NPU_DEVICE__)
#include "shmem.h"
#endif
#include "../common/mega_moe_tp_layout.hpp"
#include "../common/mega_moe_utils.hpp"
#include "../common/mega_moe_mxfp8_utils.hpp"
#include "../common/aiv1_async_events.hpp"
#include "../common/ascend_timer_v2_device.hpp"

namespace MegaMoeImpl {

using namespace AscendC;

// 子核组软同步屏障：组内每个逻辑核独占 1 个 64B cache line，按单调递增 phase 同步。
__aicore__ inline void SyncTpSubgroup(GM_ADDR syncBasePtr, uint32_t subIdx, uint32_t subCount, int32_t& phase)
{
    ++phase;
    if (subCount <= 1U) {
        return;
    }
    auto* slots = reinterpret_cast<__gm__ int32_t*>(syncBasePtr);
    WriteGmByPassDCache(slots + static_cast<uint64_t>(subIdx) * INT_CACHELINE, phase);
    for (uint32_t peer = 0U; peer < subCount; ++peer) {
        while (ReadGmByPassDCache(slots + static_cast<uint64_t>(peer) * INT_CACHELINE) < phase) {
            const int64_t startCycle = GetSystemCycle();
            while (GetSystemCycle() - startCycle < GM_FLAG_POLL_BACKOFF_CYCLES) {
            }
        }
    }
    SyncFuncStatic<HardEvent::S_MTE2, EVENT_ID0>();
    SyncFuncStatic<HardEvent::S_V, EVENT_ID0>();
}

/*
 * Phase 0：软同步通知区一次性清零 + 本卡输入准备。
 *   - 64 AIV 先并行清零连续 flag 区，完成后立刻与 32 AIC 做全核 SyncAll<false>()，
 *     使 AIC 在 BF16 共享专家直读场景下可零等待立刻启动 Shared GMM1（t ≈ 1us）；
 *   - 随后 64 AIV 并行完成本卡 x 量化/拷贝与 topk 拷贝，AIV 组内 SyncAll<true>() 后由 aivIdx=0
 *     发布 flagLocalPrepDonePtr，供 FP8 共享专家或 FP8->BF16 反量化路径的 AIC 软同步等待；
 *   - 本地 token 同时写入本卡对称窗口 agSrc*（供远端 MTE2 拉取）与本地 workspace.ag*（免去 Phase 1 自拷）。
 */
__aicore__ inline void ProcessTpInputPrep(const MegaMoeTpTilingData& tiling, uint32_t rankId, GM_ADDR xGm,
                                          GM_ADDR xScalesGm, GM_ADDR topkIdsGm, GM_ADDR topkWeightsGm,
                                          const TpPeermemInfo& localPeermem, const TpWorkspaceInfo& workspace,
                                          AscendTimerDevice& timer)
{
    if constexpr (g_coreType == AIC) {
        timer.Start(AscendTimer::AIC_WAIT_INPUT_PREP);
        SyncAll<false>();
        timer.End(AscendTimer::AIC_WAIT_INPUT_PREP);
        return;
    }

    const uint32_t aivIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t aivNum = tiling.aivNum;
    const AivJobContext aivJob{aivIdx, aivNum};

    // 1. 清零连续软同步区，完成后立刻放行 AIC（使 BF16 Shared GMM1 提前与 AIV_QUANT + CrossRankSync + AllGather 全重叠）
    timer.Start(AscendTimer::AIV_RESET);
    constexpr int32_t kResetBatchInts = 4096;  // 16 KiB UB
    LocalTensor<int32_t> zeroUb(TPosition::VECCALC, 0U, kResetBatchInts);
    Duplicate<int32_t>(zeroUb, 0, kResetBatchInts);
    SyncFuncStatic<HardEvent::V_MTE3, EVENT_ID0>();
    ResetWorkspaceRegion<INT_CACHELINE>(aivJob, workspace.flagRegionBeginPtr,
                                        static_cast<int32_t>(workspace.flagResetElementCount), kResetBatchInts, zeroUb);
    SyncFuncStatic<HardEvent::MTE3_V, EVENT_ID0>();
    SyncFuncStatic<HardEvent::MTE3_MTE2, EVENT_ID0>();
    SyncFuncStatic<HardEvent::MTE3_S, EVENT_ID0>();
    asc_sync_data_barrier(DSB_DDR);
    PipeBarrier<PIPE_ALL>();
    timer.End(AscendTimer::AIV_RESET);
    SyncAll<false>();

    // 2. 准备本卡负责的 bs 子区间
    const WorkRange myTokens = GetBalancedTokenRange(tiling.bs, aivIdx, aivNum);
    if (myTokens.count > 0U) {
            timer.Start(AscendTimer::AIV_QUANT);
            const uint32_t h = tiling.h;
            const uint32_t scaleH = static_cast<uint32_t>(CalcTpMxScaleStrideBytes(h));
            const uint32_t scaleAlign = static_cast<uint32_t>(TpCeilAlign(scaleH, ALIGN_32));
            const uint32_t topK = tiling.topK;
            const uint64_t localGlobalTokenBase = static_cast<uint64_t>(rankId) * tiling.bs + myTokens.start;

            // 2a. 拷贝本地 topk_ids 与 topk_weights 到 peermem 与 workspace
            {
                GlobalTensor<int32_t> srcIdsGm, dstPeerIdsGm, dstWsIdsGm;
                GlobalTensor<float> srcWeightsGm, dstPeerWeightsGm, dstWsWeightsGm;
                srcIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(topkIdsGm));
                dstPeerIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localPeermem.agSrcTopkIdsPtr));
                dstWsIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.agTopkIdsPtr));
                srcWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(topkWeightsGm));
                dstPeerWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(localPeermem.agSrcTopkWeightsPtr));
                dstWsWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace.agTopkWeightsPtr));

                constexpr uint32_t kRouteBatchTokens = 512U;
                const uint32_t maxItems = kRouteBatchTokens * 32U;  // topK <= 32 -> 64 KiB each
                LocalTensor<int32_t> idsUb(TPosition::VECCALC, 0U, maxItems);
                LocalTensor<float> weightsUb(TPosition::VECCALC, maxItems * sizeof(int32_t), maxItems);

                for (uint32_t off = 0U; off < myTokens.count; off += kRouteBatchTokens) {
                    const uint32_t cnt = Min(kRouteBatchTokens, myTokens.count - off);
                    const uint32_t bytes = cnt * topK * static_cast<uint32_t>(sizeof(int32_t));
                    const uint64_t srcElem = static_cast<uint64_t>(myTokens.start + off) * topK;
                    const uint64_t wsElem = (localGlobalTokenBase + off) * topK;
                    DataCopyPad(idsUb, srcIdsGm[srcElem], {1U, bytes, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                    DataCopyPad(weightsUb, srcWeightsGm[srcElem], {1U, bytes, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                    SyncFuncStatic<HardEvent::MTE2_MTE3, EVENT_ID0>();
                    DataCopyPad(dstPeerIdsGm[srcElem], idsUb, {1U, bytes, 0U, 0U, 0U});
                    DataCopyPad(dstWsIdsGm[wsElem], idsUb, {1U, bytes, 0U, 0U, 0U});
                    DataCopyPad(dstPeerWeightsGm[srcElem], weightsUb, {1U, bytes, 0U, 0U, 0U});
                    DataCopyPad(dstWsWeightsGm[wsElem], weightsUb, {1U, bytes, 0U, 0U, 0U});
                    SyncFuncStatic<HardEvent::MTE3_MTE2, EVENT_ID0>();
                }
                SyncFuncStatic<HardEvent::MTE3_V, EVENT_ID0>();
                SyncFuncStatic<HardEvent::MTE3_S, EVENT_ID0>();
            }

        // 2b. 处理本地 x / x_scales（按紧致单记录 [Align256(H) | Align32(scaleH)] 单写落地到对称窗口 agXDataPtr）
        const uint32_t dataStride = static_cast<uint32_t>(CalcTpAgTokenDataStrideBytes(h));
        const uint32_t recordBytes = static_cast<uint32_t>(CalcTpAgTokenRecordBytes(h));
        GlobalTensor<uint8_t> dstSymRecordGm;
        dstSymRecordGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(localPeermem.agXDataPtr));

        if (xScalesGm == nullptr) {
            // BF16 输入：逐 token 双缓冲 MXFP8 量化，UB 内 [outDataUb | outScaleUb] 天然紧邻构成单条 recordBytes 记录
            GlobalTensor<bfloat16_t> srcBf16Gm;
            srcBf16Gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(xGm));

            const bool writeSharedPlanarFp8 =
                (tiling.sharedExpertNum > 0U) &&
                (tiling.sharedPrecisionMode == static_cast<uint8_t>(SharedPrecisionMode::FP8_MX));
            GlobalTensor<uint8_t> dstShFp8DataGm, dstShFp8ScaleGm;
            if (writeSharedPlanarFp8) {
                dstShFp8DataGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(workspace.sharedBf16InputPtr));
                dstShFp8ScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
                    workspace.sharedBf16InputPtr + TpCeilAlign(static_cast<int64_t>(tiling.bs) * h, ALIGN_512)));
            }

            const uint32_t xInBytes = static_cast<uint32_t>(TpCeilAlign(h, ALIGN_128)) * sizeof(bfloat16_t);
            const uint32_t xOutDataBytes = dataStride;
            const uint32_t xOutScaleBytes = scaleAlign;
            const uint32_t mxTempBytes = 4096U;

            uint32_t ubCursor = 0U;
            const uint32_t mxTempAddr = ubCursor;
            ubCursor += mxTempBytes;
            const uint32_t xInAddr0 = ubCursor;
            ubCursor += xInBytes;
            const uint32_t xInAddr1 = ubCursor;
            ubCursor += xInBytes;
            const uint32_t xOutDataAddr0 = ubCursor;
            ubCursor += xOutDataBytes;
            const uint32_t xOutScaleAddr0 = ubCursor;
            ubCursor += xOutScaleBytes;
            const uint32_t xOutDataAddr1 = ubCursor;
            ubCursor += xOutDataBytes;
            const uint32_t xOutScaleAddr1 = ubCursor;
            ubCursor += xOutScaleBytes;

            LocalTensor<int16_t> scratchSpan(TPosition::VECCALC, 0U, ubCursor / sizeof(int16_t));
            Duplicate<int16_t>(scratchSpan, 0, static_cast<int32_t>(ubCursor / sizeof(int16_t)));
            SyncFuncStatic<HardEvent::V_MTE2, EVENT_ID2>();

            LocalTensor<uint16_t> mxTempUb(TPosition::VECCALC, mxTempAddr, mxTempBytes / sizeof(uint16_t));
            __ubuf__ uint16_t* maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(mxTempUb.GetPhyAddr());
            const uint32_t scaleNumAlign = static_cast<uint32_t>(TpCeilAlign(TpCeilDiv(h, ALIGN_32), 2));
            __ubuf__ uint16_t* halfScaleAddr =
                reinterpret_cast<__ubuf__ uint16_t*>(mxTempUb[TpCeilAlign(scaleNumAlign, ALIGN_32)].GetPhyAddr());

            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            for (uint32_t idx = 0U; idx < myTokens.count; ++idx) {
                const bool first = (idx % 2U == 0U);
                const TEventID ev = first ? EVENT_ID0 : EVENT_ID1;
                const uint32_t recAddr = first ? xOutDataAddr0 : xOutDataAddr1;
                LocalTensor<bfloat16_t> xInUb(TPosition::VECCALC, first ? xInAddr0 : xInAddr1,
                                              xInBytes / sizeof(bfloat16_t));
                LocalTensor<uint8_t> outRecordUb(TPosition::VECCALC, recAddr, recordBytes);
                LocalTensor<uint8_t> outDataUb(TPosition::VECCALC, recAddr, xOutDataBytes);
                LocalTensor<uint8_t> outScaleUb(TPosition::VECCALC, first ? xOutScaleAddr0 : xOutScaleAddr1,
                                                xOutScaleBytes);

                WaitFlag<HardEvent::MTE3_MTE2>(ev);
                const uint64_t localTok = myTokens.start + idx;
                const uint64_t globalTok = localGlobalTokenBase + idx;
                DataCopyPad(xInUb, srcBf16Gm[localTok * h],
                            {1U, static_cast<uint32_t>(h * sizeof(bfloat16_t)), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                SetFlag<HardEvent::MTE2_V>(ev);
                WaitFlag<HardEvent::MTE2_V>(ev);

                Mxfp8::ComputeFp8Token<bfloat16_t, fp8_e4m3fn_t>(
                    reinterpret_cast<__ubuf__ bfloat16_t*>(xInUb.GetPhyAddr()), maxExpAddr,
                    reinterpret_cast<__ubuf__ uint16_t*>(outScaleUb.GetPhyAddr()), halfScaleAddr,
                    reinterpret_cast<__ubuf__ int8_t*>(outDataUb.GetPhyAddr()), h, scaleNumAlign);

                SetFlag<HardEvent::V_MTE3>(ev);
                WaitFlag<HardEvent::V_MTE3>(ev);
                DataCopy(dstSymRecordGm[globalTok * recordBytes], outRecordUb, recordBytes);
                if (writeSharedPlanarFp8) {
                    DataCopyPad(dstShFp8DataGm[localTok * h], outDataUb, {1U, h, 0U, 0U, 0U});
                    DataCopyPad(dstShFp8ScaleGm[localTok * scaleH], outScaleUb, {1U, scaleH, 0U, 0U, 0U});
                }
                SetFlag<HardEvent::MTE3_MTE2>(ev);
            }
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        } else {
            // 预量化 (FP8, E8M0) 输入：在 UB 双缓冲组装为紧致单记录后单次连续写入对称窗口
            GlobalTensor<uint8_t> srcDataGm, srcScaleGm;
            srcDataGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(xGm));
            srcScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(xScalesGm));

            constexpr uint32_t kRecSlotCap = 112U * 1024U;
            const uint32_t batchTokens = Max(1U, kRecSlotCap / recordBytes);
            const bool aligned32In = (h % ALIGN_32 == 0U) && (scaleH % ALIGN_32 == 0U) && (h == dataStride) &&
                                     (scaleH == scaleAlign);

            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            uint32_t seq = 0U;
            for (uint32_t off = 0U; off < myTokens.count; off += batchTokens) {
                const uint32_t cnt = Min(batchTokens, myTokens.count - off);
                const uint32_t slot = seq++ % 2U;
                const TEventID ev = slot == 0U ? EVENT_ID0 : EVENT_ID1;
                LocalTensor<uint8_t> recUb(TPosition::VECCALC, slot * kRecSlotCap, cnt * recordBytes);

                WaitFlag<HardEvent::MTE3_MTE2>(ev);
                const uint64_t localTokBase = myTokens.start + off;
                const uint64_t globalTokBase = localGlobalTokenBase + off;
                for (uint32_t t = 0U; t < cnt; ++t) {
                    const uint64_t lTok = localTokBase + t;
                    const uint32_t dstOff = t * recordBytes;
                    if (aligned32In) {
                        DataCopy(recUb[dstOff], srcDataGm[lTok * h], h);
                        DataCopy(recUb[dstOff + dataStride], srcScaleGm[lTok * scaleH], scaleH);
                    } else {
                        DataCopyPad(recUb[dstOff], srcDataGm[lTok * h], {1U, h, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                        DataCopyPad(recUb[dstOff + dataStride], srcScaleGm[lTok * scaleH], {1U, scaleH, 0U, 0U, 0U},
                                    {false, 0U, 0U, 0U});
                    }
                }
                SetFlag<HardEvent::MTE2_MTE3>(ev);
                WaitFlag<HardEvent::MTE2_MTE3>(ev);
                DataCopy(dstSymRecordGm[globalTokBase * recordBytes], recUb, cnt * recordBytes);
                SetFlag<HardEvent::MTE3_MTE2>(ev);
            }
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

                // 若共享专家使用 BF16 而输入 x 是预量化 FP8，反量化本卡 x 到 sharedBf16InputPtr
                if (tiling.sharedExpertNum > 0U &&
                    tiling.sharedPrecisionMode == static_cast<uint8_t>(SharedPrecisionMode::BF16)) {
                    GlobalTensor<bfloat16_t> dstSharedBf16Gm;
                    dstSharedBf16Gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspace.sharedBf16InputPtr));
                    const uint32_t tokAligned = static_cast<uint32_t>(TpCeilAlign(h, ALIGN_256));
                    const uint32_t inRecordBytes = tokAligned + scaleAlign;
                    uint32_t dqCursor = 0U;
                    LocalTensor<uint8_t> inBytesUb(TPosition::VECCALC, dqCursor, inRecordBytes);
                    LocalTensor<bfloat16_t> inBf16View(TPosition::VECCALC, dqCursor, inRecordBytes / sizeof(bfloat16_t));
                    dqCursor += static_cast<uint32_t>(TpCeilAlign(inRecordBytes, ALIGN_256));
                    const uint32_t scaleBf16Bytes =
                        static_cast<uint32_t>(TpCeilAlign(scaleH * 2U * sizeof(bfloat16_t), ALIGN_256));
                    LocalTensor<bfloat16_t> scaleBf16Ub(TPosition::VECCALC, dqCursor, scaleBf16Bytes / sizeof(bfloat16_t));
                    dqCursor += scaleBf16Bytes;
                    const uint32_t scaleFp32Bytes =
                        static_cast<uint32_t>(TpCeilAlign(scaleH * 4U * sizeof(float), ALIGN_256));
                    LocalTensor<float> scaleFp32Ub(TPosition::VECCALC, dqCursor, scaleFp32Bytes / sizeof(float));
                    dqCursor += scaleFp32Bytes;
                    LocalTensor<float> sumFp32Ub(TPosition::VECCALC, dqCursor, h);
                    dqCursor += h * static_cast<uint32_t>(sizeof(float));
                    LocalTensor<bfloat16_t> outBf16Ub(TPosition::VECCALC, dqCursor, h);

                    for (uint32_t idx = 0U; idx < myTokens.count; ++idx) {
                        const uint64_t localTok = myTokens.start + idx;
                        DataCopyPad(inBytesUb, srcDataGm[localTok * h], {1U, h, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                        DataCopyPad(inBytesUb[tokAligned], srcScaleGm[localTok * scaleH], {1U, scaleH, 0U, 0U, 0U},
                                    {false, 0U, 0U, 0U});
                        SyncFuncStatic<HardEvent::MTE2_V, EVENT_ID0>();
                        Mxfp8::DeQuantMxFp8<fp8_e4m3fn_t, bfloat16_t>(inBf16View, sumFp32Ub, scaleBf16Ub, scaleFp32Ub,
                                                                      scaleH, h);
                        PipeBarrier<PIPE_V>();
                        Cast(outBf16Ub, sumFp32Ub, RoundMode::CAST_RINT, h);
                        SyncFuncStatic<HardEvent::V_MTE3, EVENT_ID0>();
                        DataCopy(dstSharedBf16Gm[localTok * h], outBf16Ub, h);
                        SyncFuncStatic<HardEvent::MTE3_MTE2, EVENT_ID0>();
                    }
                }
            }
        timer.End(AscendTimer::AIV_QUANT);
    }
    SyncFuncStatic<HardEvent::MTE3_S, EVENT_ID0>();
    asc_sync_data_barrier(DSB_DDR);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    if (aivIdx == 0U) {
        WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(workspace.flagLocalPrepDonePtr), 1);
    }
}

/*
 * Phase 1：64 AIV 跨卡握手 + 每 AICore 内部 AIV0/AIV1 对称分工：
 *   - AllGather 组 (每 AICore 的 AIV0，共 aicNum=32 核)：按 peer 均衡划分（TP4 下 11/11/10），
 *     单 AIV0 固定拉取单一远端 rank 的 FP8 data + E8M0 scale，避免同 AICore 双 AIV 并发抢占 BIU/MTE2；
 *   - RouteTable 组 (每 AICore 的 AIV1，共 aicNum=32 核)：拉取远端 topk_ids/weights + 三趟确定性
 *     Histogram / PrefixSum / Scatter（零原子争用）。
 * 返回本轮跨卡握手的 1-based 单调递增 epoch，供 Phase 3 免清零 rsChunkReady 标志使用。
 */
__aicore__ inline int32_t ProcessTpAllGatherAndRoute(const MegaMoeTpTilingData& tiling, uint32_t rankId,
                                                     GM_ADDR expertTokenNumsOutGm, const TpPeermemInfo& localPeermem,
                                                     const TpWorkspaceInfo& workspace, AscendTimerDevice& timer)
{
    if constexpr (g_coreType == AIC) {
        return 0;
    }
    const uint32_t aivIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t aivNum = tiling.aivNum;
    const uint32_t tpWorldSize = tiling.tpWorldSize;
    const AivJobContext aivJob{aivIdx, aivNum};

    timer.Start(AscendTimer::AIV_INPUT_CROSS_RANK_SYNC);
    asc_sync_data_barrier(DSB_DDR);
    auto* syncCount = reinterpret_cast<__gm__ int32_t*>(localPeermem.rankSyncInWorldPtr +
                                                        RANK_SYNC_COUNTER_OFFSET_BYTES +
                                                        static_cast<uint64_t>(aivIdx) * RANK_SYNC_COUNTER_SLOT_BYTES);
    const int32_t epoch =
        CrossRankHandshakeInWorldSize(localPeermem.rankSyncInWorldPtr, rankId, tpWorldSize, aivJob, syncCount);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    timer.End(AscendTimer::AIV_INPUT_CROSS_RANK_SYNC);

    const uint32_t bs = tiling.bs;
    const uint32_t h = tiling.h;
    const uint32_t scaleH = static_cast<uint32_t>(CalcTpMxScaleStrideBytes(h));
    const uint32_t scaleAlign = static_cast<uint32_t>(TpCeilAlign(scaleH, ALIGN_32));

    if (GetSubBlockIdx() == 0U) {
        // ==================== AllGather 子组（AIV0：默认 URMA 多 QP 推送，零占用 AICore MTE2/BIU 带宽） ====================
        timer.SetWave(AscendTimer::NO_WAVE);
        AscendTimerSpanScope agSpan(timer.Enabled() ? &timer : nullptr, AscendTimer::AIV_ALLGATHER);

        const uint32_t agIdx = aivIdx / 2U;
        const uint32_t agTotalCores = tiling.aicNum;
        const uint32_t numPeers = tpWorldSize - 1U;
        const uint32_t recordBytes = static_cast<uint32_t>(CalcTpAgTokenRecordBytes(h));

        if (tiling.useUrmaAllGather != 0U) {
#if defined(__NPU_DEVICE__)
            const uint32_t cfgQps =
                tiling.udmaQpsPerPeer > 0U ? static_cast<uint32_t>(tiling.udmaQpsPerPeer) : 1U;
            const uint32_t qpsPerPeer = Min(cfgQps, Max(1U, agTotalCores / numPeers));
            if (agIdx < numPeers * qpsPerPeer) {
                aclshmemx_instance_ctx_set(tiling.shmemInstanceId);
                const uint32_t peerSlot = agIdx / qpsPerPeer;
                const uint32_t qpIdx = agIdx % qpsPerPeer;
                const uint32_t peer = (rankId + 1U + peerSlot) % tpWorldSize;
                const WorkRange myRange = GetBalancedTokenRange(bs, qpIdx, qpsPerPeer);

                if (myRange.count > 0U) {
                    __gm__ uint8_t* symSlicePtr =
                        reinterpret_cast<__gm__ uint8_t*>(localPeermem.agXDataPtr) +
                        (static_cast<uint64_t>(rankId) * bs + myRange.start) * static_cast<uint64_t>(recordBytes);
                    constexpr uint64_t kMaxSinglePutBytes = 64ULL * 1024ULL * 1024ULL;
                    const uint64_t totalBytes = static_cast<uint64_t>(myRange.count) * recordBytes;
                    uint64_t sent = 0ULL;
                    while (sent < totalBytes) {
                        const uint32_t chunk = static_cast<uint32_t>(
                            (totalBytes - sent) > kMaxSinglePutBytes ? kMaxSinglePutBytes : (totalBytes - sent));
                        aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_S, ACLSHMEMX_UDMA_OP_CONFIG_DEFAULT>(
                            symSlicePtr + sent, symSlicePtr + sent, nullptr, chunk, static_cast<int>(peer), qpIdx, 0U);
                        sent += chunk;
                    }
                    aclshmemx_udma_qp_quiet(static_cast<int>(peer), qpIdx);
                }

                // 本 (rankId, qpIdx) 的 UDMA 写入在目标卡 HBM 确认完成后，发布单调递增 epoch 通知
                auto* remDoneSlot = reinterpret_cast<__gm__ int32_t*>(
                                        g_winRankAddr_[peer] + localPeermem.layout.agPeerDoneOffset) +
                                    (static_cast<uint64_t>(rankId) * TP_UDMA_MAX_QPS_PER_PEER + qpIdx) * INT_CACHELINE;
                WriteGmByPassDCache(remDoneSlot, epoch);
                asc_sync_data_barrier(DSB_DDR);

                // 等待对端 peer 推入本卡对应 (peer, qpIdx) 分片完成
                auto* locDoneSlot = reinterpret_cast<__gm__ int32_t*>(localPeermem.agPeerDonePtr) +
                                    (static_cast<uint64_t>(peer) * TP_UDMA_MAX_QPS_PER_PEER + qpIdx) * INT_CACHELINE;
                WaitUntilGmFlagAtLeast(locDoneSlot, epoch);
            }
#endif
        } else {
            const uint32_t baseCores = agTotalCores / numPeers;
            const uint32_t extraPeers = agTotalCores % numPeers;
            const uint32_t splitPoint = extraPeers * (baseCores + 1U);
            uint32_t peerSlot = 0U;
            uint32_t subIdx = 0U;
            uint32_t coresForThisPeer = baseCores;
            if (agIdx < splitPoint) {
                peerSlot = agIdx / (baseCores + 1U);
                subIdx = agIdx % (baseCores + 1U);
                coresForThisPeer = baseCores + 1U;
            } else {
                const uint32_t remIdx = agIdx - splitPoint;
                peerSlot = extraPeers + remIdx / baseCores;
                subIdx = remIdx % baseCores;
                coresForThisPeer = baseCores;
            }
            const uint32_t peer = (rankId + 1U + peerSlot) % tpWorldSize;
            const WorkRange myRange = GetBalancedTokenRange(bs, subIdx, coresForThisPeer);

            if (myRange.count > 0U) {
                GlobalTensor<uint8_t> remRecordGm, dstRecordGm;
                remRecordGm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ uint8_t*>(g_winRankAddr_[peer] + localPeermem.layout.agXDataOffset));
                dstRecordGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(localPeermem.agXDataPtr));

                constexpr uint32_t kRecSlotCap = 112U * 1024U;
                const uint32_t batchTokens = Max(1U, kRecSlotCap / recordBytes);

                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                uint32_t seq = 0U;
                for (uint32_t off = 0U; off < myRange.count; off += batchTokens) {
                    const uint32_t cnt = Min(batchTokens, myRange.count - off);
                    const uint32_t slot = seq++ % 2U;
                    const TEventID ev = slot == 0U ? EVENT_ID0 : EVENT_ID1;
                    LocalTensor<uint8_t> recUb(TPosition::VECCALC, slot * kRecSlotCap, kRecSlotCap);

                    WaitFlag<HardEvent::MTE3_MTE2>(ev);
                    const uint64_t peerGlobalTok = static_cast<uint64_t>(peer) * bs + myRange.start + off;
                    const uint32_t copyBytes = cnt * recordBytes;
                    DataCopy(recUb, remRecordGm[peerGlobalTok * recordBytes], copyBytes);
                    SetFlag<HardEvent::MTE2_MTE3>(ev);
                    WaitFlag<HardEvent::MTE2_MTE3>(ev);
                    DataCopy(dstRecordGm[peerGlobalTok * recordBytes], recUb, copyBytes);
                    SetFlag<HardEvent::MTE3_MTE2>(ev);
                }
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                SyncFuncStatic<HardEvent::MTE3_S, EVENT_ID0>();
                asc_sync_data_barrier(DSB_DDR);
            }
        }
    } else {
        // ==================== RouteTable 子组（32 个 AIV1） ====================
        timer.SetWave(AscendTimer::NO_WAVE);
        AscendTimerSpanScope routeSpan(timer.Enabled() ? &timer : nullptr, AscendTimer::AIV_ROUTE_TABLE);

        const uint32_t rIdx = aivIdx / 2U;
        const uint32_t routeCores = tiling.aicNum;
        const uint32_t topK = tiling.topK;
        const uint32_t numExperts = tiling.numExperts;
        const uint32_t expertStride = static_cast<uint32_t>(TpCeilAlign(numExperts, INT_CACHELINE));
        const uint32_t globalBs = bs * tpWorldSize;
        int32_t routeSyncPhase = 0;

        // Step A: 拉取远端 (tpWorldSize - 1) 个 rank 的 topk_ids 与 topk_weights
        {
            const uint32_t totalRemoteTokens = (tpWorldSize - 1U) * bs;
            const WorkRange remRange = GetBalancedTokenRange(totalRemoteTokens, rIdx, routeCores);
            uint32_t remCursor = remRange.start;
            const uint32_t remEnd = remRange.start + remRange.count;

            GlobalTensor<int32_t> dstIdsGm;
            GlobalTensor<float> dstWeightsGm;
            dstIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.agTopkIdsPtr));
            dstWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace.agTopkWeightsPtr));

            constexpr uint32_t kPullBatchTokens = 512U;
            constexpr uint32_t kMaxPullItems = kPullBatchTokens * 32U;
            LocalTensor<int32_t> idsUb(TPosition::VECCALC, 0U, kMaxPullItems);
            LocalTensor<float> weightsUb(TPosition::VECCALC, kMaxPullItems * sizeof(int32_t), kMaxPullItems);

            while (remCursor < remEnd) {
                const uint32_t peerSlot = remCursor / bs;
                const uint32_t tokInPeer = remCursor % bs;
                const uint32_t peer = (rankId + 1U + peerSlot) % tpWorldSize;
                const uint32_t peerRem = bs - tokInPeer;
                const uint32_t cnt = Min(kPullBatchTokens, Min(peerRem, remEnd - remCursor));
                const uint32_t bytes = cnt * topK * static_cast<uint32_t>(sizeof(int32_t));

                GlobalTensor<int32_t> srcIdsGm;
                GlobalTensor<float> srcWeightsGm;
                srcIdsGm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ int32_t*>(g_winRankAddr_[peer] + localPeermem.layout.agSrcTopkIdsOffset));
                srcWeightsGm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float*>(g_winRankAddr_[peer] + localPeermem.layout.agSrcTopkWeightsOffset));

                const uint64_t srcElem = static_cast<uint64_t>(tokInPeer) * topK;
                const uint64_t dstElem = (static_cast<uint64_t>(peer) * bs + tokInPeer) * topK;
                DataCopyPad(idsUb, srcIdsGm[srcElem], {1U, bytes, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                DataCopyPad(weightsUb, srcWeightsGm[srcElem], {1U, bytes, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                SyncFuncStatic<HardEvent::MTE2_MTE3, EVENT_ID0>();
                DataCopyPad(dstIdsGm[dstElem], idsUb, {1U, bytes, 0U, 0U, 0U});
                DataCopyPad(dstWeightsGm[dstElem], weightsUb, {1U, bytes, 0U, 0U, 0U});
                SyncFuncStatic<HardEvent::MTE3_MTE2, EVENT_ID0>();
                remCursor += cnt;
            }
            SyncFuncStatic<HardEvent::MTE3_S, EVENT_ID0>();
            asc_sync_data_barrier(DSB_DDR);
            {
                AscendTimerWaitScope wait(timer.Enabled() ? &timer : nullptr, AscendTimer::AIV_ROUTE_WAIT);
                SyncTpSubgroup(workspace.flagRouteGroupSyncPtr, rIdx, routeCores, routeSyncPhase);
            }
        }

        // Step B: 局部专家直方图统计
        const WorkRange myGlobalTokens = GetBalancedTokenRange(globalBs, rIdx, routeCores);
        {
            LocalTensor<int32_t> localCntUb(TPosition::VECCALC, 0U, expertStride);
            Duplicate<int32_t>(localCntUb, 0, static_cast<int32_t>(expertStride));
            SyncFuncStatic<HardEvent::V_S, EVENT_ID0>();

            GlobalTensor<int32_t> agIdsGm;
            agIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.agTopkIdsPtr));
            constexpr uint32_t kHistBatchTokens = 512U;
            constexpr uint32_t kMaxHistItems = kHistBatchTokens * 32U;
            LocalTensor<int32_t> topkIdsUb(TPosition::VECCALC, 16U * 1024U, kMaxHistItems);

            for (uint32_t off = 0U; off < myGlobalTokens.count; off += kHistBatchTokens) {
                const uint32_t cnt = Min(kHistBatchTokens, myGlobalTokens.count - off);
                const uint32_t itemCount = cnt * topK;
                const uint32_t bytes = itemCount * static_cast<uint32_t>(sizeof(int32_t));
                DataCopyPad(topkIdsUb, agIdsGm[static_cast<uint64_t>(myGlobalTokens.start + off) * topK],
                            {1U, bytes, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                SyncFuncStatic<HardEvent::MTE2_S, EVENT_ID0>();
                for (uint32_t i = 0U; i < itemCount; ++i) {
                    const int32_t expId = topkIdsUb.GetValue(i);
                    if (expId >= 0 && static_cast<uint32_t>(expId) < numExperts) {
                        localCntUb.SetValue(expId, localCntUb.GetValue(expId) + 1);
                    }
                }
                SyncFuncStatic<HardEvent::S_MTE2, EVENT_ID0>();
            }

            SyncFuncStatic<HardEvent::S_MTE3, EVENT_ID0>();
            GlobalTensor<int32_t> partialCountsGm;
            partialCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.routePartialCountsPtr));
            DataCopy(partialCountsGm[static_cast<uint64_t>(rIdx) * expertStride], localCntUb, expertStride);
            SyncFuncStatic<HardEvent::MTE3_S, EVENT_ID0>();
            asc_sync_data_barrier(DSB_DDR);
            {
                AscendTimerWaitScope wait(timer.Enabled() ? &timer : nullptr, AscendTimer::AIV_ROUTE_WAIT);
                SyncTpSubgroup(workspace.flagRouteGroupSyncPtr, rIdx, routeCores, routeSyncPhase);
            }
        }

        // Step C: 跨核前缀和 + 发布 expertCounts / expertOffsets / expertRevTokenNums / expert_token_nums
        {
            const WorkRange myExperts = GetBalancedTokenRange(numExperts, rIdx, routeCores);
            auto* partialCountsPtr = reinterpret_cast<__gm__ int32_t*>(workspace.routePartialCountsPtr);
            auto* partialOffsetsPtr = reinterpret_cast<__gm__ int32_t*>(workspace.routePartialOffsetsPtr);
            auto* expertCountsPtr = reinterpret_cast<__gm__ int32_t*>(workspace.expertCountsPtr);

            for (uint32_t e = myExperts.start; e < myExperts.start + myExperts.count; ++e) {
                int32_t acc = 0;
                for (uint32_t c = 0U; c < routeCores; ++c) {
                    const uint64_t idx = static_cast<uint64_t>(c) * expertStride + e;
                    const int32_t val = ReadGmByPassDCache(partialCountsPtr + idx);
                    WriteGmByPassDCache(partialOffsetsPtr + idx, acc);
                    acc += val;
                }
                WriteGmByPassDCache(expertCountsPtr + e, acc);
            }
            asc_sync_data_barrier(DSB_DDR);
            {
                AscendTimerWaitScope wait(timer.Enabled() ? &timer : nullptr, AscendTimer::AIV_ROUTE_WAIT);
                SyncTpSubgroup(workspace.flagRouteGroupSyncPtr, rIdx, routeCores, routeSyncPhase);
            }

            // 读取完整 expertCounts 到 UB，用于写出 block-major expertRevTokenNums 及全局前缀 expertOffsets
            GlobalTensor<int32_t> expertCountsGm, revTokenNumsGm;
            expertCountsGm.SetGlobalBuffer(expertCountsPtr);
            revTokenNumsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.expertRevTokenNumsPtr));

            LocalTensor<int32_t> countsUb(TPosition::VECCALC, 0U, expertStride);
            DataCopy(countsUb, expertCountsGm, expertStride);
            SyncFuncStatic<HardEvent::MTE2_S, EVENT_ID0>();
            SyncFuncStatic<HardEvent::MTE2_MTE3, EVENT_ID0>();

            if (rIdx == 0U) {
                auto* expertOffsetsPtr = reinterpret_cast<__gm__ int32_t*>(workspace.expertOffsetsPtr);
                int32_t prefix = 0;
                for (uint32_t e = 0U; e < numExperts; ++e) {
                    const int32_t cnt = countsUb.GetValue(e);
                    WriteGmByPassDCache(expertOffsetsPtr + e, prefix);
                    prefix += cnt;
                }
                WriteGmByPassDCache(expertOffsetsPtr + numExperts, prefix);

                if (expertTokenNumsOutGm != nullptr) {
                    GlobalTensor<int32_t> outCountsGm;
                    outCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expertTokenNumsOutGm));
                    DataCopyPad(outCountsGm, countsUb,
                                {1U, static_cast<uint32_t>(numExperts * sizeof(int32_t)), 0U, 0U, 0U});
                }
            }

            for (uint32_t blk = rIdx; blk < tiling.aicNum; blk += routeCores) {
                DataCopy(revTokenNumsGm[static_cast<uint64_t>(blk) * expertStride], countsUb, expertStride);
            }
            SyncFuncStatic<HardEvent::MTE3_S, EVENT_ID0>();
            SyncFuncStatic<HardEvent::MTE3_MTE2, EVENT_ID0>();
            SyncFuncStatic<HardEvent::S_MTE2, EVENT_ID0>();
            asc_sync_data_barrier(DSB_DDR);
            {
                AscendTimerWaitScope wait(timer.Enabled() ? &timer : nullptr, AscendTimer::AIV_ROUTE_WAIT);
                SyncTpSubgroup(workspace.flagRouteGroupSyncPtr, rIdx, routeCores, routeSyncPhase);
            }
        }

        // Step D: 确定性无锁写入双向路由映射 permRowToToken & tokenToPermRow（及可选 pre-quant 路由权重）
        if (myGlobalTokens.count > 0U) {
            GlobalTensor<int32_t> expertOffsetsGm, partialOffsetsGm, agIdsGm, tokenToPermRowGm;
            expertOffsetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.expertOffsetsPtr));
            partialOffsetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.routePartialOffsetsPtr));
            agIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.agTopkIdsPtr));
            tokenToPermRowGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.tokenToPermRowPtr));

            GlobalTensor<int32_t> agWeightsBitsGm;
            agWeightsBitsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.agTopkWeightsPtr));
            auto* permRowToTokenPtr = reinterpret_cast<__gm__ int32_t*>(workspace.permRowToTokenPtr);
            auto* metaInfoPtr = reinterpret_cast<__gm__ int32_t*>(workspace.metaInfoPtr);
            const bool writeMetaWeight = tiling.routeWeightBeforeQuant != 0U;

            LocalTensor<int32_t> baseOffsetsUb(TPosition::VECCALC, 0U, expertStride);
            LocalTensor<int32_t> partOffsetsUb(TPosition::VECCALC, 4U * 1024U, expertStride);
            LocalTensor<int32_t> cursorUb(TPosition::VECCALC, 8U * 1024U, expertStride);
            DataCopy(baseOffsetsUb, expertOffsetsGm, expertStride);
            DataCopy(partOffsetsUb, partialOffsetsGm[static_cast<uint64_t>(rIdx) * expertStride], expertStride);
            SyncFuncStatic<HardEvent::MTE2_V, EVENT_ID0>();
            Add(cursorUb, baseOffsetsUb, partOffsetsUb, static_cast<int32_t>(expertStride));
            SyncFuncStatic<HardEvent::V_S, EVENT_ID0>();

            constexpr uint32_t kScatterBatchTokens = 512U;
            constexpr uint32_t kMaxScatterItems = kScatterBatchTokens * 32U;
            LocalTensor<int32_t> topkIdsUb(TPosition::VECCALC, 16U * 1024U, kMaxScatterItems);
            LocalTensor<int32_t> weightsBitsUb(TPosition::VECCALC, 80U * 1024U, kMaxScatterItems);
            LocalTensor<int32_t> permRowOutUb(TPosition::VECCALC, 144U * 1024U, kMaxScatterItems);

            for (uint32_t off = 0U; off < myGlobalTokens.count; off += kScatterBatchTokens) {
                const uint32_t cnt = Min(kScatterBatchTokens, myGlobalTokens.count - off);
                const uint32_t itemCount = cnt * topK;
                const uint32_t bytes = itemCount * static_cast<uint32_t>(sizeof(int32_t));
                const uint64_t tokBase = myGlobalTokens.start + off;

                DataCopyPad(topkIdsUb, agIdsGm[tokBase * topK], {1U, bytes, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                if (writeMetaWeight) {
                    DataCopyPad(weightsBitsUb, agWeightsBitsGm[tokBase * topK], {1U, bytes, 0U, 0U, 0U},
                                {false, 0U, 0U, 0U});
                }
                SyncFuncStatic<HardEvent::MTE2_S, EVENT_ID0>();

                for (uint32_t localTok = 0U; localTok < cnt; ++localTok) {
                    const int32_t globalTok = static_cast<int32_t>(tokBase + localTok);
                    const uint32_t rowBase = localTok * topK;
                    for (uint32_t k = 0U; k < topK; ++k) {
                        const uint32_t flat = rowBase + k;
                        const int32_t expId = topkIdsUb.GetValue(flat);
                        if (expId >= 0 && static_cast<uint32_t>(expId) < numExperts) {
                            const int32_t permRow = cursorUb.GetValue(expId);
                            cursorUb.SetValue(expId, permRow + 1);
                            permRowOutUb.SetValue(flat, permRow);
                            WriteGmByPassDCache(permRowToTokenPtr + permRow, globalTok);
                            if (writeMetaWeight) {
                                WriteGmByPassDCache(
                                    metaInfoPtr + static_cast<uint64_t>(permRow) * META_INFO_SIZE + WEIGHT_INDEX,
                                    weightsBitsUb.GetValue(flat));
                            }
                        } else {
                            permRowOutUb.SetValue(flat, -1);
                        }
                    }
                }
                SyncFuncStatic<HardEvent::S_MTE2, EVENT_ID0>();
                SyncFuncStatic<HardEvent::S_MTE3, EVENT_ID0>();
                DataCopyPad(tokenToPermRowGm[tokBase * topK], permRowOutUb, {1U, bytes, 0U, 0U, 0U});
                SyncFuncStatic<HardEvent::MTE3_S, EVENT_ID0>();
                SyncFuncStatic<HardEvent::MTE3_MTE2, EVENT_ID0>();
            }
            asc_sync_data_barrier(DSB_DDR);
        }
    }

    // 64 AIV 汇合：保证 AllGather 数据与全局路由表均在 DDR 可见后再发布 flagRouteReadyPtr
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    if (aivIdx == 0U) {
        const int32_t totalRows =
            ReadGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(workspace.expertOffsetsPtr) + tiling.numExperts);
        asc_sync_data_barrier(DSB_DDR);
        WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(workspace.flagRouteReadyPtr), totalRows > 0 ? 2 : 1);
    }
    return epoch;
}

/*
 * Phase 2 AIV1 Permute 跨 Slice 不断流异步状态机：
 *   - 仅使用常驻 UB [permRecordBase, end) 与本地事件槽 0..5 (MTE2<->MTE3) + 7 (MTE3->S / MTE2->S)，
 *     与 Routed/Shared SwiGLU 的 UB [0, actBytes) 及事件 ActivationUb(0)/ActivationDone(6) 完全正交；
 *   - 跨专家 slice 保持 3 行在途 MTE2 预取不断流，批量缓存已完成 slice 并在阈值/Wave 边界统一刷盘通知。
 */
constexpr uint32_t TP_PERMUTE_PREFETCH_DEPTH = 3U;
constexpr uint32_t TP_PERMUTE_MAX_PENDING_SLICES = 24U;

struct TpPermuteSliceSpan {
    uint32_t expertIdx = 0U;
    uint32_t rowBeginInExpert = 0U;
    uint32_t rowEndInExpert = 0U;
};

struct TpPermuteAsyncState {
    uint32_t issuedRows = 0U;
    uint64_t inflightPermRow[TP_PERMUTE_PREFETCH_DEPTH]{};
    uint32_t inflightSlot[TP_PERMUTE_PREFETCH_DEPTH]{};
    uint32_t inflightHead = 0U;
    uint32_t inflightTail = 0U;
    TpPermuteSliceSpan pendingSlices[TP_PERMUTE_MAX_PENDING_SLICES]{};
    uint32_t pendingSliceCount = 0U;

    __aicore__ inline bool HasPending() const
    {
        return (inflightHead < inflightTail) || (pendingSliceCount > 0U);
    }
};

__aicore__ inline void StoreTpPermutedRow(const TpAiv1ResidentUbLayout& ubLayout, const TpWorkspaceInfo& workspace,
                                          uint32_t h, uint32_t scaleH, uint32_t slot, uint64_t permRow,
                                          bool aligned32)
{
    const TEventID ev = static_cast<TEventID>(slot);
    WaitFlag<HardEvent::MTE2_MTE3>(ev);
    const uint32_t recBase = ubLayout.permRecordBase + slot * ubLayout.permRecordBytes;
    LocalTensor<uint8_t> dataUb(TPosition::VECCALC, recBase, ubLayout.permRowDataBytes);
    LocalTensor<uint8_t> scaleUb(TPosition::VECCALC, recBase + ubLayout.permRowDataBytes, ubLayout.permRowScaleBytes);
    GlobalTensor<uint8_t> dstDataGm, dstScaleGm;
    dstDataGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(workspace.permutedXPtr));
    dstScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(workspace.permutedScalePtr));
    if (aligned32) {
        DataCopy(dstDataGm[permRow * h], dataUb, h);
        DataCopy(dstScaleGm[permRow * scaleH], scaleUb, scaleH);
    } else {
        DataCopyPad(dstDataGm[permRow * h], dataUb, {1U, h, 0U, 0U, 0U});
        DataCopyPad(dstScaleGm[permRow * scaleH], scaleUb, {1U, scaleH, 0U, 0U, 0U});
    }
    SetFlag<HardEvent::MTE3_MTE2>(ev);
}

__aicore__ inline void FlushAndRetireTpPermute(const MegaMoeTpTilingData& tiling,
                                               const TpAiv1ResidentUbLayout& ubLayout,
                                               const TpWorkspaceInfo& workspace, TpPermuteAsyncState& state)
{
    if (!state.HasPending()) {
        return;
    }
    const uint32_t h = tiling.h;
    const uint32_t scaleH = static_cast<uint32_t>(CalcTpMxScaleStrideBytes(h));
    const bool aligned32 = (h % ALIGN_32 == 0U) && (scaleH % ALIGN_32 == 0U);

    while (state.inflightHead < state.inflightTail) {
        const uint32_t idx = state.inflightHead % TP_PERMUTE_PREFETCH_DEPTH;
        StoreTpPermutedRow(ubLayout, workspace, h, scaleH, state.inflightSlot[idx], state.inflightPermRow[idx],
                           aligned32);
        ++state.inflightHead;
    }

    SetFlag<HardEvent::MTE3_S>(static_cast<TEventID>(Aiv1AsyncEvents::DispatchDone));
    WaitFlag<HardEvent::MTE3_S>(static_cast<TEventID>(Aiv1AsyncEvents::DispatchDone));
    asc_sync_data_barrier(DSB_DDR);

    for (uint32_t s = 0U; s < state.pendingSliceCount; ++s) {
        const auto& span = state.pendingSlices[s];
        if (span.rowBeginInExpert < span.rowEndInExpert) {
            __gm__ int32_t* readyBase = reinterpret_cast<__gm__ int32_t*>(workspace.flagPermuteToGmm1Ptr) +
                                        static_cast<uint64_t>(span.expertIdx) * workspace.mGroupFlagSlotsPerExpert;
            const uint32_t firstGroup = span.rowBeginInExpert / L1_TILE_M_256;
            const uint32_t lastGroup = (span.rowEndInExpert - 1U) / L1_TILE_M_256;
            for (uint32_t g = firstGroup; g <= lastGroup; ++g) {
                const uint32_t gBegin = g * L1_TILE_M_256;
                const uint32_t gEnd = gBegin + L1_TILE_M_256;
                const uint32_t ovBegin = span.rowBeginInExpert > gBegin ? span.rowBeginInExpert : gBegin;
                const uint32_t ovEnd = span.rowEndInExpert < gEnd ? span.rowEndInExpert : gEnd;
                AtomicAdd(readyBase + static_cast<uint64_t>(g) * INT_CACHELINE, static_cast<int32_t>(ovEnd - ovBegin));
            }
        }
    }
    state.pendingSliceCount = 0U;
}

__aicore__ inline void IssueTpPermuteSlice(const MegaMoeTpTilingData& tiling, const TpAiv1ResidentUbLayout& ubLayout,
                                           const TpPeermemInfo& localPeermem, const TpWorkspaceInfo& workspace,
                                           uint32_t blockIdx, const ExpertSlice& slice, uint64_t waveId,
                                           TpPermuteAsyncState& state)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || slice.rowCount == 0U) {
        return;
    }
    const uint32_t aicNum = tiling.aicNum;
    const uint32_t shift = static_cast<uint32_t>(waveId % aicNum);
    const uint32_t rotJob = blockIdx >= shift ? blockIdx - shift : blockIdx + aicNum - shift;
    const WorkRange owned = GetRotatedBalancedTokenRange(slice.rowCount, rotJob, aicNum, slice.globalRowStart);
    if (owned.count == 0U) {
        return;
    }

    if (state.pendingSliceCount >= TP_PERMUTE_MAX_PENDING_SLICES) {
        FlushAndRetireTpPermute(tiling, ubLayout, workspace, state);
    }

    const uint32_t h = tiling.h;
    const uint32_t scaleH = static_cast<uint32_t>(CalcTpMxScaleStrideBytes(h));
    const uint32_t recordBytes = ubLayout.permRecordBytes;
    const bool aligned32 = (h % ALIGN_32 == 0U) && (scaleH % ALIGN_32 == 0U);
    GlobalTensor<int32_t> permIdxGm;
    permIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.permRowToTokenPtr));
    LocalTensor<int32_t> permIdxUb(TPosition::VECCALC, ubLayout.permIdxBase, TP_PERMUTE_IDX_BATCH_CAP);
    GlobalTensor<uint8_t> srcRecordGm;
    srcRecordGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(localPeermem.agXDataPtr));

    for (uint32_t batchOff = 0U; batchOff < owned.count; batchOff += TP_PERMUTE_IDX_BATCH_CAP) {
        const uint32_t batchCnt = Min(TP_PERMUTE_IDX_BATCH_CAP, owned.count - batchOff);
        const uint64_t batchPermBase = slice.globalRowStart + owned.start + batchOff;
        DataCopyPad(permIdxUb, permIdxGm[batchPermBase],
                    {1U, batchCnt * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        SyncFuncStatic<HardEvent::MTE2_S, static_cast<TEventID>(Aiv1AsyncEvents::DispatchDone)>();

        for (uint32_t i = 0U; i < batchCnt; ++i) {
            const uint64_t permRow = batchPermBase + i;
            const uint32_t srcTok = static_cast<uint32_t>(permIdxUb.GetValue(i));
            const uint32_t slot = state.issuedRows % TP_PERMUTE_RING_SLOTS;
            const TEventID ev = static_cast<TEventID>(slot);
            if (state.issuedRows >= TP_PERMUTE_RING_SLOTS) {
                WaitFlag<HardEvent::MTE3_MTE2>(ev);
            }
            LocalTensor<uint8_t> recUb(TPosition::VECCALC, ubLayout.permRecordBase + slot * recordBytes, recordBytes);
            // 单条 32B 对齐 DataCopy 同时搬入 (FP8 data + E8M0 scale)，随机 MTE2 读事务减半!
            DataCopy(recUb, srcRecordGm[static_cast<uint64_t>(srcTok) * recordBytes], recordBytes);
            SetFlag<HardEvent::MTE2_MTE3>(ev);

            state.inflightPermRow[state.inflightTail % TP_PERMUTE_PREFETCH_DEPTH] = permRow;
            state.inflightSlot[state.inflightTail % TP_PERMUTE_PREFETCH_DEPTH] = slot;
            ++state.inflightTail;
            ++state.issuedRows;

            if (state.inflightTail - state.inflightHead >= TP_PERMUTE_PREFETCH_DEPTH) {
                const uint32_t idx = state.inflightHead % TP_PERMUTE_PREFETCH_DEPTH;
                StoreTpPermutedRow(ubLayout, workspace, h, scaleH, state.inflightSlot[idx], state.inflightPermRow[idx],
                                   aligned32);
                ++state.inflightHead;
            }
        }
        SyncFuncStatic<HardEvent::S_MTE2, static_cast<TEventID>(Aiv1AsyncEvents::DispatchDone)>();
    }

    const uint32_t rBegin = slice.rowStartInExpert + owned.start;
    state.pendingSlices[state.pendingSliceCount++] = {slice.expertIdx, rBegin, rBegin + owned.count};
}

__aicore__ inline void DrainTpPermuteBuffers(TpPermuteAsyncState& state)
{
    const uint32_t used = state.issuedRows < TP_PERMUTE_RING_SLOTS ? state.issuedRows : TP_PERMUTE_RING_SLOTS;
    for (uint32_t slot = 0U; slot < used; ++slot) {
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slot));
    }
    state.issuedRows = 0U;
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_TP_COMM_ROUTE_HPP
