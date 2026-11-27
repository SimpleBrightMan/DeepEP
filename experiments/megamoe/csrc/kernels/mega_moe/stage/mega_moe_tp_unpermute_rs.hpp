// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

/*!
 * \file mega_moe_tp_unpermute_rs.hpp
 * \brief TP MegaMoE Phase 3：64 AIV 融合 Unpermute + MTE3 Send -> Wait + Local-Reduce（含共享专家累加）。
 */

#ifndef MEGA_MOE_TP_UNPERMUTE_RS_HPP
#define MEGA_MOE_TP_UNPERMUTE_RS_HPP

#include <c_api/asc_simd.h>
#include "../common/mega_moe_tp_layout.hpp"
#include "../common/mega_moe_utils.hpp"
#include "../common/ascend_timer_v2_device.hpp"
#include "mega_moe_unpermute_fused.hpp"

namespace MegaMoeImpl {

using namespace AscendC;

#if defined(__DAV_C310__)
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
 * Phase 3：64 AIV 执行完整 Unpermute + ReduceScatter + Shared 归并。
 *   1. 等待本卡 32 个 AIC 完成全部 Routed GMM2（在打点区间外等待，泳道图自然呈现空泡）；
 *   2. AIV_UNPERMUTE_SEND（整步粒度）：
 *      按 rsTokensPerChunk (默认 64) 切分单卡 bs；每个 AIV 先按轮转顺序推送 3 个远端目标卡的完整 8KiB token，
 *      最后写本卡，并在每个 chunk 落地后以单写者写对方 rsChunkReady[srcRank, chunk] = epoch（免清零、无跨卡原子）；
 *   3. AIV_RECV_REDUCE（整步粒度，内嵌 AIV_RECV_REDUCE_WAIT 软同步等待）：
 *      轮询本卡对应 chunk 的全部 TP 分片与共享专家 256-row 计数就绪后，FP32 归并写出 BF16 yOut。
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
        //   [0, 16 KiB)      permIdxUb (最多 chunkTokens <= 128 * topK <= 32 -> 16 KiB; 若 chunkTokens 更大则按批分段)
        //   [16, 32 KiB)     weightsUb (同上, 16 KiB)
        //   [32, 48 KiB)     outRowUb (单行完整 token 组装区, H<=8192 -> 16 KiB)
        //   [48, 80 KiB)     fallback FP32 累加缓冲 (仅 H%128!=0 时使用, 2 * 4096 floats = 32 KiB)
        //   [80, 208 KiB)    2 个双缓冲 gather 槽 (每槽 64 KiB -> H=4096, topK=8 时整行一次完成无需切列!)
        constexpr uint32_t kSubBatchMaxTokens = 64U;
        constexpr uint32_t kMetaMaxItems = kSubBatchMaxTokens * 32U;  // 2048 items = 8 KiB
        LocalTensor<int32_t> permIdxUb(TPosition::VECCALC, 0U, kMetaMaxItems);
        LocalTensor<float> weightsUb(TPosition::VECCALC, 8U * 1024U, kMetaMaxItems);
        LocalTensor<bfloat16_t> outRowUb(TPosition::VECCALC, 16U * 1024U, 8192U);
        LocalTensor<float> fbAccUb(TPosition::VECCALC, 32U * 1024U, 4096U);
        LocalTensor<float> fbTmpUb(TPosition::VECCALC, 48U * 1024U, 4096U);

        constexpr uint32_t kPackBaseAddr = 64U * 1024U;
        constexpr uint32_t kPackSlotBytes = 96U * 1024U;  // 64..256 KiB 共 192 KiB -> 2 * 96 KiB!
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

        GlobalTensor<int32_t> tokenToPermRowGm;
        GlobalTensor<float> agWeightsGm;
        GlobalTensor<bfloat16_t> gmm2OutGm;
        tokenToPermRowGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace.tokenToPermRowPtr));
        agWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace.agTopkWeightsPtr));
        gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspace.gmm2OutPtr));
        gmm2OutGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        // 先按旋转链路发送远端 (tpWorldSize - 1) 个目标卡，最后写本卡
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
                    // 快路径（如 H=4096, topK=8）：单槽 64 KiB -> 192 KiB 可划 3 槽，实现 MTE2 || Vector || MTE3 三级硬件流水
                    const uint32_t slotElems = topK * h;
                    const uint32_t slotBytes =
                        static_cast<uint32_t>(TpCeilAlign(static_cast<int64_t>(slotElems) * sizeof(bfloat16_t), ALIGN_256));
                    constexpr uint32_t kPackTotalBytes = 192U * 1024U;
                    const uint32_t numSlots = Min(3U, kPackTotalBytes / slotBytes);
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
                        SyncFuncStatic<HardEvent::MTE2_S, EVENT_ID3>();

                        for (uint32_t localT = 0U; localT < subCnt; ++localT) {
                            const uint32_t slot = seq++ % numSlots;
                            const TEventID ev = static_cast<TEventID>(slot);
                            LocalTensor<bfloat16_t> packUb(
                                TPosition::VECCALC, kPackBaseAddr + slot * slotBytes, slotElems);

                            WaitFlag<HardEvent::MTE3_MTE2>(ev);
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

                            const uint64_t dstRow =
                                static_cast<uint64_t>(rankId) * maxTokens + chunkBegin + subOff + localT;
                            DataCopy(remRecvGm[dstRow * h], packUb, h);
                            SetFlag<HardEvent::MTE3_MTE2>(ev);
                        }
                        SyncFuncStatic<HardEvent::S_MTE2, EVENT_ID3>();
                        if (!preQuantWeight) {
                            SyncFuncStatic<HardEvent::V_MTE2, EVENT_ID3>();
                        }
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
