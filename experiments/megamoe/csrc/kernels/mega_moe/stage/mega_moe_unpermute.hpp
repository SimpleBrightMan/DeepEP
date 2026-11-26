/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See third-party/cann.LICENSE for the full text of the License.
 */
// Modified by SimpleBright_Man 2026

#ifndef MEGA_MOE_UNPERMUTE_H
#define MEGA_MOE_UNPERMUTE_H

#include "../common/mega_moe_mxfp8_utils.hpp"
#include "../common/mega_moe_utils.hpp"
#include "mega_moe_unpermute_fused.hpp"

namespace MegaMoeImpl {

using namespace AscendC;

struct TokenUnpermuteConfig {
    AivJobContext job;
    uint32_t quantTokenSizeBytes;
    uint32_t fullTokenChunkJobCount;
    MegaMoeUnpermuteBufferConfig fullTokenChunkConfig;
    MegaMoeUnpermuteBufferConfig tailTokenChunkConfig;
};

struct TokenUnpermuteScratch {
    LocalTensor<bfloat16_t> dataResTensor;
    LocalTensor<float> dataResFp32Tensor;
    LocalTensor<float> topKWeightsTensor;
    LocalTensor<float> fp32ScaleTensor;
    LocalTensor<bfloat16_t> bf16ScaleTensor;
    LocalTensor<bfloat16_t> topKWeightsBf16Tensor;
};

// 加载并转换一批 top-k 权重。
template <typename TopkWeightsType>
__aicore__ inline void LoadTokenUnpermuteWeights(const MoeStageCommonConfig& common, const Params& params,
                                                 TokenUnpermuteScratch& scratch, int32_t jobOffset,
                                                 int32_t batchTokenOffset, int32_t batchTokenCount)
{
    if (params.tilingData->topkWeightsPrefetch != 0U) {
        // pre-quant路径已在SwiGLU中应用过，所有Combine实现都必须避免重复加权。
        Duplicate(scratch.topKWeightsTensor, 1.0F, batchTokenCount * common.topK);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return;
    }
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGm;
        topKWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.probsGmAddr));
        DataCopyPad(scratch.topKWeightsTensor,
                    topKWeightsGm[static_cast<uint64_t>(jobOffset + batchTokenOffset) * common.topK],
                    {1U, static_cast<uint32_t>(batchTokenCount * common.topK * sizeof(float)), 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        GlobalTensor<bfloat16_t> topKWeightsGm;
        topKWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.probsGmAddr));
        DataCopyPad(scratch.topKWeightsBf16Tensor,
                    topKWeightsGm[static_cast<uint64_t>(jobOffset + batchTokenOffset) * common.topK],
                    {1U, static_cast<uint32_t>(batchTokenCount * common.topK * sizeof(bfloat16_t)), 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SyncFuncStatic<HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(scratch.topKWeightsTensor, scratch.topKWeightsBf16Tensor, RoundMode::CAST_NONE,
             batchTokenCount * common.topK);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
}

// 将一个 MoE 专家的输入搬入 UB，并按 Combine 模式完成 Cast 或 MXFP8 反量化。
template <uint8_t CombineMode>
__aicore__ inline void LoadMoeExpertInput(const TokenUnpermuteConfig& context, const MoeStageCommonConfig& common,
                                          TokenUnpermuteScratch& scratch, const GlobalTensor<bfloat16_t>& expandedX,
                                          uint64_t expertInputIndex, TEventID event,
                                          LocalTensor<bfloat16_t>& dataInBf16, LocalTensor<float>& dataInFp32)
{
    WaitFlag<HardEvent::V_MTE2>(event);
    {
        DataCopy(dataInBf16, expandedX[expertInputIndex * common.tokenHiddenDim], common.tokenHiddenDim);
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
        Cast(dataInFp32, dataInBf16, RoundMode::CAST_NONE, common.tokenHiddenDim);
    }
}

// 累加一个 token 对应的全部 MoE 专家输入。
template <uint8_t CombineMode, bool TopkWeightsPrefetch>
__aicore__ inline void AccumulateMoeExpertsForToken(const TokenUnpermuteConfig& context,
                                                    const MoeStageCommonConfig& common, const Params& params,
                                                    TokenUnpermuteScratch& scratch,
                                                    int32_t tokenIdx, int32_t localIdx,
                                                    const GlobalTensor<bfloat16_t>& expandedX,
                                                    const MegaMoeUnpermuteBufferConfig& bufferConfig)
{
    GlobalTensor<int32_t> routeIds;
    routeIds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.expertIdxGmAddr));
    bool hasRoutedValue = false;
    for (int32_t expertIdx = 0; expertIdx < static_cast<int32_t>(common.topK); ++expertIdx) {
        // -1不产生远端Combine写回，必须跳过；读取未初始化槽再乘零仍可能传播NaN。
        if (params.tilingData->variableTokenCounts != 0U &&
            routeIds.GetValue(static_cast<uint64_t>(tokenIdx) * common.topK + expertIdx) < 0) continue;
        int32_t accumulationItemIdx = localIdx * static_cast<int32_t>(common.topK + common.sharedExpertNum) + expertIdx;
        int32_t inputBufferIdx = accumulationItemIdx % bufferConfig.inputBufferCount;
        TEventID event = static_cast<TEventID>(inputBufferIdx);
        LocalTensor<bfloat16_t> dataInBf16 =
            scratch.dataResTensor[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
        LocalTensor<float> dataInFp32 =
            scratch.dataResFp32Tensor[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
        uint64_t expertInputIndex = static_cast<uint64_t>(tokenIdx) * common.topK + expertIdx;
        LoadMoeExpertInput<CombineMode>(context, common, scratch, expandedX, expertInputIndex, event, dataInBf16,
                                        dataInFp32);
        SetFlag<HardEvent::S_V>(event);
        WaitFlag<HardEvent::S_V>(event);
        PipeBarrier<PIPE_V>();
        if (!hasRoutedValue) {
            {
                float expertScale = scratch.topKWeightsTensor.GetValue(localIdx * common.topK + expertIdx);
                Muls(scratch.dataResFp32Tensor, dataInFp32, expertScale, common.tokenHiddenDim);
            }
        } else {
            {
                float expertScale = scratch.topKWeightsTensor.GetValue(localIdx * common.topK + expertIdx);
                Muls(dataInFp32, dataInFp32, expertScale, common.tokenHiddenDim);
                PipeBarrier<PIPE_V>();
            }
            Add(scratch.dataResFp32Tensor, scratch.dataResFp32Tensor, dataInFp32, common.tokenHiddenDim);
            PipeBarrier<PIPE_V>();
        }
        hasRoutedValue = true;
        SetFlag<HardEvent::V_MTE2>(event);
    }
    if (!hasRoutedValue) {
        Duplicate(scratch.dataResFp32Tensor, 0.0F, common.tokenHiddenDim);
        PipeBarrier<PIPE_V>();
    }
}

// 等待该共享专家的 GMM2 结果就绪后累加进当前 token（等待与累加是一体动作，调用方无需分两步）。
template <uint32_t Gmm1TileM>
__aicore__ inline void AccumulateSharedExpertForToken(const MoeStageCommonConfig& common, const Params& params,
                                                      TokenUnpermuteScratch& scratch, int32_t tokenIdx,
                                                      int32_t localIdx, uint32_t sharedExpertIdx,
                                                      const MegaMoeUnpermuteBufferConfig& bufferConfig)
{
    uint32_t tokenGroupIndex = static_cast<uint32_t>(tokenIdx) / Gmm1TileM;
    uint32_t tokenGroupCount = Ops::Base::CeilDiv(common.tokenNum, Gmm1TileM);
    uint64_t sharedExpertStride = static_cast<uint64_t>(tokenGroupCount) * INT_CACHELINE;
    __gm__ int32_t* counterAddr = GetCombineSyncCounterAddress(
        reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.sharedExpertGmm2TileCounterPtr) +
            static_cast<uint64_t>(sharedExpertIdx) * sharedExpertStride,
        tokenGroupIndex);
    const auto sharedPolicy = SelectSharedExpertPolicy(params.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4,
                                                       common.sharedExpertNum, common.tokenNum);
    uint32_t gmm2NTilesPerGroup = Ops::Base::CeilDiv(common.tokenHiddenDim, sharedPolicy.gmm2TileN);
    WaitUntilGmFlagEquals(counterAddr, static_cast<int32_t>(gmm2NTilesPerGroup));

    GlobalTensor<bfloat16_t> sharedResult;
    sharedResult.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.workspaceInfo.sharedExpertResultPtr));
    int32_t accumulationItemIdx = localIdx * static_cast<int32_t>(common.topK + common.sharedExpertNum) +
                                  static_cast<int32_t>(common.topK + sharedExpertIdx);
    int32_t inputBufferIdx = accumulationItemIdx % bufferConfig.inputBufferCount;
    TEventID event = static_cast<TEventID>(inputBufferIdx);
    LocalTensor<bfloat16_t> dataInBf16 =
        scratch.dataResTensor[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
    LocalTensor<float> dataInFp32 = scratch.dataResFp32Tensor[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopy(
        dataInBf16,
        sharedResult[(static_cast<uint64_t>(sharedExpertIdx) * common.tokenNum + tokenIdx) * common.tokenHiddenDim],
        common.tokenHiddenDim);
    SetFlag<HardEvent::MTE2_V>(event);
    WaitFlag<HardEvent::MTE2_V>(event);
    SetFlag<HardEvent::S_V>(event);
    WaitFlag<HardEvent::S_V>(event);
    Cast(dataInFp32, dataInBf16, RoundMode::CAST_NONE, common.tokenHiddenDim);
    PipeBarrier<PIPE_V>();
    Add(scratch.dataResFp32Tensor, scratch.dataResFp32Tensor, dataInFp32, common.tokenHiddenDim);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(event);
}

template <uint8_t CombineMode, typename TopkWeightsType, bool TopkWeightsPrefetch, uint32_t Gmm1TileM>
__aicore__ inline void ProcessTokenUnpermuteBatch(
    const TokenUnpermuteConfig& context, const MoeStageCommonConfig& common, const Params& params,
    TokenUnpermuteScratch& scratch, const MegaMoeUnpermuteBufferConfig& bufferConfig,
    const GlobalTensor<bfloat16_t>& expandedX, GlobalTensor<bfloat16_t>& output, int32_t jobOffset,
    int32_t batchTokenOffset, int32_t batchTokenCount, TEventID outputBufferEvent)
{
    {
        LoadTokenUnpermuteWeights<TopkWeightsType>(common, params, scratch, jobOffset, batchTokenOffset,
                                                   batchTokenCount);
    }
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
        SetFlag<HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
    }
    for (int32_t localIdx = 0; localIdx < batchTokenCount; ++localIdx) {
        int32_t tokenIdx = jobOffset + batchTokenOffset + localIdx;
        AccumulateMoeExpertsForToken<CombineMode, TopkWeightsPrefetch>(context, common, params, scratch, tokenIdx, localIdx,
                                                                       expandedX, bufferConfig);
        for (uint32_t sharedExpertIdx = 0; sharedExpertIdx < common.sharedExpertNum; ++sharedExpertIdx) {
            AccumulateSharedExpertForToken<Gmm1TileM>(common, params, scratch, tokenIdx, localIdx, sharedExpertIdx,
                                                      bufferConfig);
        }
        WaitFlag<HardEvent::MTE3_V>(outputBufferEvent);
        Cast(scratch.dataResTensor, scratch.dataResFp32Tensor, RoundMode::CAST_RINT, common.tokenHiddenDim);
        SyncFuncStatic<HardEvent::V_MTE3, SYNC_EVENT_ID3>();
        DataCopy(output[static_cast<uint64_t>(tokenIdx) * common.tokenHiddenDim], scratch.dataResTensor,
                 common.tokenHiddenDim);
        SetFlag<HardEvent::MTE3_V>(outputBufferEvent);
    }
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
        WaitFlag<HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
    }
}

/*
 * 构造 Unpermute 使用的 UB 视图，并返回当前 AIV 对应的 buffer 配置（普通/wave/a8w4 编排共用）。
 * UB 从地址 0 起按下列顺序连续排布（N = inputBufferCount，每段起点=上一段末尾）：
 *   [dataRes    ] (N+1) 个 bf16 槽 × bf16SlotElementCount —— 累加结果 bf16 输出
 *   [dataResFp32] (N+1) 个 fp32 槽 × fp32SlotElementCount —— fp32 累加中间态
 *   [topKWeights] topKWeightsBufferBytes —— 本批 token 的 topk 权重（fp32）
 *   [权重转换区 ] 仅 TopkWeightsType==bf16 时存在（topKWeightsConversionBufferBytes）
 *   [bf16Scale  ] 仅量化 combine 时存在，Align32(ceil(H/32)·2B·扩展系数)
 *   [fp32Scale  ] 仅量化 combine 时存在，Align32(ceil(H/32)·4B·扩展系数)
 */
template <typename TopkWeightsType, int32_t CombineQuantMode>
__aicore__ inline MegaMoeUnpermuteBufferConfig CreateTokenUnpermuteBuffers(const TokenUnpermuteConfig& context,
                                                                           uint32_t tokenHiddenDim,
                                                                           TokenUnpermuteScratch& scratch)
{
    MegaMoeUnpermuteBufferConfig bufferConfig = context.job.jobIndex < context.fullTokenChunkJobCount
                                                    ? context.fullTokenChunkConfig
                                                    : context.tailTokenChunkConfig;

    uint32_t bf16ScaleBufAlign = 0U;
    uint32_t fp32ScaleBufAlign = 0U;

    uint32_t bf16SlotBytes = bufferConfig.bf16SlotElementCount * sizeof(bfloat16_t);
    uint32_t fp32SlotBytes = bufferConfig.fp32SlotElementCount * sizeof(float);
    uint32_t dataResBufBytes = (bufferConfig.inputBufferCount + 1) * bf16SlotBytes;
    uint32_t dataResFp32BufBytes = (bufferConfig.inputBufferCount + 1) * fp32SlotBytes;
    uint32_t dataResAddr = 0U;
    scratch.dataResTensor =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr, dataResBufBytes / sizeof(bfloat16_t));
    uint32_t dataResFp32Addr = dataResAddr + dataResBufBytes;
    scratch.dataResFp32Tensor =
        LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr, dataResFp32BufBytes / sizeof(float));
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufBytes;

    scratch.topKWeightsTensor =
        LocalTensor<float>(TPosition::VECCALC, tempAddr, bufferConfig.topKWeightsBufferBytes / sizeof(float));
    tempAddr += bufferConfig.topKWeightsBufferBytes;
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        scratch.topKWeightsBf16Tensor = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, tempAddr, bufferConfig.topKWeightsConversionBufferBytes / sizeof(bfloat16_t));
        tempAddr += bufferConfig.topKWeightsConversionBufferBytes;
    }

    return bufferConfig;
}

#if defined(__DAV_C310__)
// 仅复用host已经批准的dataRes/dataResFp32前缀，不侵占其后的top-k权重区。
// 同一token的top-k结果在GM连续，以一次2D搬运收集；两个槽分别由MTE3归还。
__aicore__ inline bool TryFusedTokenUnpermute(const MoeStageCommonConfig& common, const Params& params,
                                             TokenUnpermuteScratch& scratch,
                                             const MegaMoeUnpermuteBufferConfig& buffers,
                                             const WorkRange& range,
                                             const GlobalTensor<bfloat16_t>& input,
                                             GlobalTensor<bfloat16_t>& output)
{
    const uint32_t prefixBytes = (buffers.inputBufferCount + 1U) *
        (buffers.bf16SlotElementCount * sizeof(bfloat16_t) + buffers.fp32SlotElementCount * sizeof(float));
    const uint32_t capacity = common.topK == 0U ? 0U : prefixBytes / (2U * common.topK * sizeof(bfloat16_t));
    uint32_t chunk = (capacity / 128U) * 128U;
    chunk = chunk < common.tokenHiddenDim ? chunk : common.tokenHiddenDim;
    // 非整向量尾部保留原实现；不能越界读GM或使用未初始化的BF16 lane。
    if (params.tilingData->variableTokenCounts != 0U || common.sharedExpertNum != 0U || common.tokenHiddenDim % 128U != 0U || chunk == 0U ||
        common.topK > 65535U) return false;
    const uint32_t slotElements = common.topK * chunk;
    LocalTensor<bfloat16_t> packed(TPosition::VECCALC, 0U, 2U * slotElements);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    uint32_t sequence = 0U;
    for (uint32_t batch = 0; batch < range.count; batch += buffers.tokensPerBatch) {
        const uint32_t count = Min(static_cast<uint32_t>(buffers.tokensPerBatch), range.count - batch);
        LoadTokenUnpermuteWeights<float>(common, params, scratch, range.start, batch, count);
        for (uint32_t local = 0; local < count; ++local) {
            const uint64_t token = range.start + batch + local;
            for (uint32_t channel = 0; channel < common.tokenHiddenDim; channel += chunk) {
                const uint32_t columns = Min(chunk, common.tokenHiddenDim - channel);
                const uint32_t slot = sequence++ % 2U;
                const auto event = static_cast<TEventID>(slot);
                auto tile = packed[slot * slotElements];
                WaitFlag<HardEvent::MTE3_MTE2>(event);
                DataCopyPad(tile, input[token * common.topK * common.tokenHiddenDim + channel],
                    {static_cast<uint16_t>(common.topK), columns * 2U,
                     (common.tokenHiddenDim - columns) * 2U, 0U, 0U}, {false, 0U, 0U, 0U});
                SetFlag<HardEvent::MTE2_V>(event);
                WaitFlag<HardEvent::MTE2_V>(event);
                WeightedUnpermuteVf(reinterpret_cast<__ubuf__ bfloat16_t*>(tile.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(scratch.topKWeightsTensor.GetPhyAddr()) + local * common.topK,
                    common.topK, columns);
                SetFlag<HardEvent::V_MTE3>(event);
                WaitFlag<HardEvent::V_MTE3>(event);
                DataCopy(output[token * common.tokenHiddenDim + channel], tile, columns);
                SetFlag<HardEvent::MTE3_MTE2>(event);
            }
        }
        // 下一批会覆写同一权重区，必须保证本批VF已经读完权重；不排空输出MTE3。
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    return true;
}
#endif

// 原型：MegaMoe::Unpermute。执行当前 token 分区的完整 Unpermute 流水。
template <uint8_t CombineMode, typename TopkWeightsType, bool TopkWeightsPrefetch, uint32_t Gmm1TileM>
__aicore__ inline void UnpermuteTokens(const TokenUnpermuteConfig& context, const MoeStageCommonConfig& common,
                                       const Params& params, TokenUnpermuteScratch& scratch,
                                       const MegaMoeUnpermuteBufferConfig& bufferConfig)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    WorkRange tokenRange = TilingByJobContext(common.tokenNum, context.job.jobIndex, context.job.totalJobs, 1U);
    if (tokenRange.count == 0U) {
        return;
    }
    int32_t jobLen = static_cast<int32_t>(tokenRange.count);
    int32_t jobOffset = static_cast<int32_t>(tokenRange.start);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.peermemInfo.combineSendPtr));
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.y2GmAddr));

#if defined(__DAV_C310__)
    if constexpr (CombineMode == COMBINE_NO_QUANT && !TopkWeightsPrefetch &&
                  Std::IsSame<TopkWeightsType, float>::value) {
        // 3510下DISABLE映射硬件mode4=NOTALLOC_KEEP，与DS归并读策略一致。
        // 保留已有L2命中、不为一次性消费分配新行；不改完成边界或数据布局。
        expandedX.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        if (TryFusedTokenUnpermute(common, params, scratch, bufferConfig, tokenRange, expandedX, output)) return;
    }
#endif

    constexpr TEventID outputBufferEvent = EVENT_ID0;
    SetFlag<HardEvent::MTE3_V>(outputBufferEvent);
    for (int32_t batchTokenOffset = 0; batchTokenOffset < jobLen; batchTokenOffset += bufferConfig.tokensPerBatch) {
        int32_t batchTokenCount = batchTokenOffset + bufferConfig.tokensPerBatch > jobLen ? jobLen - batchTokenOffset
                                                                                          : bufferConfig.tokensPerBatch;
        ProcessTokenUnpermuteBatch<CombineMode, TopkWeightsType, TopkWeightsPrefetch, Gmm1TileM>(
            context, common, params, scratch, bufferConfig, expandedX, output, jobOffset, batchTokenOffset,
            batchTokenCount, outputBufferEvent);
    }
    WaitFlag<HardEvent::MTE3_V>(outputBufferEvent);
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_UNPERMUTE_H
