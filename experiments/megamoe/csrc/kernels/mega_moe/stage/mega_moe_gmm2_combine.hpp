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

#ifndef MEGA_MOE_GMM2_COMBINE_H
#define MEGA_MOE_GMM2_COMBINE_H
#include "../common/mega_moe_aiv1_ub_layout.hpp"

#include <type_traits>

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../common/ascend_timer_v2_device.hpp"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "tensor_api/tensor.h"
#include "adv_api/reduce/reduce.h"
#include "../common/mega_moe_gmm_common.hpp"
#include "../common/mega_moe_utils.hpp"
#include "../common/mega_moe_mxfp8_utils.hpp"
#include "../mc2_common/mc2_kernel_utils.hpp"
#include "../mc2_common/quantize_functions.hpp"

namespace MegaMoeImpl {

using namespace AscendC;

namespace CombineImpl {

// Combine 发送路由：dispatch 阶段写入 metaInfo 的 (目标rank, 原token行, topk槽) 三元组。
struct CombineTokenRoute {
    uint32_t dstRankId;
    uint32_t tokenIdx;
    uint32_t topkIdx;
};

__aicore__ inline CombineTokenRoute LoadCombineTokenRoute(const LocalTensor<int32_t>& metaInfoTensor,
                                                          uint32_t recordIdx)
{
    uint32_t recordBase = recordIdx * META_INFO_SIZE;
    return {static_cast<uint32_t>(metaInfoTensor.GetValue(recordBase + RANK_ID)),
            static_cast<uint32_t>(metaInfoTensor.GetValue(recordBase + TOKEN_ID)),
            static_cast<uint32_t>(metaInfoTensor.GetValue(recordBase + TOPK_INDEX))};
}

// wave发送通常每核只有少量行；直接读取三个路由字段，避免为小metadata额外排入MTE2->Scalar完成边界。
// 调用者先观察同一slice的GMM2 ready；Dispatch生成的metadata此时已发布，不改变可见性协议。
__aicore__ inline CombineTokenRoute LoadGmCombineTokenRoute(const Params& params, uint64_t row)
{
    auto record = reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.metaInfoPtr) + row * META_INFO_SIZE;
    return {static_cast<uint32_t>(ReadGmByPassDCache(record + RANK_ID)),
            static_cast<uint32_t>(ReadGmByPassDCache(record + TOKEN_ID)),
            static_cast<uint32_t>(ReadGmByPassDCache(record + TOPK_INDEX))};
}

// 目标卡 combine 接收区按 (token, topk) 展开的紧凑行号。
__aicore__ inline uint64_t GetCombineDstRowIndex(const CombineTokenRoute& route, const Params& params)
{
    return static_cast<uint64_t>(route.tokenIdx) * params.tilingData->topK + route.topkIdx;
}

// 为 tile 内每个 token 行发送一段有效的 GMM2 tile 数据。
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(uint32_t nLoc, uint32_t n, LocalTensor<int32_t>& metaInfoTensor,
                                     LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, BlockShape& actualBlockShape,
                                     uint32_t ubTileN, const Params& params)
{
    // 调用方在进入该数据操作前，保证批量加载的 metadata 对 Scalar 可见。
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
    uint64_t gmRemoteBaseOffset =
        static_cast<uint64_t>(params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr);
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);
    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        CombineTokenRoute route = LoadCombineTokenRoute(metaInfoTensor, static_cast<uint32_t>(tileIdx));
        gmRemoteD.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementMMadOut2*>(GetRankWinAddrWithOffset(route.dstRankId, gmRemoteBaseOffset)));
        uint64_t gmDstOffset = GetCombineDstRowIndex(route, params) * n + nLoc;
        AscendC::DataCopyPad(gmRemoteD[gmDstOffset], l0cOutUbGMM2[tileIdx * ubTileN], ub2GmParams);
    }
}

// 发送一行完整的 Combine 数据，BF16 与 FP8 记录复用相同的定长搬运流程。
template <typename Element>
__aicore__ inline void SendCombineTokenRow(uint32_t rowElements, uint64_t gmRemoteBaseOffset,
                                           const CombineTokenRoute& route, LocalTensor<Element>& rowTensor,
                                           const Params& params)
{
    GlobalTensor<Element> gmRemoteD;
    gmRemoteD.SetGlobalBuffer(
        reinterpret_cast<__gm__ Element*>(GetRankWinAddrWithOffset(route.dstRankId, gmRemoteBaseOffset)));
    uint64_t gmDstRowOffset = GetCombineDstRowIndex(route, params) * rowElements;

    DataCopyExtParams ub2GmParams{1U, static_cast<uint32_t>(rowElements * sizeof(Element)), 0U, 0U, 0U};
    DataCopyPad(gmRemoteD[gmDstRowOffset], rowTensor, ub2GmParams);
}

}  // namespace CombineImpl

constexpr uint32_t META_INFO_TENSOR_ADDR = 200U * 1024U;
constexpr int32_t MAX_AICORE_NUM = 36;

struct QuantTokenBufferConfig {
    uint32_t quantTokenSizeBytes;
};

struct WaveCombineBufferConfig {
    uint32_t rowBytes = 0;
    uint32_t rowStrideBytes = 0;
    uint32_t slotStrideBytes = 0;
    uint32_t rowBufferCount = 0;
    uint32_t eventBase = 0;
};

struct WaveCombineScratch {
    LocalTensor<int32_t> metaInfoTensor;
    LocalTensor<bfloat16_t> rowBufferTensor;
};

constexpr uint32_t WAVE_COMBINE_MIN_ROW_BUFFER_COUNT = 2U;
constexpr uint32_t WAVE_COMBINE_STEADY_ROW_BUFFER_COUNT = 1U;
constexpr uint32_t WAVE_COMBINE_MAX_ROW_BUFFER_COUNT = 6U;
constexpr uint32_t WAVE_COMBINE_UB_BASE = 64U * 1024U;
// [64 KiB, 184 KiB) is dedicated to the Combine row ring and quant scratch.
// The ready scan starts at 184 KiB, so adaptive row buffers must stay below it.
constexpr uint32_t WAVE_COMBINE_UB_LIMIT = 184U * 1024U;
constexpr uint32_t WAVE_COMBINE_META_INFO_TOKEN_CAPACITY = Aiv1ResidentUbLayout::CombineMetaRows;

// MTE BF16 Combine 行环。常规 Wave 只有 AIV1 建立并使用这些 UB 视图；
// 最后一轮不再与 GMM1/SwiGLU 并行时，调用方才为 AIV0 补充初始化。
template <bool IncludeAiv0 = false, uint32_t MaxRowBufferCount = WAVE_COMBINE_MAX_ROW_BUFFER_COUNT>
__aicore__ inline WaveCombineBufferConfig InitWaveCombineBuffers(const MoeStageCommonConfig& common,
                                                                 WaveCombineScratch& scratch,
                                                                 uint32_t rowBase = WAVE_COMBINE_UB_BASE,
                                                                 uint32_t metaBase = META_INFO_TENSOR_ADDR)
{
    static_assert(MaxRowBufferCount > 0U && MaxRowBufferCount <= WAVE_COMBINE_MAX_ROW_BUFFER_COUNT,
                  "invalid Combine row buffer count");
    WaveCombineBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return bufferConfig;
    }
    if (!IncludeAiv0 && GetSubBlockIdx() != 1U) {
        return bufferConfig;
    }
    bufferConfig.rowBytes = common.tokenHiddenDim * sizeof(bfloat16_t);
    bufferConfig.rowStrideBytes = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(bufferConfig.rowBytes), static_cast<uint64_t>(ALIGN_32)));
    bufferConfig.slotStrideBytes = bufferConfig.rowStrideBytes;

    bufferConfig.rowBufferCount = MaxRowBufferCount;

    uint32_t rowRingBytes = bufferConfig.rowBufferCount * bufferConfig.slotStrideBytes;
    scratch.rowBufferTensor =
        LocalTensor<bfloat16_t>(TPosition::VECIN, rowBase, rowRingBytes / sizeof(bfloat16_t));
    scratch.metaInfoTensor = LocalTensor<int32_t>(TPosition::VECCALC, metaBase,
                                                  WAVE_COMBINE_META_INFO_TOKEN_CAPACITY * META_INFO_SIZE);

    return bufferConfig;
}

// 默认由 AIV1 复用既有配置，仅为 AIV0 绑定最终 Combine 的 UB 视图。steady Wave 使用较小行环，
// 调用方排空旧行环后通过 ReinitializeAllAiv 让两个 AIV 都恢复最终 Wave 的最大容量。

// 构造量化 Unpermute 输入记录的单 token 存储布局。
__aicore__ inline QuantTokenBufferConfig CreateQuantTokenBufferConfig(uint32_t tokenHiddenDim)
{
    uint32_t nScale = Ops::Base::CeilDiv(tokenHiddenDim, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
    uint32_t tokenStorageBytes = Ops::Base::CeilAlign(tokenHiddenDim, static_cast<uint32_t>(ALIGN_256));
    uint32_t storedScaleBytes = Ops::Base::CeilAlign(nScale, 2U);
    uint32_t quantTokenSizeBytes =
        Ops::Base::CeilAlign(tokenStorageBytes + storedScaleBytes, static_cast<uint32_t>(ALIGN_32));
    return {quantTokenSizeBytes};
}

// 按专家累计 token 行数滚动物理 AIV 起点，每个 AIV 在当前专家内仍负责一个连续均衡区间。
// 常规 Wave 只使用每个 AIC 配对的 AIV1；最终专家可把同组 AIV0/AIV1 展开成两个独立任务。
template <bool IncludeAiv0 = false>
__aicore__ inline WorkRange GetWaveCombineOwnedRange(const AivJobContext& job, uint32_t tokenCount,
                                                     uint64_t expertRowPrefix)
{
    if constexpr (g_coreType == AIC) {
        return {};
    }
    AivJobContext combineJob = job;
    if constexpr (IncludeAiv0) {
        constexpr uint32_t AIV_CORE_COUNT_PER_AIC = 2U;
        uint32_t subBlockIdx = static_cast<uint32_t>(GetSubBlockIdx());
        combineJob = {job.jobIndex * AIV_CORE_COUNT_PER_AIC + subBlockIdx, job.totalJobs * AIV_CORE_COUNT_PER_AIC};
    } else if (GetSubBlockIdx() != 1U) {
        return {};
    }
    if (combineJob.totalJobs == 0U || combineJob.jobIndex >= combineJob.totalJobs) {
        return {};
    }
    return GetRotatedBalancedTokenRange(tokenCount, combineJob.jobIndex, combineJob.totalJobs, expertRowPrefix);
}

// 等待并消费当前 Wave 的 Combine 发送在行环形 buffer 上产生的全部完成事件，使该环不再有在途 slot。
__aicore__ inline void DrainCombineRowBuffers(uint32_t& issuedRowCount, uint32_t rowBufferCount,
                                              uint32_t eventBase = 0U)
{
    uint32_t activeSlotCount = issuedRowCount < rowBufferCount ? issuedRowCount : rowBufferCount;
    for (uint32_t slot = 0U; slot < activeSlotCount; ++slot) {
        WaitFlag<HardEvent::MTE3_MTE2>(
            static_cast<TEventID>(eventBase + slot));
    }
    /*
     * Drain 会消费所有在途事件，下一段必须从空闲的 slot 0 重新开始；否则该段会把已消费的事件
     * 误判为 buffer 复用依赖并再次等待。W4 的计数跨 Wave 存活，A8W8 则在每次 Combine 调用时
     * 创建新计数；在这里统一重置，可以保证两种生命周期都遵守相同的行环起始状态。
     */
    issuedRowCount = 0U;
}

// 搬运并按需量化一个 wave Combine token 后发送；函数内部不包含 GMM2-ready 依赖。
template <bool IsBufferReuse>
__aicore__ inline void SendWaveCombineToken(const MoeStageCommonConfig& common,
                                            const WaveCombineBufferConfig& bufferConfig, WaveCombineScratch& scratch,
                                            const Params& params, GlobalTensor<bfloat16_t>& gmm2OutGm,
                                            uint64_t gmRemoteBaseOffset, uint32_t tokenLocal,
                                            const CombineImpl::CombineTokenRoute& route, uint32_t slot)
{
    TEventID eventId = static_cast<TEventID>(bufferConfig.eventBase + slot);
    if constexpr (IsBufferReuse) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    }
    uint32_t slotElementOffset = slot * bufferConfig.slotStrideBytes / sizeof(bfloat16_t);
    LocalTensor<bfloat16_t> rowUb = scratch.rowBufferTensor[slotElementOffset];
    DataCopyExtParams gm2UbParams{1U, bufferConfig.rowBytes, 0U, 0U, 0U};
    DataCopyPadExtParams<bfloat16_t> gm2UbPad{false, 0U, 0U, 0U};
    DataCopyPad(rowUb, gmm2OutGm[static_cast<uint64_t>(tokenLocal) * common.tokenHiddenDim], gm2UbParams, gm2UbPad);
    {
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        CombineImpl::SendCombineTokenRow<bfloat16_t>(common.tokenHiddenDim, gmRemoteBaseOffset, route, rowUb,
                                                     params);
    }
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
}

// 使用行级流水发送当前逻辑 AIV 任务负责的已就绪 token 区间。

__aicore__ inline void CombineWaveTokenRange(const MoeStageCommonConfig& common,
                                             const WaveCombineBufferConfig& bufferConfig, WaveCombineScratch& scratch,
                                             const Params& params, GM_ADDR gmm2OutGlobal, uint32_t tokenStart,
                                             uint32_t tokenCount, uint64_t metaInfoGmRowStart, uint32_t& rowSequence)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (tokenCount == 0U) {
        return;
    }
    GlobalTensor<bfloat16_t> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gmm2OutGlobal));
    uint64_t gmRemoteBaseOffset =
        static_cast<uint64_t>(params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr);
    uint32_t rowBufferCount = bufferConfig.rowBufferCount;
    uint32_t tokenIdxInSlice = 0U;
    uint32_t firstUseCount = rowSequence < rowBufferCount ? rowBufferCount - rowSequence : 0U;
    firstUseCount = firstUseCount < tokenCount ? firstUseCount : tokenCount;
    for (; tokenIdxInSlice < firstUseCount; ++tokenIdxInSlice, ++rowSequence) {
        uint32_t slot = rowSequence % rowBufferCount;
        auto route = CombineImpl::LoadGmCombineTokenRoute(params, metaInfoGmRowStart + tokenIdxInSlice);
        SendWaveCombineToken<false>(common, bufferConfig, scratch, params, gmm2OutGm, gmRemoteBaseOffset,
                                    tokenStart + tokenIdxInSlice, route, slot);
    }
    for (; tokenIdxInSlice < tokenCount; ++tokenIdxInSlice, ++rowSequence) {
        uint32_t slot = rowSequence % rowBufferCount;
        auto route = CombineImpl::LoadGmCombineTokenRoute(params, metaInfoGmRowStart + tokenIdxInSlice);
        SendWaveCombineToken<true>(common, bufferConfig, scratch, params, gmm2OutGm, gmRemoteBaseOffset,
                                   tokenStart + tokenIdxInSlice, route, slot);
    }
}

// 等待当前 GMM2 tile 对应的 SwiGLU 输入就绪。
template <bool IsShared, typename Config>
__aicore__ inline void WaitForGmm2InputReady(const GMMAddrInfo& gmmAddrInfo, const Config& config, uint32_t mLoc)
{
    if constexpr (IsShared) {
        return;
    }
    if constexpr (Config::IS_WAVE_FLAG_GRAINED) {
        uint32_t waveIdx = mLoc / L1_TILE_M_256;
        uint32_t waveStart = waveIdx * L1_TILE_M_256;
        uint32_t waveM = waveStart + L1_TILE_M_256 > config.m ? config.m - waveStart : L1_TILE_M_256;
        constexpr uint32_t sourceNFactor = Config::SOURCE_GMM1_INTERLEAVED ? ACTIVATION_N_HALF : 1U;
        uint32_t targetLoops = Ops::Base::CeilDiv(waveM, config.activationTileM) *
                               Ops::Base::CeilDiv(config.k * sourceNFactor, Config::GMM1_TILE_N);
        uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
        __gm__ int32_t* flagValueAddr = gmmAddrInfo.activationToGmm2Flag + flagOffset;
        AscendTimerWaitScope wait(gmmAddrInfo.timer, AscendTimer::AIC_GMM2_WAIT);
        WaitUntilGmFlagEquals(flagValueAddr, static_cast<int32_t>(targetLoops));
    } else {
        GmmKernel::BlockScheduler gmmBlockScheduler(
            {config.m, config.k, config.n},
            GmmKernel::BlockScheduler::Params{
                Te::MakeCoord(static_cast<int64_t>(config.activationTileM), static_cast<int64_t>(L1_TILE_N))});
        uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
        AscendTimerWaitScope wait(gmmAddrInfo.timer, AscendTimer::AIC_GMM2_WAIT);
        WaitUntilGmFlagEquals(gmmAddrInfo.activationToGmm2Flag, static_cast<int32_t>(targetLoops));
    }
}

// Notifies all AIV consumers assigned to the completed GMM2 token group.
__aicore__ inline void NotifyCombineConsumersOfTileCompletion(uint32_t rowTileOffset,
                                                              const GroupSyncSlotLayout& slotLayout,
                                                              __gm__ int32_t* expertCounterBase)
{
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);

    uint32_t tokenGroupIndex = rowTileOffset / COMBINE_TOKEN_GROUP_SIZE;
    uint32_t firstSyncSlot = 0;
    uint32_t syncSlotCount = 0;
    GetGroupSyncSlotRange(tokenGroupIndex, slotLayout, firstSyncSlot, syncSlotCount);
    for (uint32_t syncSlot = firstSyncSlot; syncSlot < firstSyncSlot + syncSlotCount; ++syncSlot) {
        AscendC::AtomicAdd(GetCombineSyncCounterAddress(expertCounterBase, syncSlot), int32_t(1));
    }
}

// Notifies the AIV consumer assigned to the completed shared-expert token group.
__aicore__ inline void NotifySharedExpertTileCompletion(uint32_t rowTileOffset, __gm__ int32_t* sharedExpertCounterBase)
{
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);

    uint32_t tokenGroupIndex = rowTileOffset / COMBINE_TOKEN_GROUP_SIZE;
    AscendC::AtomicAdd(GetCombineSyncCounterAddress(sharedExpertCounterBase, tokenGroupIndex), int32_t(1));
}

namespace GmmKernel {

// 执行通用 GMM2 tile 循环。

// 非量化 Combine 的统一入口。AIV1 直接使用 GMM2 已构造的 scheduler，逐 tile 等待配对 AIC 并消费。
// 当前 GMM2 problem 的起始地址和 M 大小共同限定 token 范围，无需再次遍历专家或重建 scheduler。
// UB 从 64 KiB 起步，避开 AIV1 Dispatch 跨 wave 保留在低区的 cumsum 状态。
template <typename ElementC, typename MakeLayoutC, typename Scheduler, typename TensorC, typename Config>
__aicore__ inline void CombineTokenRange(Scheduler& scheduler, TensorC& l0cOutGm, const Params& params,
                                         const GMMAddrInfo& gmmAddrInfo, int32_t& gmTileSequence, const Config& config,
                                         uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        AscendTimerTileScope tileTimer(gmmAddrInfo.timer, AscendTimer::AIV_COMBINE, mLoc, nLoc,
                                       Get<M_VALUE>(actualShape));

        int32_t expectedReadySequence = gmTileSequence + 1;
        {
            AscendTimerWaitScope wait(gmmAddrInfo.timer, AscendTimer::AIV_COMBINE_WAIT);
            WaitUntilGmFlagAtLeast(gmmAddrInfo.gmmToEpilogueFlag, expectedReadySequence);
        }
        auto tensorBlockGm = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc),
                                            Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        constexpr uint32_t tileUbBase = WAVE_COMBINE_UB_BASE;
        auto layoutTileUb = MakeLayoutC{}(config.tileM, L1_TILE_N);
        auto tensorBlockUb = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(tileUbBase), layoutTileUb);
        LocalTensor<ElementC> tileUb = LocalTensor<ElementC>(TPosition::VECIN, tileUbBase, config.tileM * L1_TILE_N);
        auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copyGM2UB, tensorBlockUb, tensorBlockGm);

        int32_t lenTile = Get<M_VALUE>(actualShape);
        GlobalTensor<int32_t> metaInfoGm;
        LocalTensor<int32_t> metaInfoTensor =
            LocalTensor<int32_t>(TPosition::VECCALC, META_INFO_TENSOR_ADDR, lenTile * META_INFO_SIZE);
        metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            gmmAddrInfo.metaInfoGlobal + static_cast<uint64_t>(mLoc) * META_INFO_SIZE * sizeof(int32_t)));
        AscendC::DataCopy(metaInfoTensor, metaInfoGm, lenTile * META_INFO_SIZE);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        CombineImpl::CombineTokens<ElementC, decltype(actualShape)>(nLoc, config.n, metaInfoTensor, tileUb, actualShape,
                                                                    L1_TILE_N, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        gmTileSequence = expectedReadySequence;
    }
}

// 执行 A8W4 GMM2 tile 循环。
template <bool IsShared, bool NotifyCombineTileReady, bool TokenCombine, typename BlockMmad, typename Scheduler,
          typename Config>
__aicore__ inline void Gmm2AicMmadA8W4(BlockMmad& blockMmad, Scheduler& scheduler, const GMMAddrInfo& gmmAddrInfo,
                                       const Config& config, uint32_t startLoopIdx, uint32_t tileNum,
                                       uint32_t expertTokenCount, uint32_t rowOffsetInExpert,
                                       int32_t* gmTileSequence = nullptr)
{
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    GroupSyncSlotLayout groupSyncSlotLayout{};

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        AscendTimerTileScope tileTimer(
            (IsShared || NotifyCombineTileReady || TokenCombine) ? gmmAddrInfo.timer : nullptr,
            IsShared ? AscendTimer::AIC_SHARED_GMM2
                     : (TokenCombine ? AscendTimer::AIC_GMM2_ISSUE : AscendTimer::AIC_GMM2),
            mLoc, nLoc, Get<M_VALUE>(actualShape), config.tileN);

        if constexpr (Config::IS_WAVE_FLAG_GRAINED) {
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx != lastWaveWaited) {
                WaitForGmm2InputReady<IsShared>(gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
        } else if (loopIdx == startLoopIdx) {
            WaitForGmm2InputReady<IsShared>(gmmAddrInfo, config, mLoc);
        }

        blockMmad.Compute(mLoc, nLoc, Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape));
        if constexpr (NotifyCombineTileReady && !IsShared) {
            // 只有逐tile消费者需要此处确认FIX完成。TokenCombine在slice出口排空并发布，
            // MMAD/后继tile仍由unit flag保护L0C复用，不能把每tile的Scalar排空当成依赖。
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WriteGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag, ++(*gmTileSequence));
        } else if constexpr (IsShared) {
            NotifySharedExpertTileCompletion(mLoc, gmmAddrInfo.sharedExpertGmm2TileCounter);
        }
    }
}

// 为一个 GMM2 逻辑 block 的 AIV0 路径展开一段 A8W4 权重。
template <typename BlockPrologue, typename Scheduler, typename Config>
__aicore__ inline void Gmm2Aiv0PrologueA8W4(BlockPrologue& blockPrologue, Scheduler& scheduler, const Config& config,
                                            uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        auto mL1Size = Get<M_VALUE>(actualShape);
        auto nL1Size = Get<N_VALUE>(actualShape);
        blockPrologue.Dequantize(nLoc, mL1Size, nL1Size);
    }
}

// 根据 GM 地址建立执行资源，并执行 A8W4 GMM2 阶段。
template <typename ElementC, bool IsShared, bool NotifyCombineTileReady, bool TokenCombine, typename Scheduler,
          typename Config>
__aicore__ inline void Gmm2ExecA8W4(Scheduler& scheduler, const GMMAddrInfo& gmmAddrInfo, const Config& config,
                                    uint32_t startLoopIdx, uint32_t tileNum, uint32_t expertTokenCount,
                                    uint32_t rowOffsetInExpert, const Params* params = nullptr,
                                    int32_t* gmTileSequence = nullptr, CatlassGmm2<Config>* waveKernel = nullptr)
{
    if constexpr (g_coreType == AscendC::AIC) {
        if (waveKernel != nullptr) {
            waveKernel->UpdateProblem(config, gmmAddrInfo);
            waveKernel->Begin();
            Gmm2AicMmadA8W4<IsShared, NotifyCombineTileReady, TokenCombine>(*waveKernel, scheduler, gmmAddrInfo, config,
                                                                            startLoopIdx, tileNum, expertTokenCount,
                                                                            rowOffsetInExpert, gmTileSequence);
            return;
        }
        CatlassGmm2<Config> blockMmad(config, gmmAddrInfo);
        Gmm2AicMmadA8W4<IsShared, NotifyCombineTileReady, TokenCombine>(blockMmad, scheduler, gmmAddrInfo, config,
                                                                        startLoopIdx, tileNum, expertTokenCount,
                                                                        rowOffsetInExpert, gmTileSequence);
    } else {
        if (GetSubBlockIdx() == 0U) {
            AscendTimerSpanScope prologueTimer(IsShared ? gmmAddrInfo.timer : nullptr,
                                               AscendTimer::AIV_SHARED_GMM2_PROLOGUE);
            if (waveKernel != nullptr) {
                waveKernel->UpdateProblem(config, gmmAddrInfo);
                waveKernel->Begin();
                Gmm2Aiv0PrologueA8W4(*waveKernel, scheduler, config, startLoopIdx, tileNum);
                return;
            }
            CatlassGmm2<Config> blockPrologue(config, gmmAddrInfo);
            Gmm2Aiv0PrologueA8W4(blockPrologue, scheduler, config, startLoopIdx, tileNum);
        } else if constexpr (NotifyCombineTileReady) {
            using KernelConfig = typename Config::KernelConfig;
            using MakeLayoutC = typename KernelConfig::MakeLayoutC;
            auto gmC = Te::MakeTensor(
                Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC*>(gmmAddrInfo.gmm2OutGlobal)),
                MakeLayoutC{}(config.m, config.outputN));
            CombineTokenRange<ElementC, MakeLayoutC>(scheduler, gmC, *params, gmmAddrInfo, *gmTileSequence, config,
                                                     startLoopIdx, tileNum);
        }
    }
}

}  // namespace GmmKernel

// RunGmm2A8W4：AIV0执行W4→W8 prologue，AIC执行GMM2；可选由配对AIV1逐tile Combine。
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool IsShared = false, bool IsWaveFlagGrained = false,
          bool NotifyCombineTileReady = false, bool TokenCombine = false, bool SourceGmm1Fused = false,
          typename WaveKernel = std::nullptr_t>
__aicore__ inline void RunGmm2A8W4(const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                   const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
                                   const BlockJobContext& blockJob, uint32_t expertTokenCount,
                                   uint32_t rowOffsetInExpert, const Params* params = nullptr,
                                   int32_t* gmTileSequence = nullptr, bool reverseBlocks = false,
                                   WaveKernel* waveContext = nullptr)
{
    static_assert(!IsShared || !NotifyCombineTileReady, "Shared expert GMM2 has its own completion counter");
    static_assert(!TokenCombine || (IsWaveFlagGrained && !IsShared && !NotifyCombineTileReady),
                  "Token Combine requires wave slice readiness instead of tile/shared/layered notifications");
    static_assert(!SourceGmm1Fused || (IsWaveFlagGrained && !IsShared),
                  "Fused GMM1 readiness requires routed wave counters");
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using GmmConfig = GmmKernel::Config<true, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, false,
                                        IsShared, false, IsWaveFlagGrained, SourceGmm1Fused>;
    auto config = GmmConfig::BuildGmm2ProblemConfig(problemShape, blockJob, Gmm1TileM);

    GmmKernel::BlockScheduler scheduler({config.m, config.outputN, config.k},
                                        GmmKernel::BlockScheduler::Params{Te::MakeCoord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(config.tileN))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx = GetGmmFirstOwnedTile(config.blockIdx, config.blockNum, startBlockIdx, reverseBlocks);
    if (startLoopIdx >= tileNum) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    using Kernel = GmmKernel::CatlassGmm2<decltype(config)>;
    static_assert(std::is_same_v<WaveKernel, std::nullptr_t> || std::is_same_v<WaveKernel, Kernel>,
                  "GMM2 pipeline/config types must match");
    Kernel* kernel = nullptr;
    if constexpr (!std::is_same_v<WaveKernel, std::nullptr_t>) kernel = waveContext;
    GmmKernel::Gmm2ExecA8W4<ElementC, IsShared, NotifyCombineTileReady, TokenCombine, GmmKernel::BlockScheduler,
                            decltype(config)>(scheduler, gmmAddrInfo, config, startLoopIdx, tileNum, expertTokenCount,
                                              rowOffsetInExpert, params, gmTileSequence, kernel);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool IsShared = false, bool IsWaveFlagGrained = false,
          bool NotifyCombineTileReady = false>
__aicore__ inline void RunGmm2A8W4(const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                   const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
                                   const Params* params = nullptr, int32_t* gmTileSequence = nullptr)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    RunGmm2A8W4<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, Gmm1TileM, IsShared, IsWaveFlagGrained,
                NotifyCombineTileReady>(problemShape, gmmAddrInfo, startBlockIdx, blockJob,
                                        static_cast<uint32_t>(Get<M_VALUE>(problemShape)), 0U, params, gmTileSequence);
}

// 紧凑 token 资源按累计行偏移寻址，专家固定资源按 expertIdx 寻址。
template <typename WeightType, typename ActivationOutType, typename QuantScaleType>
__aicore__ inline void UpdateMoeExpertGmm2GlobalBuffer(const GmmExecutionConfig& gmmConfig,
                                                       const MoeSyncWorkspaceLayout& syncLayout,
                                                       const WorkspaceInfo& workspace,
                                                       const ExpertWeightTensorListAddrs& weights,
                                                       GMMAddrInfo& gmmAddrInfo, const ExpertContext& expertContext,
                                                       uint32_t rowOffsetInExpert = 0U)
{
    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t activationOutputElementsPerByte = PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    uint64_t gmm1OutputDim = static_cast<uint64_t>(Get<N_VALUE>(expertContext.problemShape));
    gmmAddrInfo.expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(expertContext.problemShape));
    uint64_t tokenHiddenDim = static_cast<uint64_t>(Get<K_VALUE>(expertContext.problemShape));
    uint64_t globalTokenStartIndex = static_cast<uint64_t>(expertContext.globalTokenStartIndex) + rowOffsetInExpert;
    uint32_t expertMGroupOffset = rowOffsetInExpert / L1_TILE_M_256;
    uint64_t activationOutputWidth = gmm1OutputDim / ACTIVATION_N_HALF;
    uint64_t activationScaleWidth =
        Ops::Base::CeilDiv(activationOutputWidth, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    gmmAddrInfo.gmm2OutGlobal = workspace.gmm2MmadResPtr + globalTokenStartIndex * tokenHiddenDim * sizeof(bfloat16_t);
    gmmAddrInfo.metaInfoGlobal = workspace.metaInfoPtr + globalTokenStartIndex * META_INFO_SIZE * sizeof(int32_t);
    gmmAddrInfo.aGlobal = workspace.activationQuantDataPtr + globalTokenStartIndex * activationOutputWidth /
                                                                 activationOutputElementsPerByte *
                                                                 sizeof(ActivationOutType);
    gmmAddrInfo.aScaleGlobal =
        workspace.activationQuantScalePtr + globalTokenStartIndex * activationScaleWidth * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal =
        GetExpertWeightAddr<WeightType>(weights.weight2, gmmConfig.isPerExpertWeightTensor, expertContext.expertIdx,
                                        static_cast<uint64_t>(expertContext.expertIdx) * tokenHiddenDim *
                                            activationOutputWidth / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales2, gmmConfig.isPerExpertWeightTensor, expertContext.expertIdx,
        static_cast<uint64_t>(expertContext.expertIdx) * tokenHiddenDim * activationScaleWidth);
    gmmAddrInfo.activationToGmm2Flag =
        reinterpret_cast<__gm__ int32_t*>(workspace.flagActivationToGmm2Ptr) +
        static_cast<uint64_t>(expertContext.expertIdx) * syncLayout.activationFlagSlotCountPerExpert +
        static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    gmmAddrInfo.dispatchToGmm1Flag =
        reinterpret_cast<__gm__ int32_t*>(workspace.flagDispatchToGmm1Ptr) +
        static_cast<uint64_t>(expertContext.expertIdx) * syncLayout.dispatchFlagSlotCountPerExpert +
        static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
}

// Combine 只消费当前专家的 GMM2 输出；避免复用完整 GMM2 绑定接口时计算无关的输入、权重和 flag 地址。

template <typename ActivationType, typename WeightType, typename QuantScaleType, uint32_t Gmm1TileM>
__aicore__ inline void UpdateSharedExpertGmm2GlobalBuffer(const MoeStageCommonConfig& commonConfig,
                                                          const GmmExecutionConfig& gmmConfig,
                                                          const WorkspaceInfo& workspace,
                                                          const ExpertWeightTensorListAddrs& weights,
                                                          GMMAddrInfo& gmmAddrInfo, uint32_t sharedExpertIdx)
{
    constexpr uint32_t activationElementsPerByte = PackedElementTraits<ActivationType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    uint64_t tokenNum = commonConfig.tokenNum;
    gmmAddrInfo.expertTokenCount = 0U;  // shared不继承之前routed专家的缓存判据。
    uint64_t tokenHiddenDim = commonConfig.tokenHiddenDim;
    uint64_t activationOutputWidth = commonConfig.gmm1OutputDim / ACTIVATION_N_HALF;
    uint64_t activationScaleWidth =
        Ops::Base::CeilDiv(activationOutputWidth, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t sharedExpertTokenStartIndex = static_cast<uint64_t>(sharedExpertIdx) * tokenNum;

    gmmAddrInfo.gmm2OutGlobal =
        workspace.sharedExpertResultPtr + sharedExpertTokenStartIndex * tokenHiddenDim * sizeof(bfloat16_t);
    gmmAddrInfo.aGlobal =
        workspace.sharedExpertActivationDataPtr +
        sharedExpertTokenStartIndex * activationOutputWidth / activationElementsPerByte * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = workspace.sharedExpertActivationScalePtr +
                               sharedExpertTokenStartIndex * activationScaleWidth * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<WeightType>(
        weights.weight2, gmmConfig.isPerExpertWeightTensor, sharedExpertIdx,
        static_cast<uint64_t>(sharedExpertIdx) * tokenHiddenDim * activationOutputWidth / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales2, gmmConfig.isPerExpertWeightTensor, sharedExpertIdx,
        static_cast<uint64_t>(sharedExpertIdx) * tokenHiddenDim * activationScaleWidth);
    uint32_t tokenGroupCount = Ops::Base::CeilDiv(commonConfig.tokenNum, Gmm1TileM);
    gmmAddrInfo.sharedExpertGmm2TileCounter =
        reinterpret_cast<__gm__ int32_t*>(workspace.sharedExpertGmm2TileCounterPtr) +
        static_cast<uint64_t>(sharedExpertIdx) * tokenGroupCount * INT_CACHELINE;
}

// 供普通模板、Wave 模板和共享专家的 GMM2 阶段复用。
// 使用调用方传入的 block 任务执行 GMM2，并保持所选同步约定不变。

constexpr uint32_t WAVE_GMM2_READY_SCAN_UB_ADDR = WAVE_COMBINE_UB_LIMIT;
constexpr uint32_t WAVE_GMM2_READY_MAX_SCAN_BYTES = MAX_AICORE_NUM * INT_CACHELINE * sizeof(int32_t) > ALIGN_512
                                                        ? MAX_AICORE_NUM * INT_CACHELINE * sizeof(int32_t)
                                                        : ALIGN_512;
constexpr uint32_t WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR =
    WAVE_GMM2_READY_SCAN_UB_ADDR + ((WAVE_GMM2_READY_MAX_SCAN_BYTES + ALIGN_512 - 1U) / ALIGN_512) * ALIGN_512;
constexpr uint32_t WAVE_GMM2_READY_SUM_UB_ADDR = WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR + ALIGN_512;

// 每个 AIC 在 GMM2 写入可见后发布一个独占 cache line 的完成标记。

// 一个 AIV Combine 任务等待全部 AIC 完成指定专家。

// 输入 expert 和专家内的 slice 起始行，返回该 slice 独占的 GM 就绪计数地址。
// Wave 切片起点按 M256 对齐，尾片仅长度可不足256；按起点编号可以隔离跨 wave 通知。
// 地址步长必须与 WorkspaceInfo 分配一致，每个槽独占 INT_CACHELINE 个 int32（64B）。
__aicore__ inline __gm__ int32_t* GetTokenCombineReadyCounter(const Params& params, uint32_t expertIdx,
                                                              uint32_t rowStart)
{
    uint64_t slotsPerExpert = GetMGroupCountForRows(params.tilingData->maxOutputSize, L1_TILE_M_256);
    uint64_t slot = static_cast<uint64_t>(expertIdx) * slotsPerExpert + rowStart / L1_TILE_M_256;
    return reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.gmm2ReadyPtr) + slot * INT_CACHELINE;
}

// 计算当前 slice 的实际 GEMM2 生产核数，而不是 tile 数或发送核数。
// 独立游标/反向分核只置换 owner，循环分配仍恰有 min(tile数, AIC数) 个生产核。
// 这让少 tile、尾 slice 不必等待空 AIC；所有行、所有 N 列完成前仍不能发送。
__aicore__ inline uint32_t GetTokenCombineProducerCount(const MoeStageCommonConfig& common, const AivJobContext& job,
                                                        uint32_t rowCount)
{
    uint32_t tiles = GetMGroupCountForRows(rowCount, L1_TILE_M_256) *
                     Ops::Base::CeilDiv(common.tokenHiddenDim, static_cast<uint32_t>(L1_TILE_N));
    return tiles < job.totalJobs ? tiles : job.totalJobs;
}

// 仅实际生产核、每 slice 一次；默认由已排空输出的 AIC 发布，Wave 路径由等待真实 FIX 的 AIV1 发布。
// 调用方保证输出完成；这里只递增 ready，不重复排空，也不改变通信粒度。
// AtomicAdd 只操作就绪标志，不改变输出数值，也不改变 Dispatch/Combine 的通信粒度。
template <bool CompletedOnAiv1 = false>
__aicore__ inline void NotifyTokenCombineSliceReady(const Params& params, uint32_t expertIdx, uint32_t rowStart)
{
    if constexpr (CompletedOnAiv1 && g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) AscendC::AtomicAdd(GetTokenCombineReadyCounter(params, expertIdx, rowStart), 1);
    } else if constexpr (!CompletedOnAiv1 && g_coreType == AIC) {
        AscendC::AtomicAdd(GetTokenCombineReadyCounter(params, expertIdx, rowStart), 1);
    }
}

// 按完整 token 行发送当前专家 slice；调用方传入的 job 已包含所需的通信 owner 轮换。
// InputReadyChecked=true 仅供已探测同一 slice 就绪的优先级调度器使用。
// 其他调用保留一次阻塞等待；输出 metadata、发送目的地及 timer 协议不变。
template <bool InputReadyChecked = false>
__aicore__ inline void RunTokenCombineSlice(const MoeStageCommonConfig& common, const AivJobContext& job,
                                            const Params& params, const ExpertContext& expertContext, uint32_t rowStart,
                                            uint32_t rowCount, AscendTimerDevice& timer, WaveCombineScratch& scratch,
                                            const WaveCombineBufferConfig* residentBuffers = nullptr,
                                            uint32_t* persistentSequence = nullptr)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || rowCount == 0U) {
        return;
    }
    uint64_t globalRowStart = static_cast<uint64_t>(expertContext.globalTokenStartIndex) + rowStart;
    WorkRange owned = GetWaveCombineOwnedRange(job, rowCount, globalRowStart);
    if (owned.count == 0U) {
        return;
    }
    // 按实际单卡 BS 选择，不能用专家 slice 行数：Prefill 恢复两槽以限制并发搬运，Decode 保留六槽。
    // 常驻调用跨任务保留行环/credit；非持久调用在返回前排空，不得混用两种生命周期。
    auto buffers = residentBuffers != nullptr ? *residentBuffers :
        (common.tokenNum <= W4A8_SMALL_BATCH_MAX_TOKENS ? InitWaveCombineBuffers(common, scratch)
                                                      : InitWaveCombineBuffers<false, 2U>(common, scratch));
    GM_ADDR sliceOutput =
        params.workspaceInfo.gmm2MmadResPtr + globalRowStart * common.tokenHiddenDim * sizeof(bfloat16_t);
    uint32_t localSequence = 0U;
    uint32_t& rowSequence = persistentSequence != nullptr ? *persistentSequence : localSequence;
    timer.SetSlice(expertContext.expertIdx, rowStart, rowCount);
    for (uint32_t done = 0U; done < owned.count;) {
        uint32_t count = owned.count - done;
        count = count < WAVE_COMBINE_META_INFO_TOKEN_CAPACITY ? count : WAVE_COMBINE_META_INFO_TOKEN_CAPACITY;
        AscendTimerTileScope rows(timer.Enabled() ? &timer : nullptr,
                                  persistentSequence != nullptr ? AscendTimer::AIV_COMBINE_ISSUE : AscendTimer::AIV_COMBINE,
                                  owned.start + done,
                                  static_cast<uint32_t>(-1), count);
        if constexpr (!InputReadyChecked) {
            if (done == 0U) {
                AscendTimerWaitScope wait(timer.Enabled() ? &timer : nullptr, AscendTimer::AIV_COMBINE_WAIT);
                WaitUntilGmFlagAtLeast(GetTokenCombineReadyCounter(params, expertContext.expertIdx, rowStart),
                                       static_cast<int32_t>(GetTokenCombineProducerCount(common, job, rowCount)));
            }
        }
        CombineWaveTokenRange(common, buffers, scratch, params, sliceOutput, owned.start + done, count,
                              globalRowStart + owned.start + done,
                              rowSequence);
        // 发送完成才结束计时和复用 metadata/UB；不增加全核或全卡同步。
        // metadata已由Scalar读成路由值，后续可覆盖；行槽必须等自己的MTE3_MTE2归还。
        if (persistentSequence == nullptr) {
            DrainCombineRowBuffers(rowSequence, buffers.rowBufferCount, buffers.eventBase);
        }
        done += count;
    }
    timer.ClearSlice();
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_GMM2_COMBINE_H
