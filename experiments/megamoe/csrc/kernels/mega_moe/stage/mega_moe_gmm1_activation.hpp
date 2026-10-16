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

#ifndef MEGA_MOE_GMM1_ACTIVATION_H
#define MEGA_MOE_GMM1_ACTIVATION_H

#include <type_traits>

#include "kernel_operator.h"
#include "../common/ascend_timer_v2_device.hpp"
#include "../common/gmm1_async_completion.hpp"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "tensor_api/tensor.h"
#include "../epilogue/block_epilogue_activation_mx_quant.hpp"
#include "../common/mega_moe_gmm_common.hpp"

namespace MegaMoeImpl {

using namespace AscendC;

constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_256 = 256U * 256U;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_128 = 128U * 256U;

// AIC 等本 M-group 的实际行数全部落地；这不是外层调度 Wave 的全体屏障。
// 同一 M-group 的各 N tile 共享输入 ready；尾块按实际行数等待。共享专家输入另有准备屏障。
template <bool IsShared, typename Config>
__aicore__ inline void WaitForGmm1InputReady(const GMMAddrInfo& gmmAddrInfo, const Config& config, uint32_t mLoc)
{
    if constexpr (IsShared) {
        return;
    }
    // Compute replay 的本地输入已准备好；正常路径始终提供非空 Dispatch flag。
    if (gmmAddrInfo.dispatchToGmm1Flag == nullptr) {
        return;
    }
    uint32_t mGroupIdx = mLoc / config.tileM;
    uint32_t targetValue = (mLoc + config.tileM > config.m) ? (config.m - mLoc) : config.tileM;
    uint64_t flagOffset = static_cast<uint64_t>(mGroupIdx) * INT_CACHELINE;
    __gm__ int32_t* flagValueAddr = gmmAddrInfo.dispatchToGmm1Flag + flagOffset;
    AscendTimerWaitScope wait(gmmAddrInfo.timer, AscendTimer::AIC_GMM1_WAIT);
    WaitUntilGmFlagEquals(flagValueAddr, static_cast<int32_t>(targetValue));
}

namespace GmmKernel {

template <bool IsWaveFlagGrained>
__aicore__ inline void NotifyGmm2InputReady(__gm__ int32_t* flagBase, uint32_t flagSlotIdx = 0U)
{
    if (flagBase == nullptr) {
        return;
    }
    if constexpr (IsWaveFlagGrained) {
        flagBase += static_cast<uint64_t>(flagSlotIdx) * INT_CACHELINE;
    }
    AscendC::AtomicAdd(flagBase, 1);
}

// AIC executes a CATLASS W4A8 tile; stage-level readiness remains owned by Wave.

// 执行 A8W4 GMM1 AIC tile 循环，并维持 Dispatch/GMM1 和 GMM1/SwiGLU 的同步关系。
template <typename BlockMmad, bool IsShared, typename Scheduler, typename Config>
__aicore__ inline void Gmm1AicMmadA8W4(BlockMmad& blockMmad, Scheduler& scheduler, const GMMAddrInfo& gmmAddrInfo,
                                       const Config& config, uint32_t startLoopIdx, uint32_t tileNum,
                                       int32_t& gmm1TileReadySequence, uint32_t expertIdx)
{
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        AscendTimerTileScope tileTimer(
            gmmAddrInfo.timer,
            IsShared ? AscendTimer::AIC_SHARED_GMM1
                     : (gmmAddrInfo.gmm1AsyncCompletion ? AscendTimer::AIC_GMM1_ISSUE : AscendTimer::AIC_GMM1),
            mLoc, nLoc, Get<M_VALUE>(actualShape), config.tileN);
        uint32_t waveIdx = mLoc / config.tileM;
        if (waveIdx != lastWaveWaited) {
            WaitForGmm1InputReady<IsShared>(gmmAddrInfo, config, mLoc);
            lastWaveWaited = waveIdx;
        }

        if (gmmAddrInfo.gmm1AsyncCompletion) {
            Gmm1AsyncCompletion::AcquireProducer(gmm1TileReadySequence);
        }
        blockMmad.Compute(mLoc, nLoc, Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape));

        if (gmmAddrInfo.gmm1AsyncCompletion) {
            Gmm1AsyncCompletion::PublishCompleted(gmm1TileReadySequence);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
        }
        if (gmmAddrInfo.gmmToEpilogueFlag != nullptr) {
            // 异步分支这里只发布issued。消费者必须再等FIX full，不能把GM序号当作完成。
            AscendC::WriteGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag, ++gmm1TileReadySequence);
        }
    }
}

template <typename ElementC, typename MakeLayoutC, bool IsWaveFlagGrained, typename TensorC, typename ActivationQuantOp,
          typename Config, typename ActualShape>
__aicore__ inline void Gmm1Aiv1EpilogueTileA8W4(ActivationQuantOp& activationQuantOp, TensorC& l0cOutGm,
                                                const GMMAddrInfo& gmmAddrInfo, const Config& config,
                                                const ActualShape& actualShape, uint32_t mLoc, uint32_t nLoc)
{
    if (Get<M_VALUE>(actualShape) == 0U) {
        return;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
    const uint32_t gateLoc = Config::FUSED_GATE_UP ? nLoc * ACTIVATION_N_HALF : nLoc;
    const uint32_t upLoc = gateLoc + (Config::FUSED_GATE_UP ? Get<N_VALUE>(actualShape) : config.outputN);
    auto tensorBlockGmFirst = l0cOutGm.Slice(Te::MakeCoord(mLoc, gateLoc),
                                             Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
    auto tensorBlockGmSecond =
        l0cOutGm.Slice(Te::MakeCoord(mLoc, upLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
    auto layoutL0cUb = MakeLayoutC{}(config.tileM, L1_TILE_N);
    int64_t ubOffsetFirst = 0;
    uint32_t ubBufferSize =
        config.tileM == L1_TILE_M_128 ? MAX_SINGLE_MN_ALIGN32_NUM_128 : MAX_SINGLE_MN_ALIGN32_NUM_256;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufferSize) * sizeof(ElementC);
    auto tensorBlockUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), layoutL0cUb);
    auto tensorBlockUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), layoutL0cUb);
    auto copyGm2Ub = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
    AscendC::Te::Copy(copyGm2Ub, tensorBlockUbFirst, tensorBlockGmFirst);
    AscendC::Te::Copy(copyGm2Ub, tensorBlockUbSecond, tensorBlockGmSecond);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                                                 0, 0};
    Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
        mLoc * config.outputN + nLoc,
        mLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
            CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
        0,
        0,
        0,
        0};
    AscendC::SetCtrlSpr<60, 60>(0);
    activationQuantOp(epilogueShape, epilogueOffset);
    NotifyGmm2InputReady<IsWaveFlagGrained>(gmmAddrInfo.activationToGmm2Flag, mLoc / L1_TILE_M_256);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
}

// AIV0 按同一投影顺序展开权重：routed 单次融合，shared 使用 planar 调度。
// AIV0 按与 AIC 相同的 tile 顺序生产权重；Dequantize 内部逐 K 块转换 FP4→FP8 并写 L1。
// L1 ready/free 跨核事件约束生产与复用，不要求整个专家的权重先全部转换完成。
template <typename BlockPrologue, typename Scheduler, typename Config>
__aicore__ inline void Gmm1Aiv0PrologueA8W4(BlockPrologue& blockPrologue, Scheduler& scheduler, const Config& config,
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

template <typename ElementC, typename MakeLayoutC, bool IsWaveFlagGrained, typename WorkSet, typename Config,
          typename ActivationQuantOp>
__aicore__ inline void Gmm1Aiv1EpilogueA8W4(WorkSet& workSet, const GMMAddrInfo& gmmAddrInfo, const Config& config,
                                            uint32_t startLoopIdx, uint32_t tileNum, int32_t& gmm1TileReadySequence,
                                            ActivationQuantOp& activationQuantOp)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        AscendTimerTileScope tileTimer(gmmAddrInfo.timer, AscendTimer::AIV_SWIGLU, mLoc, nLoc,
                                       Get<M_VALUE>(actualShape), Config::GMM1_TILE_N);
        int32_t expectedReadySequence = gmm1TileReadySequence + 1;
        {
            AscendTimerWaitScope wait(gmmAddrInfo.timer, AscendTimer::AIV_SWIGLU_WAIT);
            WaitUntilGmFlagAtLeast(gmmAddrInfo.gmmToEpilogueFlag, expectedReadySequence);
        }
        Gmm1Aiv1EpilogueTileA8W4<ElementC, MakeLayoutC, IsWaveFlagGrained>(activationQuantOp, workSet.gmC, gmmAddrInfo,
                                                                           config, actualShape, mLoc, nLoc);
        gmm1TileReadySequence = expectedReadySequence;
    }
}

// 这里只管理计算资源与 tile 循环：异步 routed 流水由外层 handoff，其他路径仍在 slice 末归还。
template <bool IsShared, typename Config, typename Scheduler>
__aicore__ inline void RunGmm1ComputeTiles(CatlassGmm1<Config>& kernel, Scheduler& scheduler, const GMMAddrInfo& addr,
                                           const Config& config, uint32_t startLoopIdx, uint32_t tileNum,
                                           int32_t& readySequence, uint32_t expertIdx)
{
    kernel.UpdateProblem(config, addr);
    kernel.Begin();
    if constexpr (g_coreType == AscendC::AIC) {
        Gmm1AicMmadA8W4<CatlassGmm1<Config>, IsShared>(kernel, scheduler, addr, config, startLoopIdx, tileNum,
                                                       readySequence, expertIdx);
        if (!addr.gmm1AsyncCompletion) kernel.End(true);  // 同步路径逐tile已排空FIX。
    } else {
        Gmm1Aiv0PrologueA8W4(kernel, scheduler, config, startLoopIdx, tileNum);
        if (!addr.gmm1AsyncCompletion) kernel.End();
    }
}

// AIV1 兼容消费路径：只建立输出视图；独立优先级调度器继续使用 ready-tile 入口。
template <typename ElementC, typename MakeLayoutC, bool IsWaveFlagGrained, typename Scheduler, typename Config,
          typename ActivationQuantOp>
__aicore__ inline void RunGmm1VectorEpilogueA8W4(Scheduler& scheduler, const Params& params, const GMMAddrInfo& addr,
                                                 const Config& config, uint32_t startLoopIdx, uint32_t tileNum,
                                                 int32_t& readySequence, ActivationQuantOp& activationQuantOp,
                                                 uint32_t expertBeforeCnt, uint32_t expertIdx)
{
    // 只给实际消费结果的 AIV1 建立视图；AIC/AIV0 不需要旧 WorkSet 中的 A/B/scale/bias Tensor。
    using KernelConfig = typename Config::KernelConfig;
    auto layouts = KernelConfig::BuildLayouts(config);
    auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC*>(addr.gmm1OutGlobal)),
                              layouts.c);
    AscendC::GlobalTensor<float> metaInfoGm;

    using OutputTensor = decltype(gmC);
    using MetadataTensor = decltype(metaInfoGm);
    // 保留既有 epilogue 接口的字段名，不让它持有无关的计算输入。
    struct WorkSet {
        Scheduler& scheduler;
        OutputTensor gmC;
        MetadataTensor metaInfoGm;
    } workSet{scheduler, gmC, metaInfoGm};
    {
        Gmm1Aiv1EpilogueA8W4<ElementC, MakeLayoutC, IsWaveFlagGrained>(workSet, addr, config, startLoopIdx, tileNum,
                                                                       readySequence, activationQuantOp);
    }
}

}  // namespace GmmKernel

// 调度器已确认可消费：同步模式检查ready，异步模式检查issued并在MTE2排入FIX full等待。
// 这里只执行一个SwiGLU tile；函数不依赖物理AIV子核编号，不自行建立完成事件通道。
// 路由与共享激活均由独立 AIV1 调度器按真实完成事件消费。
// 共享 activationToGmm2Flag=nullptr，使用独立的共享激活完成标记。
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM, bool FuseGateUp = false, typename ActivationQuantOp>
__aicore__ inline void RunReadyGmm1ActivationTileA8W4(
    ActivationQuantOp& activationQuantOp, const ProblemShape& problemShape, const GMMAddrInfo& gmmAddrInfo,
    const BlockJobContext& blockJob, uint32_t loopIdx, uint32_t expertBeforeCnt,
    AscendTimer::DynamicTimingType timingType = AscendTimer::AIV_SWIGLU)
{
    using GmmConfig = GmmKernel::Config<true, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, false,
                                        false, false, true, FuseGateUp>;
    auto config = GmmConfig::BuildGmm1ProblemConfig(problemShape, blockJob, Gmm1TileM);
    auto layouts = GmmConfig::BuildLayouts(config);
    auto gmC = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC*>(gmmAddrInfo.gmm1OutGlobal)), layouts.c);
    GmmKernel::BlockScheduler scheduler(
        {config.m, config.outputN, config.k},
        GmmKernel::BlockScheduler::Params{
            Te::MakeCoord(static_cast<int64_t>(config.tileM), static_cast<int64_t>(decltype(config)::GMM1_TILE_N))});
    auto coord = scheduler.GetBlockCoord(loopIdx);
    auto shape = scheduler.GetBlockShape(coord);
    uint32_t m = Get<M_VALUE>(coord);
    uint32_t n = Get<N_VALUE>(coord);
    activationQuantOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    AscendTimerTileScope timer(gmmAddrInfo.timer, timingType, m, n, Get<M_VALUE>(shape), decltype(config)::GMM1_TILE_N);
    {
        GmmKernel::Gmm1Aiv1EpilogueTileA8W4<ElementC, typename GmmConfig::MakeLayoutC, true>(
            activationQuantOp, gmC, gmmAddrInfo, config, shape, m, n);
    }
}

// RunGmm1A8W4：执行 A8W4 prologue（W4→W8）、GMM1、SwiGLU 和量化。
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, uint32_t EpilogueTileM = Gmm1TileM, bool IsShared = false,
          bool IsWaveFlagGrained = false, bool FuseGateUp = false, typename PersistentKernel = std::nullptr_t>
__aicore__ inline void RunGmm1A8W4(
    BlockEpilogueActivationMxQuant<ElementA, ElementC, EpilogueTileM, L1_TILE_N, false>& activationQuantOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& gmm1TileReadySequence,
    const BlockJobContext& blockJob, uint32_t expertBeforeCnt, uint32_t expertIdx = 0,
    PersistentKernel* persistentKernel = nullptr)
{
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using GmmConfig = GmmKernel::Config<true, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, false,
                                        IsShared, false, IsWaveFlagGrained, FuseGateUp>;
    auto config = GmmConfig::BuildGmm1ProblemConfig(problemShape, blockJob, Gmm1TileM);

    if constexpr (g_coreType == AIV) {
        activationQuantOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    using MakeLayoutC = typename GmmConfig::MakeLayoutC;

    GmmKernel::BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                                        GmmKernel::BlockScheduler::Params{Te::MakeCoord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(config.tileN))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    if (startLoopIdx >= tileNum) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    // 配置/分核后直接按角色执行，不再经过通用 Exec 包装和重复的 AIC/AIV0 分派。
    if (g_coreType == AIC || GetSubBlockIdx() == 0U) {
        AscendTimerSpanScope sharedPrologueTimer(g_coreType == AIV && IsShared ? gmmAddrInfo.timer : nullptr,
                                                 AscendTimer::AIV_SHARED_GMM1_PROLOGUE);
        using Kernel = GmmKernel::CatlassGmm1<decltype(config)>;
        if constexpr (!std::is_same_v<PersistentKernel, std::nullptr_t>) {
            static_assert(std::is_same_v<PersistentKernel, Kernel>, "GMM1 pipeline/config types must match");
            GmmKernel::RunGmm1ComputeTiles<IsShared>(*persistentKernel, scheduler, gmmAddrInfo, config, startLoopIdx,
                                                     tileNum, gmm1TileReadySequence, expertIdx);
        } else {
            Kernel kernel;
            GmmKernel::RunGmm1ComputeTiles<IsShared>(kernel, scheduler, gmmAddrInfo, config, startLoopIdx, tileNum,
                                                     gmm1TileReadySequence, expertIdx);
        }
    } else {
        GmmKernel::RunGmm1VectorEpilogueA8W4<ElementC, MakeLayoutC, IsWaveFlagGrained>(
            scheduler, params, gmmAddrInfo, config, startLoopIdx, tileNum, gmm1TileReadySequence, activationQuantOp,
            expertBeforeCnt, expertIdx);
    }

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, uint32_t EpilogueTileM = Gmm1TileM, bool IsShared = false,
          bool IsWaveFlagGrained = false, bool FuseGateUp = false>
__aicore__ inline void RunGmm1A8W4(
    BlockEpilogueActivationMxQuant<ElementA, ElementC, EpilogueTileM, L1_TILE_N, false>& activationQuantOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& gmm1TileReadySequence, uint32_t expertBeforeCnt,
    uint32_t expertIdx = 0)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    RunGmm1A8W4<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, Gmm1TileM, EpilogueTileM, IsShared,
                IsWaveFlagGrained, FuseGateUp>(activationQuantOp, params, problemShape, gmmAddrInfo, startBlockIdx,
                                               gmm1TileReadySequence, blockJob, expertBeforeCnt, expertIdx);
}

template <typename ActivationType, typename WeightType, typename ActivationOutType, typename QuantScaleType,
          uint32_t ActivationElementsPerByte, bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void UpdateMoeExpertGmm1GlobalBuffer(
    const GmmExecutionConfig& gmmConfig, const MoeSyncWorkspaceLayout& syncLayout, const WorkspaceInfo& workspace,
    const ExpertWeightTensorListAddrs& weights, BlockEpilogue& epilogueOp, GMMAddrInfo& gmmAddrInfo,
    const ExpertContext& expertContext, uint32_t rowOffsetInExpert = 0U, uint32_t gmm1TilesPerMGroup = 0U)
{
    if constexpr (g_coreType == AIV && !EnableA8W4) {
        if (GetSubBlockIdx() != 0) {
            return;
        }
    }

    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElementsPerByte = PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    int64_t n = Get<N_VALUE>(expertContext.problemShape);
    int64_t k = Get<K_VALUE>(expertContext.problemShape);
    int64_t expertOffset = expertContext.expertIdx;
    int64_t globalTokenStartIndex = expertContext.globalTokenStartIndex + rowOffsetInExpert;
    uint32_t expertMGroupOffset = rowOffsetInExpert / L1_TILE_M_256;
    int64_t scaleK = Ops::Base::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmm1OutGlobal = workspace.gmm1MmadResPtr + globalTokenStartIndex * n * sizeof(bfloat16_t);
    }

    gmmAddrInfo.metaInfoGlobal = workspace.metaInfoPtr;
    gmmAddrInfo.activationToGmm2Flag = reinterpret_cast<__gm__ int32_t*>(workspace.flagActivationToGmm2Ptr) +
                                       expertOffset * syncLayout.activationFlagSlotCountPerExpert +
                                       static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    gmmAddrInfo.aGlobal =
        workspace.dispatchRevDataPtr + globalTokenStartIndex * k / ActivationElementsPerByte * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = workspace.dispatchRevScalePtr + globalTokenStartIndex * scaleK * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(
        weights.weight1, gmmConfig.isPerExpertWeightTensor, expertContext.expertIdx,
        static_cast<uint64_t>(expertContext.expertIdx) * n * k / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales1, gmmConfig.isPerExpertWeightTensor, expertContext.expertIdx,
        static_cast<uint64_t>(expertContext.expertIdx) * n * scaleK);
    if constexpr (g_coreType == AIV) {
        bool runsActivation = true;
        if constexpr (EnableA8W4) {
            runsActivation = GetSubBlockIdx() == 1;
        }
        if (runsActivation) {
            int64_t scaleN = Ops::Base::CeilDiv(n / ACTIVATION_N_HALF, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                             MXFP_MULTI_BASE_SIZE;
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                globalTokenStartIndex * n / ACTIVATION_N_HALF / outputElementsPerByte,
                globalTokenStartIndex * scaleN,
                expertOffset * syncLayout.activationFlagSlotCountPerExpert / INT_CACHELINE + expertMGroupOffset,
                0L,
                0L,
                0L};
            epilogueOp.UpdateGlobalAddr(vecBaseOffset);
        }
    }
    if constexpr (g_coreType == AIC) {
        gmmAddrInfo.dispatchToGmm1Flag = reinterpret_cast<__gm__ int32_t*>(workspace.flagDispatchToGmm1Ptr) +
                                         expertOffset * syncLayout.dispatchFlagSlotCountPerExpert +
                                         static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    }
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmmToEpilogueFlag = nullptr;
        if (workspace.flagGmmToEpiloguePtr != nullptr) {
            gmmAddrInfo.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t*>(workspace.flagGmmToEpiloguePtr) +
                                            static_cast<uint64_t>(gmmConfig.blockJob.jobIndex) * INT_CACHELINE;
        }
    }
}

// 原型：MegaMoe::UpdateSharedGlobalBuffer<GMM1>。构造一个共享专家的 GMM1 和 SwiGLU GM 视图。
template <typename ActivationType, typename WeightType, typename ActivationOutType, typename QuantScaleType,
          bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void UpdateSharedExpertGmm1GlobalBuffer(const MoeStageCommonConfig& commonConfig,
                                                          const GmmExecutionConfig& gmmConfig,
                                                          const WorkspaceInfo& workspace,
                                                          const ExpertWeightTensorListAddrs& weights,
                                                          BlockEpilogue& epilogueOp, GMMAddrInfo& gmmAddrInfo,
                                                          uint32_t sharedExpertIdx)
{
    constexpr uint32_t weightElemsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElemsPerByte = PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    uint64_t m = commonConfig.tokenNum;
    uint64_t n = commonConfig.gmm1OutputDim;
    uint64_t k = commonConfig.tokenHiddenDim;
    uint64_t scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t activationN = n / ACTIVATION_N_HALF;
    uint64_t scaleN = Ops::Base::CeilDiv(activationN, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t expertIdx = sharedExpertIdx;

    gmmAddrInfo.aGlobal = workspace.sharedExpertInputDataPtr;
    gmmAddrInfo.aScaleGlobal = workspace.sharedExpertInputScalePtr;
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmm1OutGlobal = workspace.sharedExpertGmm1OutPtr + expertIdx * m * n * sizeof(bfloat16_t);
    }
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(weights.weight1, gmmConfig.isPerExpertWeightTensor,
                                                              sharedExpertIdx, expertIdx * n * k / weightElemsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales1, gmmConfig.isPerExpertWeightTensor, sharedExpertIdx, expertIdx * n * scaleK);
    gmmAddrInfo.activationToGmm2Flag = nullptr;
    gmmAddrInfo.metaInfoGlobal = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmmToEpilogueFlag = nullptr;
        if (workspace.flagGmmToEpiloguePtr != nullptr) {
            gmmAddrInfo.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t*>(workspace.flagGmmToEpiloguePtr) +
                                            static_cast<uint64_t>(gmmConfig.blockJob.jobIndex) * INT_CACHELINE;
        }
    }

    if constexpr (g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            static_cast<int64_t>(expertIdx * m * activationN / outputElemsPerByte),
            static_cast<int64_t>(expertIdx * m * scaleN),
            0L,
            0L,
            0L,
            0L};
        epilogueOp.UpdateGlobalAddr(vecBaseOffset);
    }
}

// 原型：MegaMoe::GroupMatmulWithActivationQuant<true>。执行一个共享专家的 GMM1/SwiGLU 阶段。

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_GMM1_ACTIVATION_H
