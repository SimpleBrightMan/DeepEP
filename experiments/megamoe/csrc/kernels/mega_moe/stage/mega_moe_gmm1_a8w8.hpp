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

#pragma once
#include "mega_moe_gmm1_activation.hpp"

// A8W8 的 MX 计算实现。W4A8 不包含此文件，不在主路径继续做权重类型分派。
namespace MegaMoeImpl {
namespace GmmKernel {
template <typename Scheduler, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
          typename TensorBias, typename TensorC, typename TensorMetaInfo>
struct Gmm1WorkSet {
    Scheduler& scheduler;
    TensorA& gmA;
    TensorB& gmB;
    TensorScaleA& gmScaleA;
    TensorScaleB& gmScaleB;
    TensorBias& gmBias;
    TensorC& gmC;
    TensorMetaInfo& metaInfoGm;
};

template <typename BlockMmad, bool IsGmm1Interleaved, typename Config, typename TensorB, typename TensorScaleB,
          typename TensorBias, typename ActualShape, typename TensorA, typename TensorScaleA, typename TensorC>
__aicore__ inline void Gmm1AicMmadTileToGmGeneric(BlockMmad& blockMmad, const Config& config, TensorB& gmB,
                                                  TensorScaleB& gmScaleB, TensorBias& gmBias, TensorC& gmC,
                                                  const ActualShape& actualShape, uint32_t mLoc, uint32_t nLoc,
                                                  uint32_t kLoc, TensorA& gmBlockA, TensorScaleA& gmBlockScaleA)
{
    typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                               Get<K_VALUE>(actualShape), 0};

    if constexpr (IsGmm1Interleaved) {
        auto gmBlockB =
            gmB.Slice(Te::MakeCoord(kLoc, nLoc), Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB =
            gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorBlockGm =
            gmC.Slice(Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockGm, singleShape);
    } else {
        for (uint32_t weightBlock = 0; weightBlock < ACTIVATION_N_HALF; ++weightBlock) {
            uint32_t nOffset = nLoc + weightBlock * config.outputN;
            auto gmBlockB = gmB.Slice(Te::MakeCoord(kLoc, nOffset),
                                      Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB =
                gmScaleB.Slice(Te::MakeCoord(0, nOffset), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
            auto tensorBlockGm = gmC.Slice(Te::MakeCoord(mLoc, nOffset),
                                           Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockGm, singleShape);
        }
    }
}

template <typename BlockMmad, bool IsGmm1Interleaved, typename Config, typename TensorB, typename TensorScaleB,
          typename TensorBias, typename ActualShape, typename TensorA, typename TensorScaleA, typename TensorC>
__aicore__ inline void Gmm1AicMmadTileToUbGeneric(BlockMmad& blockMmad, const Config& config, TensorB& gmB,
                                                  TensorScaleB& gmScaleB, TensorBias& gmBias,
                                                  const ActualShape& actualShape, uint32_t nLoc, uint32_t kLoc,
                                                  TensorA& gmBlockA, TensorScaleA& gmBlockScaleA,
                                                  TensorC& l0cOutUbFirst, TensorC& l0cOutUbSecond, uint16_t pingpongIdx)
{
    typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                               Get<K_VALUE>(actualShape), 0};

    if constexpr (IsGmm1Interleaved) {
        auto gmBlockB =
            gmB.Slice(Te::MakeCoord(kLoc, nLoc), Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB =
            gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorUb = pingpongIdx == 0U ? l0cOutUbFirst : l0cOutUbSecond;
        auto tensorBlockUb =
            tensorUb.Slice(Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockUb, singleShape);
    } else {
        auto tensorBlockUbFirst = l0cOutUbFirst.Slice(
            Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto tensorBlockUbSecond = l0cOutUbSecond.Slice(
            Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        for (uint32_t weightBlock = 0; weightBlock < ACTIVATION_N_HALF; ++weightBlock) {
            auto gmBlockB = gmB.Slice(Te::MakeCoord(kLoc, nLoc + weightBlock * config.outputN),
                                      Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB = gmScaleB.Slice(Te::MakeCoord(0, nLoc + weightBlock * config.outputN),
                                                Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias,
                      weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, singleShape);
        }
    }
}

template <typename MmadContext, bool IsShared, bool IsGmm1Interleaved, typename WorkSet, typename Config>
__aicore__ inline void Gmm1AicMmadGeneric(WorkSet& workSet, const GMMAddrInfo& gmmAddrInfo, const Config& config,
                                          uint32_t startLoopIdx, uint32_t tileNum, int32_t& vecSetSyncCom,
                                          uint16_t& pingpongIdx, MmadContext* blockMmadContext)
{
    using BlockMmad = decltype(blockMmadContext->blockMmad);
    InitBlockMmad(*blockMmadContext, config);
    auto& blockMmad = blockMmadContext->blockMmad;

    using KernelConfig = typename Config::KernelConfig;
    using ElementC = typename KernelConfig::ElementCType;
    using MakeLayoutC = typename KernelConfig::MakeLayoutC;
    uint32_t ubBufSize = config.tileM == L1_TILE_M_128 ? MAX_SINGLE_MN_ALIGN32_NUM_128 : MAX_SINGLE_MN_ALIGN32_NUM_256;
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC);
    auto ubLayout = MakeLayoutC{}(config.tileM, L1_TILE_N);
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), ubLayout);
    auto l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), ubLayout);

    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);
        auto gmBlockA = workSet.gmA.Slice(Te::MakeCoord(mLoc, kLoc),
                                          Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        auto gmBlockScaleA =
            workSet.gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));
        uint32_t waveIdx = mLoc / config.tileM;
        if (waveIdx != lastWaveWaited) {
            WaitForGmm1InputReady<IsShared>(gmmAddrInfo, config, mLoc);
            lastWaveWaited = waveIdx;
        }
        if constexpr (IsGmm1Interleaved) {
            if (vecSetSyncCom >= 2) {
                WaitForVector(pingpongIdx);
            }
        } else if (vecSetSyncCom != 0) {
            WaitForVector();
        }

        Gmm1AicMmadTileToUbGeneric<BlockMmad, IsGmm1Interleaved>(
            blockMmad, config, workSet.gmB, workSet.gmScaleB, workSet.gmBias, actualShape, nLoc, kLoc, gmBlockA,
            gmBlockScaleA, l0cOutUbFirst, l0cOutUbSecond, pingpongIdx);

        if constexpr (IsGmm1Interleaved) {
            NotifyVector(pingpongIdx);
            vecSetSyncCom++;
            pingpongIdx = 1U - pingpongIdx;
        } else {
            NotifyVector();
            vecSetSyncCom = 1;
        }
    }
}

template <bool IsWaveFlagGrained, typename Config, typename ActivationQuantOp, typename ActualShape>
__aicore__ inline void Gmm1Aiv0InterleavedEpilogueTileGeneric(const Config& config,
                                                              ActivationQuantOp& activationQuantOp,
                                                              const GMMAddrInfo& gmmAddrInfo,
                                                              const ActualShape& actualShape, uint32_t mLoc,
                                                              uint32_t nLoc, uint16_t pingpongIdx)
{
    if (Get<M_VALUE>(actualShape) == 0U) {
        return;
    }
    uint32_t epilogueN = Get<N_VALUE>(actualShape) / ACTIVATION_N_HALF;
    uint32_t epilogueNLoc = nLoc / ACTIVATION_N_HALF;
    Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(actualShape), epilogueN, 0, 0};
    Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
        mLoc * config.outputN + epilogueNLoc,
        mLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
            CeilDiv(epilogueNLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
        static_cast<int64_t>(mLoc / L1_TILE_M_256),
        0,
        0,
        0};
    AscendC::SetCtrlSpr<60, 60>(0);
    activationQuantOp(epilogueShape, epilogueOffset, pingpongIdx);
    NotifyGmm2InputReady<IsWaveFlagGrained>(gmmAddrInfo.activationToGmm2Flag, mLoc / L1_TILE_M_256);
}

template <bool IsWaveFlagGrained, typename Config, typename ActivationQuantOp, typename ActualShape>
__aicore__ inline void Gmm1Aiv0EpilogueTileGeneric(const Config& config, ActivationQuantOp& activationQuantOp,
                                                   const GMMAddrInfo& gmmAddrInfo, const ActualShape& actualShape,
                                                   uint32_t mLoc, uint32_t nLoc)
{
    if (Get<M_VALUE>(actualShape) == 0U) {
        return;
    }
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
}

template <bool IsGmm1Interleaved, bool IsWaveFlagGrained, typename WorkSet, typename Config, typename ActivationQuantOp>
__aicore__ inline void Gmm1Aiv0EpilogueGeneric(WorkSet& workSet, const GMMAddrInfo& gmmAddrInfo, const Config& config,
                                               uint32_t startLoopIdx, uint32_t tileNum,
                                               ActivationQuantOp& activationQuantOp, uint16_t& pingpongIdx)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        if constexpr (IsGmm1Interleaved) {
            WaitForCube(pingpongIdx);
            Gmm1Aiv0InterleavedEpilogueTileGeneric<IsWaveFlagGrained>(config, activationQuantOp, gmmAddrInfo,
                                                                      actualShape, mLoc, nLoc, pingpongIdx);
            NotifyCube(pingpongIdx);
            pingpongIdx = 1U - pingpongIdx;
        } else {
            WaitForCube();
            Gmm1Aiv0EpilogueTileGeneric<IsWaveFlagGrained>(config, activationQuantOp, gmmAddrInfo, actualShape, mLoc,
                                                           nLoc);
            NotifyCube();
        }
    }
}

template <typename BlockMmad, typename ElementC, typename MakeLayoutC, bool IsShared = false,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false, typename Scheduler, typename Config,
          typename ActivationQuantOp>
__aicore__ inline void Gmm1ExecGeneric(Scheduler& scheduler, const Params& params, const GMMAddrInfo& gmmAddrInfo,
                                       const Config& config, uint32_t startLoopIdx, uint32_t tileNum,
                                       ActivationQuantOp& activationQuantOp, int32_t& vecSetSyncCom,
                                       uint32_t expertBeforeCnt, uint32_t expertIdx, uint16_t& pingpongIdx,
                                       BlockMmadContext<BlockMmad>* blockMmadContext, bool allowWeightL2Bypass)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    using BiasType = typename KernelConfig::BiasType;

    auto layouts = KernelConfig::BuildLayouts(config);
    auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)),
                              layouts.a);
    auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)),
                              layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
        layouts.scaleB);
    if constexpr (IsWaveFlagGrained && g_coreType == AscendC::AIC) {
        SetWaveWeightL2CacheHint<KernelConfig::IS_WEIGHT_NZ, KernelConfig>(config, allowWeightL2Bypass, gmB, gmScaleB);
    }
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(0UL)), layouts.bias);
    GM_ADDR cGlobal = 0UL;

    auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC*>(cGlobal)), layouts.c);
    AscendC::GlobalTensor<float> metaInfoGm;

    using WorkSetType = Gmm1WorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA), decltype(gmScaleB),
                                    decltype(gmBias), decltype(gmC), decltype(metaInfoGm)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC, metaInfoGm};
    if constexpr (g_coreType == AscendC::AIC) {
        if (blockMmadContext != nullptr) {
            {
                Gmm1AicMmadGeneric<BlockMmadContext<BlockMmad>, IsShared, IsGmm1Interleaved>(
                    workSet, gmmAddrInfo, config, startLoopIdx, tileNum, vecSetSyncCom, pingpongIdx, blockMmadContext);
            }
        } else {
            BlockMmadContext<BlockMmad> localBlockMmadContext;
            {
                Gmm1AicMmadGeneric<BlockMmadContext<BlockMmad>, IsShared, IsGmm1Interleaved>(
                    workSet, gmmAddrInfo, config, startLoopIdx, tileNum, vecSetSyncCom, pingpongIdx,
                    &localBlockMmadContext);
            }
        }
    } else {
        // AIV1 在入口处提前退出。
        {
            Gmm1Aiv0EpilogueGeneric<IsGmm1Interleaved, IsWaveFlagGrained>(workSet, gmmAddrInfo, config, startLoopIdx,
                                                                          tileNum, activationQuantOp, pingpongIdx);
        }
    }
}
}  // namespace GmmKernel

template <typename ElementA, typename EpilogueElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          uint32_t EpilogueTileM = Gmm1TileM, bool IsShared = false, bool IsGmm1Interleaved = false,
          bool IsWaveFlagGrained = false>
__aicore__ inline void RunGmm1Generic(BlockEpilogueActivationMxQuant<EpilogueElementA, ElementC, EpilogueTileM,
                                                                     L1_TILE_N, false, IsGmm1Interleaved>& epilogueOp,
                                      const Params& params,
                                      const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                      const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom,
                                      const BlockJobContext& blockJob, uint32_t expertBeforeCnt, uint32_t expertIdx,
                                      uint16_t& pingpongIdx, void* blockMmadContext = nullptr,
                                      bool allowWeightL2Bypass = false)
{
    using GmmConfig = GmmKernel::Config<false, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB,
                                        IsWeightNZ, IsShared, IsGmm1Interleaved, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm1ProblemConfig(problemShape, blockJob, Gmm1TileM);

    GmmKernel::BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                                        GmmKernel::BlockScheduler::Params{Te::MakeCoord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    if (GetSubBlockIdx() != 0) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }
    if constexpr (g_coreType == AIV) {
        epilogueOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    // Sparse/small-M experts can expose fewer tiles than logical AIC groups.
    // Idle groups must only advance the shared cursor; constructing a large-K
    // BlockMmad without a matching sibling AIV task can leave the MIX group in
    // an unmatched synchronization state.
    if (startLoopIdx >= tileNum) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }
    using BlockMmad = typename GmmConfig::BlockMmad;
    using MmadContext = GmmKernel::BlockMmadContext<BlockMmad>;
    auto* typedBlockMmadContext = reinterpret_cast<MmadContext*>(blockMmadContext);
    using MakeLayoutC = typename GmmConfig::MakeLayoutC;
    GmmKernel::Gmm1ExecGeneric<BlockMmad, ElementC, MakeLayoutC, IsShared, IsGmm1Interleaved, IsWaveFlagGrained>(
        scheduler, params, gmmAddrInfo, config, startLoopIdx, tileNum, epilogueOp, vecSetSyncCom, expertBeforeCnt,
        expertIdx, pingpongIdx, typedBlockMmadContext, allowWeightL2Bypass);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <typename ElementA, typename EpilogueElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          uint32_t EpilogueTileM = Gmm1TileM, bool IsShared = false, bool IsGmm1Interleaved = false,
          bool IsWaveFlagGrained = false>
__aicore__ inline void RunGmm1Generic(BlockEpilogueActivationMxQuant<EpilogueElementA, ElementC, EpilogueTileM,
                                                                     L1_TILE_N, false, IsGmm1Interleaved>& epilogueOp,
                                      const Params& params,
                                      const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                      const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom,
                                      uint32_t expertBeforeCnt, uint32_t expertIdx, uint16_t& pingpongIdx,
                                      void* blockMmadContext = nullptr, bool allowWeightL2Bypass = false)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    RunGmm1Generic<ElementA, EpilogueElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, IsWeightNZ,
                   Gmm1TileM, EpilogueTileM, IsShared, IsGmm1Interleaved, IsWaveFlagGrained>(
        epilogueOp, params, problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom, blockJob, expertBeforeCnt,
        expertIdx, pingpongIdx, blockMmadContext, allowWeightL2Bypass);
}

template <typename QuantOutType, typename ActivationOutType, typename QuantScaleType, uint32_t Gmm1TileM,
          uint32_t EpilogueTileM, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false, bool IsShared = false,
          typename BlockEpilogue>
__aicore__ inline void RunGmm1A8W8(const GmmExecutionConfig& gmmConfig, const Params& gmmParams,
                                   BlockEpilogue& epilogueOp, const GMMAddrInfo& gmmAddrInfo,
                                   const ProblemShape& problemShape, uint32_t globalTokenStartIndex,
                                   GmmRuntimeState& runtimeState, uint32_t expertIdx,
                                   void* persistentBlockMmadContext = nullptr, bool allowWeightL2Bypass = false)
{
    if (gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
        gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        RunGmm1Generic<QuantOutType, ActivationOutType, QuantOutType, bfloat16_t, QuantScaleType, QuantScaleType, true,
                       Gmm1TileM, EpilogueTileM, IsShared, IsGmm1Interleaved, IsWaveFlagGrained>(
            epilogueOp, gmmParams, problemShape, gmmAddrInfo, runtimeState.startBlockIdx, runtimeState.vecSetSyncCom,
            gmmConfig.blockJob, globalTokenStartIndex, expertIdx, runtimeState.pingpongIdx, persistentBlockMmadContext,
            allowWeightL2Bypass);
    } else {
        RunGmm1Generic<QuantOutType, ActivationOutType, QuantOutType, bfloat16_t, QuantScaleType, QuantScaleType, false,
                       Gmm1TileM, EpilogueTileM, IsShared, IsGmm1Interleaved, IsWaveFlagGrained>(
            epilogueOp, gmmParams, problemShape, gmmAddrInfo, runtimeState.startBlockIdx, runtimeState.vecSetSyncCom,
            gmmConfig.blockJob, globalTokenStartIndex, expertIdx, runtimeState.pingpongIdx, persistentBlockMmadContext,
            allowWeightL2Bypass);
    }
}

template <typename QuantOutType, typename WeightType, typename ActivationOutType, typename QuantScaleType,
          bool EnableA8W4, uint32_t Gmm1TileM, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false,
          typename BlockEpilogue>
__aicore__ inline void RunSharedExpertGmm1ActivationStage(
    const MoeStageCommonConfig& commonConfig, const GmmExecutionConfig& gmmConfig, const Params& gmmParams,
    BlockEpilogue& epilogueOp, const GMMAddrInfo& gmmAddrInfo, const ProblemShape& problemShape,
    GmmRuntimeState& runtimeState, uint32_t sharedExpertIdx, int32_t* gmm1TileReadySequence = nullptr,
    void* persistentBlockMmadContext = nullptr, bool allowWeightL2Bypass = false)
{
    uint32_t expertBeforeCnt = sharedExpertIdx * commonConfig.tokenNum;
    if constexpr (EnableA8W4) {
        RunGmm1A8W4<QuantOutType, WeightType, bfloat16_t, QuantScaleType, QuantScaleType, Gmm1TileM, L1_TILE_M_256,
                    true, IsWaveFlagGrained>(epilogueOp, gmmParams, problemShape, gmmAddrInfo,
                                             runtimeState.startBlockIdx, *gmm1TileReadySequence, gmmConfig.blockJob,
                                             expertBeforeCnt, sharedExpertIdx);
    } else {
        RunGmm1A8W8<QuantOutType, ActivationOutType, QuantScaleType, Gmm1TileM, L1_TILE_M_256, IsGmm1Interleaved,
                    IsWaveFlagGrained, true>(gmmConfig, gmmParams, epilogueOp, gmmAddrInfo, problemShape,
                                             expertBeforeCnt, runtimeState, sharedExpertIdx, persistentBlockMmadContext,
                                             allowWeightL2Bypass);
    }
}
}  // namespace MegaMoeImpl
