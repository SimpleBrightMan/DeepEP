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
#include "mega_moe_gmm2_combine.hpp"

// A8W8 的 MX 计算实现。W4A8 不包含此文件，不在主路径继续做权重类型分派。
namespace MegaMoeImpl {
namespace GmmKernel {
template <typename BlockMmad, bool IsShared, bool NotifyCombineTileReady = false, typename WorkSet, typename Config>
__aicore__ inline void Gmm2AicMmadGeneric(BlockMmad& blockMmad, WorkSet& workSet, const GMMAddrInfo& gmmAddrInfo,
                                          const Config& config, uint32_t startLoopIdx, uint32_t tileNum,
                                          uint32_t rowOffsetInExpert = 0U, int32_t* gmTileSequence = nullptr)
{
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    GroupSyncSlotLayout groupSyncSlotLayout{};

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        // Slice only builds tensor views. Keep it ahead of the synchronization waits so the
        // current tile's address calculation is not inserted into the compute critical path.
        auto gmBlockA = workSet.gmA.Slice(Te::MakeCoord(mLoc, kLoc),
                                          Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        /* E8M0 scales are padded to an even count because GM->L1 moves 64-K scale pairs as b16. */
        auto gmBlockScaleA =
            workSet.gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));

        if constexpr (std::remove_reference_t<decltype(config)>::IS_WAVE_FLAG_GRAINED) {
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx != lastWaveWaited) {
                WaitForGmm2InputReady<IsShared>(gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
        } else if (loopIdx == startLoopIdx) {
            WaitForGmm2InputReady<IsShared>(gmmAddrInfo, config, mLoc);
        }

        typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                                   Get<K_VALUE>(actualShape), 0};

        auto gmBlockB = workSet.gmB.Slice(Te::MakeCoord(kLoc, nLoc),
                                          Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB =
            workSet.gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto gmBlockC = workSet.gmC.Slice(Te::MakeCoord(mLoc, nLoc),
                                          Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, gmBlockC, singleShape);
        if constexpr (NotifyCombineTileReady && !IsShared) {
            // GMM2 tile 与配对 AIV1 的 Combine tile 一对一通知。
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WriteGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag, ++(*gmTileSequence));
        } else if constexpr (IsShared) {
            NotifySharedExpertTileCompletion(mLoc, gmmAddrInfo.sharedExpertGmm2TileCounter);
        }
    }
}

template <typename BlockMmad, typename ElementC, bool IsShared = false, bool NotifyCombineTileReady = false,
          typename Scheduler, typename Config>
__aicore__ inline void Gmm2ExecGeneric(Scheduler& scheduler, const GMMAddrInfo& gmmAddrInfo, const Config& config,
                                       uint32_t startLoopIdx, uint32_t tileNum,
                                       BlockMmadContext<BlockMmad>* blockMmadContext, bool allowWeightL2Bypass,
                                       uint32_t rowOffsetInExpert = 0U, const Params* params = nullptr,
                                       int32_t* gmTileSequence = nullptr)
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
    if constexpr (Config::IS_WAVE_FLAG_GRAINED && g_coreType == AscendC::AIC) {
        SetWaveWeightL2CacheHint<KernelConfig::IS_WEIGHT_NZ, KernelConfig>(config, allowWeightL2Bypass, gmB, gmScaleB);
    }
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(0UL)), layouts.bias);
    auto gmC = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC*>(gmmAddrInfo.gmm2OutGlobal)), layouts.c);

    using WorkSetType = GroupMatmulWorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                           decltype(gmScaleB), decltype(gmBias), decltype(gmC)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC};

    if constexpr (g_coreType == AscendC::AIC) {
        if (blockMmadContext != nullptr) {
            InitBlockMmad(*blockMmadContext, config);
            Gmm2AicMmadGeneric<BlockMmad, IsShared, NotifyCombineTileReady>(blockMmadContext->blockMmad, workSet,
                                                                            gmmAddrInfo, config, startLoopIdx, tileNum,
                                                                            rowOffsetInExpert, gmTileSequence);
        } else {
            BlockMmadContext<BlockMmad> localBlockMmadContext;
            InitBlockMmad(localBlockMmadContext, config);
            Gmm2AicMmadGeneric<BlockMmad, IsShared, NotifyCombineTileReady>(localBlockMmadContext.blockMmad, workSet,
                                                                            gmmAddrInfo, config, startLoopIdx, tileNum,
                                                                            rowOffsetInExpert, gmTileSequence);
        }
    } else if constexpr (NotifyCombineTileReady) {
        if (GetSubBlockIdx() == 1U) {
            using MakeLayoutC = typename KernelConfig::MakeLayoutC;
            CombineTokenRange<ElementC, MakeLayoutC>(workSet.scheduler, workSet.gmC, *params, gmmAddrInfo,
                                                     *gmTileSequence, config, startLoopIdx, tileNum);
        }
    }
}
}  // namespace GmmKernel

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256, bool IsShared = false,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false, bool NotifyCombineTileReady = false>
__aicore__ inline void RunGmm2Generic(const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                      const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
                                      const BlockJobContext& blockJob, void* blockMmadContext = nullptr,
                                      bool allowWeightL2Bypass = false, uint32_t rowOffsetInExpert = 0U,
                                      const Params* params = nullptr, int32_t* gmTileSequence = nullptr)
{
    static_assert(!IsShared || !NotifyCombineTileReady, "Shared expert GMM2 has its own completion counter");
    using GmmConfig = GmmKernel::Config<false, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB,
                                        IsWeightNZ, IsShared, IsGmm1Interleaved, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm2ProblemConfig(problemShape, blockJob, Gmm1TileM);

    GmmKernel::BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                                        GmmKernel::BlockScheduler::Params{Te::MakeCoord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    if constexpr (0 == COMBINE_NO_QUANT && g_coreType == AscendC::AIV && !NotifyCombineTileReady) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    using BlockMmad = typename GmmConfig::BlockMmad;
    using MmadContext = GmmKernel::BlockMmadContext<BlockMmad>;
    auto* typedBlockMmadContext = reinterpret_cast<MmadContext*>(blockMmadContext);
    GmmKernel::Gmm2ExecGeneric<BlockMmad, ElementC, IsShared, NotifyCombineTileReady>(
        scheduler, gmmAddrInfo, config, startLoopIdx, tileNum, typedBlockMmadContext, allowWeightL2Bypass,
        rowOffsetInExpert, params, gmTileSequence);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          bool IsWeightNZ = false, uint32_t Gmm1TileM = L1_TILE_M_256, bool IsShared = false,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false, bool NotifyCombineTileReady = false>
__aicore__ inline void RunGmm2Generic(const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
                                      const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
                                      void* blockMmadContext = nullptr, bool allowWeightL2Bypass = false,
                                      uint32_t rowOffsetInExpert = 0U, const Params* params = nullptr,
                                      int32_t* gmTileSequence = nullptr)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    RunGmm2Generic<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, IsWeightNZ, Gmm1TileM, IsShared,
                   IsGmm1Interleaved, IsWaveFlagGrained, NotifyCombineTileReady>(
        problemShape, gmmAddrInfo, startBlockIdx, blockJob, blockMmadContext, allowWeightL2Bypass, rowOffsetInExpert,
        params, gmTileSequence);
}

template <typename GenericElementA, typename QuantScaleType, uint32_t Gmm1TileM, bool IsShared,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false, bool NotifyCombineTileReady = false>
__aicore__ inline void RunGmm2A8W8(const GmmExecutionConfig& gmmConfig, const GMMAddrInfo& gmmAddrInfo,
                                   const ProblemShape& problemShape, GmmRuntimeState& runtimeState,
                                   void* persistentBlockMmadContext = nullptr, bool allowWeightL2Bypass = false,
                                   uint32_t rowOffsetInExpert = 0U, const Params* params = nullptr,
                                   int32_t* gmTileSequence = nullptr)
{
    if (gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
        gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        RunGmm2Generic<GenericElementA, GenericElementA, bfloat16_t, QuantScaleType, QuantScaleType, true, Gmm1TileM,
                       IsShared, IsGmm1Interleaved, IsWaveFlagGrained, NotifyCombineTileReady>(
            problemShape, gmmAddrInfo, runtimeState.startBlockIdx, gmmConfig.blockJob, persistentBlockMmadContext,
            allowWeightL2Bypass, rowOffsetInExpert, params, gmTileSequence);
    } else {
        RunGmm2Generic<GenericElementA, GenericElementA, bfloat16_t, QuantScaleType, QuantScaleType, false, Gmm1TileM,
                       IsShared, IsGmm1Interleaved, IsWaveFlagGrained, NotifyCombineTileReady>(
            problemShape, gmmAddrInfo, runtimeState.startBlockIdx, gmmConfig.blockJob, persistentBlockMmadContext,
            allowWeightL2Bypass, rowOffsetInExpert, params, gmTileSequence);
    }
}
}  // namespace MegaMoeImpl
