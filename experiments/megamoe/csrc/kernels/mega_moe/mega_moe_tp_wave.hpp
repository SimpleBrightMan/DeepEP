// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

/*!
 * \file mega_moe_tp_wave.hpp
 * \brief TP MegaMoE (A8W4) 端到端四阶段 Wave 编排器：
 *        Phase 0: InputPrep + FlagReset
 *        Phase 1: AIC Shared GMM1 || 64 AIV (CrossRankSync -> AllGather || RouteTable)
 *        Phase 2: AIC+AIV0 Routed GMM1/GMM2 Wave (lag=1) || AIV1 (Routed SwiGLU > Shared SwiGLU > Wave Permute)
 *        Phase 3: AIC Shared GMM2 || 64 AIV (Fused Unpermute+Send -> Wait+LocalReduce)
 */

#ifndef MEGA_MOE_TP_WAVE_HPP
#define MEGA_MOE_TP_WAVE_HPP

#include <c_api/asc_simd.h>
#include "common/mega_moe_tp_layout.hpp"
#include "common/mega_moe_utils.hpp"
#include "common/mega_moe_schedule_policy.hpp"
#include "common/gmm1_async_completion.hpp"
#include "common/aiv1_activation_ub.hpp"
#include "common/ascend_timer_v2_device.hpp"
#include "mc2_common/mc2_kernel_utils.hpp"
#include "epilogue/block_epilogue_activation_mx_quant.hpp"
#include "stage/mega_moe_gmm1_activation.hpp"
#include "stage/mega_moe_gmm2_combine.hpp"
#include "stage/mega_moe_tp_comm_route.hpp"
#include "stage/mega_moe_tp_shared.hpp"
#include "stage/mega_moe_tp_unpermute_rs.hpp"

namespace MegaMoeImpl {

using namespace AscendC;

class MegaMoeTpWave {
public:
    using QuantOutType = fp8_e4m3fn_t;
    using WeightType = fp4x2_e2m1_t;
    using ScaleType = fp8_e8m0_t;

    static constexpr uint32_t GMM1_TILE_M = L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M = Aiv1ResidentUbLayout::ActivationRows;
    static constexpr bool FUSE_GATE_UP = true;
    static constexpr uint32_t GMM1_TASK_N = GMM1_FUSED_TILE_N;  // 128 logical (paired [gate128, up128] = 256 physical)
    static constexpr bool ASYNC_GMM1 = true;

    using BlockEpilogue =
        BlockEpilogueActivationMxQuant<QuantOutType, bfloat16_t, EPILOGUE_TILE_M, L1_TILE_N, false>;
    using Gmm1Config = GmmKernel::Config<true, QuantOutType, WeightType, bfloat16_t, ScaleType, ScaleType, false, false,
                                         false, true, FUSE_GATE_UP>;
    using Gmm2Config = GmmKernel::Config<true, QuantOutType, WeightType, bfloat16_t, ScaleType, ScaleType, false, false,
                                         false, true, FUSE_GATE_UP>;

    __aicore__ inline MegaMoeTpWave() = default;

    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR xScales, GM_ADDR topkIds, GM_ADDR topkWeights,
                                GM_ADDR weight1, GM_ADDR weight2, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR sharedWeight1, GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1,
                                GM_ADDR sharedWeightScales2, GM_ADDR yOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR workspaceGm, const MegaMoeTpTilingData* tilingData, GM_ADDR timerGm)
    {
        timer_.Init(timerGm);
        tiling_ = *tilingData;
        mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
        rankId_ = mc2Context_->epRankId;
        for (uint32_t i = 0U; i < tiling_.tpWorldSize; ++i) {
            g_winRankAddr_[i] = reinterpret_cast<GM_ADDR>(mc2Context_->epHcclBuffer_[i]) + EXCEPTION_DUMP_REGION_SIZE;
        }

        xGm_ = x;
        xScalesGm_ = xScales;
        topkIdsGm_ = topkIds;
        topkWeightsGm_ = topkWeights;
        weight1Gm_ = weight1;
        weight2Gm_ = weight2;
        weightScales1Gm_ = weightScales1;
        weightScales2Gm_ = weightScales2;
        sharedWeight1Gm_ = sharedWeight1;
        sharedWeight2Gm_ = sharedWeight2;
        sharedWeightScales1Gm_ = sharedWeightScales1;
        sharedWeightScales2Gm_ = sharedWeightScales2;
        yOutGm_ = yOut;
        expertTokenNumsOutGm_ = expertTokenNumsOut;

        blockIdx_ = static_cast<uint32_t>(GetBlockIdx() / GetTaskRation());
        blockJob_ = {blockIdx_, tiling_.aicNum};
        countWorkspace_ = {blockIdx_, tiling_.aicNum};
        ubLayout_ = MakeTpAiv1ResidentUbLayout(tiling_.h);
        workspace_ = TpWorkspaceInfo(workspaceGm, &tiling_);
        localPeermem_ = TpPeermemInfo(g_winRankAddr_[rankId_], &tiling_);

        const bool preQuant = tiling_.routeWeightBeforeQuant != 0U;
        epilogueOp_.Init({.yGmAddr = workspace_.actQuantDataPtr,
                          .yScaleGmAddr = workspace_.actQuantScalePtr,
                          .clampLimit = tiling_.clampLimit,
                          .actMode = tiling_.actMode,
                          .actSubMode = tiling_.actSubMode,
                          .activationAlpha = tiling_.activationAlpha,
                          .activationBeta = tiling_.activationBeta,
                          .ubBaseBytes = 0U,
                          .routeWeightBeforeQuant = preQuant,
                          .deepInferQuant = preQuant});
        if (tiling_.sharedExpertNum > 0U &&
            tiling_.sharedPrecisionMode == static_cast<uint8_t>(SharedPrecisionMode::FP8_MX)) {
            sharedEpilogueOp_.Init({.yGmAddr = workspace_.sharedActDataPtr,
                                    .yScaleGmAddr = workspace_.sharedActScalePtr,
                                    .clampLimit = tiling_.clampLimit,
                                    .actMode = tiling_.actMode,
                                    .actSubMode = tiling_.actSubMode,
                                    .activationAlpha = tiling_.activationAlpha,
                                    .activationBeta = tiling_.activationBeta,
                                    .ubBaseBytes = 0U,
                                    .routeWeightBeforeQuant = false,
                                    .deepInferQuant = preQuant});
        }
    }

    __aicore__ inline void Process()
    {
        timer_.Start(AscendTimer::KERNEL_TIMING);
        const int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
        SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);

        // Phase 0: 连续 flag 清零（结束后立刻放行 AIC）+ AIV 本地量化/拷贝
        ProcessTpInputPrep(tiling_, rankId_, xGm_, xScalesGm_, topkIdsGm_, topkWeightsGm_, localPeermem_, workspace_,
                           timer_);

        // Phase 1: AIC 执行 Shared GMM1 || 64 AIV 执行 CrossRankSync + (AIV0 AllGather || AIV1 RouteTable)
        int32_t epoch = 0;
        if constexpr (g_coreType == AIC) {
            if (tiling_.sharedExpertNum > 0U) {
                // BF16+直读输入 或 FP8_MX+预量化输入 时直接读取用户 xGm/xScalesGm，无需等待 AIV_QUANT；
                // 其余路径通过 flagLocalPrepDonePtr 软同步等待 AIV_QUANT 落盘 sharedBf16InputPtr。
                const bool sharedBf16Direct =
                    (tiling_.sharedPrecisionMode == static_cast<uint8_t>(SharedPrecisionMode::BF16)) &&
                    (xScalesGm_ == nullptr);
                const bool sharedFp8Direct =
                    (tiling_.sharedPrecisionMode == static_cast<uint8_t>(SharedPrecisionMode::FP8_MX)) &&
                    (xScalesGm_ != nullptr);
                if (!sharedBf16Direct && !sharedFp8Direct) {
                    WaitUntilGmFlagIsNonZero(reinterpret_cast<__gm__ int32_t*>(workspace_.flagLocalPrepDonePtr));
                    PipeBarrier<PIPE_ALL>();
                }
                ProcessTpSharedGmm1(tiling_, rankId_, blockIdx_, xGm_, xScalesGm_, sharedWeight1Gm_,
                                    sharedWeightScales1Gm_, workspace_, timer_);
            }
        } else {
            epoch = ProcessTpAllGatherAndRoute(tiling_, rankId_, expertTokenNumsOutGm_, localPeermem_, workspace_,
                                               timer_);
        }

        // 等待路由表与 AllGather 就绪（AIV1 无需等待 AIC 的 Shared GMM1，可直接抢跑 Wave 0 Permute!）
        auto* routeReadyFlag = reinterpret_cast<__gm__ int32_t*>(workspace_.flagRouteReadyPtr);
        WaitUntilGmFlagIsNonZero(routeReadyFlag);
        const int32_t routeState = ReadGmByPassDCache(routeReadyFlag);
        PipeBarrier<PIPE_ALL>();
        const bool hasRoutedWork = (routeState == 2);

        // Phase 2: 路由专家 Wave 流水 + 共享专家 SwiGLU
        int32_t gmm1ReadySequence = 0;
        if (hasRoutedWork) {
            if constexpr (g_coreType == AIV) {
                if (GetSubBlockIdx() == 1U) {
                    ProcessTpAiv1PriorityStages(gmm1ReadySequence);
                } else {
                    ProcessTpGmmWaves(gmm1ReadySequence);
                }
            } else {
                ProcessTpGmmWaves(gmm1ReadySequence);
            }
        } else {
            if constexpr (g_coreType == AIV) {
                if (GetSubBlockIdx() == 1U && tiling_.sharedExpertNum > 0U) {
                    TpSharedSwiGluCursor sharedCursor{};
                    Aiv1ActivationUb::Init();
                    while (!sharedCursor.done) {
                        if (!TryRunTpSharedSwiGlu(tiling_, blockIdx_, workspace_, sharedEpilogueOp_, sharedCursor,
                                                  timer_)) {
                            const int64_t start = GetSystemCycle();
                            while (GetSystemCycle() - start < GM_FLAG_POLL_BACKOFF_CYCLES) {
                            }
                        }
                    }
                    Aiv1ActivationUb::Finish();
                }
            }
        }

        // Phase 3: AIC 执行 Shared GMM2 || 64 AIV 执行 Fused Unpermute+Send -> Wait+LocalReduce
        if constexpr (g_coreType == AIC) {
            if (tiling_.sharedExpertNum > 0U) {
                ProcessTpSharedGmm2(tiling_, blockIdx_, sharedWeight2Gm_, sharedWeightScales2Gm_, workspace_, timer_);
            }
        } else {
            ProcessTpUnpermuteAndReduceScatter(tiling_, rankId_, yOutGm_, localPeermem_, workspace_, epoch,
                                               hasRoutedWork, timer_);
        }

        PipeBarrier<PIPE_ALL>();
        SyncAll<false>();
        SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
        timer_.End(AscendTimer::KERNEL_TIMING);
        timer_.FlushDynamicCounts();
    }

private:
    struct TpActivationCursor {
        ExpertTokenPosition position{};
        ExpertTokenRange slice{};
        uint64_t wave = 0U;
        uint32_t mGroups = 0U;
        uint32_t startBlockIdx = 0U;
        uint32_t nextTile = 0U;
        uint32_t tileCount = 0U;
        uint32_t sliceSeq = 0U;
        BlockJobContext job{};
    };

    // 双重负载均衡映射：
    //   1) GMM1 正向分配 (0 -> aicNum-1)，GMM2 反向分配 (aicNum-1 -> 0)，错峰抵消单核余数尾块；
    //   2) 按 sliceSeq 奇偶翻转最低位 (^ (sliceSeq & 1U))，彻底打破偶数 tileNum 与偶数 aicNum 导致的
    //      “偶数核永远吃满 256 行 tile、奇数核永远吃尾块”的奇偶锁相。
    __aicore__ inline BlockJobContext MakeTpGmmJob(bool reverseBlocks, uint32_t sliceSeq) const
    {
        uint32_t logical = reverseBlocks ? (tiling_.aicNum - 1U - blockIdx_) : blockIdx_;
        if ((tiling_.aicNum & 1U) == 0U) {
            logical ^= (sliceSeq & 1U);
        }
        return {logical, tiling_.aicNum};
    }

    __aicore__ inline bool HasTpWaveCapacity(const ExpertTokenPosition& pos, uint32_t usedMGroups) const
    {
        return MegaMoeImpl::HasWaveCapacity(pos, tiling_.numExperts, usedMGroups, tiling_.mGroupsPerWave);
    }

    __aicore__ inline void BindTpRoutedGmm1Addr(const ExpertContext& expertCtx, uint32_t rowOffsetInExpert,
                                                GMMAddrInfo& addr)
    {
        const uint64_t n1 = tiling_.routedGmm1OutDim;
        const uint64_t k = tiling_.h;
        const uint64_t scaleK = static_cast<uint64_t>(CalcTpMxScaleStrideBytes(k));
        const uint64_t interN = n1 / ACTIVATION_N_HALF;
        const uint64_t scaleInter = static_cast<uint64_t>(CalcTpMxScaleStrideBytes(interN));
        const uint64_t expIdx = expertCtx.expertIdx;
        const uint64_t globalRow = static_cast<uint64_t>(expertCtx.globalTokenStartIndex) + rowOffsetInExpert;
        const uint32_t mGroupOffset = rowOffsetInExpert / L1_TILE_M_256;

        addr.expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(expertCtx.problemShape));
        addr.aGlobal = workspace_.permutedXPtr + globalRow * k;
        addr.aScaleGlobal = workspace_.permutedScalePtr + globalRow * scaleK;
        addr.bGlobal = weight1Gm_ + expIdx * n1 * (k / 2U);
        addr.bScaleGlobal = weightScales1Gm_ + expIdx * n1 * scaleK;
        addr.gmm1OutGlobal = workspace_.gmm1OutPtr + globalRow * n1 * sizeof(bfloat16_t);
        addr.metaInfoGlobal = workspace_.metaInfoPtr + globalRow * META_INFO_SIZE * sizeof(int32_t);
        addr.activationToGmm2Flag = reinterpret_cast<__gm__ int32_t*>(workspace_.flagActToGmm2Ptr) +
                                    expIdx * workspace_.mGroupFlagSlotsPerExpert +
                                    static_cast<uint64_t>(mGroupOffset) * INT_CACHELINE;
        addr.dispatchToGmm1Flag = reinterpret_cast<__gm__ int32_t*>(workspace_.flagPermuteToGmm1Ptr) +
                                  expIdx * workspace_.mGroupFlagSlotsPerExpert +
                                  static_cast<uint64_t>(mGroupOffset) * INT_CACHELINE;
        addr.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t*>(workspace_.flagGmm1IssuedPtr) +
                                 static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;

        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U) {
                AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                    static_cast<int64_t>(globalRow * interN),
                    static_cast<int64_t>(globalRow * scaleInter),
                    static_cast<int64_t>(expIdx * (workspace_.mGroupFlagSlotsPerExpert / INT_CACHELINE) + mGroupOffset),
                    0L, 0L, 0L};
                epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
            }
        }
    }

    __aicore__ inline void BindTpRoutedGmm2Addr(const ExpertContext& expertCtx, uint32_t rowOffsetInExpert,
                                                GMMAddrInfo& addr)
    {
        const uint64_t n1 = tiling_.routedGmm1OutDim;
        const uint64_t interK = n1 / ACTIVATION_N_HALF;
        const uint64_t scaleInter = static_cast<uint64_t>(CalcTpMxScaleStrideBytes(interK));
        const uint64_t outN = tiling_.h;
        const uint64_t expIdx = expertCtx.expertIdx;
        const uint64_t globalRow = static_cast<uint64_t>(expertCtx.globalTokenStartIndex) + rowOffsetInExpert;
        const uint32_t mGroupOffset = rowOffsetInExpert / L1_TILE_M_256;

        addr.expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(expertCtx.problemShape));
        addr.aGlobal = workspace_.actQuantDataPtr + globalRow * interK;
        addr.aScaleGlobal = workspace_.actQuantScalePtr + globalRow * scaleInter;
        addr.bGlobal = weight2Gm_ + expIdx * outN * (interK / 2U);
        addr.bScaleGlobal = weightScales2Gm_ + expIdx * outN * scaleInter;
        addr.gmm2OutGlobal = workspace_.gmm2OutPtr + globalRow * outN * sizeof(bfloat16_t);
        addr.metaInfoGlobal = workspace_.metaInfoPtr + globalRow * META_INFO_SIZE * sizeof(int32_t);
        addr.activationToGmm2Flag = reinterpret_cast<__gm__ int32_t*>(workspace_.flagActToGmm2Ptr) +
                                    expIdx * workspace_.mGroupFlagSlotsPerExpert +
                                    static_cast<uint64_t>(mGroupOffset) * INT_CACHELINE;
        addr.dispatchToGmm1Flag = nullptr;
        addr.gmmToEpilogueFlag = nullptr;
    }

    __aicore__ inline bool NextTpActivationSlice(TpActivationCursor& cursor)
    {
        const uint32_t gmm1NTiles =
            Ops::Base::CeilDiv(tiling_.routedGmm1OutDim / ACTIVATION_N_HALF, static_cast<uint32_t>(GMM1_TASK_N));
        while (true) {
            if (!HasTpWaveCapacity(cursor.position, cursor.mGroups) && cursor.mGroups != 0U) {
                cursor.mGroups = 0U;
                ++cursor.wave;
            }
            if (cursor.position.expertIdx >= tiling_.numExperts) {
                return false;
            }
            cursor.slice = PlanNextExpertTokenRangeInWave<GMM1_TILE_M>(
                workspace_.expertRevTokenNumsPtr, countWorkspace_, tiling_.numExperts, tiling_.mGroupsPerWave,
                cursor.mGroups, cursor.position);
            cursor.position = cursor.slice.end;
            const uint32_t rows =
                static_cast<uint32_t>(cursor.slice.end.globalTokenIndex - cursor.slice.begin.globalTokenIndex);
            if (rows == 0U) {
                continue;
            }
            cursor.job = MakeTpGmmJob(false, cursor.sliceSeq++);
            cursor.tileCount = GetMGroupCountForRows(rows, GMM1_TILE_M) * gmm1NTiles;
            cursor.nextTile = (cursor.job.jobIndex + tiling_.aicNum - cursor.startBlockIdx) % tiling_.aicNum;
            cursor.startBlockIdx = (cursor.startBlockIdx + cursor.tileCount) % tiling_.aicNum;
            if (cursor.nextTile < cursor.tileCount) {
                return true;
            }
        }
    }

    __aicore__ inline bool RetireTpRoutedActivation(__gm__ int32_t*& readyFlag, bool& pending)
    {
        if (!pending) {
            return false;
        }
        timer_.SetWave(AscendTimer::NO_WAVE);
        const int boundary = timer_.BeginBoundary(AscendTimer::AIV_SWIGLU_SYNC);
        WaitFlag<HardEvent::MTE3_S>(static_cast<TEventID>(Aiv1AsyncEvents::ActivationDone));
        if (readyFlag != nullptr) {
            AtomicAdd(readyFlag, 1);
        }
        pending = false;
        timer_.EndBoundary(AscendTimer::AIV_SWIGLU_SYNC, boundary);
        return true;
    }

    __aicore__ inline bool TryRunTpRoutedActivation(TpActivationCursor& cursor, int32_t& readySequence,
                                                    __gm__ int32_t*& outReadyFlag, bool& outPending)
    {
        auto* issuedFlag = reinterpret_cast<__gm__ int32_t*>(workspace_.flagGmm1IssuedPtr) +
                           static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
        if (ReadGmByPassDCache(issuedFlag) < readySequence + 1) {
            return false;
        }

        const auto& begin = cursor.slice.begin;
        const uint32_t rows = static_cast<uint32_t>(cursor.slice.end.globalTokenIndex - begin.globalTokenIndex);
        ExpertContext expertCtx{};
        expertCtx.expertIdx = begin.expertIdx;
        expertCtx.globalTokenStartIndex = begin.globalTokenIndex - begin.tokenIndexInExpert;
        Get<M_VALUE>(expertCtx.problemShape) = rows;
        Get<N_VALUE>(expertCtx.problemShape) = tiling_.routedGmm1OutDim;
        Get<K_VALUE>(expertCtx.problemShape) = tiling_.h;

        GMMAddrInfo addr{};
        BindTpRoutedGmm1Addr(expertCtx, begin.tokenIndexInExpert, addr);
        addr.timer = timer_.Enabled() ? &timer_ : nullptr;

        timer_.SetWave(static_cast<int64_t>(cursor.wave));
        timer_.SetSlice(static_cast<int>(begin.expertIdx), begin.tokenIndexInExpert, rows);
        if constexpr (ASYNC_GMM1) {
            Gmm1AsyncCompletion::WaitConsumer(static_cast<unsigned>(readySequence));
        }
        outReadyFlag =
            RunReadyGmm1ActivationTileA8W4<QuantOutType, WeightType, bfloat16_t, ScaleType, ScaleType, GMM1_TILE_M,
                                           FUSE_GATE_UP>(epilogueOp_, expertCtx.problemShape, addr, cursor.job,
                                                         cursor.nextTile, static_cast<uint32_t>(begin.globalTokenIndex),
                                                         AscendTimer::AIV_SWIGLU_ISSUE, true);
        SetFlag<HardEvent::MTE3_S>(static_cast<TEventID>(Aiv1AsyncEvents::ActivationDone));
        outPending = true;
        timer_.ClearSlice();
        if constexpr (ASYNC_GMM1) {
            Gmm1AsyncCompletion::ReleaseConsumer(static_cast<unsigned>(readySequence));
        }
        ++readySequence;
        return true;
    }

    // Phase 2 AIV1 优先级循环：Routed SwiGLU (最高) > Shared SwiGLU > Wave Permute（WAVE 粒度打点）
    __aicore__ inline void ProcessTpAiv1PriorityStages(int32_t& readySequence)
    {
        TpActivationCursor actCursor{};
        bool hasAct = NextTpActivationSlice(actCursor);
        __gm__ int32_t* actReadyFlag = nullptr;
        bool actPending = false;

        TpSharedSwiGluCursor sharedCursor{};
        sharedCursor.done = (tiling_.sharedExpertNum == 0U);

        ExpertTokenPosition permPos{};
        uint32_t permGroups = 0U;
        uint64_t permWave = 0U;
        TpPermuteAsyncState permAsync{};
        bool permTimerOpen = false;
        uint64_t permTimerWave = 0U;

        TpEarlyUnpermCursor earlyUnperm{};
        earlyUnperm.Init(tiling_, blockIdx_);
        bool allGmm2Done = false;
        auto checkAllGmm2Done = [&]() __aicore__ -> bool {
            if (allGmm2Done) {
                return true;
            }
            auto* doneFlags = reinterpret_cast<__gm__ int32_t*>(workspace_.flagRoutedGmm2DonePtr);
            for (uint32_t c = 0U; c < tiling_.aicNum; ++c) {
                if (ReadGmByPassDCache(doneFlags + static_cast<uint64_t>(c) * INT_CACHELINE) < 1) {
                    return false;
                }
            }
            allGmm2Done = true;
            return true;
        };

        auto closePermTimer = [&]() __aicore__ {
            if (permTimerOpen) {
                timer_.EndSpan(AscendTimer::AIV_PERMUTE);
                permTimerOpen = false;
            }
        };
        auto ensurePermTimer = [&](uint64_t wave) __aicore__ {
            if (permTimerOpen && permTimerWave != wave) {
                timer_.EndSpan(AscendTimer::AIV_PERMUTE);
                permTimerOpen = false;
            }
            if (!permTimerOpen) {
                timer_.SetWave(static_cast<int64_t>(wave));
                timer_.StartSpan(AscendTimer::AIV_PERMUTE);
                permTimerOpen = true;
                permTimerWave = wave;
            }
        };

        if constexpr (ASYNC_GMM1) {
            Gmm1AsyncCompletion::InitConsumer();
        }
        Aiv1ActivationUb::Init();

        while (hasAct || actPending || !sharedCursor.done || permAsync.HasPending() ||
               permPos.expertIdx < tiling_.numExperts || (!earlyUnperm.done && !checkAllGmm2Done())) {
            bool progressed = false;

            if (actPending) {
                closePermTimer();
                progressed |= RetireTpRoutedActivation(actReadyFlag, actPending);
            }

            // Routed GMM1 tile 是否已发布：先探测再刷出在途 Permute 并关闭区间
            if (hasAct) {
                auto* issuedFlag = reinterpret_cast<__gm__ int32_t*>(workspace_.flagGmm1IssuedPtr) +
                                   static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
                if (ReadGmByPassDCache(issuedFlag) >= readySequence + 1) {
                    if (permAsync.HasPending()) {
                        ensurePermTimer(permWave);
                        FlushAndRetireTpPermute(tiling_, ubLayout_, workspace_, permAsync);
                    }
                    closePermTimer();
                    if (TryRunTpRoutedActivation(actCursor, readySequence, actReadyFlag, actPending)) {
                        actCursor.nextTile += tiling_.aicNum;
                        if (actCursor.nextTile >= actCursor.tileCount) {
                            hasAct = NextTpActivationSlice(actCursor);
                        }
                        progressed = true;
                        continue;
                    }
                }
            }

            // Shared SwiGLU 探测（仅在 Wave 0 Permute 完成后再抢占，避免拖慢首 Wave 路由输入供给）
            if (!sharedCursor.done && (permWave > 0U || permPos.expertIdx >= tiling_.numExperts)) {
                bool canRunShared = sharedCursor.gmm1Ready;
                if (!canRunShared) {
                    canRunShared = true;
                    auto* flags = reinterpret_cast<__gm__ int32_t*>(workspace_.flagSharedGmm1DonePtr);
                    for (uint32_t c = 0U; c < tiling_.aicNum; ++c) {
                        if (ReadGmByPassDCache(flags + static_cast<uint64_t>(c) * INT_CACHELINE) < 1) {
                            canRunShared = false;
                            break;
                        }
                    }
                }
                if (canRunShared) {
                    if (permAsync.HasPending()) {
                        ensurePermTimer(permWave);
                        FlushAndRetireTpPermute(tiling_, ubLayout_, workspace_, permAsync);
                    }
                    closePermTimer();
                    if (TryRunTpSharedSwiGlu(tiling_, blockIdx_, workspace_, sharedEpilogueOp_, sharedCursor, timer_)) {
                        progressed = true;
                        continue;
                    }
                }
            }

            // 跨 Slice 不断流 Permute：6 槽环跨专家 slice 保持在途，仅在批次阈值或 Wave 末尾刷盘发布
            if (permPos.expertIdx < tiling_.numExperts) {
                if (!HasTpWaveCapacity(permPos, permGroups) && permGroups != 0U) {
                    if (permAsync.HasPending()) {
                        ensurePermTimer(permWave);
                        FlushAndRetireTpPermute(tiling_, ubLayout_, workspace_, permAsync);
                    }
                    closePermTimer();
                    permGroups = 0U;
                    ++permWave;
                }
                const ExpertTokenRange range = PlanNextExpertTokenRangeInWave<GMM1_TILE_M>(
                    workspace_.expertRevTokenNumsPtr, countWorkspace_, tiling_.numExperts, tiling_.mGroupsPerWave,
                    permGroups, permPos);
                permPos = range.end;
                const uint32_t rows =
                    static_cast<uint32_t>(range.end.globalTokenIndex - range.begin.globalTokenIndex);
                if (rows > 0U) {
                    ensurePermTimer(permWave);
                    const ExpertSlice slice{range.begin.expertIdx, range.begin.tokenIndexInExpert, rows,
                                            range.begin.globalTokenIndex};
                    IssueTpPermuteSlice(tiling_, ubLayout_, workspace_, blockIdx_, slice, permWave, permAsync);
                    const uint32_t flushThreshold = (permWave == 0U) ? 4U : 8U;
                    const bool waveDone =
                        !HasTpWaveCapacity(permPos, permGroups) || (permPos.expertIdx >= tiling_.numExperts);
                    if (waveDone || permAsync.pendingSliceCount >= flushThreshold) {
                        FlushAndRetireTpPermute(tiling_, ubLayout_, workspace_, permAsync);
                    }
                    if (waveDone) {
                        closePermTimer();
                        if (permPos.expertIdx < tiling_.numExperts) {
                            permGroups = 0U;
                            ++permWave;
                        }
                    }
                } else if (permPos.expertIdx >= tiling_.numExperts) {
                    if (permAsync.HasPending()) {
                        ensurePermTimer(permWave);
                        FlushAndRetireTpPermute(tiling_, ubLayout_, workspace_, permAsync);
                    }
                    closePermTimer();
                }
                progressed = true;
                continue;
            }

            if (permAsync.HasPending()) {
                ensurePermTimer(permWave);
                FlushAndRetireTpPermute(tiling_, ubLayout_, workspace_, permAsync);
                closePermTimer();
                progressed = true;
                continue;
            }

            // P1：Permute 与 Shared SwiGLU 全部完成后，利用 Phase 2 后半段 AIV1 空闲窗口抢跑已完成专家的部分 Unpermute
            if (!earlyUnperm.done && sharedCursor.done) {
                if (!earlyUnperm.permDrained) {
                    DrainTpPermuteBuffers(permAsync);
                    earlyUnperm.permDrained = true;
                }
                if (!checkAllGmm2Done() &&
                    StepTpEarlySplitUnpermute(tiling_, rankId_, workspace_, earlyUnperm, timer_)) {
                    progressed = true;
                    continue;
                }
            }

            if (!progressed) {
                const int64_t start = GetSystemCycle();
                while (GetSystemCycle() - start < GM_FLAG_POLL_BACKOFF_CYCLES) {
                }
            }
        }

        closePermTimer();
        Aiv1ActivationUb::Finish();
        if (!earlyUnperm.permDrained) {
            DrainTpPermuteBuffers(permAsync);
        }
        timer_.ClearSlice();
        timer_.SetWave(AscendTimer::NO_WAVE);
    }

    __aicore__ inline void PrepareTpExpertContext(ExpertContext& ctx, uint32_t expertIdx)
    {
        const uint32_t cnt = GetExpertTokenCountFromWorkspace(workspace_.expertRevTokenNumsPtr, countWorkspace_,
                                                              tiling_.numExperts, expertIdx);
        UpdateExpertContext(ctx, expertIdx, cnt);
    }

    template <typename Gmm1Kernel>
    __aicore__ inline void RunTpGmm1Slice(const ExpertContext& expertCtx, const ExpertSlice& slice,
                                          GMMAddrInfo& gmm1Addr, uint32_t& startBlockIdx, int32_t& readySeq,
                                          uint32_t sliceSeq, Gmm1Kernel& gmm1Kernel)
    {
        const BlockJobContext gmm1Job = MakeTpGmmJob(false, sliceSeq);
        const uint32_t gmm1NTiles =
            Ops::Base::CeilDiv(tiling_.routedGmm1OutDim / ACTIVATION_N_HALF, static_cast<uint32_t>(GMM1_TASK_N));
        const uint32_t tileCount = GetMGroupCountForRows(slice.rowCount, GMM1_TILE_M) * gmm1NTiles;
        if (HandleWaveProblemWithoutWork(tileCount, gmm1Job, startBlockIdx)) {
            return;
        }
        BindTpRoutedGmm1Addr(expertCtx, slice.rowStartInExpert, gmm1Addr);
        gmm1Addr.timer = timer_.Enabled() ? &timer_ : nullptr;
        gmm1Addr.gmm1AsyncCompletion = ASYNC_GMM1;

        ProblemShape sliceShape = expertCtx.problemShape;
        Get<M_VALUE>(sliceShape) = slice.rowCount;
        Params dummyParams{};

        timer_.SetSlice(static_cast<int>(slice.expertIdx), slice.rowStartInExpert, slice.rowCount);
        RunGmm1A8W4<QuantOutType, WeightType, bfloat16_t, ScaleType, ScaleType, GMM1_TILE_M, EPILOGUE_TILE_M, false,
                    true, FUSE_GATE_UP, Gmm1Kernel, true>(
            epilogueOp_, dummyParams, sliceShape, gmm1Addr, startBlockIdx, readySeq, gmm1Job,
            static_cast<uint32_t>(slice.globalRowStart), slice.expertIdx, &gmm1Kernel);
        timer_.ClearSlice();
    }

    template <typename Gmm2Kernel>
    __aicore__ inline void RunTpGmm2Slice(const ExpertContext& expertCtx, const ExpertSlice& slice,
                                          GMMAddrInfo& gmm2Addr, uint32_t& startBlockIdx, uint32_t sliceSeq,
                                          Gmm2Kernel& gmm2Kernel)
    {
        const BlockJobContext gmm2Job = MakeTpGmmJob(true, sliceSeq);
        const uint32_t nTiles = Ops::Base::CeilDiv(tiling_.h, static_cast<uint32_t>(L1_TILE_N));
        const uint32_t tileCount = GetMGroupCountForRows(slice.rowCount, GMM1_TILE_M) * nTiles;
        if (HandleWaveProblemWithoutWork(tileCount, gmm2Job, startBlockIdx, false)) {
            return;
        }
        BindTpRoutedGmm2Addr(expertCtx, slice.rowStartInExpert, gmm2Addr);
        gmm2Addr.timer = timer_.Enabled() ? &timer_ : nullptr;

        ProblemShape sliceShape = expertCtx.problemShape;
        Get<M_VALUE>(sliceShape) = slice.rowCount;

        timer_.SetSlice(static_cast<int>(slice.expertIdx), slice.rowStartInExpert, slice.rowCount);
        RunGmm2A8W4<QuantOutType, WeightType, bfloat16_t, ScaleType, ScaleType, GMM1_TILE_M, false, true, false, true,
                    FUSE_GATE_UP, Gmm2Kernel, true>(
            sliceShape, gmm2Addr, startBlockIdx, gmm2Job,
            static_cast<uint32_t>(Get<M_VALUE>(expertCtx.problemShape)), slice.rowStartInExpert, nullptr, nullptr,
            false, &gmm2Kernel);
        timer_.ClearSlice();
    }

    template <typename Gmm1Kernel, typename Gmm2Kernel>
    __aicore__ inline void RunTpGmm2Wave(const ExpertWaveRange& wave, ExpertContext& gmm2Ctx, GMMAddrInfo& gmm2Addr,
                                         uint32_t& startBlockIdx, uint32_t& sliceSeq, Gmm1Kernel& gmm1Kernel,
                                         Gmm2Kernel& gmm2Kernel)
    {
        const uint32_t endExpExclusive = wave.end.expertIdx + (wave.end.tokenIndexInExpert == 0U ? 0U : 1U);
        if constexpr (ASYNC_GMM1) {
            gmm1Kernel.HandoffTo(gmm2Kernel);
        }
        for (uint32_t expIdx = wave.begin.expertIdx; expIdx < endExpExclusive; ++expIdx) {
            const bool startsAtBoundary = (expIdx != wave.begin.expertIdx) || (wave.begin.tokenIndexInExpert == 0U);
            if (startsAtBoundary) {
                PrepareTpExpertContext(gmm2Ctx, expIdx);
            }
            const uint32_t expTokens = static_cast<uint32_t>(Get<M_VALUE>(gmm2Ctx.problemShape));
            const ExpertSlice slice = GetExpertSliceInWave(wave, expIdx, expTokens, gmm2Ctx.globalTokenStartIndex);
            if (slice.rowCount > 0U) {
                RunTpGmm2Slice(gmm2Ctx, slice, gmm2Addr, startBlockIdx, sliceSeq++, gmm2Kernel);
            }
        }
        if constexpr (ASYNC_GMM1) {
            gmm2Kernel.HandoffTo(gmm1Kernel);
        }
    }

    // Phase 2 AIC + AIV0 确定性 GMM1/GMM2 Wave 交错流水（局部构造 CATLASS 对象，严格隔离 Phase 1/3 硬件事件）
    __aicore__ inline void ProcessTpGmmWaves(int32_t& gmm1ReadySequence)
    {
        GmmKernel::CatlassGmm1<typename Gmm1Config::ProblemConfig> gmm1Kernel;
        GmmKernel::CatlassGmm2<typename Gmm2Config::ProblemConfig> gmm2Kernel;

        ExpertContext gmm1Ctx{};
        Get<N_VALUE>(gmm1Ctx.problemShape) = tiling_.routedGmm1OutDim;
        Get<K_VALUE>(gmm1Ctx.problemShape) = tiling_.h;
        ExpertContext gmm2Ctx = gmm1Ctx;

        GMMAddrInfo gmm1Addr{};
        GMMAddrInfo gmm2Addr{};
        uint32_t gmm1StartBlock = 0U;
        uint32_t gmm2StartBlock = 0U;
        uint32_t gmm1SliceSeq = 0U;
        uint32_t gmm2SliceSeq = 0U;
        uint64_t gmm1WaveCount = 0U;
        uint64_t gmm2WaveCount = 0U;

        const bool gmm2Lag = (tiling_.bs * tiling_.tpWorldSize >= 512U);
        const uint32_t splitExpert = tiling_.numExperts > 1U
                                         ? Min(tiling_.numExperts - 1U, Max(1U, tiling_.numExperts / 2U))
                                         : 1U;
        bool splitPublished = !gmm2Lag || (tiling_.numExperts < 2U);
        ExpertWaveRange pendingWave{{}, {}, tiling_.numExperts};
        ExpertTokenPosition gmm1Cursor{};

        while (gmm1Cursor.expertIdx < tiling_.numExperts) {
            ExpertWaveRange wave{gmm1Cursor, gmm1Cursor, tiling_.numExperts};
            uint32_t usedMGroups = 0U;
            while (HasTpWaveCapacity(wave.end, usedMGroups)) {
                if (wave.end.tokenIndexInExpert == 0U) {
                    PrepareTpExpertContext(gmm1Ctx, wave.end.expertIdx);
                }
                const uint32_t expTokens = static_cast<uint32_t>(Get<M_VALUE>(gmm1Ctx.problemShape));
                const ExpertSlice slice =
                    TakeNextExpertSliceInWave<GMM1_TILE_M>(expTokens, tiling_.mGroupsPerWave, usedMGroups, wave.end);
                if (slice.rowCount > 0U) {
                    wave.lastActiveExpert = slice.expertIdx;
                    timer_.SetWave(static_cast<int64_t>(gmm1WaveCount));
                    RunTpGmm1Slice(gmm1Ctx, slice, gmm1Addr, gmm1StartBlock, gmm1ReadySequence, gmm1SliceSeq++,
                                   gmm1Kernel);
                }
            }
            gmm1Cursor = wave.end;
            ++gmm1WaveCount;

            if (!gmm2Lag || gmm1WaveCount > 1U) {
                const ExpertWaveRange& curGmm2Wave = gmm2Lag ? pendingWave : wave;
                timer_.SetWave(static_cast<int64_t>(gmm2WaveCount));
                RunTpGmm2Wave(curGmm2Wave, gmm2Ctx, gmm2Addr, gmm2StartBlock, gmm2SliceSeq, gmm1Kernel, gmm2Kernel);
                ++gmm2WaveCount;
                if (!splitPublished && curGmm2Wave.end.expertIdx >= splitExpert) {
                    splitPublished = true;
                    if constexpr (g_coreType == AIC) {
                        SetFlag<HardEvent::FIX_S>(0);
                        WaitFlag<HardEvent::FIX_S>(0);
                        asc_sync_data_barrier(DSB_DDR);
                        WriteGmByPassDCache(
                            reinterpret_cast<__gm__ int32_t*>(workspace_.flagGmm2SplitDonePtr) +
                                static_cast<uint64_t>(blockIdx_) * INT_CACHELINE,
                            static_cast<int32_t>(curGmm2Wave.end.globalTokenIndex) + 1);
                    }
                }
            }
            if (gmm2Lag) {
                pendingWave = wave;
            }
        }

        if (gmm2WaveCount < gmm1WaveCount) {
            timer_.SetWave(static_cast<int64_t>(gmm2WaveCount));
            RunTpGmm2Wave(pendingWave, gmm2Ctx, gmm2Addr, gmm2StartBlock, gmm2SliceSeq, gmm1Kernel, gmm2Kernel);
            ++gmm2WaveCount;
        }

        gmm1Kernel.End();
        if constexpr (g_coreType == AIC) {
            if constexpr (ASYNC_GMM1) {
                Gmm1AsyncCompletion::DrainProducer();
            } else {
                SetFlag<HardEvent::FIX_S>(0);
                WaitFlag<HardEvent::FIX_S>(0);
            }
            asc_sync_data_barrier(DSB_DDR);
            WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(workspace_.flagRoutedGmm2DonePtr) +
                                    static_cast<uint64_t>(blockIdx_) * INT_CACHELINE,
                                1);
        }
        timer_.SetWave(AscendTimer::NO_WAVE);
    }

    MegaMoeTpTilingData tiling_{};
    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    uint32_t rankId_{0U};
    uint32_t blockIdx_{0U};
    BlockJobContext blockJob_{};
    BlockWorkspaceContext countWorkspace_{};
    TpAiv1ResidentUbLayout ubLayout_{};
    TpWorkspaceInfo workspace_{};
    TpPeermemInfo localPeermem_{};

    GM_ADDR xGm_{nullptr};
    GM_ADDR xScalesGm_{nullptr};
    GM_ADDR topkIdsGm_{nullptr};
    GM_ADDR topkWeightsGm_{nullptr};
    GM_ADDR weight1Gm_{nullptr};
    GM_ADDR weight2Gm_{nullptr};
    GM_ADDR weightScales1Gm_{nullptr};
    GM_ADDR weightScales2Gm_{nullptr};
    GM_ADDR sharedWeight1Gm_{nullptr};
    GM_ADDR sharedWeight2Gm_{nullptr};
    GM_ADDR sharedWeightScales1Gm_{nullptr};
    GM_ADDR sharedWeightScales2Gm_{nullptr};
    GM_ADDR yOutGm_{nullptr};
    GM_ADDR expertTokenNumsOutGm_{nullptr};

    BlockEpilogue epilogueOp_;
    BlockEpilogue sharedEpilogueOp_;
    AscendTimerDevice timer_;
};

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_TP_WAVE_HPP
