// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

/*!
 * \file mega_moe_wave_a8w4.h
 * \brief MegaMoe A8W4 动态 Wave 调度
 */

#ifndef MEGA_MOE_WAVE_A8W4_H
#define MEGA_MOE_WAVE_A8W4_H

#include "common/mega_moe_utils.hpp"
#include "common/small_linear_task_schedule.hpp"
#include "mega_moe_arch35.hpp"

namespace MegaMoeImpl {

#define TemplateMegaMoeA8W4WaveTypeClass                                                                    \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        bool FuseGateUp, bool UseCv
#define TemplateMegaMoeA8W4WaveTypeFunc XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, FuseGateUp, UseCv

template <TemplateMegaMoeA8W4WaveTypeClass>
class MegaMoeA8W4Wave : public MegaMoe<XType, OutputType, TopkWeightsType, Weight1Type, QuantMode> {
private:
    using MegaMoeBase = MegaMoe<XType, OutputType, TopkWeightsType, Weight1Type, QuantMode>;
    friend MegaMoeBase;

public:
    using MegaMoeBase::Init;
    __aicore__ inline void Process();

private:
    using QuantOutType = typename MegaMoeBase::QuantOutType;
    using ActivationType = typename MegaMoeBase::ActivationType;
    using QuantScaleOutType = typename MegaMoeBase::QuantScaleOutType;
    using ActivationQuantOutType = typename MegaMoeBase::ActivationQuantOutType;

    static constexpr uint32_t A_ELEMS_PER_BYTE = MegaMoeBase::A_ELEMS_PER_BYTE;
    static constexpr uint32_t GMM1_TILE_M = MegaMoeBase::GMM1_TILE_M;
    static constexpr uint32_t EPILOGUE_TILE_M = MegaMoeBase::EPILOGUE_TILE_M;
    // 一个逻辑任务对应一对 gate/up；生产、SwiGLU 和 GMM2 ready 必须使用同一粒度。
    static constexpr uint32_t GMM1_TASK_N = FuseGateUp ? GMM1_FUSED_TILE_N : L1_TILE_N;
    static constexpr bool ASYNC_GMM1 = true;
    // 本消融对齐DS两层同向分核；执行、空任务判断与完成通知必须共用这个方向。
    static constexpr bool GMM2_REVERSE_BLOCKS = false;
    static_assert(!UseCv || FuseGateUp, "CV requires paired weights");

    using Gmm1Config = GmmKernel::Config<true, QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType,
                                         QuantScaleOutType, false, false, false, true, FuseGateUp>;
    GmmKernel::CatlassGmm1<typename Gmm1Config::ProblemConfig> gmm1CatlassContext_;
    using Gmm2Config = GmmKernel::Config<true, ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType,
                                         QuantScaleOutType, false, false, false, true, FuseGateUp>;
    GmmKernel::CatlassGmm2<typename Gmm2Config::ProblemConfig> gmm2WaveContext_;
    using CvActivationOp = BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t,
        Gmm1CvPlan::StepRows, Gmm1CvPlan::Columns, false, true>;
    CvActivationOp cvActivationOp_;
    Gmm1CvTask cvActivation_{};
    static constexpr auto CvScratch = Detail::BuildActivationMxQuantUbOffsets<
        bfloat16_t, Gmm1CvPlan::StepRows, Gmm1CvPlan::Columns, true>();
    static_assert(CvScratch.reciprocalScaleOffsetBytes + CvScratch.scaleElementCapacity * sizeof(uint16_t)
                  <= Gmm1CvPlan::ScratchBytes);

    using MegaMoeBase::commonConfig_;
    using MegaMoeBase::computeReplay_;
    using MegaMoeBase::countWorkspace_;
    using MegaMoeBase::DispatchBuffInit;
    using MegaMoeBase::epilogueOp_;
    using MegaMoeBase::exceptionDump_;
    using MegaMoeBase::gmmExecutionConfig_;
    using MegaMoeBase::gmmLoopCount_;
    using MegaMoeBase::mGroupsPerWave_;
    using MegaMoeBase::moeWeightTensorListAddrs_;
    using MegaMoeBase::params_;
    using MegaMoeBase::RunGmm2CombineForExpert;
    using MegaMoeBase::syncWorkspaceLayout_;
    using MegaMoeBase::timer_;
    using MegaMoeBase::tokenDispatchConfig_;
    using MegaMoeBase::tokenDispatchScratch_;
    using MegaMoeBase::waveCombineJob_;
    using MegaMoeBase::waveCombineScratch_;
    using MegaMoeBase::waveCombineBuffers_;

    using SharedActivationCursor = typename MegaMoeBase::SharedActivationCursor;
    using MegaMoeBase::sharedExpertNum_;
    using MegaMoeBase::TryRunSharedExpertActivation;

    // 各任务流保留一个待办项，不分配随 BS 增长的任务队列。
    struct AivTaskCursor {
        ExpertTokenPosition position{};
        ExpertTokenRange slice{};
        uint64_t wave = 0U;
        uint32_t mGroups = 0U;
        uint32_t startBlockIdx = 0U;
        uint32_t nextTile = 0U;
        uint32_t tileCount = 0U;
        uint32_t nextRow = 0U;
    };
    enum class AivTaskKind {
        Activation,
        Combine,
    };
    DispatchAsyncState dispatchAsync_{};
    uint32_t combineRowSequence_ = 0U;
    __gm__ int32_t* activationReadyFlag_ = nullptr;
    bool activationPending_ = false;
    __aicore__ inline bool RetireActivation();
    __aicore__ inline bool NextAivSlice(AivTaskCursor& cursor, AivTaskKind kind);
    __aicore__ inline bool TryRunReadyActivation(AivTaskCursor& cursor, int32_t& readySequence);
    __aicore__ inline bool TryRunCvActivation(AivTaskCursor& cursor, int32_t& readySequence);
    __aicore__ inline bool TryRunReadyCombine(const AivTaskCursor& cursor);
    __aicore__ inline void ProcessAiv1PriorityStages(int32_t& readySequence);
    __aicore__ inline bool HasWaveCapacity(const ExpertTokenPosition& position, uint32_t waveMGroupCount) const
    {
        return MegaMoeImpl::HasWaveCapacity(position, commonConfig_.moeExpertPerRank, waveMGroupCount, mGroupsPerWave_);
    }

    __aicore__ inline void RunGmm1ActivationForExpert(const ExpertContext& expertContext, const ExpertSlice& slice,
                                                      GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
                                                      int32_t& gmm1TileReadySequence, uint32_t gmm1TilesPerMGroup);
    __aicore__ inline void DispatchNextExpertSlice(ExpertTokenPosition& position, uint32_t& waveMGroupCount,
                                                   uint64_t waveId);
    __aicore__ inline void RunGmm2CombineForWaveRange(const ExpertWaveRange& wave, ExpertContext& gmm2ExpertContext,
                                                      GMMAddrInfo& gmm2AddrInfo, uint32_t& startBlockIdx);
    __aicore__ inline void ProcessMoeExpertStages(int32_t& gmm1TileReadySequence);
    __aicore__ inline void ProcessGmmWaves(int32_t& gmm1TileReadySequence);
    __aicore__ inline uint32_t SmallLinearBlockCount();
    __aicore__ inline void AdvanceSmallLinearExpert(ExpertContext& expert, uint32_t block,
                                                     uint32_t& blockEnd, uint32_t& nextExpert);
    __aicore__ inline void ProcessSmallLinearTasks(uint32_t blocks, int32_t& readySequence);
    __aicore__ inline void ProcessSmallLinearAiv1(uint32_t blocks, int32_t& readySequence);

    __aicore__ inline void RunNextGmm1Slice(ExpertWaveRange& wave, uint32_t& usedMGroups,
                                            ExpertContext& gmm1ExpertContext, GMMAddrInfo& gmm1AddrInfo,
                                            uint32_t& startBlockIdx, int32_t& readySequence, uint64_t waveId);
    __aicore__ inline ExpertWaveRange RunGmm1Wave(ExpertTokenPosition& gmm1Cursor, ExpertContext& gmm1ExpertContext,
                                                  GMMAddrInfo& gmm1AddrInfo, uint32_t& startBlockIdx,
                                                  int32_t& readySequence, uint64_t waveId);
};

// 同一 slice：AIC 做 GMM1，AIV0 做权重 Prologue；AIV1 不从此入口执行激活。
// 现役非量化路径的 AIV1 由 ProcessAiv1PriorityStages 独立消费 GMM1 ready 序号。
// ExpertContext 只读地描述完整专家；slice 固定本次计算范围，不受下一任务游标推进影响。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::RunGmm1ActivationForExpert(
    const ExpertContext& expertContext, const ExpertSlice& slice, GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
    int32_t& gmm1TileReadySequence, uint32_t gmm1TilesPerMGroup)
{
    uint32_t problemTileCount = GetMGroupCountForRows(slice.rowCount, GMM1_TILE_M) * gmm1TilesPerMGroup;
    if (HandleWaveProblemWithoutWork(problemTileCount, gmmExecutionConfig_.blockJob, startBlockIdx)) {
        return;
    }
    UpdateMoeExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                    A_ELEMS_PER_BYTE, true>(
        gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_,
        gmmAddrInfo, expertContext, slice.rowStartInExpert, gmm1TilesPerMGroup);
    gmmAddrInfo.timer = timer_.Enabled() ? &timer_ : nullptr;
    gmmAddrInfo.gmm1AsyncCompletion = ASYNC_GMM1;
    gmmAddrInfo.gmm1CvOutput = UseCv;
    if (computeReplay_) {
        // 输入已在 priming 中完成；本次没有 Dispatch producer，不能等待其被清零的 flag。
        gmmAddrInfo.dispatchToGmm1Flag = nullptr;
    }
    ProblemShape sliceProblemShape = expertContext.problemShape;
    Get<M_VALUE>(sliceProblemShape) = slice.rowCount;
    timer_.SetSlice(slice.expertIdx, slice.rowStartInExpert, slice.rowCount);
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 0U) {
            timer_.StartSpan(AscendTimer::AIV_GMM1_PROLOGUE);
        }
    }
    RunGmm1A8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                EPILOGUE_TILE_M, false, true, FuseGateUp>(
        epilogueOp_, params_, sliceProblemShape, gmmAddrInfo, startBlockIdx, gmm1TileReadySequence,
        gmmExecutionConfig_.blockJob, static_cast<uint32_t>(slice.globalRowStart), slice.expertIdx,
        &gmm1CatlassContext_);
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 0U) {
            timer_.EndSpan(AscendTimer::AIV_GMM1_PROLOGUE);
        }
    }
    timer_.ClearSlice();
}

// 只由 AIV1 调用。复用原有分片、分核和完成边界。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::DispatchNextExpertSlice(
    ExpertTokenPosition& position, uint32_t& waveMGroupCount, uint64_t waveId)
{
    if (dispatchAsync_.pending) {
        timer_.SetWave(static_cast<int64_t>(waveId));
        timer_.SetSlice(dispatchAsync_.expertIdx, dispatchAsync_.completed.localExpertRowBegin,
                        dispatchAsync_.completed.localExpertRowEnd - dispatchAsync_.completed.localExpertRowBegin);
        const int boundary = timer_.BeginBoundary(AscendTimer::AIV_DISPATCH_SYNC);
        RetireDispatchReady<GMM1_TILE_M>(syncWorkspaceLayout_, params_, dispatchAsync_);
        timer_.EndBoundary(AscendTimer::AIV_DISPATCH_SYNC, boundary);
        timer_.ClearSlice();
        return;
    }
    ExpertTokenRange range = PlanNextExpertTokenRangeInWave<GMM1_TILE_M>(
        params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_, commonConfig_.moeExpertPerRank, mGroupsPerWave_,
        waveMGroupCount, position);
    if (range.end.globalTokenIndex > range.begin.globalTokenIndex) {
        const ExpertSlice slice{range.begin.expertIdx, range.begin.tokenIndexInExpert,
                                static_cast<uint32_t>(range.end.globalTokenIndex - range.begin.globalTokenIndex),
                                range.begin.globalTokenIndex};
        // Prefix与两种消费者的工作区互不重叠，保持本launch的UB快照，不再从GM恢复。
        timer_.SetWave(static_cast<int64_t>(waveId));
        timer_.StartSpan(AscendTimer::AIV_DISPATCH_ISSUE);
        // AIC-balance使用独立GMM1/GMM2游标；只映射通信job，不能修改计算游标或物理核号。
        auto dispatchJob = gmmExecutionConfig_.blockJob;
        {
            if (dispatchJob.totalJobs != 0U && dispatchJob.jobIndex < dispatchJob.totalJobs) {
                // 只轮转搬运 owner，不改 peer 配额或 GMM/SwiGLU 共用的物理核编号。
                // 同 wave 的所有 slice 使用同一偏移；取调度器 waveId，不依赖 timer。
                uint32_t shift = static_cast<uint32_t>(waveId % dispatchJob.totalJobs);
                dispatchJob.jobIndex = dispatchJob.jobIndex >= shift
                                           ? dispatchJob.jobIndex - shift
                                           : dispatchJob.jobIndex + dispatchJob.totalJobs - shift;
            }
        }
        // 小 batch 每核仅几行，直接二分源 rank，避免空 peer 的配额统计/扫描；索引预取仍保留。
        // 只改变行排列，token metadata 随行写出；GMM 和 Combine 的 slice/ready 边界不变。
        if (commonConfig_.tokenNum <= W4A8_SMALL_BATCH_MAX_TOKENS) {
            DispatchExpertSlice<ActivationType, QuantScaleOutType, GMM1_TILE_M, false>(
                tokenDispatchConfig_, commonConfig_, dispatchJob, syncWorkspaceLayout_, params_, g_winRankAddr_,
                tokenDispatchScratch_, slice, &dispatchAsync_);
        } else {
            DispatchExpertSlice<ActivationType, QuantScaleOutType, GMM1_TILE_M, false, true>(
                tokenDispatchConfig_, commonConfig_, dispatchJob, syncWorkspaceLayout_, params_, g_winRankAddr_,
                tokenDispatchScratch_, slice, &dispatchAsync_);
        }
        timer_.EndSpan(AscendTimer::AIV_DISPATCH_ISSUE);
    }
    // 这里提交issue游标；completed范围另存dispatchAsync_，下次访问真实完成后才发布ready。
    position = range.end;
}

// 与DS一样把量化提交和就绪发布拆开；只在已有MTE3完成边界观察，不增加全局屏障。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline bool MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::RetireActivation()
{
    if (!activationPending_) return false;
    timer_.SetWave(AscendTimer::NO_WAVE);
    const int boundary = timer_.BeginBoundary(AscendTimer::AIV_SWIGLU_SYNC);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(Aiv1AsyncEvents::ActivationDone));
    if (activationReadyFlag_ != nullptr) AscendC::AtomicAdd(activationReadyFlag_, 1);
    activationPending_ = false;
    timer_.EndBoundary(AscendTimer::AIV_SWIGLU_SYNC, boundary);
    return true;
}

// 惰性枚举原 Wave 的 slice；SwiGLU 只重放 AIC 的独立 GMM1 分核游标，
// 即使本核没有当前 slice 的 tile，也不能漏掉它对后续分核起点的影响。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline bool MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::NextAivSlice(AivTaskCursor& cursor,
                                                                                      AivTaskKind kind)
{
    const bool activation = kind == AivTaskKind::Activation;
    const uint32_t gmm1N =
        Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(GMM1_TASK_N));
    const auto& job = gmmExecutionConfig_.blockJob;
    while (true) {
        if (!HasWaveCapacity(cursor.position, cursor.mGroups) && cursor.mGroups != 0U) {
            cursor.mGroups = 0U;
            ++cursor.wave;
            UpdateGmmLoopCount(gmmLoopCount_, activation ? LoopCountIndex::GMM1 : LoopCountIndex::GMM2, cursor.wave);
        }
        if (cursor.position.expertIdx >= commonConfig_.moeExpertPerRank) {
            return false;
        }
        cursor.slice = PlanNextExpertTokenRangeInWave<GMM1_TILE_M>(params_.workspaceInfo.expertRevTokenNumsPtr,
                                                                   countWorkspace_, commonConfig_.moeExpertPerRank,
                                                                   mGroupsPerWave_, cursor.mGroups, cursor.position);
        cursor.position = cursor.slice.end;
        uint32_t rows = static_cast<uint32_t>(cursor.slice.end.globalTokenIndex - cursor.slice.begin.globalTokenIndex);
        if (rows == 0U) {
            continue;
        }
        if (!activation) {
            return true;
        }
        cursor.tileCount = GetMGroupCountForRows(rows, GMM1_TILE_M) * gmm1N;
        cursor.nextTile = (job.jobIndex + job.totalJobs - cursor.startBlockIdx) % job.totalJobs;
        cursor.startBlockIdx = (cursor.startBlockIdx + cursor.tileCount) % job.totalJobs;
        if (cursor.nextTile < cursor.tileCount) {
            return true;
        }
    }
}

// 只探测本核下一个GMM1 tile；异步模式的GM序号仅表示已提交，随后由MTE2等待真实FIX完成。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline bool MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::TryRunReadyActivation(
    AivTaskCursor& cursor, int32_t& readySequence)
{
    if constexpr (UseCv) return TryRunCvActivation(cursor, readySequence);
    const auto& begin = cursor.slice.begin;
    {
        auto flag = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.flagGmmToEpiloguePtr) +
                    static_cast<uint64_t>(gmmExecutionConfig_.blockJob.jobIndex) * INT_CACHELINE;
        if (AscendC::ReadGmByPassDCache(flag) < readySequence + 1) {
            return false;
        }
    }

    uint32_t rows = static_cast<uint32_t>(cursor.slice.end.globalTokenIndex - begin.globalTokenIndex);
    ExpertContext expertContext = CreateExpertContext(commonConfig_);
    expertContext.expertIdx = begin.expertIdx;
    expertContext.globalTokenStartIndex = begin.globalTokenIndex - begin.tokenIndexInExpert;
    Get<M_VALUE>(expertContext.problemShape) = rows;
    GMMAddrInfo addr{};
    uint32_t tilesPerM =
        Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(GMM1_TASK_N));
    UpdateMoeExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                    A_ELEMS_PER_BYTE, true>(
        gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_, addr,
        expertContext, begin.tokenIndexInExpert, tilesPerM);
    addr.timer = timer_.Enabled() ? &timer_ : nullptr;
    timer_.SetWave(static_cast<int64_t>(cursor.wave));
    timer_.SetSlice(begin.expertIdx, begin.tokenIndexInExpert, rows);
    if constexpr (ASYNC_GMM1) {
        // issued探测成功才排入等待，避免抢占AIV1后阻止尚未提交任务所需的Dispatch。
        Gmm1AsyncCompletion::WaitConsumer(readySequence);
    }
    activationReadyFlag_ = RunReadyGmm1ActivationTileA8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                                   GMM1_TILE_M, FuseGateUp>(epilogueOp_, expertContext.problemShape, addr,
                                                            gmmExecutionConfig_.blockJob, cursor.nextTile,
                                                            static_cast<uint32_t>(begin.globalTokenIndex),
                                                            AscendTimer::AIV_SWIGLU_ISSUE, true);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(Aiv1AsyncEvents::ActivationDone));
    activationPending_ = true;
    timer_.ClearSlice();
    if constexpr (ASYNC_GMM1) {
        // 归还事件槽；输出GM独立寻址不复用，SwiGLU的读入/量化/写出均保持原依赖。
        Gmm1AsyncCompletion::ReleaseConsumer(readySequence);
    }
    {
        ++readySequence;
    }
    return true;
}

// DS式：FIX双槽输入常驻，V直接等待full；每次最多32行，然后把Scalar交还通信调度。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline bool MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::TryRunCvActivation(
    AivTaskCursor& cursor, int32_t& readySequence)
{
    if (cursor.nextRow % Gmm1CvPlan::Rows == 0U) {
        const auto issued = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.flagGmmToEpiloguePtr) +
            static_cast<uint64_t>(gmmExecutionConfig_.blockJob.jobIndex) * INT_CACHELINE;
        if (AscendC::ReadGmByPassDCache(issued) < readySequence + 1) return false;
    }
    const auto& begin = cursor.slice.begin;
    if (cursor.nextRow == 0U) {
        const uint32_t rows = static_cast<uint32_t>(cursor.slice.end.globalTokenIndex - begin.globalTokenIndex);
        auto expert = CreateExpertContext(commonConfig_);
        expert.expertIdx = begin.expertIdx;
        expert.globalTokenStartIndex = begin.globalTokenIndex - begin.tokenIndexInExpert;
        Get<M_VALUE>(expert.problemShape) = rows;
        GMMAddrInfo addr{};
        const uint32_t tilesPerM = Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF,
                                                     static_cast<uint32_t>(GMM1_TASK_N));
        UpdateMoeExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
            A_ELEMS_PER_BYTE, true>(gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo,
                moeWeightTensorListAddrs_, cvActivationOp_, addr, expert, begin.tokenIndexInExpert, tilesPerM);
        auto config = Gmm1Config::BuildGmm1ProblemConfig(expert.problemShape, gmmExecutionConfig_.blockJob, GMM1_TILE_M);
        GmmKernel::BlockScheduler scheduler({config.m, config.outputN, config.k},
            GmmKernel::BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(config.tileM),
                                                            static_cast<int64_t>(GMM1_TASK_N))});
        const auto coord = scheduler.GetBlockCoord(cursor.nextTile);
        const auto shape = scheduler.GetBlockShape(coord);
        cvActivation_ = {static_cast<uint32_t>(Get<M_VALUE>(coord)), static_cast<uint32_t>(Get<N_VALUE>(coord)),
                         static_cast<uint32_t>(Get<M_VALUE>(shape)), static_cast<uint32_t>(Get<N_VALUE>(shape)), 0U};
        cvActivationOp_.UpdateNextProblem({config.m, config.outputN, config.k, 0});
        activationReadyFlag_ = addr.activationToGmm2Flag +
            static_cast<uint64_t>(cvActivation_.m / L1_TILE_M_256) * INT_CACHELINE;
    }
    if (cvActivation_.StageBegin()) Gmm1AsyncCompletion::WaitVectorConsumer(static_cast<uint32_t>(readySequence));
    const uint32_t count = cvActivation_.StepCount();
    const uint32_t row = cvActivation_.nextRow;
    timer_.SetWave(static_cast<int64_t>(cursor.wave));
    timer_.SetSlice(begin.expertIdx, begin.tokenIndexInExpert, cursor.slice.end.globalTokenIndex - begin.globalTokenIndex);
    {
        AscendTimerTileScope step(timer_.Enabled() ? &timer_ : nullptr, AscendTimer::AIV_SWIGLU_ISSUE,
            cvActivation_.m + row, cvActivation_.n, count, cvActivation_.columns);
        // MTE3最后读完输出scratch，V才可覆盖；不等待Scalar，不重读GM中的GMM1结果。
        Gmm1CvScratch::Acquire();
        const uint32_t input = Gmm1CvPlan::SlotOffset(static_cast<uint32_t>(readySequence)) +
            (row % Gmm1CvPlan::Rows) * Gmm1CvPlan::Columns * sizeof(bfloat16_t);
        const uint32_t outputN = commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF;
        const uint32_t scaleN = CeilDiv(outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        AscendC::SetCtrlSpr<60, 60>(0);
        cvActivationOp_({count, cvActivation_.columns, 0, 0},
            {(cvActivation_.m + row) * outputN + cvActivation_.n,
             (cvActivation_.m + row) * scaleN + CeilDiv(cvActivation_.n, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
             0, 0, 0, 0}, 0, false, input);
    }
    if (cvActivation_.Advance(count)) {
        // 仅该stage的最后一组读取/写出入队后归还EMPTY；另一stage仍由其自己的credit保护。
        Gmm1AsyncCompletion::ReleaseConsumer(static_cast<uint32_t>(readySequence));
        ++readySequence;
    }
    cursor.nextRow = cvActivation_.Complete() ? 0U : cvActivation_.nextRow;
    if (cvActivation_.Complete()) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(Aiv1AsyncEvents::ActivationDone));
        activationPending_ = true;
    }
    timer_.ClearSlice();
    return true;
}

template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline bool MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::TryRunReadyCombine(const AivTaskCursor& cursor)
{
    const auto& begin = cursor.slice.begin;
    uint32_t rows = static_cast<uint32_t>(cursor.slice.end.globalTokenIndex - begin.globalTokenIndex);
    // 只轮换当前 AIV1 的通信任务，不修改成员中的物理核编号。
    // 输入是 Combine 自己的逻辑 wave（cursor.wave），不是当前 Dispatch/GMM1 的进度；
    // BS>=4096 时 GMM2 滞后一轮，使用其他阶段的 wave 会把同一批输出分给错误的 owner。
    // GMM2 反向分核、通信 owner 轮换都不改变 slice 的身份和实际生产核数量。
    auto combineJob = waveCombineJob_;
    if (combineJob.totalJobs != 0U && combineJob.jobIndex < combineJob.totalJobs) {
        uint32_t shift = static_cast<uint32_t>(cursor.wave % combineJob.totalJobs);
        // 与 Dispatch 相同：物理核 c 在 wave w 执行逻辑核 (c-w) mod 核数 的连续行任务。
        // 条件减法避免无符号下溢；先取模再转换，保证较大的 wave 编号也能正确轮换。
        combineJob.jobIndex = combineJob.jobIndex >= shift ? combineJob.jobIndex - shift
                                                           : combineJob.jobIndex + combineJob.totalJobs - shift;
    }
    // 本次只重映射完整任务，不新增 peer 分段。结合原有全局行前缀，等价首 owner 为
    // (sliceGlobalRowStart + cursor.wave) mod 核数；同 wave 跨专家的 slice 使用同一偏移。
    // 空任务判断必须使用轮换后的 owner，否则小 batch/尾片可能被错误跳过而漏发。
    WorkRange owned = GetWaveCombineOwnedRange(combineJob, rows, begin.globalTokenIndex);
    if (owned.count == 0U) {
        return true;
    }
    // 只读取当前 expert/slice 的计数；不同 slice 不共用槽，快核后续通知不能代偿慢核。
    // 没就绪就返回，让优先级调度器继续 SwiGLU/Dispatch，不能在探测函数里阻塞。
    auto ready = GetTokenCombineReadyCounter(params_, begin.expertIdx, begin.tokenIndexInExpert);
    if (AscendC::ReadGmByPassDCache(ready) <
        static_cast<int32_t>(GetTokenCombineProducerCount(commonConfig_, waveCombineJob_, rows))) {
        return false;
    }
    ExpertContext expertContext = CreateExpertContext(commonConfig_);
    expertContext.expertIdx = begin.expertIdx;
    expertContext.globalTokenStartIndex = begin.globalTokenIndex - begin.tokenIndexInExpert;
    timer_.SetWave(static_cast<int64_t>(cursor.wave));
    // 将就绪检查使用的同一份 job 传给实际发送，保证判定和发送的行范围一致。
    // 当前 slice 已检查就绪，模板 true 跳过发送入口内的重复等待。计数只增不减，
    // 到本次算子结束前不会重置，因此检查与发送之间无需再次轮询。
    // timer 仍记录物理核；完整行发送及 UB 排空规则保持原样。
    RunTokenCombineSlice<true>(commonConfig_, combineJob, params_, expertContext, begin.tokenIndexInExpert, rows,
                               timer_, waveCombineScratch_, &waveCombineBuffers_, &combineRowSequence_);
    return true;
}

// 三套UB常驻后，每轮推进一个就绪激活，再交替服务Dispatch/Combine。
// Combine未就绪时立即退回Dispatch；不进入阻塞等待，不抢占正在执行的tile/slice。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::ProcessAiv1PriorityStages(
    int32_t& readySequence)
{
    AivTaskCursor activation{}, combine{};
    bool hasActivation = NextAivSlice(activation, AivTaskKind::Activation);
    bool hasCombine = !computeReplay_ && NextAivSlice(combine, AivTaskKind::Combine);
    SharedActivationCursor shared{};
    shared.done = sharedExpertNum_ == 0U;
    ExpertTokenPosition dispatch{};
    if (computeReplay_) {
        dispatch.expertIdx = commonConfig_.moeExpertPerRank;
    }
    uint32_t dispatchGroups = 0U;
    if constexpr (ASYNC_GMM1) Gmm1AsyncCompletion::InitConsumer();
    // 一次初始化、逐块消费/归还，最后统一消费末个credit。初始化也覆盖零激活路径。
    if constexpr (UseCv) {
        const auto* t = params_.tilingData;
        cvActivationOp_.Init({.yGmAddr = params_.workspaceInfo.activationQuantDataPtr,
            .yScaleGmAddr = params_.workspaceInfo.activationQuantScalePtr, .clampLimit = t->clampLimit,
            .actMode = t->actMode, .actSubMode = t->actSubMode, .activationAlpha = t->activationAlpha,
            .activationBeta = t->activationBeta, .ubBaseBytes = Gmm1CvPlan::ScratchBase});
        Gmm1CvScratch::Init();
    } else {
        Aiv1ActivationUb::Init();
    }
    uint64_t dispatchWave = 0U;
    bool preferCombine = false;
    while (hasActivation || activationPending_ || hasCombine || !shared.done || dispatchAsync_.pending ||
           dispatch.expertIdx < commonConfig_.moeExpertPerRank) {
        bool progressed = RetireActivation();
        if (hasActivation && TryRunReadyActivation(activation, readySequence)) {
            if (activation.nextRow == 0U) {
                activation.nextTile += gmmExecutionConfig_.blockJob.totalJobs;
                if (activation.nextTile >= activation.tileCount) {
                    hasActivation = NextAivSlice(activation, AivTaskKind::Activation);
                }
            }
            progressed = true;
        }
        if (!shared.done && TryRunSharedExpertActivation(shared)) {
            progressed = true;
        }
        const bool hasDispatch = dispatchAsync_.pending || dispatch.expertIdx < commonConfig_.moeExpertPerRank;
        if (hasCombine && (preferCombine || !hasDispatch) &&
            TryRunReadyCombine(combine)) {
            hasCombine = NextAivSlice(combine, AivTaskKind::Combine);
            preferCombine = false;
            progressed = true;
        } else if (hasDispatch) {
            if (!dispatchAsync_.pending && dispatchGroups >= mGroupsPerWave_) {
                dispatchGroups = 0U;
                ++dispatchWave;
            }
            DispatchNextExpertSlice(dispatch, dispatchGroups, dispatchWave);
            preferCombine = true;
            progressed = true;
        }
        if (progressed) {
            continue;
        }
        // 没有可执行任务才退避；不能进入 Combine 的阻塞等待，否则可能挡住 GMM2 所需的 SwiGLU。
        int64_t start = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - start < GM_FLAG_POLL_BACKOFF_CYCLES) {
        }
    }
    // 生命周期的唯一收尾。先前slice的ready已发布；不重置仍在飞行的cursor或复用事件。
    if constexpr (UseCv) Gmm1CvScratch::Finish();
    else Aiv1ActivationUb::Finish();
    DrainDispatchCopyBuffers<ActivationType, QuantScaleOutType, false>(
        tokenDispatchConfig_, params_, tokenDispatchScratch_, dispatchAsync_.progress);
    DrainCombineRowBuffers(combineRowSequence_, waveCombineBuffers_.rowBufferCount, waveCombineBuffers_.eventBase);
    timer_.ClearSlice();
    timer_.SetWave(AscendTimer::NO_WAVE);
}

// 消费一个已完成 Wave 的 GMM2/Combine（调度本体在基类 RunGmm2CombineForExpert）。
// Wave 在专家内结束时需包含该专家；在专家边界结束时，waveEndPosition 已指向下一专家。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::RunGmm2CombineForWaveRange(
    const ExpertWaveRange& wave, ExpertContext& gmm2ExpertContext, GMMAddrInfo& gmm2AddrInfo, uint32_t& startBlockIdx)
{
    const auto& waveBeginPosition = wave.begin;
    const auto& waveEndPosition = wave.end;
    uint32_t waveGmm2ExpertEndExclusive =
        waveEndPosition.expertIdx + (waveEndPosition.tokenIndexInExpert == 0U ? 0U : 1U);
    const uint32_t waveStartBlockIdx = startBlockIdx;
    if constexpr (ASYNC_GMM1) {
        // AIC/AIV0 使用完全相同的 L1/L0/UB 布局；不排空、不重置槽号。
        gmm1CatlassContext_.HandoffTo(gmm2WaveContext_);
    }
    for (uint32_t expertIdx = waveBeginPosition.expertIdx; expertIdx < waveGmm2ExpertEndExclusive; ++expertIdx) {
        const bool startsAtExpertBoundary =
            expertIdx != waveBeginPosition.expertIdx || waveBeginPosition.tokenIndexInExpert == 0U;
        // WAVE 从专家起点开始时，准备该专家的 GMM2 状态；跨 WAVE slice 沿用已有状态。
        if (startsAtExpertBoundary) {
            this->PrepareGmmExpertContext(gmm2ExpertContext, expertIdx);
        }
        const uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(gmm2ExpertContext.problemShape));
        const ExpertSlice slice =
            GetExpertSliceInWave(wave, expertIdx, expertTokenCount, gmm2ExpertContext.globalTokenStartIndex);
        if (slice.rowCount != 0U) {
            this->template RunGmm2CombineForExpert<FuseGateUp>(gmm2ExpertContext, slice, gmm2AddrInfo, startBlockIdx,
                                                               gmm2WaveContext_, GMM2_REVERSE_BLOCKS);
        }
    }
    {
        // ready 仍在真正的 FIX 完成之后；异步路径只省去资源销毁/重建。
        if constexpr (ASYNC_GMM1) {
            gmm2WaveContext_.DrainOutput();
        } else {
            gmm2WaveContext_.End();
        }
        if constexpr (g_coreType == AIC) {
            // 重放轻量分核计算，避免分配随专家数增长的 pending 数组。不重新执行 GEMM。
            uint32_t notifyStartBlockIdx = waveStartBlockIdx;
            const auto& job = gmmExecutionConfig_.blockJob;
            const uint32_t nTiles = Ops::Base::CeilDiv(commonConfig_.tokenHiddenDim, uint32_t(L1_TILE_N));
            for (uint32_t expert = waveBeginPosition.expertIdx; expert < waveGmm2ExpertEndExclusive; ++expert) {
                const uint32_t rowBegin =
                    expert == waveBeginPosition.expertIdx ? waveBeginPosition.tokenIndexInExpert : 0U;
                const uint32_t rowEnd =
                    expert == waveEndPosition.expertIdx
                        ? waveEndPosition.tokenIndexInExpert
                        : GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_,
                                                           commonConfig_.moeExpertPerRank, expert);
                const uint32_t tiles = GetMGroupCountForRows(rowEnd - rowBegin, GMM1_TILE_M) * nTiles;
                if (GetGmmFirstOwnedTile(job.jobIndex, job.totalJobs, notifyStartBlockIdx, GMM2_REVERSE_BLOCKS) < tiles) {
                    NotifyTokenCombineSliceReady(params_, expert, rowBegin);
                }
                notifyStartBlockIdx = (notifyStartBlockIdx + tiles) % job.totalJobs;
            }
        }
    }

    if constexpr (ASYNC_GMM1) {
        gmm2WaveContext_.HandoffTo(gmm1CatlassContext_);
    }
}

// count 表由 AIV1 准备；计算侧只等待一次元数据发布，不增加全核屏障。
// 非量化 Combine 的 AIV1 独立推进；其余角色进入确定顺序的 GMM Wave 流水。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::ProcessMoeExpertStages(
    int32_t& gmm1TileReadySequence)
{
    // GMM1/GMM2 交错流水只记录一次阶段入口，各 Wave 完成轮次由独立计数记录。
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::MOE_GMM1_ACTIVATION);
    DispatchBuffInit();
    if (!computeReplay_) {
        PrepareMoeExpertTokenCountTable<false>(commonConfig_, countWorkspace_, params_, tokenDispatchScratch_);
        if (GetSubBlockIdx() == 0U) {
            WaitForMoeExpertTokenCountReady(params_.workspaceInfo.flagSendCntCalToUpdParamsPtr, countWorkspace_, 0U);
        }
    }
    if constexpr (UseCv) {
        const uint32_t blocks = SmallLinearBlockCount();
        if (blocks != static_cast<uint32_t>(-1)) {
            if constexpr (g_coreType == AIV) {
                if (GetSubBlockIdx() == 1U) {
                    ProcessSmallLinearAiv1(blocks,gmm1TileReadySequence);
                    return;
                }
            }
            ProcessSmallLinearTasks(blocks,gmm1TileReadySequence);
            return;
        }
    }
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            ProcessAiv1PriorityStages(gmm1TileReadySequence);
            return;
        }
    }
    ProcessGmmWaves(gmm1TileReadySequence);
}

// 只推进一个专家 slice。wave.end 是下一位置，slice 则固定本次计算的专家身份和范围。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::RunNextGmm1Slice(
    ExpertWaveRange& wave, uint32_t& usedMGroups, ExpertContext& gmm1ExpertContext, GMMAddrInfo& gmm1AddrInfo,
    uint32_t& startBlockIdx, int32_t& readySequence, uint64_t waveId)
{
    if (wave.end.tokenIndexInExpert == 0U) {
        this->PrepareGmmExpertContext(gmm1ExpertContext, wave.end.expertIdx);
    }
    const uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(gmm1ExpertContext.problemShape));
    const ExpertSlice slice =
        TakeNextExpertSliceInWave<GMM1_TILE_M>(expertTokenCount, mGroupsPerWave_, usedMGroups, wave.end);
    if (slice.rowCount != 0U) {
        wave.lastActiveExpert = slice.expertIdx;
        const uint32_t tilesPerMGroup =
            Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(GMM1_TASK_N));
        timer_.SetWave(static_cast<int64_t>(waveId));
        RunGmm1ActivationForExpert(gmm1ExpertContext, slice, gmm1AddrInfo, startBlockIdx, readySequence,
                                   tilesPerMGroup);
    }
}

// 返回本 Wave 的固定边界，调用者用它立即执行 GMM2 或保存为 pendingWave。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline ExpertWaveRange MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::RunGmm1Wave(
    ExpertTokenPosition& gmm1Cursor, ExpertContext& gmm1ExpertContext, GMMAddrInfo& gmm1AddrInfo,
    uint32_t& startBlockIdx, int32_t& readySequence, uint64_t waveId)
{
    ExpertWaveRange wave{gmm1Cursor, gmm1Cursor, commonConfig_.moeExpertPerRank};

    uint32_t usedMGroups = 0U;
    while (HasWaveCapacity(wave.end, usedMGroups)) {
        RunNextGmm1Slice(wave, usedMGroups, gmm1ExpertContext, gmm1AddrInfo, startBlockIdx, readySequence, waveId);
    }
    gmm1Cursor = wave.end;
    return wave;
}

// AIC/AIV0 同序推进 GMM1/Prologue、GMM2/Prologue；数据就绪仍由各 stage 内部的 flag 保证。
// BS<4096：G1(w)→G2(w)；BS>=4096：G1(w)→G2(w-1)，最后补齐 G2。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::ProcessGmmWaves(int32_t& gmm1TileReadySequence)
{
    uint64_t gmm1Count = 0U;
    uint64_t gmm2Count = 0U;

    // 完整专家元数据与本次 slice 分开：跨 Wave 复用专家状态，不把 slice 行数写回专家总行数。
    ExpertContext gmm1ExpertContext = CreateExpertContext(commonConfig_);
    ExpertContext gmm2ExpertContext = CreateExpertContext(commonConfig_);
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};

    // 两阶段各自跨专家、跨 Wave 轮转。GMM1 从物理核 0 正向分配，
    // GMM2 在本消融中也同向分配；完成发布使用同一个方向常量。
    uint32_t gmm1StartBlockIdx = 0U;
    uint32_t gmm2StartBlockIdx = 0U;

    // 按本卡输入 BS 判断，不按专家接收行数或当前 slice 行数判断。
    const bool gmm2LagActive = commonConfig_.tokenNum >= 4096U;
    ExpertWaveRange pendingWave{{}, {}, commonConfig_.moeExpertPerRank};

    // cursor 总是指向下一待规划位置；本次计算使用 TakeNextExpertSliceInWave 返回的固定 slice。
    ExpertTokenPosition gmm1Cursor{};
    while (gmm1Cursor.expertIdx < commonConfig_.moeExpertPerRank) {
        const ExpertWaveRange wave = RunGmm1Wave(gmm1Cursor, gmm1ExpertContext, gmm1AddrInfo, gmm1StartBlockIdx,
                                                 gmm1TileReadySequence, gmm1Count);
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM1, ++gmm1Count);
        // 非滞后路径消费当前 Wave；滞后路径从第二轮起消费上一 Wave，首轮只保存边界。
        if (!gmm2LagActive || gmm1Count > 1U) {
            const ExpertWaveRange& gmm2Wave = gmm2LagActive ? pendingWave : wave;
            timer_.SetWave(static_cast<int64_t>(gmm2Count));
            RunGmm2CombineForWaveRange(gmm2Wave, gmm2ExpertContext, gmm2AddrInfo, gmm2StartBlockIdx);
            UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
        }
        if (gmm2LagActive) {
            pendingWave = wave;
        }
    }
    // 计算侧补跑最后一个 Wave；非量化 AIV1 仍按自己的 ready 游标发送 Combine。
    if (gmm2Count < gmm1Count) {
        timer_.SetWave(static_cast<int64_t>(gmm2Count));
        RunGmm2CombineForWaveRange(pendingWave, gmm2ExpertContext, gmm2AddrInfo, gmm2StartBlockIdx);
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
    }
    if constexpr (ASYNC_GMM1) {
        // 每次 GMM2 后资源归还给 GMM1 owner，包括尾轮；最终且仅最终回收硬件 credit。
        gmm1CatlassContext_.End();
        if constexpr (g_coreType == AIC) Gmm1AsyncCompletion::DrainProducer();
    }
    timer_.SetWave(AscendTimer::NO_WAVE);
}

// 只迁移当前闭合的单M块协议：偏斜产生大专家时保留原Wave，不能用期望M冒充实际容量。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline uint32_t MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::SmallLinearBlockCount()
{
    if (computeReplay_ || sharedExpertNum_ != 0U ||
        Ops::Base::CeilDiv(commonConfig_.tokenHiddenDim,uint32_t(L1_TILE_N)) > gmmExecutionConfig_.blockJob.totalJobs)
        return static_cast<uint32_t>(-1);
    uint32_t blocks = 0U;
    for (uint32_t expert=0; expert<commonConfig_.moeExpertPerRank; ++expert) {
        const uint32_t rows = GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr,
            countWorkspace_, commonConfig_.moeExpertPerRank, expert);
        if (rows > Gmm1CvPlan::Rows) return static_cast<uint32_t>(-1);
        blocks += rows != 0U;
    }
    return blocks;
}

template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::AdvanceSmallLinearExpert(
    ExpertContext& expert, uint32_t block, uint32_t& blockEnd, uint32_t& nextExpert)
{
    while (block >= blockEnd) {
        this->PrepareGmmExpertContext(expert, nextExpert++);
        if (Get<M_VALUE>(expert.problemShape) != 0U) ++blockEnd;
    }
}

// 计算侧两层共享issued序列与两份epilogue credit，AIV1按同一task流消费真实FIX完成。
// GMM2不再在phase边界Scalar排空；唯一收尾仍确认所有最终credit已归还。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::ProcessSmallLinearTasks(
    uint32_t blocks, int32_t& readySequence)
{
    const auto& job = gmmExecutionConfig_.blockJob;
    const uint32_t n1 = Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF,uint32_t(GMM1_TASK_N));
    const uint32_t n2 = Ops::Base::CeilDiv(commonConfig_.tokenHiddenDim,uint32_t(L1_TILE_N));
    SmallLinearTaskSchedule schedule{blocks,n1,n2,job.totalJobs};
    auto first = CreateExpertContext(commonConfig_), second = CreateExpertContext(commonConfig_);
    uint32_t firstEnd=0U, firstNext=0U, secondEnd=0U, secondNext=0U;
    SmallLinearTask task{};
    while (schedule.Next(job.jobIndex,task)) {
        timer_.SetWave(task.index / job.totalJobs);
        GMMAddrInfo addr{};
        if (task.gmm1) {
            gmm2WaveContext_.HandoffTo(gmm1CatlassContext_);
            AdvanceSmallLinearExpert(first,task.index/n1,firstEnd,firstNext);
            const auto config = Gmm1Config::BuildGmm1ProblemConfig(first.problemShape,job,GMM1_TILE_M);
            const uint32_t rows = static_cast<uint32_t>(Get<M_VALUE>(first.problemShape));
            const uint32_t column = task.index%n1 * GMM1_TASK_N;
            const uint32_t columns = Min(uint32_t(GMM1_TASK_N),uint32_t(config.outputN)-column);
            UpdateMoeExpertGmm1GlobalBuffer<ActivationType,Weight1Type,ActivationQuantOutType,QuantScaleOutType,
                A_ELEMS_PER_BYTE,true>(gmmExecutionConfig_,syncWorkspaceLayout_,params_.workspaceInfo,
                moeWeightTensorListAddrs_,epilogueOp_,addr,first,0U,n1);
            addr.timer = timer_.Enabled() ? &timer_ : nullptr;
            gmm1CatlassContext_.UpdateProblem(config,addr);
            gmm1CatlassContext_.Begin();
            timer_.SetSlice(first.expertIdx,0U,rows);
            if constexpr (g_coreType == AIC) {
                AscendTimerTileScope event(addr.timer,AscendTimer::AIC_GMM1_ISSUE,0U,column,rows,columns);
                WaitForGmm1InputReady<false>(addr,config,0U);
                gmm1CatlassContext_.ComputeCv(0U,column,rows,columns,uint32_t(readySequence),addr.gmmToEpilogueFlag);
                readySequence += Gmm1CvPlan::ChunkCount(rows);
            } else {
                AscendTimerSpanScope event(addr.timer,AscendTimer::AIV_GMM1_PROLOGUE);
                gmm1CatlassContext_.Dequantize(column,rows,columns);
            }
        } else {
            gmm1CatlassContext_.HandoffTo(gmm2WaveContext_);
            AdvanceSmallLinearExpert(second,task.index/n2,secondEnd,secondNext);
            const auto config = Gmm2Config::BuildGmm2ProblemConfig(second.problemShape,job,GMM1_TILE_M);
            const uint32_t rows = static_cast<uint32_t>(Get<M_VALUE>(second.problemShape));
            const uint32_t column = task.index%n2 * L1_TILE_N;
            const uint32_t columns = Min(uint32_t(L1_TILE_N),uint32_t(config.outputN)-column);
            UpdateMoeExpertGmm2GlobalBuffer<Weight1Type,ActivationQuantOutType,QuantScaleOutType>(
                gmmExecutionConfig_,syncWorkspaceLayout_,params_.workspaceInfo,moeWeightTensorListAddrs_,addr,second);
            addr.timer = timer_.Enabled() ? &timer_ : nullptr;
            gmm2WaveContext_.UpdateProblem(config,addr);
            gmm2WaveContext_.Begin();
            timer_.SetSlice(second.expertIdx,0U,rows);
            if constexpr (g_coreType == AIC) {
                AscendTimerTileScope event(addr.timer,AscendTimer::AIC_GMM2_ISSUE,0U,column,rows,columns);
                WaitForGmm2InputReady<false>(addr,config,0U);
                Gmm1AsyncCompletion::AcquireProducer(uint32_t(readySequence));
                gmm2WaveContext_.Compute(0U,column,rows,columns);
                Gmm1AsyncCompletion::PublishCompleted(uint32_t(readySequence));
                auto issued = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.flagGmmToEpiloguePtr) +
                    static_cast<uint64_t>(job.jobIndex)*INT_CACHELINE;
                AscendC::WriteGmByPassDCache(issued,++readySequence);
            } else {
                AscendTimerSpanScope event(addr.timer,AscendTimer::AIV_GMM2_PROLOGUE);
                gmm2WaveContext_.Dequantize(column,rows,columns);
            }
        }
        timer_.ClearSlice();
    }
    gmm2WaveContext_.HandoffTo(gmm1CatlassContext_);
    gmm1CatlassContext_.End();
    if constexpr (g_coreType == AIC) Gmm1AsyncCompletion::DrainProducer();
    timer_.SetWave(AscendTimer::NO_WAVE);
}

// AIV1 epilogue必须与AIC重放相同的两层task流；独立只消费GMM1会在共享FIX credit上形成环。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::ProcessSmallLinearAiv1(
    uint32_t blocks, int32_t& readySequence)
{
    const auto& job = gmmExecutionConfig_.blockJob;
    const uint32_t n1 = Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim/ACTIVATION_N_HALF,uint32_t(GMM1_TASK_N));
    const uint32_t n2 = Ops::Base::CeilDiv(commonConfig_.tokenHiddenDim,uint32_t(L1_TILE_N));
    SmallLinearTaskSchedule schedule{blocks,n1,n2,job.totalJobs};
    auto first = CreateExpertContext(commonConfig_), second = CreateExpertContext(commonConfig_);
    uint32_t firstEnd=0U, firstNext=0U, secondEnd=0U, secondNext=0U;
    SmallLinearTask task{};
    AivTaskCursor activation{},combine{};
    bool hasTask=false, submitted=false;
    auto next = [&]() __aicore__ {
        hasTask=schedule.Next(job.jobIndex,task);
        submitted=false;
        if (!hasTask || !task.gmm1) return;
        AdvanceSmallLinearExpert(first,task.index/n1,firstEnd,firstNext);
        const uint32_t rows=static_cast<uint32_t>(Get<M_VALUE>(first.problemShape));
        const uint64_t globalStart=static_cast<uint64_t>(first.globalTokenStartIndex);
        activation.slice={{first.expertIdx,0U,globalStart},{first.expertIdx,rows,globalStart+rows}};
        activation.nextTile=task.index%n1;
        activation.nextRow=0U;
        activation.wave=task.index/job.totalJobs;
    };
    Gmm1AsyncCompletion::InitConsumer();
    const auto* t=params_.tilingData;
    cvActivationOp_.Init({.yGmAddr=params_.workspaceInfo.activationQuantDataPtr,
        .yScaleGmAddr=params_.workspaceInfo.activationQuantScalePtr,.clampLimit=t->clampLimit,
        .actMode=t->actMode,.actSubMode=t->actSubMode,.activationAlpha=t->activationAlpha,
        .activationBeta=t->activationBeta,.ubBaseBytes=Gmm1CvPlan::ScratchBase});
    Gmm1CvScratch::Init();
    next();
    bool hasCombine=NextAivSlice(combine,AivTaskKind::Combine),preferCombine=false;
    ExpertTokenPosition dispatch{};
    uint32_t dispatchGroups=0U;
    uint64_t dispatchWave=0U;
    const auto issued=reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.flagGmmToEpiloguePtr) +
        static_cast<uint64_t>(job.jobIndex)*INT_CACHELINE;
    while (hasTask || activationPending_ || hasCombine || dispatchAsync_.pending ||
           dispatch.expertIdx<commonConfig_.moeExpertPerRank) {
        bool progressed=RetireActivation();
        if (submitted && !activationPending_) next();
        if (hasTask && !submitted) {
            if (task.gmm1) {
                if (TryRunCvActivation(activation,readySequence)) {
                    submitted=activation.nextRow==0U;
                    progressed=true;
                }
            } else if (AscendC::ReadGmByPassDCache(issued)>=readySequence+1) {
                AdvanceSmallLinearExpert(second,task.index/n2,secondEnd,secondNext);
                {
                    AscendTimerWaitScope wait(timer_.Enabled()?&timer_:nullptr,AscendTimer::AIV_COMBINE_WAIT);
                    Gmm1AsyncCompletion::WaitScalarConsumer(uint32_t(readySequence));
                }
                // 真正FIX完成后发布该N tile的贡献；小M且N tile<=核数保证每核每专家至多一次。
                asc_sync_data_barrier(DSB_DDR);
                AscendC::AtomicAdd(GetTokenCombineReadyCounter(params_,second.expertIdx,0U),1);
                Gmm1AsyncCompletion::ReleaseScalarConsumer(uint32_t(readySequence));
                ++readySequence;
                next();
                progressed=true;
            }
        }
        const bool hasDispatch=dispatchAsync_.pending || dispatch.expertIdx<commonConfig_.moeExpertPerRank;
        if (hasCombine && (preferCombine || !hasDispatch) && TryRunReadyCombine(combine)) {
            hasCombine=NextAivSlice(combine,AivTaskKind::Combine);
            preferCombine=false;
            progressed=true;
        } else if (hasDispatch) {
            if (!dispatchAsync_.pending && dispatchGroups>=mGroupsPerWave_) {
                dispatchGroups=0U;
                ++dispatchWave;
            }
            DispatchNextExpertSlice(dispatch,dispatchGroups,dispatchWave);
            preferCombine=true;
            progressed=true;
        }
        if (!progressed) {
            const int64_t start=AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle()-start<GM_FLAG_POLL_BACKOFF_CYCLES) {}
        }
    }
    Gmm1CvScratch::Finish();
    DrainDispatchCopyBuffers<ActivationType,QuantScaleOutType,false>(
        tokenDispatchConfig_,params_,tokenDispatchScratch_,dispatchAsync_.progress);
    DrainCombineRowBuffers(combineRowSequence_,waveCombineBuffers_.rowBufferCount,waveCombineBuffers_.eventBase);
    timer_.ClearSlice();
    timer_.SetWave(AscendTimer::NO_WAVE);
}

template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline void MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::Process()
{
    this->ProcessWave(*this);
}

}  // namespace MegaMoeImpl

#undef TemplateMegaMoeA8W4WaveTypeClass
#undef TemplateMegaMoeA8W4WaveTypeFunc

#endif
