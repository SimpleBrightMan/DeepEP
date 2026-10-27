// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

/*!
 * \file mega_moe_wave_a8w4.h
 * \brief MegaMoe A8W4 动态 Wave 调度
 */

#ifndef MEGA_MOE_WAVE_A8W4_H
#define MEGA_MOE_WAVE_A8W4_H

#include "common/mega_moe_utils.hpp"
#include "common/mega_moe_schedule_policy.hpp"
#include "common/gmm2_aiv_completion.hpp"
#include "mega_moe_arch35.hpp"

namespace MegaMoeImpl {

#define TemplateMegaMoeA8W4WaveTypeClass                                                                    \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        bool FuseGateUp
#define TemplateMegaMoeA8W4WaveTypeFunc XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, FuseGateUp

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
    // 两层同向分核；执行、空任务判断与完成通知必须共用这个方向。
    static constexpr bool GMM2_REVERSE_BLOCKS = false;

    using Gmm1Config = GmmKernel::Config<true, QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType,
                                         QuantScaleOutType, false, false, false, true, FuseGateUp>;
    GmmKernel::CatlassGmm1<typename Gmm1Config::ProblemConfig> gmm1CatlassContext_;
    using Gmm2Config = GmmKernel::Config<true, ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType,
                                         QuantScaleOutType, false, false, false, true, FuseGateUp>;
    GmmKernel::CatlassGmm2<typename Gmm2Config::ProblemConfig> gmm2WaveContext_;

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
    };
    enum class AivTaskKind {
        Activation,
        Combine,
    };
    DispatchAsyncState dispatchAsync_{};
    uint32_t combineRowSequence_ = 0U;
    int32_t gmm2IssuedSequence_ = 0;
    __gm__ int32_t* activationReadyFlag_ = nullptr;
    bool activationPending_ = false;
    __aicore__ inline bool RetireActivation();
    __aicore__ inline bool NextAivSlice(AivTaskCursor& cursor, AivTaskKind kind);
    __aicore__ inline bool TryRunReadyActivation(AivTaskCursor& cursor, int32_t& readySequence);
    __aicore__ inline bool TryRunReadyCombine(const AivTaskCursor& cursor);
    __aicore__ inline bool NextGmm2CompletionSlice(AivTaskCursor& cursor);
    __aicore__ inline bool TryReceiveGmm2Completion(const AivTaskCursor& cursor, int32_t& completedSequence);
    __aicore__ inline void ProcessAiv1PriorityStages(int32_t& readySequence);
    __aicore__ inline bool HasWaveCapacity(const ExpertTokenPosition& position, uint32_t waveMGroupCount) const
    {
        return MegaMoeImpl::HasWaveCapacity(position, commonConfig_.moeExpertPerRank, waveMGroupCount, mGroupsPerWave_);
    }

    __aicore__ inline void RunGmm1ActivationForExpert(const ExpertContext& expertContext, const ExpertSlice& slice,
                                                      GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
                                                      int32_t& gmm1TileReadySequence, uint32_t gmm1TilesPerMGroup);
    __aicore__ inline void DispatchNextExpertSlice(ExpertTokenPosition& position, uint32_t& waveMGroupCount,
                                                   uint64_t waveId, bool stridedSmall = false);
    __aicore__ inline void RunGmm2CombineForWaveRange(const ExpertWaveRange& wave, ExpertContext& gmm2ExpertContext,
                                                      GMMAddrInfo& gmm2AddrInfo, uint32_t& startBlockIdx);
    __aicore__ inline void ProcessMoeExpertStages(int32_t& gmm1TileReadySequence);
    __aicore__ inline void ProcessGmmWaves(int32_t& gmm1TileReadySequence);

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
    ExpertTokenPosition& position, uint32_t& waveMGroupCount, uint64_t waveId, bool stridedSmall)
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
        // 小batch每核仅几行，直接二分源rank，避免空peer配额扫描；索引供给由公共Dispatch叶子负责。
        // 只改变行排列，token metadata 随行写出；GMM 和 Combine 的 slice/ready 边界不变。
        if (stridedSmall) {
            DispatchStridedSmallExpertSlice<ActivationType,QuantScaleOutType,GMM1_TILE_M>(
                tokenDispatchConfig_,commonConfig_,gmmExecutionConfig_.blockJob,syncWorkspaceLayout_,
                params_,g_winRankAddr_,tokenDispatchScratch_,slice,dispatchAsync_);
        } else if (commonConfig_.tokenNum <= W4A8_SMALL_BATCH_MAX_TOKENS) {
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

// 量化提交和就绪发布分开；只在已有MTE3完成边界观察，不增加全局屏障。
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

template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline bool MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::TryRunReadyCombine(const AivTaskCursor& cursor)
{
    const auto& begin = cursor.slice.begin;
    uint32_t rows = static_cast<uint32_t>(cursor.slice.end.globalTokenIndex - begin.globalTokenIndex);
    // 只轮换当前 AIV1 的通信任务，不修改成员中的物理核编号。
    // 输入是 Combine 自己的逻辑 wave（cursor.wave），不是当前 Dispatch/GMM1 的进度；
    // GMM2 滞后一轮时，使用其他阶段的 wave 会把同一批输出分给错误的 owner。
    // GMM2计算核方向与通信owner轮换都不改变slice身份和实际生产核数量。
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

// 完成流只枚举本核实际生产的 slice；空核也推进分核游标，避免消费不存在的 FULL。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline bool MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::NextGmm2CompletionSlice(AivTaskCursor& cursor)
{
    const auto& job = gmmExecutionConfig_.blockJob;
    const uint32_t nTiles = Ops::Base::CeilDiv(commonConfig_.tokenHiddenDim, uint32_t(L1_TILE_N));
    while (NextAivSlice(cursor, AivTaskKind::Combine)) {
        const uint32_t rows = static_cast<uint32_t>(cursor.slice.end.globalTokenIndex - cursor.slice.begin.globalTokenIndex);
        const uint32_t tiles = GetMGroupCountForRows(rows, GMM1_TILE_M) * nTiles;
        const bool owns = GetGmmFirstOwnedTile(job.jobIndex, job.totalJobs, cursor.startBlockIdx,
                                               GMM2_REVERSE_BLOCKS) < tiles;
        cursor.startBlockIdx = (cursor.startBlockIdx + tiles) % job.totalJobs;
        if (owns) return true;
    }
    return false;
}

// issued 只允许排入本 slice 的真实 FIX 等待；完成后仅为该 slice 贡献一次 ready。
template <TemplateMegaMoeA8W4WaveTypeClass>
__aicore__ inline bool MegaMoeA8W4Wave<TemplateMegaMoeA8W4WaveTypeFunc>::TryReceiveGmm2Completion(
    const AivTaskCursor& cursor, int32_t& completedSequence)
{
    const auto& job = gmmExecutionConfig_.blockJob;
    auto issued = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.flagGmm2IssuedPtr) +
                  static_cast<uint64_t>(job.jobIndex) * INT_CACHELINE;
    if (!Gmm2AivCompletion::IsIssued(issued, completedSequence + 1)) return false;
    const auto& begin = cursor.slice.begin;
    timer_.SetWave(static_cast<int64_t>(cursor.wave));
    timer_.SetSlice(begin.expertIdx, begin.tokenIndexInExpert,
                    cursor.slice.end.globalTokenIndex - begin.globalTokenIndex);
    {
        AscendTimerWaitScope wait(timer_.Enabled() ? &timer_ : nullptr, AscendTimer::AIV_COMBINE_WAIT);
        Gmm2AivCompletion::WaitConsumer();
    }
    NotifyTokenCombineSliceReady<true>(params_, begin.expertIdx, begin.tokenIndexInExpert);
    Gmm2AivCompletion::ReleaseConsumer();
    ++completedSequence;
    timer_.ClearSlice();
    return true;
}

// 三套UB常驻后，每轮推进一个就绪激活，再交替服务Dispatch/Combine。
// Combine未就绪时退回Dispatch；只有已提交完整Wave的FIX事件可以进入完成等待。
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
    AivTaskCursor completion{};
    bool hasCompletion = NextGmm2CompletionSlice(completion);
    int32_t completedSequence = 0;
    Gmm2AivCompletion::InitConsumer();
    if constexpr (ASYNC_GMM1) Gmm1AsyncCompletion::InitConsumer();
    // 一次初始化、逐块消费/归还，最后统一消费末个credit。初始化也覆盖零激活路径。
    Aiv1ActivationUb::Init();
    uint64_t dispatchWave = 0U;
    bool preferCombine = false;
    while (hasActivation || activationPending_ || hasCombine || !shared.done || dispatchAsync_.pending ||
           dispatch.expertIdx < commonConfig_.moeExpertPerRank ||
           hasCompletion) {
        bool progressed = RetireActivation();
        // 此时 AIC 已通过该 Wave 全部输入 ready；保留原激活/Dispatch/Combine 的轮转公平性。
        // FIX队列里排在该GMM2之前的GMM1必须已安排消费，避免Scalar等待FULL反过来堵住EMPTY归还。
        // 非lag先消费同wave；lag先消费下一wave。更后面的激活不属于该完成事件的前置依赖。
        const bool priorActivation = hasActivation && activation.wave <= completion.wave +
            static_cast<uint64_t>(UseGmm2WaveLag(commonConfig_.tokenNum));
        if (hasCompletion && !priorActivation && TryReceiveGmm2Completion(completion, completedSequence)) {
            hasCompletion = NextGmm2CompletionSlice(completion);
            progressed = true;
        }
        if (hasActivation && TryRunReadyActivation(activation, readySequence)) {
            activation.nextTile += gmmExecutionConfig_.blockJob.totalJobs;
            if (activation.nextTile >= activation.tileCount) {
                hasActivation = NextAivSlice(activation, AivTaskKind::Activation);
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
    Aiv1ActivationUb::Finish();
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
            const auto& job = gmmExecutionConfig_.blockJob;
            const uint32_t tiles = GetMGroupCountForRows(slice.rowCount, GMM1_TILE_M) *
                                   Ops::Base::CeilDiv(commonConfig_.tokenHiddenDim, uint32_t(L1_TILE_N));
            const bool owns = GetGmmFirstOwnedTile(job.jobIndex, job.totalJobs, startBlockIdx,
                                                   GMM2_REVERSE_BLOCKS) < tiles;
            this->template RunGmm2CombineForExpert<FuseGateUp>(gmm2ExpertContext, slice, gmm2AddrInfo, startBlockIdx,
                                                               gmm2WaveContext_, GMM2_REVERSE_BLOCKS);
            if constexpr (g_coreType == AIC) {
                if (owns) {
                    // FULL 排在本 slice 的全部 FIX 后、下一个 slice 的输出前；不排空到 Scalar。
                    auto issued = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.flagGmm2IssuedPtr) +
                                  static_cast<uint64_t>(job.jobIndex) * INT_CACHELINE;
                    Gmm2AivCompletion::PublishCompleted(issued, ++gmm2IssuedSequence_);
                }
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
    // routed调度统一走Wave；数据完成仍由各stage内部的真实依赖保证。
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
// 所有EP在BS>=512时滞后一波，最后补齐G2。
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
    // GMM2同向分配；完成发布使用同一个方向常量。
    uint32_t gmm1StartBlockIdx = 0U;
    uint32_t gmm2StartBlockIdx = 0U;

    // 小BS保留较早的下游启动；阈值按本卡实际BS，不按容量或专家接收行数判断。
    const bool gmm2LagActive = UseGmm2WaveLag(commonConfig_.tokenNum);
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
        // AIV1 已消费最后的 GMM2 完成事件后，才回收流水的物理资源。
        if constexpr (g_coreType == AIC) Gmm2AivCompletion::DrainProducer();
        gmm1CatlassContext_.End();
        if constexpr (g_coreType == AIC) Gmm1AsyncCompletion::DrainProducer();
    }
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
