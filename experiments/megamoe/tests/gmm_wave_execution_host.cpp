// Execute production Wave orchestration on CPU; GEMM/communication leaves only record calls.
// LegacyProcessGmmWaves is frozen from c1c4633 to detect order/range/rolling-owner regressions.
#include "common/mega_moe_wave_schedule.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

using namespace MegaMoeImpl;
using ProblemShape = std::array<int64_t, 4>;
constexpr uint32_t M_VALUE = 0, N_VALUE = 1, K_VALUE = 2;
constexpr int AIC = 0, AIV = 1, COMBINE_NO_QUANT = 0;
constexpr uint32_t ACTIVATION_N_HALF = 2;
template <size_t I, typename T> auto& Get(T& value) { return std::get<I>(value); }
namespace Ops::Base {
template <typename T> T CeilDiv(T a, T b) { return a / b + (a % b != 0); }
}
struct MoeStageCommonConfig {
    uint32_t moeExpertPerRank, tokenNum;
    uint32_t gmm1OutputDim = 6144, tokenHiddenDim = 7168;
};
#include "expert_context_methods.inc"

using ExpertLoopState = ExpertContext;  // Frozen baseline spelling, not a production alias.
ExpertContext CreateExpertLoopState(const MoeStageCommonConfig& c) { return CreateExpertContext(c); }
struct GMMAddrInfo {};
struct FakeParams {
    int workspaceInfo = 0;
    operator int() const { return 0; }
};
struct WaveCombineBufferConfig { uint32_t rowBufferCount = 0; };
enum class LoopCountIndex { GMM1, GMM2 };
struct GmmLoopCount { std::array<uint64_t, 2> value{}; };
void UpdateGmmLoopCount(GmmLoopCount* c, LoopCountIndex i, uint64_t value) {
    c->value[static_cast<uint32_t>(i)] = value;
}
namespace AscendTimer { constexpr int64_t NO_WAVE = -1; constexpr int AIV_COMBINE = 1; }
enum class HardEvent { MTE3_MTE2 };
constexpr int SYNC_EVENT_ID0 = 0;
struct Timer {
    int64_t wave = -1;
    void SetWave(int64_t value) { wave = value; }
    void StartSpan(int) {}
    void EndSpan(int) {}
};
using Event = std::array<uint64_t, 9>;

template <int Core, int CombineQuantMode, bool FuseGateUp = false>
struct WaveHarness {
    static constexpr int g_coreType = Core;
    static constexpr uint32_t GMM1_TILE_M = 256;
    static constexpr uint32_t GMM1_TASK_N = FuseGateUp ? 128 : 256;
    static constexpr bool ENABLE_A8W4 = true;
    uint32_t sharedExpertNum_ = 0;
    FakeParams params_;
    int waveCombineJob_ = 0;
    MoeStageCommonConfig commonConfig_;
    std::vector<uint32_t> counts;
    uint32_t mGroupsPerWave_, subBlock;
    Timer timer_;
    int waveCombineScratch_ = 0;
    GmmLoopCount loops;
    GmmLoopCount* gmmLoopCount_ = &loops;
    std::vector<Event> events;

    WaveHarness(std::vector<uint32_t> c, uint32_t bs, uint32_t groups, uint32_t sub)
        : commonConfig_{static_cast<uint32_t>(c.size()), bs}, counts(std::move(c)),
          mGroupsPerWave_(groups), subBlock(sub) {}
    uint32_t GetSubBlockIdx() const { return subBlock; }
    bool HasWaveCapacity(const ExpertTokenPosition& p, uint32_t used) const {
        return MegaMoeImpl::HasWaveCapacity(p, counts.size(), used, mGroupsPerWave_);
    }
    void PrepareGmmExpertContext(ExpertContext& context, uint32_t e) {
        UpdateExpertContext(context, e, counts.at(e));
    }
    template <bool Wait> void PrepareGmmExpertState(ExpertContext& context, uint32_t e) {
        PrepareGmmExpertContext(context, e);
    }
    template <int Mode> WaveCombineBufferConfig InitWaveCombineBuffers(const MoeStageCommonConfig&, int&) {
        return {2};
    }
    void DrainCombineRowBuffers(uint32_t sequence, uint32_t slots) {
        events.push_back({3, sequence, slots, 0, 0, 0, 0, 0, 0});
    }
    void NotifyWaveGmm2Ready(int, int, uint32_t e) {
        events.push_back({5, e, 0, 0, 0, 0, 0, 0, 0});
    }
    void UpdateMoeExpertCombineGlobalBuffer(int, GMMAddrInfo&, const ExpertContext& context) {
        events.push_back({6, context.expertIdx, 0, 0, 0, 0, 0, 0, 0});
    }
    template <HardEvent EventType, int Id> void SyncFuncStatic() {
        events.push_back({7, 0, 0, 0, 0, 0, 0, 0, 0});
    }
    template <int Mode>
    WaveCombineBufferConfig PrepareFinalWaveCombineBuffers(const MoeStageCommonConfig&,
                                                           WaveCombineBufferConfig buffers, int&) {
        events.push_back({8, 0, 0, 0, 0, 0, 0, 0, 0});
        buffers.rowBufferCount = 1;
        return buffers;
    }
    template <int Mode, bool Final = false>
    void RunWaveCombineStage(const MoeStageCommonConfig&, int, const WaveCombineBufferConfig& buffers, int&, int,
                              GMMAddrInfo&, const ExpertContext&, uint32_t e, uint32_t& sequence) {
        events.push_back({9, e, buffers.rowBufferCount, Final, sequence++, 0, 0, 0, 0});
    }
    void DispatchFirstWave(ExpertTokenPosition& cursor) {
        if constexpr (Core == AIV) {
            if (subBlock != 1) return;
            const uint64_t first = cursor.globalTokenIndex;
            uint32_t groups = 0;
            while (HasWaveCapacity(cursor, groups)) {
                AdvanceExpertTokenPositionInWave<256>(counts.at(cursor.expertIdx), mGroupsPerWave_, groups, cursor);
            }
            events.push_back({0, 0, 0, cursor.globalTokenIndex - first, first, 0, 0, 0, 0});
        }
    }
    void DispatchNextExpertSlice(ExpertTokenPosition& cursor, uint32_t& groups, uint64_t wave) {
        // The production planner skips empty experts before choosing one nonempty slice.
        while (cursor.expertIdx < counts.size() && cursor.tokenIndexInExpert >= counts[cursor.expertIdx]) {
            ++cursor.expertIdx;
            cursor.tokenIndexInExpert = 0;
        }
        if (cursor.expertIdx == counts.size()) return;
        const auto slice = TakeNextExpertSliceInWave<256>(counts[cursor.expertIdx], mGroupsPerWave_, groups, cursor);
        events.push_back({4, slice.expertIdx, slice.rowStartInExpert, slice.rowCount, slice.globalRowStart,
                          wave, 0, 0, 0});
    }
    void RunGmm1ActivationForExpert(const ExpertContext& context, const ExpertSlice& slice, GMMAddrInfo&,
                                    uint32_t& start, int32_t& ready, uint32_t tasksPerM) {
        assert(context.expertIdx == slice.expertIdx);
        assert(context.globalTokenStartIndex + slice.rowStartInExpert == slice.globalRowStart);
        const auto before = start;
        const uint32_t tasks = Ops::Base::CeilDiv(slice.rowCount, 256U) * tasksPerM;
        start = (start + tasks) % 32;
        ++ready;
        events.push_back({1, slice.expertIdx, slice.rowStartInExpert, slice.rowCount, slice.globalRowStart,
                          static_cast<uint64_t>(timer_.wave), before, start, static_cast<uint64_t>(ready)});
    }
    // Original entry signature used only by the frozen scheduler.
    void RunGmm1ActivationForExpert(const ExpertContext& context, GMMAddrInfo& addr, uint32_t& start, int32_t& ready,
                                    uint32_t rowStart, uint32_t rowCount, uint32_t tasksPerM) {
        const ExpertSlice slice{context.expertIdx, rowStart, rowCount,
                                static_cast<uint64_t>(context.globalTokenStartIndex) + rowStart};
        RunGmm1ActivationForExpert(context, slice, addr, start, ready, tasksPerM);
    }
    template <bool SourceFused>
    void RunGmm2CombineForExpert(const ExpertContext& context, const ExpertSlice& slice, GMMAddrInfo&,
                                uint32_t& start, WaveCombineBufferConfig&, uint32_t& rowSequence,
                                int32_t&, bool final, bool reverse) {
        assert(reverse);
        assert(context.expertIdx == slice.expertIdx);
        assert(context.globalTokenStartIndex + slice.rowStartInExpert == slice.globalRowStart);
        const auto before = start;
        start = (start + Ops::Base::CeilDiv(slice.rowCount, 256U) * 28) % 32;
        if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
            if (slice.rowStartInExpert + slice.rowCount == static_cast<uint32_t>(context.problemShape[0])) {
                rowSequence += context.problemShape[0];
            }
        }
        events.push_back({2, slice.expertIdx, slice.rowStartInExpert, slice.rowCount, slice.globalRowStart,
                          static_cast<uint64_t>(timer_.wave), before, start, final});
    }

#include "gmm_wave_methods.inc"

    void LegacyQuantizedCombineForExpert(const ExpertContext& expertContext, const ExpertSlice& slice,
                                         GMMAddrInfo& gmmAddrInfo, WaveCombineBufferConfig& combineBufferConfig,
                                         uint32_t& combineRowSequence, bool isFinalCombine) {
        const uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(expertContext.problemShape));
// 量化 Combine 保持原 W4 语义：跨 WAVE 的专家必须等最后一个 slice 完成后再消费。
    if (slice.rowStartInExpert + slice.rowCount < expertTokenCount) {
        return;
    }

    NotifyWaveGmm2Ready(waveCombineJob_, params_, expertContext.expertIdx);
    UpdateMoeExpertCombineGlobalBuffer(params_.workspaceInfo, gmmAddrInfo, expertContext);
    if (isFinalCombine && !(ENABLE_A8W4 && sharedExpertNum_ > 0U)) {
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 0U) {
                // W4 prologue 完成后，末轮 AIV0 才能通过 MTE2 复用同一 UB 区域。
                SyncFuncStatic<HardEvent::MTE3_MTE2, SYNC_EVENT_ID0>();
            }
        }
        combineBufferConfig =
            PrepareFinalWaveCombineBuffers<CombineQuantMode>(commonConfig_, combineBufferConfig, waveCombineScratch_);
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U) {
                timer_.StartSpan(AscendTimer::AIV_COMBINE);
            }
        }
        RunWaveCombineStage<CombineQuantMode, true>(commonConfig_, waveCombineJob_, combineBufferConfig,
                                                    waveCombineScratch_, params_, gmmAddrInfo, expertContext, expertContext.expertIdx,
                                                    combineRowSequence);
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U) {
                timer_.EndSpan(AscendTimer::AIV_COMBINE);
            }
        }
        return;
    }
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            timer_.StartSpan(AscendTimer::AIV_COMBINE);
        }
    }
    RunWaveCombineStage<CombineQuantMode>(commonConfig_, waveCombineJob_, combineBufferConfig, waveCombineScratch_,
                                          params_, gmmAddrInfo, expertContext, expertContext.expertIdx, combineRowSequence);
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            timer_.EndSpan(AscendTimer::AIV_COMBINE);
        }
    }
    }

    void LegacyProcessGmmWaves(int32_t& gmm1TileReadySequence)
{
    uint64_t gmm1Count = 0U;
    uint64_t gmm2Count = 0U;
    WaveCombineBufferConfig combineBufferConfig{};
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        combineBufferConfig = InitWaveCombineBuffers<CombineQuantMode>(commonConfig_, waveCombineScratch_);
    }
    uint32_t combineRowSequence = 0U;

    // 完整专家元数据与本次 slice 分开：跨 Wave 复用专家状态，不把 slice 行数写回专家总行数。
    ExpertLoopState gmm1State = CreateExpertLoopState(commonConfig_);
    ExpertLoopState gmm2State = CreateExpertLoopState(commonConfig_);
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};

    // 两阶段各自跨专家、跨 Wave 轮转。GMM1 从物理核 0 正向分配，
    // GMM2 的逻辑核号反转，从物理核 blockNum-1 反向分配。
    uint32_t gmm1StartBlockIdx = 0U;
    uint32_t gmm2StartBlockIdx = 0U;

    const uint32_t gmm1TilesPerMGroup =
        Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(GMM1_TASK_N));

    ExpertTokenPosition dispatchPosition{};

    // 按本卡输入 BS 判断，不按专家接收行数或当前 slice 行数判断。
    const bool gmm2LagActive = commonConfig_.tokenNum >= 4096U;
    ExpertWaveRange pendingWave{{}, {}, commonConfig_.moeExpertPerRank};
    bool hasPendingGmm2Wave = false;

    DispatchFirstWave(dispatchPosition);

    // position 总是指向下一待规划位置；本次计算使用 TakeNextExpertSliceInWave 返回的固定 slice。
    ExpertTokenPosition gmm1Position{};
    while (gmm1Position.expertIdx < commonConfig_.moeExpertPerRank) {
        ExpertWaveRange wave{gmm1Position, gmm1Position, commonConfig_.moeExpertPerRank};
        uint32_t currentWaveMGroupCount = 0U;
        bool currentWaveNeedsGmm1 = true;

        uint32_t nextDispatchWaveMGroupCount = 0U;
        bool nextWaveNeedsDispatch = false;
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U) {
                nextWaveNeedsDispatch = HasWaveCapacity(dispatchPosition, nextDispatchWaveMGroupCount);
            }
        }

        while (currentWaveNeedsGmm1 || nextWaveNeedsDispatch) {
            // 仅量化路径的 AIV1：先 Dispatch 下一 Wave 的 slice，再激活当前 Wave 的 slice。
            if constexpr (g_coreType == AIV) {
                if (GetSubBlockIdx() == 1U && nextWaveNeedsDispatch) {
                    DispatchNextExpertSlice(dispatchPosition, nextDispatchWaveMGroupCount, gmm1Count + 1U);
                    nextWaveNeedsDispatch = HasWaveCapacity(dispatchPosition, nextDispatchWaveMGroupCount);
                }
            }

            if (currentWaveNeedsGmm1) {
                if (gmm1Position.tokenIndexInExpert == 0U) {
                    this->template PrepareGmmExpertState<true>(gmm1State, gmm1Position.expertIdx);
                }
                const uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(gmm1State.problemShape));
                const ExpertSlice slice = TakeNextExpertSliceInWave<GMM1_TILE_M>(
                    expertTokenCount, mGroupsPerWave_, currentWaveMGroupCount, gmm1Position);
                if (slice.rowCount != 0U) {
                    wave.lastActiveExpert = slice.expertIdx;
                    timer_.SetWave(static_cast<int64_t>(gmm1Count));
                    RunGmm1ActivationForExpert(gmm1State, slice, gmm1AddrInfo, gmm1StartBlockIdx,
                                               gmm1TileReadySequence, gmm1TilesPerMGroup);
                }
                wave.end = gmm1Position;
                currentWaveNeedsGmm1 = HasWaveCapacity(gmm1Position, currentWaveMGroupCount);
            }
        }
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM1, ++gmm1Count);
        if (gmm2LagActive) {
            // 滞后一拍：本轮消费上一 Wave 的 GMM2/Combine；当前 Wave 边界缓存到下一轮。
            if (hasPendingGmm2Wave) {
                timer_.SetWave(static_cast<int64_t>(gmm2Count));
                RunGmm2CombineForWaveRange(pendingWave, gmm2State, gmm2AddrInfo, gmm2StartBlockIdx, combineBufferConfig,
                                           combineRowSequence, gmm1TileReadySequence);
                UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
            }
            pendingWave = wave;
            hasPendingGmm2Wave = true;
        } else {
            // 当前 Wave 完成后立即消费，使其 GMM2/Combine 与下一 Wave 的输入预取重叠。
            timer_.SetWave(static_cast<int64_t>(gmm2Count));
            RunGmm2CombineForWaveRange(wave, gmm2State, gmm2AddrInfo, gmm2StartBlockIdx, combineBufferConfig,
                                       combineRowSequence, gmm1TileReadySequence);
            UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
        }
    }
    // 计算侧补跑最后一个 Wave；非量化 AIV1 仍按自己的 ready 游标发送 Combine。
    if (hasPendingGmm2Wave) {
        timer_.SetWave(static_cast<int64_t>(gmm2Count));
        RunGmm2CombineForWaveRange(pendingWave, gmm2State, gmm2AddrInfo, gmm2StartBlockIdx, combineBufferConfig,
                                   combineRowSequence, gmm1TileReadySequence);
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
    }
    timer_.SetWave(AscendTimer::NO_WAVE);
}
};

template <int Core, int Mode, bool Fused = false>
static void Check(const std::vector<uint32_t>& counts, uint32_t bs, uint32_t groups, uint32_t sub = 0) {
    WaveHarness<Core, Mode, Fused> current(counts, bs, groups, sub), legacy(counts, bs, groups, sub);
    int32_t ready = 0, oldReady = 0;
    current.ProcessGmmWaves(ready);
    legacy.LegacyProcessGmmWaves(oldReady);
    assert(current.events == legacy.events);
    assert(current.loops.value == legacy.loops.value);
    assert(ready == oldReady);
    assert(current.timer_.wave == legacy.timer_.wave);
    assert(current.loops.value[0] == current.loops.value[1]);
}

static void CheckRoles(const std::vector<uint32_t>& counts, uint32_t bs, uint32_t groups) {
    Check<AIC, 0>(counts, bs, groups);
    Check<AIV, 0>(counts, bs, groups);
    Check<AIC, 1>(counts, bs, groups);
    Check<AIV, 1>(counts, bs, groups);
    Check<AIV, 1>(counts, bs, groups, 1);
    Check<AIC, 0, true>(counts, bs, groups);
    Check<AIV, 1, true>(counts, bs, groups, 1);
}
template <int Core> static void CheckQuantizedCombine(uint32_t sub) {
    for (uint32_t shared : {0U, 1U}) {
        for (bool final : {false, true}) {
            for (uint32_t end : {1536U, 4096U}) {
                WaveHarness<Core, 1> current({4096}, 4096, 6, sub), legacy({4096}, 4096, 6, sub);
                current.sharedExpertNum_ = legacy.sharedExpertNum_ = shared;
                ExpertContext context = CreateExpertContext(current.commonConfig_);
                UpdateExpertContext(context, 0, 4096);
                const ExpertSlice slice{0, 0, end, 0};
                GMMAddrInfo addr;
                WaveCombineBufferConfig buffers{2}, oldBuffers{2};
                uint32_t sequence = 0, oldSequence = 0;
                current.RunQuantizedCombineForExpert(context, slice, addr, buffers, sequence, final);
                legacy.LegacyQuantizedCombineForExpert(context, slice, addr, oldBuffers, oldSequence, final);
                assert(current.events == legacy.events);
                assert(sequence == oldSequence && buffers.rowBufferCount == oldBuffers.rowBufferCount);
                if (end < 4096) assert(current.events.empty());
            }
        }
    }
}
int main() {
    CheckQuantizedCombine<AIC>(0);
    CheckQuantizedCombine<AIV>(0);
    CheckQuantizedCombine<AIV>(1);
    for (uint32_t bs : {96U, 4095U, 4096U, 8192U}) {
        CheckRoles(std::vector<uint32_t>(6, 4096), bs, 6);
        CheckRoles({0, 0, 0}, bs, 6);
        CheckRoles({0, 255, 0, 256, 257, 0, 4097, 0}, bs, 3);
        CheckRoles({1}, bs, 1);
        CheckRoles({}, bs, 6);
    }
    uint32_t seed = 20261015;
    for (uint32_t i = 0; i < 160; ++i) {
        std::vector<uint32_t> counts;
        for (uint32_t e = 0; e < 1 + i % 12; ++e) {
            seed = seed * 1664525U + 1013904223U;
            counts.push_back(e % 4 == 0 ? 0U : seed % 8193U);
        }
        CheckRoles(counts, i % 2 ? 4096 : 96, 1 + i % 8);
    }
}
