// Run the real orchestration body; stubs only model task readiness and completion.
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include "common/aiv1_async_events.hpp"
using MegaMoeImpl::Aiv1AsyncEvents;
using TEventID = unsigned;
struct AivTaskCursor { uint32_t nextTile = 0, tileCount = 1, index = 0, nextRow = 0; };
enum class AivTaskKind { Activation, Combine };
struct SharedActivationCursor { bool done = false; };
struct ExpertTokenPosition { uint32_t expertIdx = 0; };
struct Gmm1AsyncCompletion { static void InitConsumer() {} };
struct Gmm2AivCompletion { static void InitConsumer() {} };
namespace AscendC {
enum class HardEvent { V_MTE2, MTE3_V };
bool activationCredit = false;
template <HardEvent> void SetFlag(TEventID) { assert(!activationCredit); activationCredit = true; }
template <HardEvent> void WaitFlag(TEventID) { assert(activationCredit); activationCredit = false; }
int64_t GetSystemCycle() {
    static int64_t ticks = 0;
    if (++ticks > 10000) throw std::runtime_error("scheduler failed to terminate");
    return ticks;
}
}
#define __aicore__
#include "common/aiv1_activation_ub.hpp"
#include "common/gmm1_cv_plan.hpp"
using MegaMoeImpl::Aiv1ActivationUb;
using MegaMoeImpl::Gmm1CvPlan;
namespace AscendTimer { constexpr int NO_WAVE = -1; }
constexpr int GM_FLAG_POLL_BACKOFF_CYCLES = 1;
template <typename A, typename S, bool P, class... T> void DrainDispatchCopyBuffers(T...) {}
void DrainCombineRowBuffers(uint32_t& n, uint32_t count, uint32_t base) { assert(count==4 && base==2); n=0; }
struct Scheduler {
    using ActivationType = uint8_t;
    using QuantScaleOutType = uint8_t;
    static constexpr bool ASYNC_GMM1 = true;
    static constexpr bool FuseGateUp = false;
    static constexpr bool UseCv = false;
    struct CvParams {
        int yGmAddr, yScaleGmAddr;
        float clampLimit;
        unsigned actMode, actSubMode;
        float activationAlpha, activationBeta;
        unsigned ubBaseBytes;
    };
    struct { void Init(CvParams) { assert(false); } } cvActivationOp_;
    struct Tiling { float clampLimit=0; unsigned actMode=0, actSubMode=0; float activationAlpha=1, activationBeta=1; } tiling_;
    struct { struct { int activationQuantDataPtr=0, activationQuantScalePtr=0; } workspaceInfo; Tiling* tilingData=nullptr; } params_;
    bool computeReplay_ = false;
    bool delayActivation = false, delayCombine = false, delayCompletion = false;
    uint32_t sharedExpertNum_ = 0, mGroupsPerWave_ = 2;
    struct { uint32_t moeExpertPerRank = 6; } commonConfig_;
    struct { struct { uint32_t totalJobs = 1; } blockJob; } gmmExecutionConfig_;
    struct { void ClearSlice() {} void SetWave(int) {} } timer_;
    struct { bool pending = false; int progress = 0; } dispatchAsync_;
    struct { uint32_t rowBufferCount = 4, eventBase = 2; } waveCombineBuffers_;
    int tokenDispatchConfig_ = 0, tokenDispatchScratch_ = 0;
    uint32_t combineRowSequence_ = 0;
    bool activationPending_ = false;
    uint32_t completedActivations = 0, issuedDispatches = 0, completedGmm2 = 0;
    uint32_t dispatched = 0, activated = 0, combined = 0, nextA = 0, nextC = 0;
    std::vector<char> order;
    bool NextAivSlice(AivTaskCursor& c, AivTaskKind kind) {
        auto& next = kind == AivTaskKind::Activation ? nextA : nextC;
        if (next == commonConfig_.moeExpertPerRank) return false;
        c = {0, 1, next++};
        return true;
    }
    bool TryRunReadyActivation(AivTaskCursor& c, int32_t& sequence) {
        if (delayActivation && dispatched < commonConfig_.moeExpertPerRank) return false;
        if (!computeReplay_ && dispatched <= c.index) return false;
        assert(c.index == activated++);
        assert(!activationPending_);
        activationPending_ = true;
        ++sequence;
        order.push_back('A');
        return true;
    }
    bool TryReceiveGmm2Completion(ExpertTokenPosition& p, uint32_t& start, int32_t& sequence) {
        if (p.expertIdx >= commonConfig_.moeExpertPerRank) return false;
        if (delayCompletion && dispatched < commonConfig_.moeExpertPerRank) return false;
        const auto end = std::min(p.expertIdx + mGroupsPerWave_, commonConfig_.moeExpertPerRank);
        if (completedActivations < end) return false;
        p.expertIdx = end; completedGmm2 = end; ++sequence; ++start; order.push_back('F');
        return true;
    }
    bool TryRunReadyCombine(AivTaskCursor& c) {
        if (delayCombine && dispatched < commonConfig_.moeExpertPerRank) return false;
        if (completedGmm2 <= c.index) return false;
        assert(c.index == combined++);
        order.push_back('C');
        return true;
    }
    bool TryRunSharedExpertActivation(SharedActivationCursor&) { assert(false); return false; }
    bool RetireActivation() {
        if (!activationPending_) return false;
        activationPending_ = false;
        ++completedActivations;
        order.push_back('P');
        return true;
    }
    void DispatchNextExpertSlice(ExpertTokenPosition& d, uint32_t& groups, uint64_t wave) {
        if (dispatchAsync_.pending) {
            dispatchAsync_.pending = false;
            ++dispatched;
            order.push_back('R');
            return;
        }
        assert(wave == issuedDispatches / mGroupsPerWave_);
        assert(groups < mGroupsPerWave_);
        ++groups; ++issuedDispatches; ++d.expertIdx;
        dispatchAsync_.pending = true;
        order.push_back('D');
    }
#include "aiv1_loop.inc"
};
int main() {
    Scheduler s;
    int32_t sequence = 0;
    s.ProcessAiv1PriorityStages(sequence);
    assert(sequence == 6 && s.dispatched == 6 && s.activated == 6 && s.combined == 6);
    // The first result is serviced while inputs remain; strict producer priority fails here.
    assert(std::find(s.order.begin(), s.order.end(), 'C') < std::find(s.order.rbegin(), s.order.rend(), 'D').base()-1);
    assert(!s.activationPending_ && !s.dispatchAsync_.pending && !AscendC::activationCredit);
    for (bool delayA : {false, true}) {
        Scheduler stalled;
        stalled.delayActivation = delayA;
        stalled.delayCombine = true;
        sequence = 0;
        stalled.ProcessAiv1PriorityStages(sequence);
        assert(sequence == 6 && stalled.dispatched == 6 && stalled.combined == 6);
    }
    Scheduler delayedFix;
    delayedFix.delayCompletion = true;
    sequence = 0;
    delayedFix.ProcessAiv1PriorityStages(sequence);
    assert(sequence == 6 && delayedFix.dispatched == 6 && delayedFix.combined == 6);
    assert(delayedFix.completedGmm2 == 6);
    Scheduler replay;
    replay.computeReplay_ = true;
    sequence = 0;
    replay.ProcessAiv1PriorityStages(sequence);
    assert(sequence == 6 && replay.dispatched == 0 && replay.combined == 0);
    Scheduler empty;
    empty.commonConfig_.moeExpertPerRank = 0;
    sequence = 0;
    empty.ProcessAiv1PriorityStages(sequence);
    assert(sequence == 0 && empty.order.empty());
}
