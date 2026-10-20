// Run the real orchestration body; stubs only model task readiness and completion.
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
struct AivTaskCursor { uint32_t nextTile = 0, tileCount = 1, index = 0; };
enum class AivTaskKind { Activation, Combine };
struct SharedActivationCursor { bool done = false; };
struct ExpertTokenPosition { uint32_t expertIdx = 0; };
struct Gmm1AsyncCompletion { static void InitConsumer() {} };
namespace AscendC {
int64_t GetSystemCycle() {
    static int64_t ticks = 0;
    if (++ticks > 10000) throw std::runtime_error("scheduler failed to terminate");
    return ticks;
}
}
namespace AscendTimer { constexpr int NO_WAVE = -1; }
constexpr int GM_FLAG_POLL_BACKOFF_CYCLES = 1;
struct Scheduler {
    static constexpr bool ASYNC_GMM1 = true;
    bool computeReplay_ = false;
    bool delayActivation = false, delayCombine = false;
    uint32_t sharedExpertNum_ = 0, mGroupsPerWave_ = 2;
    struct { uint32_t moeExpertPerRank = 6; } commonConfig_;
    struct { struct { uint32_t totalJobs = 1; } blockJob; } gmmExecutionConfig_;
    struct { void ClearSlice() {} void SetWave(int) {} } timer_;
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
        ++sequence;
        order.push_back('A');
        return true;
    }
    bool TryRunReadyCombine(AivTaskCursor& c) {
        if (delayCombine && dispatched < commonConfig_.moeExpertPerRank) return false;
        if (activated <= c.index) return false;
        assert(c.index == combined++);
        order.push_back('C');
        return true;
    }
    bool TryRunSharedExpertActivation(SharedActivationCursor&) { assert(false); return false; }
    void DispatchNextExpertSlice(ExpertTokenPosition& d, uint32_t& groups, uint64_t wave) {
        assert(wave == dispatched / mGroupsPerWave_);
        assert(groups < mGroupsPerWave_);
        ++groups; ++dispatched; ++d.expertIdx;
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
    assert(s.order[0] == 'D' && s.order[1] == 'A' && s.order[2] == 'C');
    for (bool delayA : {false, true}) {
        Scheduler stalled;
        stalled.delayActivation = delayA;
        stalled.delayCombine = true;
        sequence = 0;
        stalled.ProcessAiv1PriorityStages(sequence);
        assert(sequence == 6 && stalled.dispatched == 6 && stalled.combined == 6);
    }
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
