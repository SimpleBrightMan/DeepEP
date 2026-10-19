// Copyright (c) 2026, SimpleBright_Man
// The Python test inserts the actual full-kernel TryRunReadyCombine body.
#include <cstdint>
#undef NDEBUG
#include <cassert>
#include <iostream>
#include <vector>

struct Job {
    uint32_t jobIndex = 0U;
    uint32_t totalJobs = 28U;
};
struct Position {
    uint32_t expertIdx = 0U;
    uint32_t tokenIndexInExpert = 0U;
    uint64_t globalTokenIndex = 0U;
};
struct Slice {
    Position begin{}, end{};
};
struct AivTaskCursor {
    Slice slice{};
    uint64_t wave = 0U;
};
struct ExpertLoopState {
    uint32_t expertIdx = 0U;
    uint64_t globalTokenStartIndex = 0U;
};
struct WorkRange {
    uint32_t start = 0U;
    uint32_t count = 0U;
};
struct Owned {
    uint32_t rows = 0U;
};
struct Common {};
struct Params {};
struct DispatchScratch {};
struct CombineScratch {};
struct Execution {
    Job blockJob{};
};
namespace AscendTimer {
constexpr uint32_t AIV_COMBINE_COMM = 17U;
}
constexpr uint32_t PREFILL_COMM_INTRA_CORES = 4U;

enum class Call { Probe, ProducerCount, Reload, Plan, Split, Original, TimerStart, TimerEnd };
static std::vector<Call> calls;
static int32_t readyCount = 0;
static uint32_t expectedPhysicalCore = 0U;
static uint32_t selectedRows = 11U;
static uint32_t lastOriginalOwner = 0U;
static uint64_t lastRotation = 0U;
static uint32_t lastExpert = 0U;
static uint32_t lastStart = 0U;
static uint32_t lastCount = 0U;

struct Timer {
    void SetWave(int64_t value)
    {
        assert(value == 7);
    }
    void StartCommunication(uint32_t type)
    {
        assert(type == AscendTimer::AIV_COMBINE_COMM);
        calls.push_back(Call::TimerStart);
    }
    void EndCommunication()
    {
        calls.push_back(Call::TimerEnd);
    }
    void SetCommunicationRows(uint32_t expert, uint32_t start, uint32_t rows)
    {
        lastExpert = expert;
        lastStart = start;
        lastCount = rows;
    }
};

static int32_t* GetTokenCombineReadyCounter(const Params&, uint32_t expert, uint32_t row)
{
    assert(expert == 3U && row == 512U);
    return &readyCount;
}
namespace AscendC {
static int32_t ReadGmByPassDCache(int32_t* value)
{
    calls.push_back(Call::Probe);
    return *value;
}
}  // namespace AscendC
static uint32_t GetTokenCombineProducerCount(const Common&, const Job& job, uint32_t rows)
{
    calls.push_back(Call::ProducerCount);
    assert(job.jobIndex == expectedPhysicalCore && job.totalJobs == 28U);
    assert(rows == 1024U);
    return 28U;
}
static void ReloadDispatchCumsumRange(const Common&, DispatchScratch&, uint32_t first, uint32_t last)
{
    assert(readyCount >= 28 && first == 3U && last == 3U);
    calls.push_back(Call::Reload);
}
static Owned PlanCommSplitSlice(const Common&, const Params&, DispatchScratch&, uint32_t expert, uint32_t start,
                                uint32_t stop, uint32_t physicalCore, uint32_t intra, uint64_t rotation, bool xorOrder)
{
    calls.push_back(Call::Plan);
    assert(physicalCore == expectedPhysicalCore && intra == 4U && xorOrder);
    assert(expert == 3U && start == 512U && stop == 1536U);
    lastRotation = rotation;
    return {selectedRows};
}
static ExpertLoopState CreateExpertLoopState(const Common&)
{
    return {};
}
static void RunCommSplitCombineSlice(const Common&, const Params&, const ExpertLoopState& state, uint32_t start,
                                     const Owned& owned, Timer&, CombineScratch&)
{
    assert(readyCount >= 28 && state.expertIdx == 3U && state.globalTokenStartIndex == 3000U);
    assert(start == 512U && owned.rows == selectedRows);
    calls.push_back(Call::Split);
}
static WorkRange GetWaveCombineOwnedRange(const Job& job, uint32_t rows, uint64_t global)
{
    assert(job.totalJobs == 28U && rows == 1024U && global == 3512U);
    lastOriginalOwner = job.jobIndex;
    return {2U, selectedRows};
}
template <bool InputReadyChecked>
static void RunTokenCombineSlice(const Common&, const Job& job, const Params&, const ExpertLoopState& state,
                                 uint32_t start, uint32_t rows, Timer&, CombineScratch&)
{
    static_assert(InputReadyChecked);
    assert(readyCount >= 28 && job.jobIndex == lastOriginalOwner && job.totalJobs == 28U);
    assert(state.expertIdx == 3U && state.globalTokenStartIndex == 3000U && start == 512U && rows == 1024U);
    calls.push_back(Call::Original);
}

struct Kernel {
    bool optimized = true;
    Common commonConfig_{};
    Params params_{};
    Job waveCombineJob_{};
    Execution gmmExecutionConfig_{};
    DispatchScratch tokenDispatchScratch_{};
    CombineScratch waveCombineScratch_{};
    Timer timer_{};
    bool UseOptimizedPrefillCommunication() const
    {
        return optimized;
    }
    bool TryRunReadyCombine(const AivTaskCursor& cursor)
    // INSERT_ACTUAL_READY_PROBE
};

int main()
{
    const AivTaskCursor cursor{{{3U, 512U, 3512U}, {3U, 1536U, 4536U}}, 7U};
    for (uint32_t core = 0U; core < 28U; ++core) {
        expectedPhysicalCore = core;
        Kernel kernel;
        kernel.waveCombineJob_.jobIndex = core;
        kernel.gmmExecutionConfig_.blockJob.jobIndex = core;
        for (bool optimized : {true, false}) {
            kernel.optimized = optimized;
            selectedRows = 11U;
            for (int32_t ready : {0, 1, 27}) {
                readyCount = ready;
                calls.clear();
                assert(!kernel.TryRunReadyCombine(cursor));
                assert((calls == std::vector<Call>{Call::Probe, Call::ProducerCount}));
            }
            readyCount = 28;
            calls.clear();
            assert(kernel.TryRunReadyCombine(cursor));
            if (optimized) {
                assert((calls == std::vector<Call>{Call::Probe, Call::ProducerCount, Call::TimerStart, Call::Reload,
                                                   Call::Plan, Call::Split, Call::TimerEnd}));
                assert(lastRotation == 3519U && lastExpert == 3U && lastStart == 512U && lastCount == selectedRows);
            } else {
                assert((calls == std::vector<Call>{Call::Probe, Call::ProducerCount, Call::TimerStart, Call::Original,
                                                   Call::TimerEnd}));
                assert(lastOriginalOwner == (core + 28U - 7U) % 28U);
            }
            assert(kernel.waveCombineJob_.jobIndex == core && kernel.gmmExecutionConfig_.blockJob.jobIndex == core);
        }
    }
    std::cout << "nonblocking ready probe, original GMM2 producer team and full-path fallback passed\n";
}
