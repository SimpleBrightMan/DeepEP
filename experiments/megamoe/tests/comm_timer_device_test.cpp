// Copyright (c) 2026, SimpleBright_Man
// CPU execution of the actual timer writer, with sparse GM and clock mocks.
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>

#define __aicore__
#define __gm__
#define ASCEND_IS_AIV (AscendC::hardwareCore % 3 != 0)
#define DEEPEP_MEGAMOE_TIMER
using GM_ADDR = uint8_t*;
constexpr int PIPE_ALL = 0;
namespace AscendC {
static int barriers = 0;
static int64_t cycle = 100;
static int hardwareCore = 2;
inline int GetBlockIdx()
{
    return hardwareCore % 3 == 0 ? hardwareCore / 3 : hardwareCore / 3 * 2 + hardwareCore % 3 - 1;
}
inline int GetTaskRation()
{
    return hardwareCore % 3 == 0 ? 1 : 2;
}
inline int GetSubBlockIdx()
{
    return hardwareCore % 3 - 1;
}
inline int64_t GetSystemCycle()
{
    return cycle += 10;
}
template <int>
inline void PipeBarrier()
{
    ++barriers;
}
template <typename T>
class GlobalTensor {
public:
    static constexpr int CACHE_LINE_WORDS = 128 / sizeof(T);
    struct Line {
        std::array<T, CACHE_LINE_WORDS> values{};
        bool dirty = false;
    };
    static std::map<int64_t, T> words;
    static std::map<int, std::map<int64_t, Line>> caches;
    static bool cacheEnabled;
    void SetGlobalBuffer(T*) {}
    T GetValue(int64_t index) const
    {
        return cacheEnabled ? CacheLine(index).values[index % CACHE_LINE_WORDS] : words[index];
    }
    T& operator()(int64_t index)
    {
        if (cacheEnabled) {
            auto& line = CacheLine(index);
            line.dirty = true;
            return line.values[index % CACHE_LINE_WORDS];
        }
        return words[index];
    }
    static void FlushCore(int core)
    {
        // Model non-coherent per-core scalar caches: a dirty eviction writes
        // the entire 128-byte line, including stale words from other writers.
        for (auto& [base, line] : caches[core]) {
            if (!line.dirty) continue;
            for (int i = 0; i < CACHE_LINE_WORDS; ++i) words[base + i] = line.values[i];
            line.dirty = false;
        }
    }

private:
    static Line& CacheLine(int64_t index)
    {
        const int64_t base = index / CACHE_LINE_WORDS * CACHE_LINE_WORDS;
        auto& cache = caches[hardwareCore];
        auto [entry, inserted] = cache.try_emplace(base);
        if (inserted) {
            for (int i = 0; i < CACHE_LINE_WORDS; ++i) entry->second.values[i] = words[base + i];
        }
        return entry->second;
    }
};
template <typename T>
std::map<int64_t, T> GlobalTensor<T>::words;
template <typename T>
std::map<int, std::map<int64_t, typename GlobalTensor<T>::Line>> GlobalTensor<T>::caches;
template <typename T>
bool GlobalTensor<T>::cacheEnabled = false;
}  // namespace AscendC

#include "common/ascend_timer_v2_device.hpp"

static void Require(bool ok)
{
    if (!ok) std::abort();
}

static void CheckScalarCacheLineIsolation()
{
    using namespace AscendTimer;
    using Tensor = AscendC::GlobalTensor<int64_t>;
    auto& words = Tensor::words;
    Tensor tensor;
    Tensor::cacheEnabled = true;
    words.clear();
    Tensor::caches.clear();

    // Negative control reproduces the old packed layout: all engines read the
    // enable line, then idle cores write zero beside AIV1's nonzero count. A
    // later idle-core eviction loses AIV1's count and the wave budget.
    constexpr int oldEnable = 0;
    constexpr int oldBudget = oldEnable + 1;
    constexpr int oldCount = oldEnable + 2;
    words[oldEnable] = 1;
    for (int core = 0; core < N_CORE_COUNT; ++core) {
        AscendC::hardwareCore = core;
        Require(tensor.GetValue(oldEnable) == 1);
    }
    for (int core = 0; core < N_CORE_COUNT; ++core) {
        AscendC::hardwareCore = core;
        if (core == 2) tensor(oldBudget) = 12;
        tensor(oldCount + core) = core % 3 == 2 ? 2 : 0;
    }
    for (int core = N_CORE_COUNT - 1; core >= 0; --core) Tensor::FlushCore(core);
    Require(words[oldCount + 2] == 0);
    Require(words[oldBudget] == 0);

    // Execute the actual fixed writer under the same dirty-cacheline model.
    // Include idle AIC/AIV0 writers and both eviction orders, without adding
    // device synchronization or changing any communication timing boundary.
    for (bool reverse : {false, true}) {
        words.clear();
        Tensor::caches.clear();
        words[COMM_AUDIT_ENABLE_OFFSET] = 1;
        std::array<AscendTimerDevice, N_CORE_COUNT> timers;
        for (int core = 0; core < N_CORE_COUNT; ++core) {
            AscendC::hardwareCore = core;
            timers[core].Init(reinterpret_cast<GM_ADDR>(8));
            Require(timers[core].AuditEnabled());
        }
        for (int core = 0; core < N_CORE_COUNT; ++core) {
            AscendC::hardwareCore = core;
            auto& timer = timers[core];
            timer.SetCommunicationWaveBudget(12);
            timer.SetWave(7);
            if (core % 3 == 2) {
                timer.AuditTransfer(0, 3, core, 11, core, 7);
                timer.AuditTransfer(1, 3, core, 11, core, 7);
            }
            timer.FlushDynamicCounts();
        }
        for (int order = 0; order < N_CORE_COUNT; ++order) {
            Tensor::FlushCore(reverse ? N_CORE_COUNT - 1 - order : order);
        }
        Require(words[COMM_AUDIT_ENABLE_OFFSET] == 1);
        Require(words[COMM_M_GROUPS_PER_WAVE_OFFSET] == 12);
        for (int core = 0; core < N_CORE_COUNT; ++core) {
            Require(words[COMM_AUDIT_COUNT_OFFSET + core * COMM_AUDIT_COUNT_STRIDE] == (core % 3 == 2 ? 2 : 0));
            if (core % 3 != 2) continue;
            const int record =
                COMM_AUDIT_RECORD_OFFSET + core * COMM_AUDIT_MAX_RECORDS_PER_CORE * COMM_AUDIT_RECORD_WORDS;
            Require(words[record] == 7);
            Require(words[record + 1] == ((int64_t(3) << 32) | core));
            Require(words[record + 2] == ((int64_t(11) << 48) | (int64_t(7) << 32) | core));
            Require(words[record + COMM_AUDIT_RECORD_WORDS + 1] == ((int64_t(1) << 56) | (int64_t(3) << 32) | core));
        }
    }
    Tensor::cacheEnabled = false;
    Tensor::caches.clear();
    AscendC::hardwareCore = 2;
    std::cout << "comm timer: packed-count loss reproduced; isolated scalar cache lines passed\n";
}

int main()
{
    using namespace AscendTimer;
    auto& words = AscendC::GlobalTensor<int64_t>::words;
    words[COMM_AUDIT_ENABLE_OFFSET] = 0x0101010101010101LL;
    AscendTimerDevice timer;
    timer.Init(reinterpret_cast<GM_ADDR>(8));
    timer.SetWave(17);
    timer.SetCommunicationWaveBudget(12);
    timer.StartCommunication(AIV_DISPATCH_COMM);
    timer.SetCommunicationRows(3, 512, 7);
    timer.AuditTransfer(0, 3, 513, 11, 8191, 7);
    timer.EndCommunication();
    Require(AscendC::barriers == 0);
    timer.StartCommunication(AIV_COMBINE_COMM);
    timer.SetCommunicationRows(3, 512, 0);
    timer.EndCommunication();
    timer.StartCommunication(AIV_COMBINE_COMM);
    timer.SetCommunicationRows(3, 512, 7);
    // Legacy nested tile timing still uses its existing completion barrier;
    // the outer communication interval must not clobber its active state.
    timer.SetSlice(3, 512, 7);
    Require(timer.StartTile(AIV_COMBINE, 0, static_cast<uint32_t>(-1), 7));
    timer.AuditTransfer(1, 3, 513, 11, 8191, 7);
    timer.EndTile();
    timer.EndCommunication();
    timer.FlushDynamicCounts();
    const int core = 2;
    Require(words[COMM_M_GROUPS_PER_WAVE_OFFSET] == 12);
    Require(words[COMM_AUDIT_COUNT_OFFSET + core * COMM_AUDIT_COUNT_STRIDE] == 2);
    for (int type : {AIV_DISPATCH_COMM, AIV_COMBINE_COMM}) {
        Require(words[N_TIMING_COUNTER + core * DYNAMIC_ITER_PER_CORE_ALIGN + type] == 1);
        const int metadata = TASK_METADATA_OFFSET + (core * DYNAMIC_TYPE_COUNT + type) * MAX_DYNAMIC_ITER * 2;
        Require(words[metadata] == ((int64_t(4) << 32) | 512));
        Require(words[metadata + 1] == 7);
        const int logical = FIXED_TIMING_COUNT + type * MAX_DYNAMIC_ITER;
        Require(words[WAVE_ID_OFFSET + core * WAVE_ID_PER_CORE_ALIGN + logical] == 17);
        const int timing = core * N_TIMING_COUNTER_PER_CORE_ALIGN + logical * 2;
        Require(words[timing + 1] > words[timing]);
    }
    const int record = COMM_AUDIT_RECORD_OFFSET + core * COMM_AUDIT_MAX_RECORDS_PER_CORE * COMM_AUDIT_RECORD_WORDS;
    Require(words[record] == 17);
    Require(words[record + 1] == ((int64_t(3) << 32) | 513));
    Require(words[record + 2] == ((int64_t(11) << 48) | (int64_t(7) << 32) | 8191));
    Require(words[record + COMM_AUDIT_RECORD_WORDS + 1] == ((int64_t(1) << 56) | (int64_t(3) << 32) | 513));
    Require(AscendC::barriers == 1);

    // Overflow preserves the true count but cannot spill into another core.
    words.clear();
    words[COMM_AUDIT_ENABLE_OFFSET] = 1;
    AscendTimerDevice overflow;
    overflow.Init(reinterpret_cast<GM_ADDR>(8));
    for (uint32_t i = 0; i <= COMM_AUDIT_MAX_RECORDS_PER_CORE; ++i) {
        overflow.AuditTransfer(0, 0, i, 0, i, 0);
    }
    overflow.FlushDynamicCounts();
    Require(words[COMM_AUDIT_COUNT_OFFSET + core * COMM_AUDIT_COUNT_STRIDE] == COMM_AUDIT_MAX_RECORDS_PER_CORE + 1);
    Require(words.rbegin()->first < record + COMM_AUDIT_MAX_RECORDS_PER_CORE * COMM_AUDIT_RECORD_WORDS);

    // A normal timing allocation never writes beyond the normal buffer.
    words.clear();
    AscendTimerDevice normal;
    normal.Init(reinterpret_cast<GM_ADDR>(8));
    normal.AuditTransfer(0, 0, 0, 0, 0, 0);
    normal.FlushDynamicCounts();
    Require(words.rbegin()->first < TOTAL_BUFFER_SIZE);
    AscendTimerDevice disabled;
    disabled.Init(nullptr);
    const auto before = words.size();
    disabled.StartCommunication(AIV_DISPATCH_COMM);
    disabled.AuditTransfer(0, 0, 0, 0, 0, 0);
    disabled.EndCommunication();
    disabled.FlushDynamicCounts();
    Require(words.size() == before);
    std::cout << "comm timer: exact wire, independent spans, zero rows, audit isolation passed\n";
    CheckScalarCacheLineIsolation();
}
