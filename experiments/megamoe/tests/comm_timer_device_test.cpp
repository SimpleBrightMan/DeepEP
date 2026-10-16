// Copyright (c) 2026, SimpleBright_Man
// CPU execution of the actual timer writer, with sparse GM and clock mocks.
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>

#define __aicore__
#define __gm__
#define ASCEND_IS_AIV (true)
#define DEEPEP_MEGAMOE_TIMER
using GM_ADDR = uint8_t*;
constexpr int PIPE_ALL = 0;
namespace AscendC {
static int barriers = 0;
static int64_t cycle = 100;
inline int GetBlockIdx()
{
    return 0;
}
inline int GetTaskRation()
{
    return 2;
}
inline int GetSubBlockIdx()
{
    return 1;
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
    static std::map<int64_t, T> words;
    void SetGlobalBuffer(T*) {}
    T GetValue(int64_t index) const
    {
        return words[index];
    }
    T& operator()(int64_t index)
    {
        return words[index];
    }
};
template <typename T>
std::map<int64_t, T> GlobalTensor<T>::words;
}  // namespace AscendC

#include "common/ascend_timer_v2_device.hpp"

static void Require(bool ok)
{
    if (!ok) std::abort();
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
    Require(words[COMM_AUDIT_COUNT_OFFSET + core] == 2);
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
    Require(words[COMM_AUDIT_COUNT_OFFSET + core] == COMM_AUDIT_MAX_RECORDS_PER_CORE + 1);
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
}
