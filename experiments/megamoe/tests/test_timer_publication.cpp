// Copyright (c) 2026, SimpleBright_Man
// Exercise the actual timer writer with private, write-back scalar caches.
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>

#define __aicore__
#define __gm__
#define ASCEND_IS_AIV (simulation::core % 3 != 0)
#define DEEPEP_MEGAMOE_TIMER 1
using GM_ADDR = uint8_t*;
constexpr int PIPE_ALL = 0;

namespace simulation {
constexpr int words = 16;
int core = 0;
int flushes = 0;
std::map<uintptr_t, int64_t> memory;
using Line = std::array<int64_t, words>;
std::array<std::map<uintptr_t, Line>, 96> caches;

int64_t& Access(uintptr_t address)
{
    const uintptr_t base = address / 128 * 128;
    auto& cache = caches[core];
    auto it = cache.find(base);
    if (it == cache.end()) {
        Line line{};
        for (int i = 0; i < words; ++i) line[i] = memory[base + i * 8];
        it = cache.emplace(base, line).first;
    }
    return it->second[(address - base) / 8];
}

void Flush(uintptr_t address)
{
    const uintptr_t base = address / 128 * 128;
    auto& cache = caches[core];
    const auto it = cache.find(base);
    assert(it != cache.end());
    for (int i = 0; i < words; ++i) memory[base + i * 8] = it->second[i];
    cache.erase(it);
    ++flushes;
}
}  // namespace simulation

namespace AscendC {
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
template <typename T>
class GlobalTensor {
public:
    void SetGlobalBuffer(T* pointer)
    {
        address = reinterpret_cast<uintptr_t>(pointer);
    }
    int64_t& operator()(int32_t index)
    {
        return simulation::Access(address + index * sizeof(T));
    }
    uintptr_t address = 0;
};
template <typename T, CacheLine, DcciDst>
void DataCacheCleanAndInvalid(GlobalTensor<T> tensor)
{
    simulation::Flush(tensor.address);
}
inline int GetBlockIdx()
{
    return simulation::core % 3 == 0 ? simulation::core / 3 : simulation::core / 3 * 2 + simulation::core % 3 - 1;
}
inline int GetTaskRation()
{
    return simulation::core % 3 == 0 ? 1 : 2;
}
inline int GetSubBlockIdx()
{
    return simulation::core % 3 - 1;
}
inline int64_t GetSystemCycle()
{
    return 100;
}
template <int>
void PipeBarrier()
{
}
}  // namespace AscendC

#include "kernels/mega_moe/common/ascend_timer_v2_device.hpp"

int main()
{
    // Negative control: packed per-core words lose earlier updates even when
    // each core flushes. The second cache still contains stale neighbour data.
    constexpr uintptr_t oldBase = 0x1000;
    simulation::core = 0;
    simulation::Access(oldBase) = 1;
    simulation::core = 1;
    simulation::Access(oldBase + 8) = 1;
    simulation::core = 0;
    simulation::Flush(oldBase);
    simulation::core = 1;
    simulation::Flush(oldBase + 8);
    assert(simulation::memory[oldBase] == 0);

    // Actual storage makes pointer arithmetic in the production writer valid.
    auto storage = std::make_unique<uint8_t[]>(AscendTimer::TOTAL_BUFFER_SIZE * sizeof(int64_t) + 128);
    const uintptr_t base = (reinterpret_cast<uintptr_t>(storage.get()) + 127) / 128 * 128;
    const auto address = [base](int offset, int core) {
        return base + static_cast<uintptr_t>(offset + core * AscendTimer::TIMER_STAMP_STRIDE) * 8;
    };
    std::array<AscendTimerDevice, AscendTimer::N_CORE_COUNT> timers;
    for (int round = 0; round < 2; ++round) {
        for (int core = 0; core < AscendTimer::N_CORE_COUNT; ++core) {
            simulation::core = core;
            timers[core].Init(reinterpret_cast<GM_ADDR>(base));
            assert(simulation::memory[address(AscendTimer::TIMER_SCHEMA_OFFSET, core)] ==
                   AscendTimer::TIMER_SCHEMA_VERSION);
            assert(simulation::memory[address(AscendTimer::TIMER_COMPLETION_OFFSET, core)] == 0);
        }
        for (int core = AscendTimer::N_CORE_COUNT - 1; core >= 0; --core) {
            simulation::core = core;
            timers[core].FlushDynamicCounts();
        }
        for (int core = 0; core < AscendTimer::N_CORE_COUNT; ++core) {
            assert(simulation::memory[address(AscendTimer::TIMER_SCHEMA_OFFSET, core)] ==
                   AscendTimer::TIMER_SCHEMA_VERSION);
            assert(simulation::memory[address(AscendTimer::TIMER_COMPLETION_OFFSET, core)] == 1);
        }
    }
    const int before = simulation::flushes;
    for (auto& timer : timers) {
        timer.Init(nullptr);
        timer.FlushDynamicCounts();
    }
    assert(simulation::flushes == before);
    std::cout << "96 private cache writers, repeated publication, disabled timer passed\n";
}
