// Exercise the current producer/prologue loops and synchronous/asynchronous ownership.
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
constexpr unsigned M_VALUE = 0, N_VALUE = 1;
constexpr int g_coreType = TEST_CORE;
template <unsigned I> auto Get(const std::array<uint32_t, 2>& a) { return a[I]; }
std::vector<int> events;
namespace AscendTimer { enum { AIC_SHARED_GMM1, AIC_GMM1, AIC_GMM1_ISSUE }; }
struct AscendTimerTileScope { template <typename... T> AscendTimerTileScope(T...) {} };
namespace AscendC {
constexpr int AIC = 0, AIV = 1;
enum class HardEvent { FIX_S };
template <HardEvent> void SetFlag(int) { events.push_back(4); }
template <HardEvent> void WaitFlag(int) { events.push_back(5); }
void WriteGmByPassDCache(int32_t* flag, int32_t value) { *flag = value; events.push_back(6); }
}
struct GMMAddrInfo { void* timer = nullptr; bool gmm1AsyncCompletion; int32_t* gmmToEpilogueFlag; };
struct Config { uint32_t blockNum = 2, tileM = 256, tileN = 256; };
struct Scheduler {
    std::array<uint32_t, 2> GetBlockCoord(uint32_t i) { return {i / 2 * 256, i % 2 * 256}; }
    std::array<uint32_t, 2> GetBlockShape(std::array<uint32_t, 2>) { return {256,256}; }
};
template <class C> struct CatlassGmm1 {
    bool active = false;
    unsigned begins = 0, ends = 0;
    void UpdateProblem(const C&, const GMMAddrInfo&) {}
    void Begin() { if (!active) { ++begins; active = true; } }
    void End(bool = false) { if (active) { ++ends; active = false; } }
    template <typename... T> void Compute(T...) { assert(active); events.push_back(2); }
    template <typename... T> void Dequantize(T...) { assert(active); events.push_back(8); }
};
struct Gmm1AsyncCompletion {
    static void AcquireProducer(unsigned) { events.push_back(1); }
    static void PublishCompleted(unsigned) { events.push_back(3); }
};
template <bool Shared, typename... T> void WaitForGmm1InputReady(T...) { events.push_back(0); }
template <typename... T> void NotifyGmm1TileStatus(T...) { events.push_back(7); }
#include "gmm1_role_methods.inc"
int main() {
    for (bool async : {false, true}) {
        int32_t flag = 0, sequence = 0;
        GMMAddrInfo addr{nullptr, async, &flag};
        Config config;
        Scheduler scheduler;
        CatlassGmm1<Config> kernel;
        events.clear();
        for (unsigned slice = 0; slice < 2; ++slice) {
            RunGmm1ComputeTiles<false>(kernel, scheduler, addr, config, 0, 4, sequence, 0);
        }
        assert(kernel.begins == (async ? 1U : 2U));
        assert(kernel.ends == (async ? 0U : 2U));
        assert(kernel.active == async);
        if constexpr (TEST_CORE == 0) {
            const std::vector<int> task = async ? std::vector<int>{0,1,2,3,6} : std::vector<int>{0,2,4,5,6};
            std::vector<int> expected;
            for (unsigned i = 0; i < 4; ++i) expected.insert(expected.end(), task.begin(), task.end());
            assert(events == expected && flag == 4 && sequence == 4);
        } else {
            assert(events == std::vector<int>({8,8,8,8}) && sequence == 0);
        }
        kernel.End();
        assert(!kernel.active);
    }
}
