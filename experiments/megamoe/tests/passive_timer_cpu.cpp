// 本桩刻意不提供任何 barrier/flag API；实际 timer 头文件必须只依赖时钟和私有 GM 写。
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#define __gm__
#define DEEPEP_MEGAMOE_TIMER
using GM_ADDR = unsigned char*;
static bool isAiv = false;
static int blockId = 0;
#define ASCEND_IS_AIV (isAiv)
namespace AscendC {
inline int GetBlockIdx() { return blockId; }
inline int GetTaskRation() { return isAiv ? 2 : 1; }
inline int GetSubBlockIdx() { return 0; }
inline int64_t GetSystemCycle() { static int64_t cycle = 100; return ++cycle; }
template<class T> struct GlobalTensor {
    T* ptr;
    void SetGlobalBuffer(T* p) { ptr = p; }
    T& operator()(int index) { return ptr[index]; }
};
}
#include "../csrc/kernels/mega_moe/common/ascend_timer_v2_device.hpp"
int main() {
    std::vector<int64_t> data(AscendTimer::TOTAL_BUFFER_SIZE, 0);
    AscendTimerDevice timer;
    timer.Init(nullptr);
    assert(!timer.Enabled());
    timer.Start(AscendTimer::KERNEL_TIMING);
    assert(!timer.StartTile(AscendTimer::AIC_GMM1_ISSUE,0,0,96));
    timer.Init(reinterpret_cast<GM_ADDR>(data.data()));
    timer.Start(AscendTimer::KERNEL_TIMING);
    timer.Start(AscendTimer::AIV_QUANT);
    timer.End(AscendTimer::AIV_QUANT);
    timer.End(AscendTimer::KERNEL_TIMING);
    assert(data[0] < data[AscendTimer::AIV_QUANT*2]);
    assert(data[1] > data[AscendTimer::AIV_QUANT*2+1]);
    timer.SetWave(4);timer.SetSlice(2,256,96);
    assert(timer.StartTile(AscendTimer::AIC_GMM1_ISSUE,0,128,96,128));
    assert(timer.BeginWait(AscendTimer::AIC_GMM1_WAIT));
    assert(!timer.BeginWait(AscendTimer::AIC_GMM1_WAIT));
    timer.EndWait();timer.EndTile();timer.EndTile();
    int base=(AscendTimer::FIXED_TIMING_COUNT+AscendTimer::AIC_GMM1_ISSUE*AscendTimer::MAX_DYNAMIC_ITER)*2;
    assert(data[base+1]>data[base]);
    timer.StartSpan(AscendTimer::AIV_GMM1_PROLOGUE);
    timer.BeginWait(AscendTimer::AIC_GMM1_WAIT);timer.EndWait();
    timer.EndSpan(AscendTimer::AIV_GMM1_PROLOGUE);
    int boundary=timer.BeginBoundary(AscendTimer::AIC_GMM1_BEGIN);
    timer.EndBoundary(AscendTimer::AIC_GMM1_BEGIN,boundary);
    timer.FlushDynamicCounts();
    assert(data[AscendTimer::N_TIMING_COUNTER+AscendTimer::AIC_GMM1_ISSUE]==1);
    assert(data[AscendTimer::N_TIMING_COUNTER+AscendTimer::AIV_GMM1_PROLOGUE]==2);
    timer.Init(reinterpret_cast<GM_ADDR>(data.data()));
    for(int i=0;i<AscendTimer::MAX_DYNAMIC_ITER+1;++i) {
        int index=timer.BeginBoundary(AscendTimer::AIC_GMM1_BEGIN);
        timer.EndBoundary(AscendTimer::AIC_GMM1_BEGIN,index);
    }
    timer.FlushDynamicCounts();
    assert(data[AscendTimer::N_TIMING_COUNTER+AscendTimer::AIC_GMM1_ISSUE]==0);
    assert(data[AscendTimer::N_TIMING_COUNTER+AscendTimer::AIC_GMM1_BEGIN]==AscendTimer::MAX_DYNAMIC_ITER+1);
    isAiv=true;
    timer.Init(reinterpret_cast<GM_ADDR>(data.data()));
    timer.Start(AscendTimer::KERNEL_TIMING);timer.End(AscendTimer::KERNEL_TIMING);
    assert(data[AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN]>0);
    isAiv=false;blockId=AscendTimer::DYNAMIC_TRACE_GROUPS;
    timer.Init(reinterpret_cast<GM_ADDR>(data.data()));
    assert(timer.Enabled() && !timer.DynamicEnabled());
    timer.SetSlice(0,0,96);
    assert(!timer.StartTile(AscendTimer::AIC_GMM1_ISSUE,0,0,96));
    assert(timer.BeginBoundary(AscendTimer::AIC_GMM1_BEGIN)==-1);
    timer.StartSpan(AscendTimer::AIV_DISPATCH);timer.EndSpan(AscendTimer::AIV_DISPATCH);
    timer.Start(AscendTimer::KERNEL_TIMING);timer.End(AscendTimer::KERNEL_TIMING);
    timer.FlushDynamicCounts();
    const int core=blockId*3;
    assert(data[core*AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN]>0);
    assert(data[AscendTimer::N_TIMING_COUNTER+core*AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN+AscendTimer::AIC_GMM1_ISSUE]==0);
}
