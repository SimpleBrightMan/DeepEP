#pragma once
#include <cstdint>

// [start,end] 协议沿用 MegaMoE timer；C API 实现位于单独的 device header。
namespace AscendTimer {
// v2 为无新增同步的 Scalar 观察协议；事件含义/数量改变，禁止将 v1 raw 按 v2 解码。
static constexpr int TIMER_PROTOCOL_VERSION = 2;
enum FixedTiming {
    KERNEL_TIMING_IDX = 0,
    AIV_RESET,
    AIV_INPUT_CROSS_RANK_SYNC,
    AIV_ROUTE_MASK,
    AIV_COUNT_WAIT,
    AIC_METADATA_WAIT,
    AIV_FINAL_PREP,
    AIV_FINAL_CROSS_RANK_WAIT,
    AIC_END_SYNC,
    FIXED_TIMING_COUNT
};
enum DynamicTimingType {
    AIV_DISPATCH_ISSUE,
    AIV_SWIGLU_ISSUE,
    AIV_COMBINE_WAIT,
    AIV_COMBINE_ISSUE,
    AIV_COMBINE_SYNC,
    AIV_UNPERMUTE_ISSUE,
    AIV_GMM1_PROLOGUE_ISSUE,
    AIV_GMM2_PROLOGUE_ISSUE,
    AIC_GMM1_WAIT,
    AIC_GMM1_ISSUE,
    AIC_GMM2_WAIT,
    AIC_GMM2_ISSUE,
    AIC_RING_WAIT,
    AIC_SHARED_GMM1_ISSUE,
    AIC_SHARED_GMM2_ISSUE,
    AIV_DISPATCH_SYNC,
    AIV_SWIGLU_SYNC,
    DYNAMIC_TYPE_COUNT_ENUM
};
// EP8/BS4096 均衡路由：每 AIC 72 个 GMM1 tile、84 个 GMM2 tile。
// 不记录逐 K-block L1 等待及非 Scalar 管线的等待发射，不改变原有硬件依赖。
// 8192 为协程 step/输出 chunk 和不均衡/大 BS 留余量，不按均衡 case 缩小容量。
// 超容量仍写实际 count，解析器必须拒绝，不能将截断的图当完整流水。
static constexpr int MAX_DYNAMIC_ITER = 8192;
static constexpr int N_CORE_COUNT = 96;
static constexpr int COUNTERS_PER_CORE =
    ((FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER) * 2 + 15) / 16 * 16;
static constexpr int N_TIMING_COUNTER = COUNTERS_PER_CORE * N_CORE_COUNT;
static constexpr int DYNAMIC_ITER_PER_CORE = (DYNAMIC_TYPE_COUNT_ENUM + 15) / 16 * 16;
static constexpr int TOTAL_BUFFER_SIZE = N_TIMING_COUNTER + DYNAMIC_ITER_PER_CORE * N_CORE_COUNT;
} // namespace AscendTimer
