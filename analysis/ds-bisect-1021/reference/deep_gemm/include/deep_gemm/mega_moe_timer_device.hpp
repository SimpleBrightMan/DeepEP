#pragma once
#include <c_api/asc_simd.h>
#include <deep_gemm/mega_moe_timer.hpp>

namespace deep_gemm::mega_moe {
using namespace AscendTimer;

// 只读 Scalar 时钟并写本核独占槽，不发出任何管线同步或 barrier。
// 已有 Scalar 等待记录观察区间；异步任务仅记录 ISSUE，不声称硬件已完成。
template <bool Enabled>
struct PipelineTimer {
    __gm__ int64_t* data;
    uint32_t core;
    uint32_t counts[DYNAMIC_TYPE_COUNT_ENUM]{};

    __aicore__ PipelineTimer(__gm__ int64_t* ptr, uint32_t lane) : data(ptr), core(lane) {}
    __aicore__ bool enabled() const { return Enabled && data != nullptr; }
    __aicore__ void write(uint32_t slot, bool end) {
        if constexpr (Enabled) {
            if (data != nullptr)
                asc_store_dev(data + core * COUNTERS_PER_CORE + slot * 2 + end,
                              static_cast<int64_t>(asc_get_system_cycle()));
        }
    }
    __aicore__ void start(FixedTiming type) { write(type, false); }
    __aicore__ void end(FixedTiming type) { write(type, true); }
    __aicore__ uint32_t begin(DynamicTimingType type) {
        if constexpr (Enabled) {
            if (data != nullptr) {
                const uint32_t index = counts[type]++;
                if (index < MAX_DYNAMIC_ITER)
                    write(FIXED_TIMING_COUNT + type * MAX_DYNAMIC_ITER + index, false);
                return index;
            }
        }
        return 0;
    }
    __aicore__ void finish(DynamicTimingType type, uint32_t index) {
        if constexpr (Enabled) {
            if (data != nullptr) {
                if (index < MAX_DYNAMIC_ITER)
                    write(FIXED_TIMING_COUNT + type * MAX_DYNAMIC_ITER + index, true);
            }
        }
    }
    __aicore__ void flush() {
        if constexpr (Enabled) {
            if (data != nullptr) {
                for (uint32_t i = 0; i < DYNAMIC_TYPE_COUNT_ENUM; ++i)
                    asc_store_dev(data + N_TIMING_COUNTER + core * DYNAMIC_ITER_PER_CORE + i,
                                  static_cast<int64_t>(counts[i]));
                // 主机在 kernel/stream 完成后读取；不为 timer 新增设备侧 drain。
            }
        }
    }
};

// 关闭态连计数数组都不分配，计时辅助的寄存器/栈初始化不进入性能特化。
template <>
struct PipelineTimer<false> {
    __aicore__ PipelineTimer(__gm__ int64_t*, uint32_t) {}
    __aicore__ bool enabled() const { return false; }
    __aicore__ void start(FixedTiming) {}
    __aicore__ void end(FixedTiming) {}
    __aicore__ uint32_t begin(DynamicTimingType) { return 0; }
    __aicore__ void finish(DynamicTimingType, uint32_t) {}
    __aicore__ void flush() {}
};

template <bool Enabled>
struct TimerScope {
    PipelineTimer<Enabled>& timer;
    DynamicTimingType type;
    uint32_t index;
    __aicore__ TimerScope(PipelineTimer<Enabled>& timer, DynamicTimingType type)
        : timer(timer), type(type), index(timer.begin(type)) {}
    __aicore__ ~TimerScope() { timer.finish(type, index); }
};
} // namespace deep_gemm::mega_moe
