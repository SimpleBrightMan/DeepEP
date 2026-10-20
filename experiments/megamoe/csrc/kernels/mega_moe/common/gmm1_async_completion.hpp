// Copyright (c) 2026, Lu Lu
#ifndef MEGA_MOE_GMM1_ASYNC_COMPLETION_HPP
#define MEGA_MOE_GMM1_ASYNC_COMPLETION_HPP
#include <c_api/asc_simd.h>

namespace MegaMoeImpl {
// AIC<->AIV1专用两槽完成通道，不与AIV0权重通道(6..9)共享。
// GM输出仍按专家/tile独立寻址；credit保护完成事件槽的复用，不改变输出布局。
struct Gmm1AsyncCompletion {
    static constexpr unsigned STAGES = 2;
    static constexpr unsigned FULL = 12;
    static constexpr unsigned EMPTY = 14;
    static constexpr unsigned AIV1 = 16;

    __aicore__ static inline void InitConsumer()
    {
        for (unsigned i = 0; i < STAGES; ++i) asc_sync_intra_arrive(PIPE_MTE3, AIV1 | (EMPTY + i));
    }
    __aicore__ static inline void AcquireProducer(unsigned sequence)
    {
        // 等待排在FIX队列，不阻塞Scalar；必须在当前tile的FIX写出之前。
        asc_sync_intra_wait(PIPE_FIX, AIV1 | (EMPTY + sequence % STAGES));
    }
    __aicore__ static inline void PublishCompleted(unsigned sequence)
    {
        // FIX队列顺序保证gate/up两个投影均已写GM，才发出full事件。
        asc_sync_intra_arrive(PIPE_FIX, AIV1 | (FULL + sequence % STAGES));
    }
    __aicore__ static inline void WaitConsumer(unsigned sequence)
    {
        // AIV等待时flag使用本地低4位；GM->UB读取必须位于这个MTE2依赖之后。
        asc_sync_intra_wait(PIPE_MTE2, FULL + sequence % STAGES);
    }
    __aicore__ static inline void WaitVectorConsumer(unsigned sequence)
    {
        // CV数据已经由FIX写UB，不再经过MTE2；真正的FIX full直接约束V读输入。
        asc_sync_intra_wait(PIPE_V, FULL + sequence % STAGES);
    }
    __aicore__ static inline void ReleaseConsumer(unsigned sequence)
    {
        asc_sync_intra_arrive(PIPE_MTE3, AIV1 | (EMPTY + sequence % STAGES));
    }
    __aicore__ static inline void DrainProducer()
    {
        // 消耗两槽最终credit，包含从未使用的槽；不得遗留到下次kernel调用。
        for (unsigned i = 0; i < STAGES; ++i) asc_sync_intra_wait(PIPE_FIX, AIV1 | (EMPTY + i));
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
    }
};
} // namespace MegaMoeImpl
#endif
