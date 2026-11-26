// Copyright (c) 2026, Lu Lu
#ifndef MEGA_MOE_GMM2_AIV_COMPLETION_HPP
#define MEGA_MOE_GMM2_AIV_COMPLETION_HPP
#include "kernel_operator.h"
#include "../catlass_gmm/a8w4_k_plan.hpp"

namespace MegaMoeImpl {
// 专供 routed W4A8 Wave：AIC FIX -> 配对 AIV1 Scalar -> 原 slice ready 计数。
// 0/1 不与 AIV0 权重通道 4..11、AIV1 GMM1 通道 12..15 共用。
// 一份 credit 限制未消费的 slice 完成事件；等待排在 FIX 队列，AIC Scalar 不逐 slice 排空。
struct Gmm2AivCompletion {
    static constexpr unsigned MODE = 4U;
    static constexpr unsigned FULL = 0U;
    static constexpr unsigned EMPTY = 1U;
    static constexpr unsigned AIV1 = 16U;
    static_assert(FULL != EMPTY && EMPTY < GmmKernel::A8W4BufferPlan::BFull);

    __aicore__ static inline void InitConsumer()
    {
        if constexpr (g_coreType == AscendC::AIV) {
            if (AscendC::GetSubBlockIdx() == 1U) AscendC::CrossCoreSetFlag<MODE, PIPE_S>(EMPTY);
        }
    }
    __aicore__ static inline void PublishCompleted(__gm__ int32_t* issued, int32_t sequence)
    {
        if constexpr (g_coreType == AscendC::AIC) {
            // 只保护事件槽；此前全部 GM 写回均排在同一 FIX 队列中。
            AscendC::CrossCoreWaitFlag<MODE, PIPE_FIX>(AIV1 | EMPTY);
            AscendC::CrossCoreSetFlag<MODE, PIPE_FIX>(AIV1 | FULL);
            // Scalar 只通知本核下一个有效 slice 的写回/事件已提交；它不是数据完成标记。
            AscendC::WriteGmByPassDCache(issued, sequence);
        }
    }
    __aicore__ static inline bool IsIssued(__gm__ int32_t* issued, int32_t sequence)
    {
        return AscendC::ReadGmByPassDCache(issued) >= sequence;
    }
    __aicore__ static inline void WaitConsumer()
    {
        if constexpr (g_coreType == AscendC::AIV) {
            // 仅在 IsIssued 成功后进入；必须等真实 FIX 完成才由 Scalar 发布 slice-ready。
            if (AscendC::GetSubBlockIdx() == 1U) AscendC::CrossCoreWaitFlag<MODE, PIPE_S>(FULL);
        }
    }
    __aicore__ static inline void ReleaseConsumer()
    {
        if constexpr (g_coreType == AscendC::AIV) {
            if (AscendC::GetSubBlockIdx() == 1U) AscendC::CrossCoreSetFlag<MODE, PIPE_S>(EMPTY);
        }
    }
    __aicore__ static inline void DrainProducer()
    {
        if constexpr (g_coreType == AscendC::AIC) {
            // 仅在整个 routed 流水退出时回收最后一份 credit；零 Wave 也消耗初始化 credit。
            AscendC::CrossCoreWaitFlag<MODE, PIPE_FIX>(AIV1 | EMPTY);
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
        }
    }
};
}  // namespace MegaMoeImpl
#endif
