// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#ifndef MEGA_MOE_A8W4_K_PLAN_H
#define MEGA_MOE_A8W4_K_PLAN_H
#include <cstdint>

namespace MegaMoeImpl::GmmKernel {
// 相同资源图供GMM1/GMM2交接；四个B槽容纳两个K512/N128配对窗口。
struct A8W4BufferPlan {
    static constexpr uint32_t AStages = 3, BStages = 4, ScaleStages = 2;
    static constexpr uint32_t ABytes = 64U * 1024U, BBytes = 64U * 1024U;
    static constexpr uint32_t ScaleBytes = 16U * 1024U;
    static constexpr uint32_t ABase = 0, BBase = AStages * ABytes;
    static constexpr uint32_t ScaleABase = BBase + BStages * BBytes;
    static constexpr uint32_t ScaleBBase = ScaleABase + ScaleStages * ScaleBytes;
    static constexpr uint16_t BFull = 4, BEmpty = 8;
    static_assert(ScaleBBase + ScaleStages * ScaleBytes == 512U * 1024U);
    static_assert(BFull + BStages <= BEmpty && BEmpty + BStages <= 12);
};
// Shared by AIC and AIV0: B must use exactly the same K window on both cores.
struct A8W4KPlan {
    uint32_t ka;
    uint32_t kb;
    uint32_t bColumns;
};
__aicore__ constexpr inline A8W4KPlan MakeA8W4KPlan(uint32_t m, uint32_t n, bool allowDynamicB)
{
    (void)allowDynamicB;
    (void)n;
    // 每个B槽最多64KiB，不能把K512*N256误当成一槽。
    // AIC/AIV0都使用此计划，确保K外层、N半块内层的生产消费顺序一致。
    if (m <= 128U) {
        return {512U, 512U, 128U};
    }
    return {256U, 256U, 256U};
}
}  // namespace MegaMoeImpl::GmmKernel
#endif
