// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#ifndef MEGA_MOE_A8W4_K_PLAN_H
#define MEGA_MOE_A8W4_K_PLAN_H
#include <cstdint>

namespace MegaMoeImpl::GmmKernel {
// Shared by AIC and AIV0: B must use exactly the same K window on both cores.
struct A8W4KPlan {
    uint32_t ka;
    uint32_t kb;
};
__aicore__ constexpr inline A8W4KPlan MakeA8W4KPlan(uint32_t m, uint32_t n, bool allowDynamicB,
                                                bool prefetchA = false)
{
    const uint32_t kb = allowDynamicB && m <= 256U && n <= 128U ? 512U : 256U;
    const uint32_t alignedM = (m + 15U) / 16U * 16U;
    if (alignedM == 0U || (m >= n && !prefetchA)) {
        return {kb, kb};
    }
    // GMM2利用已有128KiB A槽预装两个B窗口，不改变B生产序列或L0 K128粒度。
    const uint32_t balancedDepth = m >= n ? 2U : (n + 2U * alignedM - 1U) / (2U * alignedM);
    const uint32_t capacityDepth = (128U * 1024U) / (alignedM * kb);
    const uint32_t depth = balancedDepth < capacityDepth ? balancedDepth : capacityDepth;
    return {depth * kb, kb};
}
}  // namespace MegaMoeImpl::GmmKernel
#endif
