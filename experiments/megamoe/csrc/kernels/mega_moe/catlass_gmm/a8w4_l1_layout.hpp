// Copyright (c) 2026, Lu Lu
#ifndef MEGA_MOE_A8W4_L1_LAYOUT_H
#define MEGA_MOE_A8W4_L1_LAYOUT_H
#include <cstdint>

namespace MegaMoeImpl::GmmKernel {
// AIC与AIV0必须共享物理地址和跨核flag映射。GMM2将闲置的A容量换给四槽B。
template <uint32_t BStages>
struct A8W4L1Layout {
    static_assert(BStages == 2 || BStages == 4);
    static constexpr uint32_t A_BYTES = (BStages == 4 ? 64U : 128U) * 1024U;
    static constexpr uint32_t FULL_FLAG = 6;
    static constexpr uint32_t EMPTY_FLAG = BStages == 4 ? 10 : 8;
    static constexpr uint32_t B(uint32_t i) { return (BStages == 4 ? i * 64U : i * 384U) * 1024U; }
    static constexpr uint32_t A(uint32_t i) { return (BStages == 4 ? 256U + i * 64U : 128U + i * 128U) * 1024U; }
    static constexpr uint32_t SA(uint32_t i) { return (BStages == 4 ? 384U + i * 32U : 64U + i * 384U) * 1024U; }
    static constexpr uint32_t SB(uint32_t i) { return (BStages == 4 ? 448U + i * 32U : 96U + i * 384U) * 1024U; }
};
}
#endif
