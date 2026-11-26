// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include <cstdint>
#if defined(__CCE_AICORE__) || defined(__NPU_DEVICE__)
#define MEGAMOE_PACKED_INLINE __aicore__ constexpr inline
#else
#define MEGAMOE_PACKED_INLINE constexpr inline
#endif
namespace MegaMoeImpl::GmmKernel {
// [N256 block,K512 window,N128 half,K32,N128,16B]，尾K窗口不增加GM padding。
MEGAMOE_PACKED_INLINE uint32_t PackedWeightWindowK(uint32_t fullK,uint32_t k)
{
    const uint32_t left=fullK-k/512U*512U;
    return left<512U?left:512U;
}
MEGAMOE_PACKED_INLINE uint64_t PackedWeightByteOffset(uint32_t fullK,uint32_t n,uint32_t k)
{
    return uint64_t(n/256U)*fullK*128U + uint64_t(k/512U)*65536U +
           uint64_t(n%256U/128U)*PackedWeightWindowK(fullK,k)*64U + uint64_t(k%512U)*64U;
}
MEGAMOE_PACKED_INLINE uint64_t PackedWeightLogicalOffset(uint32_t fullK,uint32_t n,uint32_t k)
{
    // GlobalTensor<fp4x2>以4bit元素寻址，内部再除2，不能直接传字节偏移。
    return 2U*PackedWeightByteOffset(fullK,n,k);
}
}
#undef MEGAMOE_PACKED_INLINE
