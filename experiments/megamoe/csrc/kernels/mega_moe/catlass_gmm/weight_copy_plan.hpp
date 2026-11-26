#pragma once
#include <cstdint>

namespace MegaMoeImpl::GmmKernel {
// FP8 NZ物理顺序[K/32,N/16,16,32]；每个K32面板只读取当前N列。
struct Fp8NzCopyPlan {
    uint64_t sourceOffset;
    uint32_t blocks;
    uint32_t blockBytes;
    uint32_t sourceGapBytes;
};
__aicore__ constexpr inline Fp8NzCopyPlan MakeFp8NzCopyPlan(
    uint32_t fullN, uint32_t startN, uint32_t startK, uint32_t tileK, uint32_t tileN)
{
    return {static_cast<uint64_t>(startK / 32U) * fullN * 32U + static_cast<uint64_t>(startN) * 32U,
            tileK / 32U, tileN * 32U, (fullN - tileN) * 32U};
}
__aicore__ constexpr inline uint64_t WeightFixScale(bool packedFp4)
{
    return packedFp4 ? 64U : 1U;
}
}  // namespace MegaMoeImpl::GmmKernel
