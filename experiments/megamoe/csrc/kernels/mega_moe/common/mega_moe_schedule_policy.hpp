// Copyright (c) 2026, Lu Lu
#pragma once
#include "mega_moe_constants.hpp"

#if defined(__CCE_AICORE__) || defined(__NPU_DEVICE__)
#define MEGAMOE_SCHEDULE_INLINE __host__ __aicore__ constexpr inline
#else
#define MEGAMOE_SCHEDULE_INLINE constexpr inline
#endif

namespace MegaMoeImpl {
namespace schedule_detail {
MEGAMOE_SCHEDULE_INLINE uint64_t CeilDiv(uint64_t n, uint64_t d)
{
    return (n + d - 1U) / d;
}
}
// 参数均为已校验的本次launch配置；outputDim是2I，不能传容量或某个专家M。
MEGAMOE_SCHEDULE_INLINE uint32_t CalcWaveMGroupsPerWave(
    bool isA8W4, uint32_t bs, uint32_t outputDim, uint32_t hidden, uint32_t aic)
{
    const uint64_t gmm2Tiles = schedule_detail::CeilDiv(hidden, L1_TILE_N);
    if (isA8W4) {
        const uint64_t gmm1Tasks = schedule_detail::CeilDiv(outputDim / ACTIVATION_N_HALF, GMM1_FUSED_TILE_N);
        const uint64_t twoRounds = schedule_detail::CeilDiv(uint64_t(aic) * 2U, gmm1Tasks);
        if (bs <= W4A8_SMALL_BATCH_MAX_TOKENS) return static_cast<uint32_t>(twoRounds * 2U);
        const uint64_t oneGmm2Round = schedule_detail::CeilDiv(aic, gmm2Tiles);
        return static_cast<uint32_t>(twoRounds > oneGmm2Round ? twoRounds : oneGmm2Round);
    }
    const uint64_t gmm1Tiles = schedule_detail::CeilDiv(outputDim, L1_TILE_N);
    const uint64_t fourRounds = schedule_detail::CeilDiv(uint64_t(aic) * 4U, gmm1Tiles);
    const uint64_t oneGmm2Round = schedule_detail::CeilDiv(aic, gmm2Tiles);
    return static_cast<uint32_t>(fourRounds > oneGmm2Round ? fourRounds : oneGmm2Round);
}

MEGAMOE_SCHEDULE_INLINE bool UseGmm2WaveLag(uint32_t bs)
{
    return bs >= W4A8_GMM2_LAG_MIN_TOKENS;
}
} // namespace MegaMoeImpl
#undef MEGAMOE_SCHEDULE_INLINE
