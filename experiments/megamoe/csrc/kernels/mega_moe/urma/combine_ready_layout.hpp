// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include <cstdint>

namespace MegaMoeUrma {
#if defined(__NPU_DEVICE__)
#define URMA_READY_INLINE __aicore__ inline
#else
#define URMA_READY_INLINE inline
#endif

// Retain the existing per-source maxPeerChunks * 128-byte allocation. Only
// ceil(maxPeerChunks/64) cache lines are used. Each bit belongs to one CV.
URMA_READY_INLINE constexpr uint64_t CombineReadyWordIndex(uint32_t maxPeerChunks, uint32_t source, uint32_t cv)
{
    return static_cast<uint64_t>(source) * maxPeerChunks + cv / 64U;
}

URMA_READY_INLINE constexpr bool CombineRangeFitsWord(uint32_t firstCv, uint32_t count)
{
    return count != 0U && count <= 64U - firstCv % 64U;
}

URMA_READY_INLINE constexpr uint64_t CombineReadyMask(uint32_t firstCv, uint32_t count)
{
    if (!CombineRangeFitsWord(firstCv, count)) return 0U;
    const uint64_t low = count == 64U ? ~uint64_t{0} : (uint64_t{1} << count) - 1U;
    return low << (firstCv % 64U);
}

URMA_READY_INLINE constexpr bool CombineReadyBit(uint64_t word, uint32_t cv)
{
    return (word & (uint64_t{1} << (cv % 64U))) != 0U;
}

#undef URMA_READY_INLINE
} // namespace MegaMoeUrma
