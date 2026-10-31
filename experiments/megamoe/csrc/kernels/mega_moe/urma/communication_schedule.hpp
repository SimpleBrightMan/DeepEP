// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "abi.hpp"

namespace MegaMoeUrma {
#if defined(__NPU_DEVICE__)
#define URMA_COMM_INLINE __aicore__ inline
#else
#define URMA_COMM_INLINE inline
#endif
URMA_COMM_INLINE constexpr uint32_t DispatchBatchLimit(uint32_t step, uint32_t remaining, bool packFinished)
{
    // Preserve the two startup deliveries even when the entire case fits them.
    const uint32_t cap = step < 2U ? 1U : packFinished ? remaining : step == 2U ? 2U : step == 3U ? 4U : 8U;
    return cap < remaining ? cap : remaining;
}

// After submitting batch `step`, prepare one NEXT batch's original CV quotas
// before quiet. The two startup rounds are already complete and never repeated.
URMA_COMM_INLINE constexpr uint32_t DispatchPackLimit(uint32_t step, uint32_t remainingCvs,
                                                     uint32_t nextPackRound, uint32_t totalPackRounds)
{
    if (nextPackRound >= totalPackRounds) return nextPackRound;
    const uint32_t count = DispatchBatchLimit(step + 1U, remainingCvs, false);
    const uint32_t left = totalPackRounds - nextPackRound;
    return nextPackRound + (count < left ? count : left);
}

URMA_COMM_INLINE constexpr bool AdjacentChunks(const Chunk& first, uint64_t rows, uint32_t count, const Chunk& next)
{
    return next.peer == first.peer && next.cv == first.cv + count &&
           next.localRow == first.localRow + rows && next.remoteRow == first.remoteRow + rows;
}
#undef URMA_COMM_INLINE
} // namespace MegaMoeUrma
