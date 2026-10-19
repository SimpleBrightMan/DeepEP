// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"

namespace MegaMoeUrma {

// Pure host/device scheduling arithmetic. Fragments retain the peer/CV row
// order; an expert boundary always starts a new fragment, not a new Wave.
constexpr uint32_t kComputeCores = kAic;
constexpr uint32_t kComputeTileM = 256U;
constexpr uint32_t kComputeTileN = 256U;

#if defined(__NPU_DEVICE__)
#define MEGAMOE_URMA_COMPUTE_INLINE __aicore__ inline
#else
#define MEGAMOE_URMA_COMPUTE_INLINE inline
#endif

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t ComputeTileCount(uint32_t rows, uint32_t columns)
{
    return ((rows + kComputeTileM - 1U) / kComputeTileM) * ((columns + kComputeTileN - 1U) / kComputeTileN);
}

// All stage task IDs are global prefix sums. Sparse experts cannot repeatedly
// give the same physical core their first tile. AIC and its AIV0 call this with
// identical inputs; the independent AIV1 activation team uses workerCount=13.
MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t FirstOwnedComputeTile(uint64_t tileBegin, uint32_t worker,
                                                                     uint32_t workerCount)
{
    const uint32_t startOwner = tileBegin % workerCount;
    return worker >= startOwner ? worker - startOwner : worker + workerCount - startOwner;
}

#undef MEGAMOE_URMA_COMPUTE_INLINE
}  // namespace MegaMoeUrma
