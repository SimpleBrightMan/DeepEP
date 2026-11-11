// Copyright (c) 2026, Lu Lu
// Modified by SimpleBrightMan 2026
#pragma once
#include <cstdint>

namespace MegaMoeUrma {
#if defined(__NPU_DEVICE__)
#define SHARED_SCHEDULE_INLINE __aicore__ inline
#else
#define SHARED_SCHEDULE_INLINE inline
#endif
constexpr uint32_t kSharedSplitMinBs = 4096U;
SHARED_SCHEDULE_INLINE constexpr bool UseSplitSharedGmm(uint32_t bs)
{
    return bs >= kSharedSplitMinBs;
}
SHARED_SCHEDULE_INLINE constexpr uint32_t SharedGroupCount(uint32_t bs)
{
    return (bs + 255U) / 256U;
}
// Never split a group: an odd group count gives the leading half one extra.
SHARED_SCHEDULE_INLINE constexpr uint32_t SharedHeadGroupCount(uint32_t bs)
{
    return (SharedGroupCount(bs) + 1U) / 2U;
}
SHARED_SCHEDULE_INLINE constexpr uint32_t SharedTaskCount(uint32_t bs, uint32_t intermediate)
{
    return SharedGroupCount(bs) * (intermediate / 256U) * 2U;
}
SHARED_SCHEDULE_INLINE constexpr uint32_t SharedOwnedTasks(uint32_t total, uint32_t worker)
{
    return worker < total ? (total - 1U - worker) / 28U + 1U : 0U;
}
struct SharedTask {
    uint32_t group, tile, slice;
};
SHARED_SCHEDULE_INLINE constexpr SharedTask SharedTaskAt(uint32_t task, uint32_t intermediate)
{
    const uint32_t n = intermediate / 256U;
    return {task / (n * 2U), (task / 2U) % n, task % 2U};
}
#undef SHARED_SCHEDULE_INLINE
}  // namespace MegaMoeUrma
