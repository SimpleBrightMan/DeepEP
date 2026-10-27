// Copyright (c) 2026, Lu Lu
// AIV1 GM resident scratch: host/device share one byte-accurate plan.
#ifndef MEGA_MOE_AIV1_UB_LAYOUT_HPP
#define MEGA_MOE_AIV1_UB_LAYOUT_HPP

#include <cstdint>

namespace MegaMoeImpl {
struct Aiv1ResidentUbLayout {
    static constexpr uint32_t ActivationRows = 64U;
    static constexpr uint32_t ActivationColumns = 256U;
    static constexpr uint32_t ActivationBytes = ActivationRows * ActivationColumns * 2U * 2U;
    static constexpr uint32_t CombineMetaRows = 384U;
    static constexpr uint32_t MetaWords = 8U;
    uint64_t dispatchBase;
    uint64_t routeIndicesBase;
    uint64_t dispatchCopyBase;
    uint64_t expertCountsBase;
    uint64_t dispatchMetaBase;
    uint64_t dispatchEnd;
    uint64_t combineRowsBase;
    uint64_t combineMetaBase;
    uint64_t end;
    uint32_t combineRowCount;
    bool valid;
};

#if defined(__CCE_AICORE__)
#define MEGAMOE_UB_PLAN_INLINE __host__ __aicore__ constexpr inline
#else
#define MEGAMOE_UB_PLAN_INLINE constexpr inline
#endif

MEGAMOE_UB_PLAN_INLINE uint64_t AlignResidentUb(uint64_t n)
{
    return (n + 31U) / 32U * 32U;
}

// 与DispatchBuffInit逐项相同；不能只预算token ring而漏掉index、prefix和metadata。
MEGAMOE_UB_PLAN_INLINE Aiv1ResidentUbLayout MakeAiv1ResidentUbLayout(
    uint32_t hidden, uint32_t world, uint32_t experts, uint32_t routeItems,
    uint32_t dispatchSlots, uint32_t tokenScaleBytes, uint32_t combineSlots)
{
    const uint64_t activationBytes = Aiv1ResidentUbLayout::ActivationBytes;
    const uint64_t routeBase = activationBytes + AlignResidentUb(uint64_t(world) * experts * 4U);
    const uint64_t copyBase = routeBase + AlignResidentUb(uint64_t(routeItems) * 4U);
    const uint64_t expertBase = copyBase + uint64_t(dispatchSlots) * tokenScaleBytes;
    const uint64_t meta = expertBase + AlignResidentUb(uint64_t(experts) * 4U);
    const uint64_t dispatchEnd = meta + uint64_t(dispatchSlots) * Aiv1ResidentUbLayout::MetaWords * 4U;
    const uint64_t rowsBase = AlignResidentUb(dispatchEnd);
    const uint64_t metaBase = rowsBase + uint64_t(combineSlots) * AlignResidentUb(uint64_t(hidden) * 2U);
    return {activationBytes, routeBase, copyBase, expertBase, meta, dispatchEnd, rowsBase, metaBase,
            metaBase + uint64_t(Aiv1ResidentUbLayout::CombineMetaRows) * Aiv1ResidentUbLayout::MetaWords * 4U,
            combineSlots, hidden != 0U && world != 0U && experts != 0U && routeItems != 0U &&
            dispatchSlots >= 1U && dispatchSlots <= 6U && combineSlots >= 1U && combineSlots <= 6U &&
            tokenScaleBytes >= hidden};
}
#undef MEGAMOE_UB_PLAN_INLINE
} // namespace MegaMoeImpl
#endif
