// Copyright (c) 2026, Lu Lu
// AIV1 resident scratch experiment: host/device share one byte-accurate plan.
#ifndef MEGA_MOE_AIV1_UB_LAYOUT_HPP
#define MEGA_MOE_AIV1_UB_LAYOUT_HPP

#include <cstdint>
#include "gmm1_cv_plan.hpp"

namespace MegaMoeImpl {
struct Aiv1ResidentUbLayout {
    static constexpr uint32_t ActivationRows = Gmm1CvPlan::StepRows;
    static constexpr uint32_t ActivationColumns = 256U;
    static constexpr uint32_t ActivationBytes = Gmm1CvPlan::End;
    static constexpr uint32_t GmActivationBytes = 64U * 256U * 2U * 2U;
    static constexpr uint32_t CombineMetaRows = 384U;
    static constexpr uint32_t MetaWords = 8U;
    uint64_t dispatchBase;
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

// CV两槽各128行：均衡期望M<=128时可保留两完整tile；大M保留GM解耦。
// 先只推广到已验证EP8。偏斜路由可能超过期望M，此规则不是所有路由的最优保证。
MEGAMOE_UB_PLAN_INLINE bool UseGmm1CvForLaunch(uint32_t bs, uint32_t topk, uint32_t world,
                                              uint32_t expertsPerRank, uint32_t pairedMaxTokens)
{
    return world == 8U && bs <= pairedMaxTokens && topk != 0U && expertsPerRank != 0U &&
        uint64_t(bs) * topk <= uint64_t(expertsPerRank) * Gmm1CvPlan::Rows;
}

// 与DispatchBuffInit逐项相同；不能只预算token ring而漏掉index、prefix和metadata。
MEGAMOE_UB_PLAN_INLINE Aiv1ResidentUbLayout MakeAiv1ResidentUbLayout(
    uint32_t hidden, uint32_t world, uint32_t experts, uint32_t routeItems,
    uint32_t dispatchSlots, uint32_t tokenScaleBytes, uint32_t combineSlots, bool cv = true)
{
    const uint64_t activationBytes = cv ? Aiv1ResidentUbLayout::ActivationBytes : Aiv1ResidentUbLayout::GmActivationBytes;
    const uint64_t dispatchEnd = activationBytes +
        AlignResidentUb(uint64_t(world) * experts * 4U) + AlignResidentUb(uint64_t(routeItems) * 4U) +
        uint64_t(dispatchSlots) * tokenScaleBytes + AlignResidentUb(uint64_t(experts) * 4U) +
        uint64_t(dispatchSlots) * Aiv1ResidentUbLayout::MetaWords * 4U;
    const uint64_t rowsBase = AlignResidentUb(dispatchEnd);
    const uint64_t metaBase = rowsBase + uint64_t(combineSlots) * AlignResidentUb(uint64_t(hidden) * 2U);
    return {activationBytes, dispatchEnd, rowsBase, metaBase,
            metaBase + uint64_t(Aiv1ResidentUbLayout::CombineMetaRows) * Aiv1ResidentUbLayout::MetaWords * 4U,
            combineSlots, hidden != 0U && world != 0U && experts != 0U && routeItems != 0U &&
            dispatchSlots >= 1U && dispatchSlots <= 6U && combineSlots >= 1U && combineSlots <= 6U &&
            tokenScaleBytes >= hidden};
}
#undef MEGAMOE_UB_PLAN_INLINE
} // namespace MegaMoeImpl
#endif
