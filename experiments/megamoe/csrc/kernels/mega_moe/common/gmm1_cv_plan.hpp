// Copyright (c) 2026, Lu Lu
#ifndef MEGAMOE_GMM1_CV_PLAN_HPP
#define MEGAMOE_GMM1_CV_PLAN_HPP
#include <cstdint>
#if defined(__CCE_AICORE__) || defined(__NPU_DEVICE__)
#define MEGAMOE_CV_INLINE __aicore__ constexpr inline
#else
#define MEGAMOE_CV_INLINE constexpr inline
#endif
namespace MegaMoeImpl {
// 物理N256包含gate/up各128列。两个FIX输入槽常驻，量化scratch独立，不能覆盖在途CV数据。
struct Gmm1CvPlan {
    static constexpr uint32_t Rows = 128U;
    static constexpr uint32_t Columns = 256U;
    static constexpr uint32_t Stages = 2U;
    static constexpr uint32_t StepRows = 32U;
    static constexpr uint32_t SlotBytes = Rows * Columns * 2U;
    static constexpr uint32_t ScratchBase = Stages * SlotBytes;
    static constexpr uint32_t ScratchBytes = 16U * 1024U;
    static constexpr uint32_t End = ScratchBase + ScratchBytes;
    static MEGAMOE_CV_INLINE uint32_t ChunkCount(uint32_t rows) { return (rows + Rows - 1U) / Rows; }
    static MEGAMOE_CV_INLINE uint32_t SlotOffset(uint32_t seq) { return seq % Stages * SlotBytes; }
};
// 当前逻辑tile的持久状态；只在任务边界构造地址/shape，32行yield不重复重建专家上下文。
struct Gmm1CvTask {
    uint32_t m = 0U, n = 0U, rows = 0U, columns = 0U, nextRow = 0U;
    MEGAMOE_CV_INLINE uint32_t StepCount() const
    {
        const uint32_t left = rows - nextRow;
        return left < Gmm1CvPlan::StepRows ? left : Gmm1CvPlan::StepRows;
    }
    MEGAMOE_CV_INLINE bool StageBegin() const { return nextRow % Gmm1CvPlan::Rows == 0U; }
    MEGAMOE_CV_INLINE bool Advance(uint32_t count)
    {
        nextRow += count;
        return nextRow == rows || nextRow % Gmm1CvPlan::Rows == 0U;
    }
    MEGAMOE_CV_INLINE bool Complete() const { return nextRow == rows; }
};
} // namespace MegaMoeImpl
#undef MEGAMOE_CV_INLINE
#endif
