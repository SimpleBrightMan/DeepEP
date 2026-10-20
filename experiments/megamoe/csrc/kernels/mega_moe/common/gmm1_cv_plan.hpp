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
// 物理N256包含gate/up各128列；量化结果只压缩到已消费前缀，EMPTY仍由stage写出完成归还。
struct Gmm1CvPlan {
    static constexpr uint32_t Rows = 128U;
    static constexpr uint32_t Columns = 256U;
    static constexpr uint32_t Stages = 2U;
    static constexpr uint32_t StepRows = 32U;
    static constexpr uint32_t SlotBytes = Rows * Columns * 2U;
    static constexpr uint32_t ScratchBase = Stages * SlotBytes;
    static constexpr uint32_t ScratchBytes = 16U * 1024U;
    static constexpr uint32_t End = ScratchBase + ScratchBytes;
    static constexpr uint32_t ScaleBytes = Rows * Columns / 64U;
    static constexpr uint32_t ScaleBase = End - Stages * ScaleBytes;
    static MEGAMOE_CV_INLINE uint32_t ChunkCount(uint32_t rows) { return (rows + Rows - 1U) / Rows; }
    static MEGAMOE_CV_INLINE uint32_t SlotOffset(uint32_t seq) { return seq % Stages * SlotBytes; }
    static MEGAMOE_CV_INLINE uint32_t ScaleOffset(uint32_t seq) { return ScaleBase + seq % Stages * ScaleBytes; }
};
struct Gmm1CvStep {
    uint32_t sequence, row, rows, stageRows;
    MEGAMOE_CV_INLINE bool Last() const { return row + rows == stageRows; }
    MEGAMOE_CV_INLINE uint32_t InputOffset() const {
        return Gmm1CvPlan::SlotOffset(sequence) + row * Gmm1CvPlan::Columns * 2U;
    }
    MEGAMOE_CV_INLINE uint32_t OutputOffset() const {
        return Gmm1CvPlan::SlotOffset(sequence) + row * Gmm1CvPlan::Columns / 2U;
    }
    MEGAMOE_CV_INLINE uint32_t ScaleOffset() const {
        return Gmm1CvPlan::ScaleOffset(sequence) + row * Gmm1CvPlan::Columns / 64U;
    }
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
