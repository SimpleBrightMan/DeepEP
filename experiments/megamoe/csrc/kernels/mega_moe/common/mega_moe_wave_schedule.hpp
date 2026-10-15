// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
// See third-party/cann.LICENSE. Modified by SimpleBright_Man 2026.
#pragma once

#include <cstdint>

#if defined(__NPU_DEVICE__)
#define MEGAMOE_WAVE_INLINE __aicore__ inline
#else
#define MEGAMOE_WAVE_INLINE inline
#endif

namespace MegaMoeImpl {

// 专家内行号与本卡紧凑 GM 行号一起推进；后者不是跨卡全局 token ID。
struct ExpertTokenPosition {
    uint32_t expertIdx = 0U;
    uint32_t tokenIndexInExpert = 0U;
    uint64_t globalTokenIndex = 0U;
};

// 左闭右开区间；end 可以落在专家内部，也可以指向下一专家的第 0 行。
struct ExpertTokenRange {
    ExpertTokenPosition begin{};
    ExpertTokenPosition end{};
};

// 一次计算只处理一个专家内的 slice。globalRowStart 是本卡紧凑 GM 行号，不是跨卡 token ID。
// 与可变的下一任务 position 分开保存，避免跨专家推进后丢失本次计算的专家身份。
struct ExpertSlice {
    uint32_t expertIdx = 0U;
    uint32_t rowStartInExpert = 0U;
    uint32_t rowCount = 0U;
    uint64_t globalRowStart = 0U;
};

// 纯调度描述，不代表数据 ready。lastActiveExpert 用于识别最终非空 Combine。
struct ExpertWaveRange {
    ExpertTokenPosition begin{};
    ExpertTokenPosition end{};
    uint32_t lastActiveExpert = 0U;
};

// 只判断专家/预算边界，不读取 GM flag；空专家由推进函数跳过。
MEGAMOE_WAVE_INLINE constexpr bool HasWaveCapacity(const ExpertTokenPosition& position, uint32_t expertCount,
                                                   uint32_t usedMGroups, uint32_t targetMGroups)
{
    return position.expertIdx < expertCount && usedMGroups < targetMGroups;
}

// 前置条件：position 在该专家有效范围内，used <= target，TileM > 0。
// 一个 slice 不跨专家；尾片不足 TileM 也占一个 M-group，不与下一专家合并。
template <uint32_t TileM>
MEGAMOE_WAVE_INLINE uint32_t AdvanceExpertTokenPositionInWave(uint32_t expertTokenCount, uint32_t waveMGroupTarget,
                                                             uint32_t& waveMGroupCount, ExpertTokenPosition& position)
{
    static_assert(TileM > 0U);
    const uint32_t remainingRows = expertTokenCount - position.tokenIndexInExpert;
    const uint32_t waveCapacity = (waveMGroupTarget - waveMGroupCount) * TileM;
    const uint32_t rows = remainingRows < waveCapacity ? remainingRows : waveCapacity;
    waveMGroupCount += rows / TileM + (rows % TileM != 0U);
    position.tokenIndexInExpert += rows;
    position.globalTokenIndex += rows;
    if (position.tokenIndexInExpert >= expertTokenCount) {
        ++position.expertIdx;
        position.tokenIndexInExpert = 0U;
    }
    return rows;
}

// 先保存当前 slice 的身份，再推进下一任务游标；专家尾片也独占一个 M-group。
template <uint32_t TileM>
MEGAMOE_WAVE_INLINE ExpertSlice TakeNextExpertSliceInWave(uint32_t expertTokenCount, uint32_t waveMGroupTarget,
                                                         uint32_t& waveMGroupCount, ExpertTokenPosition& position)
{
    ExpertSlice slice{position.expertIdx, position.tokenIndexInExpert, 0U, position.globalTokenIndex};
    slice.rowCount = AdvanceExpertTokenPositionInWave<TileM>(expertTokenCount, waveMGroupTarget, waveMGroupCount,
                                                           position);
    return slice;
}

// GMM2 重放已保存的 Wave：边界专家只取局部范围，中间专家取完整范围；不修改 Wave 或计算游标。
// 调用方按顺序准备专家元数据，并保证 expertIdx 位于该 Wave 覆盖的专家范围内。
MEGAMOE_WAVE_INLINE ExpertSlice GetExpertSliceInWave(const ExpertWaveRange& wave, uint32_t expertIdx,
                                                     uint32_t expertTokenCount, uint64_t expertGlobalRowStart)
{
    const uint32_t begin = expertIdx == wave.begin.expertIdx ? wave.begin.tokenIndexInExpert : 0U;
    const uint32_t end = expertIdx == wave.end.expertIdx ? wave.end.tokenIndexInExpert : expertTokenCount;
    return {expertIdx, begin, end - begin, expertGlobalRowStart + begin};
}

}  // namespace MegaMoeImpl
#undef MEGAMOE_WAVE_INLINE
