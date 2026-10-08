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

}  // namespace MegaMoeImpl
#undef MEGAMOE_WAVE_INLINE
