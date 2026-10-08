// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include <cstdint>

#if defined(__NPU_DEVICE__)
#define MEGAMOE_SHARED_POLICY_INLINE __aicore__ inline
#else
#define MEGAMOE_SHARED_POLICY_INLINE inline
#endif

namespace MegaMoeImpl {
// bs 始终是本卡实际输入行数，不是接收容量，也不是某个专家 slice 的 M。
constexpr uint32_t SHARED_DECODE_MAX_BS = 256U;
MEGAMOE_SHARED_POLICY_INLINE constexpr bool UsesCountEpochInputReady(bool isA8W4, uint32_t bs)
{
    return isA8W4 && bs <= SHARED_DECODE_MAX_BS;
}

// 输入协议与路由分工只有这三种有效组合；不是随执行进度变化的 ready 状态。
enum class InputPreparationMode {
    CountEpoch,         // 双 AIV 先准备输入/路由；由逐槽 count/epoch 到达检查代替输入跨卡握手。
    AllAivRankBarrier,  // 双 AIV 先准备输入/路由，再做输入跨卡握手。
    Aiv1RouteOverlap,   // 路由延后交给 AIV1，与 AIC/AIV0 的共享专家 GMM1 重叠。
};

MEGAMOE_SHARED_POLICY_INLINE constexpr InputPreparationMode SelectInputPreparationMode(
    bool isA8W4, uint32_t sharedExperts, uint32_t bs)
{
    if (UsesCountEpochInputReady(isA8W4, bs)) {
        return InputPreparationMode::CountEpoch;
    }
    return isA8W4 && sharedExperts > 0U ? InputPreparationMode::Aiv1RouteOverlap :
                                        InputPreparationMode::AllAivRankBarrier;
}

// 共享专家的 GEMM 分块策略；不再混入路由执行核/同步协议。
struct SharedExpertPolicy {
    bool useSmallBatchTiles;
    uint32_t gmm1TileN;
    uint32_t gmm2TileN;
};
MEGAMOE_SHARED_POLICY_INLINE constexpr SharedExpertPolicy SelectSharedExpertPolicy(bool isA8W4, uint32_t sharedExperts,
                                                                                   uint32_t bs)
{
    const bool smallBatch = sharedExperts > 0U && UsesCountEpochInputReady(isA8W4, bs);
    return {smallBatch, smallBatch ? 192U : 256U, smallBatch ? 224U : 256U};
}
}  // namespace MegaMoeImpl
#undef MEGAMOE_SHARED_POLICY_INLINE
