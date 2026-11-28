// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

/*!
 * \file mega_moe_workspace.h
 * \brief
 */

#ifndef MEGA_MOE_WORKSPACE_H
#define MEGA_MOE_WORKSPACE_H

// peermem 窗口的布局与尺寸（含 HOST_DEVICE / GM_ADDR 宏与基础 include）。
#include "mega_moe_peermem.hpp"

namespace MegaMoeImpl {

constexpr int64_t EXCEPTION_DUMP_REGION_SIZE = 60 * 1024LL;  // 异常dump区
constexpr int64_t SIZE_INT_8 = 1U;
constexpr int64_t SIZE_INT_32 = 4U;
constexpr int64_t SIZE_BF_16 = 2U;

struct WorkspaceInfo {
    GM_ADDR dispatchRevDataPtr;
    GM_ADDR dispatchRevScalePtr;
    GM_ADDR activationQuantDataPtr;
    GM_ADDR activationQuantScalePtr;
    GM_ADDR expertRevTokenNumsPtr;
    GM_ADDR metaInfoPtr;
    // Per-(sourceRank, token) 64B record: [0..7]=local rows (-1 if unowned), [8..15]=cached float32 weights.
    GM_ADDR dedupRouteRowPtr{nullptr};
    // Per-(sourceRank, token) first materialized local row (-1 if not yet dispatched) for O(1) reuse lookup.
    GM_ADDR dedupFirstRowPtr{nullptr};
    GM_ADDR flagActivationToGmm2Ptr;
    GM_ADDR flagDispatchToGmm1Ptr;
    GM_ADDR flagSendCntCalToUpdParamsPtr;
    GM_ADDR flagGmmToEpiloguePtr{nullptr};
    GM_ADDR gmm2ReadyPtr{nullptr};
    GM_ADDR flagGmm2IssuedPtr{nullptr};
    GM_ADDR gmm2CombineSyncCounterPtr{nullptr};
    GM_ADDR cumsumInfoPtr{nullptr};
    GM_ADDR gmm1MmadResPtr{nullptr};
    GM_ADDR gmm2MmadResPtr{nullptr};
    GM_ADDR sharedExpertResultPtr{nullptr};
    GM_ADDR sharedExpertGmm1OutPtr{nullptr};
    GM_ADDR sharedExpertInputDataPtr{nullptr};
    GM_ADDR sharedExpertInputScalePtr{nullptr};
    GM_ADDR sharedExpertActivationDataPtr{nullptr};
    GM_ADDR sharedExpertActivationScalePtr{nullptr};
    GM_ADDR sharedExpertGmm2TileCounterPtr{nullptr};

    // 连续 flag 通知区（自 flagActivationToGmm2Ptr 起）的 int32 元素总数。
    // 在分配处顺手记账，保证 ResetSyncStatus 的清零范围与分配范围恒同源。
    GM_ADDR flagTopkValidIndexSyncPtr{nullptr};
    GM_ADDR flagRoutedFfnDonePtr{nullptr};
    GM_ADDR flagSharedActivationDonePtr{nullptr};
    GM_ADDR flagSharedGmm1DonePtr{nullptr};
    int64_t flagResetElementCount{0};

    int64_t workspaceSize;
    HOST_DEVICE WorkspaceInfo() = default;
    HOST_DEVICE WorkspaceInfo(GM_ADDR base, const MegaMoeTilingData* tilingData, uint32_t serverNum = 1)
    {
        InitializeWorkspaceLayout(base, tilingData, serverNum);
    }

private:
    HOST_DEVICE void InitializeWorkspaceLayout(GM_ADDR base, const MegaMoeTilingData* tilingData, uint32_t serverNum)
    {
        workspaceSize = 0;
        dispatchRevDataPtr = base;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->h, ALIGN_512);
        dispatchRevScalePtr = base + workspaceSize;

        int64_t dispatchScaleElementsPerToken =
            Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->h), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        workspaceSize +=
            Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * dispatchScaleElementsPerToken, ALIGN_512);

        activationQuantDataPtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_8 * tilingData->maxOutputSize * tilingData->hiddenDim / ACTIVATION_N_HALF, ALIGN_512);

        activationQuantScalePtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_8 * tilingData->maxOutputSize * tilingData->hiddenDim / ACTIVATION_N_HALF / MXFP_SCALE_GROUP_NUM,
            ALIGN_512);

        expertRevTokenNumsPtr = base + workspaceSize;
        int64_t blockMajorCountStride = static_cast<int64_t>(
            Ops::Base::CeilAlign(tilingData->moeExpertPerRank, static_cast<uint32_t>(INT_CACHELINE)));
        int64_t blockMajorCountBytes =
            blockMajorCountStride * static_cast<int64_t>(sizeof(int32_t)) * tilingData->aicNum;
        workspaceSize += Ops::Base::CeilAlign(blockMajorCountBytes, ALIGN_512);

        metaInfoPtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(tilingData->maxOutputSize * ALIGN_32, ALIGN_512);

        if (UseDeduplicatedRoutes(tilingData)) {
            int64_t sourceTokenCount =
                static_cast<int64_t>(tilingData->epWorldSize) * tilingData->numMaxTokensPerRank;
            dedupRouteRowPtr = base + workspaceSize;
            workspaceSize +=
                Ops::Base::CeilAlign(SIZE_INT_32 * sourceTokenCount * DEDUP_ROUTE_STRIDE_WORDS, ALIGN_512);
            dedupFirstRowPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * sourceTokenCount, ALIGN_512);
        }

        // 以下三组 flag 仅由 MoE 专家使用；共享专家路径不使用这些 flag。
        constexpr bool useMteWaveCombine = true;
        int64_t maxWavesPerExpert =
            Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize), static_cast<int64_t>(L1_TILE_M_256));
        int64_t waveFlagSlotsPerExpert = maxWavesPerExpert * static_cast<int64_t>(INT_CACHELINE);
        // Dispatch 与 GMM2 输入都按 256-token M-group 发布。
        int64_t activationFlagSlotsPerExpert = waveFlagSlotsPerExpert;
        int64_t moeExpertCount = static_cast<int64_t>(tilingData->moeExpertPerRank);

        // 所有 Scalar 通知都放在同一段连续 workspace 中。每个逻辑槽独占一个 64B cache line，
        // 因而 ResetFlagList 可以按任务分区一次清理完整区域。
        int64_t flagRegionBeginOffset = workspaceSize;
        flagActivationToGmm2Ptr = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * moeExpertCount * activationFlagSlotsPerExpert;
        flagDispatchToGmm1Ptr = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * moeExpertCount * waveFlagSlotsPerExpert;

        // 每(expert, aiCore)单独占一个cache_line
        flagSendCntCalToUpdParamsPtr = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * INT_CACHELINE * moeExpertCount * tilingData->aicNum;

        // 每个 AIC 的就绪序列与前面的 flag 连续存放，使 ResetFlagList 能用同一次 MTE reset 清理；
        // 每个序列独占一个 64B cache line。
        // W4 GMM1 activation 始终使用该序号；采用 tile Combine 的非量化路径也复用它。
        {
            flagGmmToEpiloguePtr = base + workspaceSize;
            workspaceSize += static_cast<int64_t>(tilingData->aicNum) * INT_CACHELINE * SIZE_INT_32;
        }

        if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4) {
            // A8W4 非量化 Combine：每个 slice 按首 M-group 独占一个 64B 计数槽。
            // 用最大 M-group 数预留地址，避免跨 wave 复用槽而让快核提前放行旧 slice。
            // 本区位于连续 flag reset 范围内，每次算子调用统一清零。
            gmm2ReadyPtr = base + workspaceSize;
            int64_t readySlotsPerExpert = waveFlagSlotsPerExpert;
            workspaceSize += SIZE_INT_32 * moeExpertCount * readySlotsPerExpert;
            // 每个 AIC 独占 cache line；仅为提交序号，不能替代 FIX 完成。纳入原连续 reset 区。
            flagGmm2IssuedPtr = base + workspaceSize;
            workspaceSize += SIZE_INT_32 * INT_CACHELINE * static_cast<int64_t>(tilingData->aicNum);
        }

        // Shared expert GMM2 tile counter: tile 级 flag counter, 每 shared expert 一组 slot。
        // 纳入连续 flag 区以便 ResetFlagList 一次性清零。
        if (tilingData->sharedExpertNum > 0) {
            sharedExpertGmm2TileCounterPtr = base + workspaceSize;
            int64_t tokenGroupCount =
                Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->bs), static_cast<int64_t>(L1_TILE_M_256));
            workspaceSize +=
                SIZE_INT_32 * tokenGroupCount * static_cast<int64_t>(tilingData->sharedExpertNum) * INT_CACHELINE;
        }
        if (useMteWaveCombine && tilingData->sharedExpertNum > 0 &&
            tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4) {
            // Independent protocols: input AIV1 subgroup, routed AIC completion, shared activation, shared AIC GMM1
            // completion.
            const int64_t coreFlagBytes = static_cast<int64_t>(tilingData->aicNum) * INT_CACHELINE * SIZE_INT_32;
            flagTopkValidIndexSyncPtr = base + workspaceSize;
            workspaceSize += coreFlagBytes;
            flagRoutedFfnDonePtr = base + workspaceSize;
            workspaceSize += coreFlagBytes;
            flagSharedActivationDonePtr = base + workspaceSize;
            workspaceSize += coreFlagBytes;
            flagSharedGmm1DonePtr = base + workspaceSize;
            workspaceSize += coreFlagBytes;
        }
        flagResetElementCount = (workspaceSize - flagRegionBeginOffset) / SIZE_INT_32;

        // W4 Dispatch 跨激活保留 cumsum；Activation 会覆盖对应 UB，
        // 因此需要逐物理 block 的 GM 备份。A8W8 的 cumsum 常驻 AIV1 UB 并按 wave 连续推进，不需要该备份。
        cumsumInfoPtr = nullptr;
        gmm1MmadResPtr = nullptr;
        gmm2MmadResPtr = nullptr;
        bool activationOverwritesDispatchCumsum = tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4;
        if (activationOverwritesDispatchCumsum) {
            // cumsumInfo：逐核备份 cumsum 状态，每核 moeExpertPerRank × epWorldSize 个 int32。
            cumsumInfoPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(static_cast<int64_t>(SIZE_INT_32 * tilingData->moeExpertPerRank *
                                                                       tilingData->epWorldSize),
                                                  ALIGN_32) *
                             tilingData->aicNum;
        }
        if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4) {
            // gmm1MmadRes：GMM1 matmul 输出，布局为 maxOutputSize × hiddenDim 个 BF16。
            gmm1MmadResPtr = base + workspaceSize;
            workspaceSize += SIZE_BF_16 * tilingData->maxOutputSize * tilingData->hiddenDim;
        }
        // gmm2MmadRes：GMM2 matmul 输出，布局为 maxOutputSize × h 个 BF16；
        // 保持原 MTE 布局：W4/A8W8 都预留整段结果容量，不引入新的 ring 复用协议。
        {
            if (useMteWaveCombine) {
                workspaceSize = Ops::Base::CeilAlign(workspaceSize, static_cast<int64_t>(ALIGN_512));
            }
            gmm2MmadResPtr = base + workspaceSize;
            // MTE wave 路径为每个 MoE token 行分配独立的 [H] BF16
            // 区域，生命周期覆盖整个 kernel。它不是 batch 或 ring slot，
            // 因此 AIC 推进游标前不需要等待 consumer ACK。
            int64_t gmm2OutputBytes = static_cast<int64_t>(SIZE_BF_16) *
                                      static_cast<int64_t>(tilingData->maxOutputSize) *
                                      static_cast<int64_t>(tilingData->h);
            workspaceSize += useMteWaveCombine ? Ops::Base::CeilAlign(gmm2OutputBytes, static_cast<int64_t>(ALIGN_512))
                                               : gmm2OutputBytes;
        }

        // 共享专家 workspace buffer。
        if (tilingData->sharedExpertNum > 0) {
            // sharedExpertResult：共享专家 GMM2 输出 [sharedExpertNum × bs × h]。
            sharedExpertResultPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_BF_16 * tilingData->bs * tilingData->sharedExpertNum * tilingData->h, ALIGN_512);
            // sharedExpertGmm1Out：共享专家 GMM1 输出 [sharedExpertNum × bs × hiddenDim]。
            sharedExpertGmm1OutPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_BF_16 * tilingData->bs * tilingData->sharedExpertNum * tilingData->hiddenDim, ALIGN_512);
            // sharedExpertInputData：GMM1 输入数据 [bs × h]（按 BF16 预留以同时兼容 FP8/BF16 共享专家）。
            sharedExpertInputDataPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(SIZE_BF_16 * tilingData->bs * tilingData->h, ALIGN_512);
            // sharedExpertInputScale：GMM1 输入 scale [bs × CeilDiv(h, 32) × 2]。
            sharedExpertInputScalePtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->bs *
                                                      Ops::Base::CeilDiv(static_cast<uint32_t>(tilingData->h),
                                                                         static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM)) *
                                                      MXFP_MULTI_BASE_SIZE,
                                                  ALIGN_512);
            // sharedExpertActivationData：SwiGLU 输出 [sharedExpertNum × bs × hiddenDim / 2]（按 BF16 预留）。
            sharedExpertActivationDataPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_BF_16 * tilingData->bs * tilingData->sharedExpertNum * tilingData->hiddenDim / ACTIVATION_N_HALF,
                ALIGN_512);
            // sharedExpertActivationScale：SwiGLU scale [sharedExpertNum × bs × hiddenDim / 2 / 32]。
            sharedExpertActivationScalePtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->bs * tilingData->sharedExpertNum *
                                                      tilingData->hiddenDim / ACTIVATION_N_HALF / MXFP_SCALE_GROUP_NUM,
                                                  ALIGN_512);
        }
    }
};

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_WORKSPACE_H
