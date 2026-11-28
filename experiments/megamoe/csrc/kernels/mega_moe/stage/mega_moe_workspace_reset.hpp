/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See third-party/cann.LICENSE for the full text of the License.
 */
// Modified by SimpleBright_Man 2026

#ifndef MEGA_MOE_WORKSPACE_RESET_H
#define MEGA_MOE_WORKSPACE_RESET_H

#include "../common/mega_moe_utils.hpp"

namespace MegaMoeImpl {

using namespace AscendC;

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
// 清理连续 flag workspace；prefetch 路径额外清理独立分配的 GMM1 tile 状态区。
// 清零范围一律取自 WorkspaceInfo 在分配处记账的元素数，与布局恒同源。
template <bool TopkWeightsPrefetch>
__aicore__ inline void ResetSyncStatus(const AivJobContext& job, const Params& params, int32_t resetBatchElementCount,
                                       LocalTensor<int32_t>& resetTensor)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs || resetBatchElementCount <= 0) {
        return;
    }
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    ResetWorkspaceRegion<1>(job, params.workspaceInfo.flagActivationToGmm2Ptr,
                            static_cast<int32_t>(params.workspaceInfo.flagResetElementCount), resetBatchElementCount,
                            resetTensor);

}

// The route map is consumed with scalar bypass-cache reads, so initialize it to
// an explicit invalid row instead of relying on allocator contents from a prior launch.
__aicore__ inline void ResetDedupRouteMap(const AivJobContext& job, const Params& params,
                                          int32_t resetBatchElementCount, LocalTensor<int32_t>& resetTensor)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (params.workspaceInfo.dedupRouteRowPtr == nullptr || resetBatchElementCount <= 0) {
        return;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_V, SYNC_EVENT_ID2>();
    int32_t alignedBatch = static_cast<int32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount), static_cast<uint64_t>(INT32_PER_256B)));
    Duplicate(resetTensor, int32_t(-1), alignedBatch);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    int32_t sourceTokenCount = static_cast<int32_t>(
        static_cast<uint64_t>(params.tilingData->epWorldSize) * params.tilingData->numMaxTokensPerRank);
    int32_t recordWordCount = static_cast<int32_t>(Ops::Base::CeilAlign(
        static_cast<uint64_t>(sourceTokenCount) * DEDUP_ROUTE_STRIDE_WORDS, static_cast<uint64_t>(INT_CACHELINE)));
    int32_t firstRowWordCount = static_cast<int32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(sourceTokenCount), static_cast<uint64_t>(INT_CACHELINE)));
    ResetWorkspaceRegion<INT_CACHELINE>(job, params.workspaceInfo.dedupRouteRowPtr, recordWordCount, alignedBatch,
                                        resetTensor);
    if (params.workspaceInfo.dedupFirstRowPtr != nullptr) {
        ResetWorkspaceRegion<INT_CACHELINE>(job, params.workspaceInfo.dedupFirstRowPtr, firstRowWordCount, alignedBatch,
                                            resetTensor);
    }
}
#endif

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_WORKSPACE_RESET_H
