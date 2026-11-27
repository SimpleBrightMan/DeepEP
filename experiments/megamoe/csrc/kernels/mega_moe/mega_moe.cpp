// Copyright (c) 2026, Lu Lu
// MegaMoE direct-call entry for the shared DeepEP kernel library.
// Keep the wave implementations and their tiling layout independent of the launcher.
// Modified by SimpleBright_Man 2026
#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#endif
#include "basic_api/kernel_basic_intf.h"
#include "kernel_launch.hpp"
#include "mega_moe_tiling.hpp"
#include "common/mega_moe_constants.hpp"
#include "common/mega_moe_aiv1_ub_layout.hpp"
#include "common/mega_moe_tp_layout.hpp"
#include "common/aiv1_async_events.hpp"
#include <stdexcept>
#include <string>
#if defined(__NPU_DEVICE__)
#include "mega_moe_wave_a8w8.hpp"
#include "mega_moe_wave_a8w4.hpp"
#include "mega_moe_tp_wave.hpp"
#endif

using namespace AscendC;

// Explicit types replace asc_opc dtype/format/tiling-key specialisation.
// A8W4's existing matmul prologue continues to interpret weights as NZ/C0_32.
// Batch scheduling preserves the old ACL schemMode=1 cross-core contract.
template <bool IsA8W4, bool FuseGateUp = false>
__schedmode__(1) __global__
    __mix__(1, 2) void mega_moe_kernel(GM_ADDR context, GM_ADDR x, GM_ADDR x_scales, GM_ADDR topk_ids,
                                       GM_ADDR topk_weights, GM_ADDR weight1, GM_ADDR weight2, GM_ADDR weight_scales1,
                                       GM_ADDR weight_scales2, GM_ADDR shared_weight1, GM_ADDR shared_weight2,
                                       GM_ADDR shared_weight_scales1, GM_ADDR shared_weight_scales2, GM_ADDR y,
                                       GM_ADDR expert_token_nums, GM_ADDR workspace, GM_ADDR timer,
                                       MegaMoeTilingData tiling, uint64_t ffts_addr, bool compute_replay)
{
#if defined(__NPU_DEVICE__)
    InitSocState();
    util_set_ffts_config(ffts_addr);
    constexpr int32_t kDispatchE4M3 = 4;
    if constexpr (IsA8W4) {
        MegaMoeImpl::MegaMoeA8W4Wave<bfloat16_t, bfloat16_t, float, fp4x2_e2m1_t, kDispatchE4M3, FuseGateUp>
            op;
        op.Init(context, x, topk_ids, topk_weights, weight1, weight2, nullptr, weight_scales1, weight_scales2, x_scales,
                shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
                workspace, &tiling, nullptr, timer);
        op.SetComputeReplay(compute_replay);
        op.Process();
    } else {
        MegaMoeImpl::MegaMoeA8W8Wave<bfloat16_t, bfloat16_t, float, fp8_e4m3fn_t, kDispatchE4M3>
            op;
        op.Init(context, x, topk_ids, topk_weights, weight1, weight2, nullptr, weight_scales1, weight_scales2, x_scales,
                shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
                workspace, &tiling, nullptr);
        op.Process();
    }
#endif
}

extern "C" __attribute__((visibility("default"))) void mega_moe_kernel_do(
    uint32_t block_dim, uint32_t ub_bytes, void* stream, bool is_a8w4, uint8_t* context, uint8_t* x, uint8_t* x_scales,
    uint8_t* topk_ids, uint8_t* topk_weights, uint8_t* weight1, uint8_t* weight2, uint8_t* weight_scales1,
    uint8_t* weight_scales2, uint8_t* shared_weight1, uint8_t* shared_weight2, uint8_t* shared_weight_scales1,
    uint8_t* shared_weight_scales2, uint8_t* y, uint8_t* expert_token_nums, uint8_t* workspace, uint8_t* timer,
    const uint8_t* tiling_host, uint64_t ffts_addr, bool compute_replay)
{
    const auto& tiling = *reinterpret_cast<const MegaMoeTilingData*>(tiling_host);
    static_assert(sizeof(MegaMoeTilingData) == 224, "MegaMoE tiling ABI changed");
    if (is_a8w4) {
        using MegaMoeImpl::Aiv1ResidentUbLayout;
        static_assert(Aiv1ResidentUbLayout::MetaWords == MegaMoeImpl::META_INFO_SIZE &&
                      Aiv1ResidentUbLayout::MetaWords == MegaMoeImpl::INT32_PER_256B);
        const uint32_t slots = MegaMoeImpl::Aiv1AsyncEvents::CombineSlots;
        const auto& config = tiling.dispatchBufferConfig;
        const auto resident = MegaMoeImpl::MakeAiv1ResidentUbLayout(
            tiling.h, tiling.epWorldSize, tiling.moeExpertPerRank, config.routeItemsPerBatch,
            config.bufferCount, config.copyBufferBytes, slots);
        // 共享/路由激活使用同一64KiB credit；通信区容量错误在发射NPU前报告。
        if (!resident.valid ||
            config.bufferCount > MegaMoeImpl::Aiv1AsyncEvents::DispatchSlots ||
            resident.end > ub_bytes || resident.end > 256U * 1024U) {
            throw std::invalid_argument("AIV1 resident UB unsupported: BS=" + std::to_string(tiling.bs) +
                " H=" + std::to_string(tiling.h) + " EP=" + std::to_string(tiling.epWorldSize) +
                " local_experts=" + std::to_string(tiling.moeExpertPerRank) +
                " shared=" + std::to_string(tiling.sharedExpertNum) + " required=" + std::to_string(resident.end) +
                " available=" + std::to_string(ub_bytes));
        }
    }
    // 只按权重类型派发；A8W4的BS差异由公共Wave策略处理。
    if (is_a8w4) {
        // 所有合法BS均使用配对权重；与host两轮Wave预算、Python权重选择一致。
        mega_moe_kernel<true, true><<<block_dim, ub_bytes, stream>>>(
            context, x, x_scales, topk_ids, topk_weights, weight1, weight2, weight_scales1, weight_scales2,
            shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
            workspace, timer, tiling, ffts_addr, compute_replay);
    } else {
        mega_moe_kernel<false><<<block_dim, ub_bytes, stream>>>(
            context, x, x_scales, topk_ids, topk_weights, weight1, weight2, weight_scales1, weight_scales2,
            shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
            workspace, timer, tiling, ffts_addr, compute_replay);
    }
}

__schedmode__(1) __global__
    __mix__(1, 2) void mega_moe_tp_kernel(GM_ADDR context, GM_ADDR x, GM_ADDR x_scales, GM_ADDR topk_ids,
                                          GM_ADDR topk_weights, GM_ADDR weight1, GM_ADDR weight2,
                                          GM_ADDR weight_scales1, GM_ADDR weight_scales2, GM_ADDR shared_weight1,
                                          GM_ADDR shared_weight2, GM_ADDR shared_weight_scales1,
                                          GM_ADDR shared_weight_scales2, GM_ADDR y, GM_ADDR expert_token_nums,
                                          GM_ADDR workspace, GM_ADDR timer,
                                          MegaMoeImpl::MegaMoeTpTilingData tiling, uint64_t ffts_addr)
{
#if defined(__NPU_DEVICE__)
    InitSocState();
    util_set_ffts_config(ffts_addr);
    MegaMoeImpl::MegaMoeTpWave op;
    op.Init(context, x, x_scales, topk_ids, topk_weights, weight1, weight2, weight_scales1, weight_scales2,
            shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
            workspace, &tiling, timer);
    op.Process();
#endif
}

extern "C" __attribute__((visibility("default"))) void mega_moe_tp_kernel_do(
    uint32_t block_dim, uint32_t ub_bytes, void* stream, uint8_t* context, uint8_t* x, uint8_t* x_scales,
    uint8_t* topk_ids, uint8_t* topk_weights, uint8_t* weight1, uint8_t* weight2, uint8_t* weight_scales1,
    uint8_t* weight_scales2, uint8_t* shared_weight1, uint8_t* shared_weight2, uint8_t* shared_weight_scales1,
    uint8_t* shared_weight_scales2, uint8_t* y, uint8_t* expert_token_nums, uint8_t* workspace, uint8_t* timer,
    const uint8_t* tiling_host, uint64_t ffts_addr)
{
    const auto& tiling = *reinterpret_cast<const MegaMoeImpl::MegaMoeTpTilingData*>(tiling_host);
    static_assert(sizeof(MegaMoeImpl::MegaMoeTpTilingData) == 80, "MegaMoeTpTilingData ABI changed");
    const auto resident = MegaMoeImpl::MakeTpAiv1ResidentUbLayout(tiling.h);
    if (!resident.valid || resident.end > ub_bytes) {
        throw std::invalid_argument("TP MegaMoE AIV1 resident UB unsupported: H=" + std::to_string(tiling.h) +
                                    " required=" + std::to_string(resident.end) +
                                    " available=" + std::to_string(ub_bytes));
    }
    mega_moe_tp_kernel<<<block_dim, ub_bytes, stream>>>(
        context, x, x_scales, topk_ids, topk_weights, weight1, weight2, weight_scales1, weight_scales2,
        shared_weight1, shared_weight2, shared_weight_scales1, shared_weight_scales2, y, expert_token_nums,
        workspace, timer, tiling, ffts_addr);
}
