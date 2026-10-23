// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#endif
#include "basic_api/kernel_basic_intf.h"
#include "abi.hpp"
#include <type_traits>
#if defined(__NPU_DEVICE__)
#include "compute.hpp"

namespace MegaMoeUrma {
// Keep the production CATLASS configurations, segmented row IO, tile ownership,
// per-expert contexts and AIV0 FP4 prologue. Inputs for BOTH stages already exist.
// No GM readiness flags, publication counters, SwiGLU or transport are executed.
template <bool First>
__aicore__ inline void RunIsolatedFragment(const Arguments& args, uint32_t block, uint32_t index, uint32_t firstTile,
                                           typename std::conditional<First, UrmaGmm1, UrmaGmm2>::type& gmm,
                                           bool endContext)
{
    const Fragment fragment = LoadFragment(args, index);
    const MegaMoeImpl::BlockJobContext job{block, kAic};
    const auto shape = FragmentShape(args, fragment);
    const auto config = First ? UrmaGmmConfig::BuildGmm1ProblemConfig(shape, job, kComputeTileM)
                              : UrmaGmmConfig::BuildGmm2ProblemConfig(shape, job, kComputeTileM);
    MegaMoeImpl::GmmKernel::BlockScheduler scheduler(
        {config.m, config.schedulerN, config.k},
        MegaMoeImpl::GmmKernel::BlockScheduler::Params{
            Te::MakeCoord(static_cast<int64_t>(config.tileM), static_cast<int64_t>(config.tileN))});
    gmm.UpdateProblem(config, First ? Gmm1Addresses(args, fragment) : Gmm2Addresses(args, fragment));
    if constexpr (First) {
        gmm.SetRowIO(FragmentRowIO<UrmaInputRows>(args, fragment));
    } else {
        gmm.SetRowIO(FragmentRowIO<UrmaOutputRows>(args, fragment));
    }
    gmm.Begin();
    for (uint32_t tile = firstTile; tile < scheduler.GetTileNum(); tile += kAic) {
        const auto coord = scheduler.GetBlockCoord(tile);
        const auto tileShape = scheduler.GetBlockShape(coord);
        const uint32_t m = Get<MegaMoeImpl::M_VALUE>(coord);
        const uint32_t n = Get<MegaMoeImpl::N_VALUE>(coord);
        const uint32_t rows = Get<MegaMoeImpl::M_VALUE>(tileShape);
        const uint32_t columns = Get<MegaMoeImpl::N_VALUE>(tileShape);
        if constexpr (g_coreType == AIC) {
            gmm.Compute(m, n, rows, columns);
            // Production GMM1 drains each tile to publish gmm1Done. With no
            // activation consumer that fence and its GM write are unnecessary.
        } else {
            gmm.Dequantize(n, rows, columns);
        }
    }
    if (endContext) {
        gmm.End();                 // Includes FIX drain: the per-tile publication drain was removed.
        CompleteComputeContext();  // Required AIC/AIV0 buffer-credit handshake.
    }
}

__aicore__ inline void RunIsolatedGmms(const Arguments& args, uint32_t block, uint32_t mode)
{
    UrmaGmm1 gmm1;
    UrmaGmm2 gmm2;
    for (uint32_t step = 0U; step < args.config.computeTaskCount; ++step) {
        const auto task = LoadComputeTask(args, step);
        if (mode != 0U && mode != static_cast<uint32_t>(task.stage)) continue;
        const auto range = LoadComputeBatch(args, task.batchIndex);
        if (range.begin == range.end) continue;
        const auto first = LoadFragment(args, range.begin);
        const bool isGmm1 = task.stage == ComputeBatchStage::Gmm1;
        const uint64_t tileBegin = isGmm1 ? first.gmm1TileBegin : first.gmm2TileBegin;
        const uint32_t tilesPerGroup = isGmm1 ? FragmentGmm1Tiles(args, first) : FragmentGmm2Tiles(args, first);
        const uint64_t tileLimit = static_cast<uint64_t>(range.end - range.begin) * tilesPerGroup;
        for (uint64_t tile = FirstOwnedComputeTile(tileBegin, block, kAic); tile < tileLimit;) {
            const auto owned = OwnedComputeGroupAt(tile, tilesPerGroup, kAic);
            const uint64_t nextTile = tile + static_cast<uint64_t>(owned.tileCount) * kAic;
            const uint32_t index = range.begin + owned.group;
            if (isGmm1) {
                RunIsolatedFragment<true>(args, block, index, owned.firstTile, gmm1, nextTile >= tileLimit);
            } else {
                RunIsolatedFragment<false>(args, block, index, owned.firstTile, gmm2, nextTile >= tileLimit);
            }
            tile = nextTile;
        }
    }
}
}  // namespace MegaMoeUrma
#endif

using namespace AscendC;
__schedmode__(1) __global__ __mix__(1, 2) void mega_moe_gmm_only_kernel(MegaMoeUrma::Arguments args, uint32_t mode)
{
#if defined(__NPU_DEVICE__)
    InitSocState();
    util_set_ffts_config(args.ffts);
    SetSysWorkspaceForce(reinterpret_cast<GM_ADDR>(args.systemWorkspace));
    const auto original = GetCtrlSpr<MegaMoeImpl::OVERFLOW_MODE_CTRL, MegaMoeImpl::OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<MegaMoeImpl::OVERFLOW_MODE_CTRL, MegaMoeImpl::OVERFLOW_MODE_CTRL>(0);
    if constexpr (g_coreType == AIC) {
        MegaMoeUrma::RunIsolatedGmms(args, GetBlockIdx() / GetTaskRation(), mode);
    } else if (GetSubBlockIdx() == 0U) {
        MegaMoeUrma::RunIsolatedGmms(args, GetBlockIdx() / GetTaskRation(), mode);
    }
    // AIV1 is idle. No all-core or rank barrier; each active engine drains its
    // own CATLASS context before exiting. The runtime joins the MIX task.
    PipeBarrier<PIPE_ALL>();
    SetCtrlSpr<MegaMoeImpl::OVERFLOW_MODE_CTRL, MegaMoeImpl::OVERFLOW_MODE_CTRL>(original);
#endif
}

extern "C" __attribute__((visibility("default"))) void mega_moe_gmm_only_kernel_do(
    uint32_t block_dim, uint32_t ub_bytes, void* stream, const MegaMoeUrma::Arguments* arguments, uint32_t mode)
{
    mega_moe_gmm_only_kernel<<<block_dim, ub_bytes, stream>>>(*arguments, mode);
}
