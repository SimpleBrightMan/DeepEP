// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"
#include "compute_contract.hpp"
#include "transport.hpp"
#include "timer.hpp"
#include "../stage/mega_moe_gmm1_activation.hpp"

namespace MegaMoeUrma {

using namespace AscendC;
using MegaMoeImpl::Get;

// Every matrix has exactly the same peer/CV row order. A fragment is the
// intersection with one expert, so no expert-major permutation or old Wave
// cursor is used anywhere in this compute path.
__aicore__ inline Fragment LoadFragment(const Arguments& args, uint32_t index)
{
    const auto* gm = reinterpret_cast<__gm__ Fragment*>(args.fragments) + index;
    return {gm->expert,          gm->sourceRank,    gm->rowBegin,     gm->rows,
            gm->inputReadyIndex, gm->gmm1TileBegin, gm->gmm2TileBegin};
}

__aicore__ inline __gm__ int32_t* ComputeFlag(uint64_t base, uint64_t index)
{
    return reinterpret_cast<__gm__ int32_t*>(base + index * kFlagBytes);
}

__aicore__ inline void WaitComputeFlag(uint64_t base, uint64_t index, uint32_t expected)
{
    auto* flag = ComputeFlag(base, index);
    while (ReadGmByPassDCache(flag) < static_cast<int32_t>(expected)) {
    }
}

__aicore__ inline uint32_t FragmentGmm1Tiles(const Arguments& args, const Fragment& fragment)
{
    return ComputeTileCount(fragment.rows, args.config.intermediate);
}

__aicore__ inline uint32_t FragmentGmm2Tiles(const Arguments& args, const Fragment& fragment)
{
    return ComputeTileCount(fragment.rows, args.config.hidden);
}

__aicore__ inline MegaMoeImpl::GMMAddrInfo Gmm1Addresses(const Arguments& args, const Fragment& fragment)
{
    const uint64_t h = args.config.hidden;
    const uint64_t n = static_cast<uint64_t>(args.config.intermediate) * 2U;
    const uint64_t scaleH = h / 32U;
    const uint64_t expert = fragment.expert;
    MegaMoeImpl::GMMAddrInfo addr{};
    addr.aGlobal = reinterpret_cast<GM_ADDR>(args.symmetric + args.config.receiveXOffset + fragment.rowBegin * h);
    addr.aScaleGlobal =
        reinterpret_cast<GM_ADDR>(args.symmetric + args.config.receiveScaleOffset + fragment.rowBegin * scaleH);
    addr.bGlobal = reinterpret_cast<GM_ADDR>(args.weight1 + expert * h * n / 2U);
    addr.bScaleGlobal = reinterpret_cast<GM_ADDR>(args.weightScale1 + expert * scaleH * n);
    addr.gmm1OutGlobal = reinterpret_cast<GM_ADDR>(args.gmm1Output + fragment.rowBegin * n * sizeof(bfloat16_t));
    return addr;
}

__aicore__ inline MegaMoeImpl::GMMAddrInfo Gmm2Addresses(const Arguments& args, const Fragment& fragment)
{
    const uint64_t h = args.config.hidden;
    const uint64_t i = args.config.intermediate;
    const uint64_t scaleI = i / 32U;
    const uint64_t expert = fragment.expert;
    MegaMoeImpl::GMMAddrInfo addr{};
    addr.aGlobal = reinterpret_cast<GM_ADDR>(args.activation + fragment.rowBegin * i);
    addr.aScaleGlobal = reinterpret_cast<GM_ADDR>(args.activationScale + fragment.rowBegin * scaleI);
    addr.bGlobal = reinterpret_cast<GM_ADDR>(args.weight2 + expert * i * h / 2U);
    addr.bScaleGlobal = reinterpret_cast<GM_ADDR>(args.weightScale2 + expert * scaleI * h);
    addr.gmm2OutGlobal = reinterpret_cast<GM_ADDR>(args.gmm2Output + fragment.rowBegin * h * sizeof(bfloat16_t));
    return addr;
}

using UrmaGmmConfig = MegaMoeImpl::GmmKernel::Config<true, 0, fp8_e4m3fn_t, fp4x2_e2m1_t, bfloat16_t, fp8_e8m0_t,
                                                     fp8_e8m0_t, false, false, false, false, false, false, false>;
using UrmaProblemConfig = UrmaGmmConfig::ProblemConfig;
using UrmaActivation =
    MegaMoeImpl::BlockEpilogueActivationMxQuant<fp8_e4m3fn_t, bfloat16_t, kActivationTileM, kComputeTileN, false>;

// Tie the host-testable memory contract to the shared arithmetic's real UB
// layout. Top-k weights are consumed by Reduce, so no prefetch buffer exists.
constexpr auto kActualActivationUb =
    MegaMoeImpl::Detail::BuildActivationMxQuantUbOffsets<bfloat16_t, kActivationTileM, kComputeTileN, false>();
static_assert(kActualActivationUb.firstInputOffsetBytes == ActivationUbContract::gateOffset &&
                  kActualActivationUb.secondInputOffsetBytes == ActivationUbContract::upOffset &&
                  kActualActivationUb.activationOutputOffsetBytes == ActivationUbContract::activationOffset &&
                  kActualActivationUb.quantOutputOffsetBytes == ActivationUbContract::quantOffset &&
                  kActualActivationUb.quantScaleOffsetBytes == ActivationUbContract::scaleOffset &&
                  kActualActivationUb.maxExpOffsetBytes == ActivationUbContract::maxExpOffset &&
                  kActualActivationUb.reciprocalScaleOffsetBytes == ActivationUbContract::reciprocalOffset &&
                  kActualActivationUb.scaleElementCapacity == ActivationUbContract::scaleElements,
              "URMA activation UB contract must match the shared epilogue");

__aicore__ inline MegaMoeImpl::ProblemShape FragmentShape(const Arguments& args, const Fragment& fragment)
{
    return {fragment.rows, static_cast<int64_t>(args.config.intermediate) * 2, args.config.hidden, 0};
}

__aicore__ inline TraceMeta FragmentTraceMeta(const Arguments& args, const Fragment& fragment, uint32_t index)
{
    TraceMeta meta;
    meta.peer = fragment.sourceRank;
    // The host encodes inputReadyIndex as sourceRank * maxPeerChunks + cv.
    meta.cv = fragment.inputReadyIndex % args.config.maxPeerChunks;
    meta.fragment = index;
    meta.expert = fragment.expert;
    meta.row = fragment.rowBegin;
    meta.rows = fragment.rows;
    return meta;
}

__aicore__ inline TraceMeta ComputeTileTraceMeta(const TraceMeta& fragmentMeta, uint32_t m, uint32_t n, uint32_t rows,
                                                 uint32_t columns)
{
    TraceMeta meta = fragmentMeta;
    meta.row += m;
    meta.rows = rows;
    // These are actual scheduler element coordinates, not linear tile IDs.
    meta.tileM = m;
    meta.tileN = n;
    meta.columns = columns;
    return meta;
}

// A new context must not republish the initial 8/9 free-slot credits until
// AIV0 has consumed the old context's final credits. The acknowledgement is
// ordered on those same MTE pipelines; it needs neither a scalar fence nor a
// full-core barrier. Only this AIC and its paired AIV0 use flag 10.
__aicore__ inline void CompleteComputeContext()
{
    constexpr uint16_t kContextEndFlag = 10U;
    if constexpr (g_coreType == AIC) {
        CrossCoreWaitFlag<4, PIPE_MTE1>(kContextEndFlag);
    } else {
        CrossCoreSetFlag<4, PIPE_MTE3>(kContextEndFlag);
    }
}

using UrmaGmm1 = MegaMoeImpl::GmmKernel::CatlassGmm1<UrmaProblemConfig, true>;

__aicore__ inline void RunGmm1Fragment(const Arguments& args, uint32_t block, uint32_t fragmentIndex, UrmaGmm1& gmm1,
                                       Timer& timer)
{
    const MegaMoeImpl::BlockJobContext job{block, kAic};
    const Fragment fragment = LoadFragment(args, fragmentIndex);
    const auto fragmentMeta = FragmentTraceMeta(args, fragment, fragmentIndex);
    const auto shape = FragmentShape(args, fragment);
    auto config1 = UrmaGmmConfig::BuildGmm1ProblemConfig(shape, job, kComputeTileM);
    MegaMoeImpl::GmmKernel::BlockScheduler schedule1(
        {config1.m, config1.schedulerN, config1.k},
        MegaMoeImpl::GmmKernel::BlockScheduler::Params{
            Te::MakeCoord(static_cast<int64_t>(config1.tileM), static_cast<int64_t>(config1.tileN))});
    const uint32_t count1 = schedule1.GetTileNum();
    const uint32_t first1 = FirstOwnedComputeTile(fragment.gmm1TileBegin, block, kAic);
    if (first1 >= count1) return;
    if constexpr (g_coreType == AIC) {
        const uint64_t waitStart = timer.TraceBegin();
        WaitLocalDone(reinterpret_cast<__gm__ uint8_t*>(args.symmetric + args.config.dispatchReadyOffset +
                                                        fragment.inputReadyIndex * kFlagBytes),
                      args.config.epoch);
        timer.TraceEnd(TraceEvent::WaitDispatch, waitStart, fragmentMeta);
    }
    timer.Begin(Phase::Gmm1);
    const auto addr = Gmm1Addresses(args, fragment);
    gmm1.UpdateProblem(config1, addr);
    gmm1.Begin();
    for (uint32_t tile = first1; tile < count1; tile += kAic) {
        const auto coord = schedule1.GetBlockCoord(tile);
        const auto tileShape = schedule1.GetBlockShape(coord);
        const uint32_t m = Get<MegaMoeImpl::M_VALUE>(coord);
        const uint32_t n = Get<MegaMoeImpl::N_VALUE>(coord);
        const uint32_t rows = Get<MegaMoeImpl::M_VALUE>(tileShape);
        const uint32_t columns = Get<MegaMoeImpl::N_VALUE>(tileShape);
        const auto tileMeta = ComputeTileTraceMeta(fragmentMeta, m, n, rows, columns);
        if constexpr (g_coreType == AIC) {
            // Compute submits asynchronous work and may wait on its
            // existing internal dependencies; it is not a FIX drain.
            const uint64_t computeStart = timer.TraceBegin();
            gmm1.Compute(m, n, rows, columns);
            timer.TraceEnd(TraceEvent::Gmm1Tile, computeStart, tileMeta);
            const uint64_t drainStart = timer.TraceBegin();
            SetFlag<HardEvent::FIX_S>(EVENT_ID0);
            WaitFlag<HardEvent::FIX_S>(EVENT_ID0);
            timer.TraceEnd(TraceEvent::Gmm1TileDrain, drainStart, tileMeta);
            WriteGmByPassDCache(ComputeFlag(args.gmm1Done, fragment.gmm1TileBegin + tile), 1);
        } else {
            const uint64_t dequantStart = timer.TraceBegin();
            gmm1.Dequantize(n, rows, columns);
            timer.TraceEnd(TraceEvent::Gmm1Dequant, dequantStart, tileMeta);
        }
    }
    const uint64_t endStart = timer.TraceBegin();
    gmm1.End(g_coreType == AIC);
    CompleteComputeContext();
    timer.TraceEnd(TraceEvent::Gmm1Drain, endStart, fragmentMeta);
    timer.End(Phase::Gmm1);
}

__aicore__ inline void RunGmm2Fragment(const Arguments& args, uint32_t block, uint32_t fragmentIndex, Timer& timer)
{
    const MegaMoeImpl::BlockJobContext job{block, kAic};
    const Fragment fragment = LoadFragment(args, fragmentIndex);
    const auto fragmentMeta = FragmentTraceMeta(args, fragment, fragmentIndex);
    const auto shape = FragmentShape(args, fragment);
    auto config2 = UrmaGmmConfig::BuildGmm2ProblemConfig(shape, job, kComputeTileM);
    MegaMoeImpl::GmmKernel::BlockScheduler schedule2(
        {config2.m, config2.schedulerN, config2.k},
        MegaMoeImpl::GmmKernel::BlockScheduler::Params{
            Te::MakeCoord(static_cast<int64_t>(config2.tileM), static_cast<int64_t>(config2.tileN))});
    const uint32_t count2 = schedule2.GetTileNum();
    const uint32_t first2 = FirstOwnedComputeTile(fragment.gmm2TileBegin, block, kAic);
    if (first2 >= count2) return;
    if constexpr (g_coreType == AIC) {
        // The 13 independent activation workers collectively publish one
        // contribution per original GMM1 tile after both FP8 planes drain.
        const uint64_t waitStart = timer.TraceBegin();
        WaitComputeFlag(args.activationDone, fragmentIndex, FragmentGmm1Tiles(args, fragment));
        timer.TraceEnd(TraceEvent::WaitActivation, waitStart, fragmentMeta);
    }
    timer.Begin(Phase::Gmm2);
    uint32_t completed = 0U;
    uint64_t drainStart = 0U;
    {
        const auto addr = Gmm2Addresses(args, fragment);
        MegaMoeImpl::GmmKernel::CatlassGmm2<UrmaProblemConfig> gmm2(config2, addr);
        for (uint32_t tile = first2; tile < count2; tile += kAic) {
            const auto coord = schedule2.GetBlockCoord(tile);
            const auto tileShape = schedule2.GetBlockShape(coord);
            const uint32_t m = Get<MegaMoeImpl::M_VALUE>(coord);
            const uint32_t n = Get<MegaMoeImpl::N_VALUE>(coord);
            const uint32_t rows = Get<MegaMoeImpl::M_VALUE>(tileShape);
            const uint32_t columns = Get<MegaMoeImpl::N_VALUE>(tileShape);
            const auto tileMeta = ComputeTileTraceMeta(fragmentMeta, m, n, rows, columns);
            if constexpr (g_coreType == AIC) {
                // Submission time only: output completion belongs to
                // the existing context destructor below.
                const uint64_t computeStart = timer.TraceBegin();
                gmm2.Compute(m, n, rows, columns);
                timer.TraceEnd(TraceEvent::Gmm2Tile, computeStart, tileMeta);
            } else {
                const uint64_t dequantStart = timer.TraceBegin();
                gmm2.Dequantize(n, rows, columns);
                timer.TraceEnd(TraceEvent::Gmm2Dequant, dequantStart, tileMeta);
            }
            ++completed;
        }
        drainStart = timer.TraceBegin();
    }  // CATLASS destructor drains FIX and the paired L1 event tokens.
    CompleteComputeContext();
    timer.TraceEnd(TraceEvent::Gmm2Drain, drainStart, fragmentMeta);
    if constexpr (g_coreType == AIC) {
        AtomicAdd(ComputeFlag(args.gmm2Done, fragmentIndex), static_cast<int32_t>(completed));
    }
    timer.End(Phase::Gmm2);
}

__aicore__ inline ComputeBatchRange LoadComputeBatch(const Arguments& args, uint32_t index)
{
    const auto* gm = reinterpret_cast<__gm__ Chunk*>(args.recvChunks) + index;
    Chunk chunk{};
    chunk.fragmentBegin = gm->fragmentBegin;
    chunk.fragmentCount = gm->fragmentCount;
    return ComputeBatchFragments(chunk);
}

// Only an AIC and its paired AIV0 enter this routine. They use the identical
// CV schedule and original global tile ownership. SwiGLU remains independent:
// running the next CV's GMM1 gives it time to finish the preceding CV.
__aicore__ inline void RunCubeAndPrologue(const Arguments& args, uint32_t block, Timer& timer)
{
    // URMA updates receive GM outside this core's cache hierarchy. Keep input
    // and scale reads uncached; every existing MTE caller still defaults off.
    UrmaGmm1 gmm1;
    const uint64_t taskCount = ComputeBatchTaskCount(args.config.recvChunkCount);
    for (uint64_t step = 0U; step < taskCount; ++step) {
        const auto task = ComputeBatchTaskAt(step, args.config.recvChunkCount);
        const auto range = LoadComputeBatch(args, task.chunkIndex);
        for (uint32_t fragmentIndex = range.begin; fragmentIndex < range.end; ++fragmentIndex) {
            if (task.stage == ComputeBatchStage::Gmm1) {
                RunGmm1Fragment(args, block, fragmentIndex, gmm1, timer);
            } else {
                RunGmm2Fragment(args, block, fragmentIndex, timer);
            }
        }
    }
}

// AIV1 blocks 15..27 use an independent 13-way tile partition. In particular,
// the original same-block GMM1/SwiGLU readiness sequence is not reused.
__aicore__ inline void RunActivation(const Arguments& args, uint32_t worker, Timer& timer)
{
    UrmaActivation epilogue;
    epilogue.Init({.yGmAddr = reinterpret_cast<GM_ADDR>(args.activation),
                   .yScaleGmAddr = reinterpret_cast<GM_ADDR>(args.activationScale)});
    const MegaMoeImpl::BlockJobContext job{worker, kActivationWorkers};
    for (uint32_t fragmentIndex = 0U; fragmentIndex < args.config.fragmentCount; ++fragmentIndex) {
        const Fragment fragment = LoadFragment(args, fragmentIndex);
        const auto fragmentMeta = FragmentTraceMeta(args, fragment, fragmentIndex);
        const uint32_t tileCount = FragmentGmm1Tiles(args, fragment);
        const uint32_t first = FirstOwnedComputeTile(fragment.gmm1TileBegin, worker, kActivationWorkers);
        if (first >= tileCount) continue;
        const auto shape = FragmentShape(args, fragment);
        auto addr = Gmm1Addresses(args, fragment);
        // The existing epilogue emits only arithmetic here. Its old Wave
        // completion pointer stays null; this backend owns completion below.
        addr.activationToGmm2Flag = nullptr;
        epilogue.UpdateGlobalAddr({static_cast<int64_t>(fragment.rowBegin * args.config.intermediate),
                                   static_cast<int64_t>(fragment.rowBegin * (args.config.intermediate / 32U)), 0, 0, 0,
                                   0});
        auto config = UrmaGmmConfig::BuildGmm1ProblemConfig(shape, job, kComputeTileM);
        const auto layouts = UrmaGmmConfig::BuildLayouts(config);
        auto gmC = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ bfloat16_t*>(addr.gmm1OutGlobal)), layouts.c);
        MegaMoeImpl::GmmKernel::BlockScheduler scheduler(
            {config.m, config.outputN, config.k},
            MegaMoeImpl::GmmKernel::BlockScheduler::Params{
                Te::MakeCoord(static_cast<int64_t>(kComputeTileM), static_cast<int64_t>(kComputeTileN))});
        epilogue.UpdateNextProblem({config.m, config.outputN, config.k, 0});
        // The scheduler and ready IDs above stay M256. Only the staging copy
        // helper's UB shape changes, matching UrmaActivation's two 64KiB inputs.
        config.tileM = kActivationTileM;
        for (uint32_t tile = first; tile < tileCount; tile += kActivationWorkers) {
            // Decode the original M256 scheduler before recording the wait;
            // no readiness or work ownership changes for the M128 slices.
            const auto coord = scheduler.GetBlockCoord(tile);
            const auto tileShape = scheduler.GetBlockShape(coord);
            const uint32_t m = Get<MegaMoeImpl::M_VALUE>(coord);
            const uint32_t n = Get<MegaMoeImpl::N_VALUE>(coord);
            const uint32_t rows = Get<MegaMoeImpl::M_VALUE>(tileShape);
            const uint32_t columns = Get<MegaMoeImpl::N_VALUE>(tileShape);
            const auto tileMeta = ComputeTileTraceMeta(fragmentMeta, m, n, rows, columns);
            const uint64_t waitStart = timer.TraceBegin();
            WaitComputeFlag(args.gmm1Done, fragment.gmm1TileBegin + tile, 1U);
            timer.TraceEnd(TraceEvent::WaitGmm1, waitStart, tileMeta);
            timer.Begin(Phase::Activation);
            for (uint32_t offset = 0U; offset < rows; offset += kActivationTileM) {
                const auto slice = ActivationSlice(m, rows, offset);
                const MegaMoeImpl::GmmKernel::BlockScheduler::BlockShape sliceShape{slice.rows, columns, config.k};
                const auto sliceMeta = ComputeTileTraceMeta(fragmentMeta, slice.rowBegin, n, slice.rows, columns);
                const uint64_t sliceStart = timer.TraceBegin();
                MegaMoeImpl::GmmKernel::Gmm1Aiv1EpilogueTileA8W4<bfloat16_t, UrmaGmmConfig::MakeLayoutC, true>(
                    epilogue, gmC, addr, config, sliceShape, slice.rowBegin, n);
                // Unlike MMAD submission, the shared epilogue already drains
                // both its FP8 and scale MTE3 stores before returning.
                timer.TraceEnd(TraceEvent::ActivationSlice, sliceStart, sliceMeta);
            }
            // Each sub-slice drains MTE3 before reusing UB. Publish exactly one
            // contribution for the original GMM1 tile, after all its rows.
            AtomicAdd(ComputeFlag(args.activationDone, fragmentIndex), 1);
            timer.End(Phase::Activation);
        }
    }
}

}  // namespace MegaMoeUrma
