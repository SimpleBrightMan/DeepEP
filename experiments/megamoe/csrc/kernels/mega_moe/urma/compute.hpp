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

// Only an AIC and its paired AIV0 enter this routine. They enumerate precisely
// the same global tile IDs, including empty ownership, preserving CATLASS's
// cross-core weight-prologue/L1 event protocol without involving AIV1.
__aicore__ inline void RunCubeAndPrologue(const Arguments& args, uint32_t block, Timer& timer)
{
    const MegaMoeImpl::BlockJobContext job{block, kAic};
    // URMA updates receive GM outside this core's cache hierarchy. Keep input
    // and scale reads uncached across repeated invocations; weights retain
    // their original cache policy and every existing MTE caller defaults off.
    MegaMoeImpl::GmmKernel::CatlassGmm1<UrmaProblemConfig, true> gmm1;
    for (uint32_t fragmentIndex = 0U; fragmentIndex < args.config.fragmentCount; ++fragmentIndex) {
        const Fragment fragment = LoadFragment(args, fragmentIndex);
        const auto shape = FragmentShape(args, fragment);
        auto config1 = UrmaGmmConfig::BuildGmm1ProblemConfig(shape, job, kComputeTileM);
        MegaMoeImpl::GmmKernel::BlockScheduler schedule1(
            {config1.m, config1.schedulerN, config1.k},
            MegaMoeImpl::GmmKernel::BlockScheduler::Params{
                Te::MakeCoord(static_cast<int64_t>(config1.tileM), static_cast<int64_t>(config1.tileN))});
        const uint32_t count1 = schedule1.GetTileNum();
        const uint32_t first1 = FirstOwnedComputeTile(fragment.gmm1TileBegin, block, kAic);
        if (first1 < count1) {
            if constexpr (g_coreType == AIC) {
                WaitLocalDone(reinterpret_cast<__gm__ uint8_t*>(args.symmetric + args.config.dispatchReadyOffset +
                                                                fragment.inputReadyIndex * kFlagBytes),
                              args.config.epoch);
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
                if constexpr (g_coreType == AIC) {
                    gmm1.Compute(m, n, rows, columns);
                    SetFlag<HardEvent::FIX_S>(EVENT_ID0);
                    WaitFlag<HardEvent::FIX_S>(EVENT_ID0);
                    WriteGmByPassDCache(ComputeFlag(args.gmm1Done, fragment.gmm1TileBegin + tile), 1);
                } else {
                    gmm1.Dequantize(n, rows, columns);
                }
            }
            gmm1.End(g_coreType == AIC);
            timer.End(Phase::Gmm1);
        }

        auto config2 = UrmaGmmConfig::BuildGmm2ProblemConfig(shape, job, kComputeTileM);
        MegaMoeImpl::GmmKernel::BlockScheduler schedule2(
            {config2.m, config2.schedulerN, config2.k},
            MegaMoeImpl::GmmKernel::BlockScheduler::Params{
                Te::MakeCoord(static_cast<int64_t>(config2.tileM), static_cast<int64_t>(config2.tileN))});
        const uint32_t count2 = schedule2.GetTileNum();
        const uint32_t first2 = FirstOwnedComputeTile(fragment.gmm2TileBegin, block, kAic);
        if (first2 < count2) {
            if constexpr (g_coreType == AIC) {
                // The 13 independent activation workers collectively publish
                // one contribution per GMM1 tile after both FP8 planes drain.
                WaitComputeFlag(args.activationDone, fragmentIndex, count1);
            }
            timer.Begin(Phase::Gmm2);
            uint32_t completed = 0U;
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
                    if constexpr (g_coreType == AIC) {
                        gmm2.Compute(m, n, rows, columns);
                    } else {
                        gmm2.Dequantize(n, rows, columns);
                    }
                    ++completed;
                }
            }  // CATLASS destructor drains FIX and the paired L1 event tokens.
            if constexpr (g_coreType == AIC) {
                AtomicAdd(ComputeFlag(args.gmm2Done, fragmentIndex), static_cast<int32_t>(completed));
            }
            timer.End(Phase::Gmm2);
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
            WaitComputeFlag(args.gmm1Done, fragment.gmm1TileBegin + tile, 1U);
            timer.Begin(Phase::Activation);
            const auto coord = scheduler.GetBlockCoord(tile);
            const auto tileShape = scheduler.GetBlockShape(coord);
            const uint32_t m = Get<MegaMoeImpl::M_VALUE>(coord);
            const uint32_t n = Get<MegaMoeImpl::N_VALUE>(coord);
            const uint32_t rows = Get<MegaMoeImpl::M_VALUE>(tileShape);
            const uint32_t columns = Get<MegaMoeImpl::N_VALUE>(tileShape);
            for (uint32_t offset = 0U; offset < rows; offset += kActivationTileM) {
                const auto slice = ActivationSlice(m, rows, offset);
                const MegaMoeImpl::GmmKernel::BlockScheduler::BlockShape sliceShape{slice.rows, columns, config.k};
                MegaMoeImpl::GmmKernel::Gmm1Aiv1EpilogueTileA8W4<bfloat16_t, UrmaGmmConfig::MakeLayoutC, true>(
                    epilogue, gmC, addr, config, sliceShape, slice.rowBegin, n);
            }
            // Each sub-slice drains MTE3 before reusing UB. Publish exactly one
            // contribution for the original GMM1 tile, after all its rows.
            AtomicAdd(ComputeFlag(args.activationDone, fragmentIndex), 1);
            timer.End(Phase::Activation);
        }
    }
}

}  // namespace MegaMoeUrma
