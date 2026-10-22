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

// A fragment is a compute M256 group of one expert. Row segments preserve its
// input/output correspondence with the original source-major communication CVs.
__aicore__ inline Fragment LoadFragment(const Arguments& args, uint32_t index)
{
    const auto* gm = reinterpret_cast<__gm__ Fragment*>(args.fragments) + index;
    return {gm->expert,       gm->segmentBegin,  gm->rowBegin,     gm->rows,
            gm->segmentCount, gm->gmm1TileBegin, gm->gmm2TileBegin};
}

__aicore__ inline RowSegment LoadRowSegment(const Arguments& args, uint32_t index)
{
    const auto* gm = reinterpret_cast<__gm__ RowSegment*>(args.segments) + index;
    return {gm->sourceRank, gm->cv,   gm->receiveRow,      gm->returnRow,
            gm->tileRow,    gm->rows, gm->inputReadyIndex, gm->reserved};
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
    // X is gathered directly from source-major receive rows. MX scale rows
    // have been packed once by AIV1 to preserve the normal aligned L1 loader.
    addr.aGlobal = reinterpret_cast<GM_ADDR>(args.symmetric + args.config.receiveXOffset);
    addr.aScaleGlobal = reinterpret_cast<GM_ADDR>(args.gatheredInputScale + fragment.rowBegin * scaleH);
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
    // Segmented row IO scatters remote rows into this original source-major
    // output plane and self rows directly into their final symmetric return.
    addr.gmm2OutGlobal = reinterpret_cast<GM_ADDR>(args.gmm2Output);
    return addr;
}

using UrmaGmmConfig = MegaMoeImpl::GmmKernel::Config<true, 0, fp8_e4m3fn_t, fp4x2_e2m1_t, bfloat16_t, fp8_e8m0_t,
                                                     fp8_e8m0_t, false, false, false, false, false, false, false>;
using UrmaProblemConfig = UrmaGmmConfig::ProblemConfig;
using UrmaInputRows = MegaMoeImpl::GmmKernel::SegmentedRowIO<RowSegment, true, false>;
using UrmaOutputRows = MegaMoeImpl::GmmKernel::SegmentedRowIO<RowSegment, false, true>;
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

template <typename RowIO>
__aicore__ inline RowIO FragmentRowIO(const Arguments& args, const Fragment& fragment)
{
    return {reinterpret_cast<const __gm__ RowSegment*>(args.segments) + fragment.segmentBegin,
            fragment.segmentCount,
            args.config.rank,
            reinterpret_cast<GM_ADDR>(args.symmetric + args.config.receiveXOffset),
            reinterpret_cast<GM_ADDR>(args.symmetric + args.config.receiveScaleOffset),
            args.config.hidden,
            args.config.hidden / 32U,
            reinterpret_cast<GM_ADDR>(args.gmm2Output),
            reinterpret_cast<GM_ADDR>(args.symmetric + args.config.returnYOffset),
            args.config.hidden};
}

__aicore__ inline MegaMoeImpl::ProblemShape FragmentShape(const Arguments& args, const Fragment& fragment)
{
    return {fragment.rows, static_cast<int64_t>(args.config.intermediate) * 2, args.config.hidden, 0};
}

__aicore__ inline TraceMeta FragmentTraceMeta(const Fragment& fragment, uint32_t index)
{
    TraceMeta meta;
    // A compute group may span several peers/CVs. Its aggregate input-ready
    // wait uses the fragment as the stable compute-group ID.
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

using UrmaGmm1 = MegaMoeImpl::GmmKernel::CatlassGmm1<UrmaProblemConfig, false, UrmaInputRows>;
using UrmaGmm2 = MegaMoeImpl::GmmKernel::CatlassGmm2<UrmaProblemConfig, false, UrmaOutputRows>;

__aicore__ inline void RunGmm1Fragment(const Arguments& args, uint32_t block, uint32_t fragmentIndex, UrmaGmm1& gmm1,
                                       bool endContext, ComputeBatchTimer& batchTimer)
{
    const MegaMoeImpl::BlockJobContext job{block, kAic};
    const Fragment fragment = LoadFragment(args, fragmentIndex);
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
        // The AIV1 gather owner checks every segment's current dispatch epoch,
        // then drains the compact scale stores before publishing this flag.
        // It therefore covers both X arrival and scale readiness; rereading
        // every segment here only repeats that fan-in on each owning AIC.
        // Forward clears inputScaleDone on the launch stream every invocation.
        WaitComputeFlag(args.inputScaleDone, fragmentIndex, 1U);
    }
    batchTimer.BeginOnce();
    const auto addr = Gmm1Addresses(args, fragment);
    gmm1.UpdateProblem(config1, addr);
    gmm1.SetRowIO(FragmentRowIO<UrmaInputRows>(args, fragment));
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
            // Required for publishing gmm1Done to SwiGLU, independent of timer.
            SetFlag<HardEvent::FIX_S>(EVENT_ID0);
            WaitFlag<HardEvent::FIX_S>(EVENT_ID0);
            WriteGmByPassDCache(ComputeFlag(args.gmm1Done, fragment.gmm1TileBegin + tile), 1);
        } else {
            gmm1.Dequantize(n, rows, columns);
        }
    }
    if (endContext) {
        gmm1.End(g_coreType == AIC);
        CompleteComputeContext();
    }
}

__aicore__ inline void RunGmm2Fragment(const Arguments& args, uint32_t block, uint32_t fragmentIndex, UrmaGmm2& gmm2,
                                       uint32_t batchBegin, bool endContext, ComputeBatchTimer& batchTimer)
{
    const MegaMoeImpl::BlockJobContext job{block, kAic};
    const Fragment fragment = LoadFragment(args, fragmentIndex);
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
        WaitComputeFlag(args.activationDone, fragmentIndex, FragmentGmm1Tiles(args, fragment));
    }
    batchTimer.BeginOnce();
    const auto addr = Gmm2Addresses(args, fragment);
    gmm2.UpdateProblem(config2, addr);
    gmm2.SetRowIO(FragmentRowIO<UrmaOutputRows>(args, fragment));
    gmm2.Begin();
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
    }
    if (endContext) {
        // One FIX drain per expert batch. Intermediate M256 groups keep
        // submitting; the block's FIX_M events still protect each L0C slot.
        gmm2.End();
        CompleteComputeContext();
        if constexpr (g_coreType == AIC) {
            // Every group (including a short M tail) has the same N tile
            // count. Recover each group's contribution without rereading GM
            // descriptors. Publish only after ALL this core's stores drain.
            const uint64_t batchTileBegin =
                fragment.gmm2TileBegin - static_cast<uint64_t>(fragmentIndex - batchBegin) * count2;
            for (uint32_t index = batchBegin; index <= fragmentIndex; ++index) {
                const uint64_t tileBegin = batchTileBegin + static_cast<uint64_t>(index - batchBegin) * count2;
                const uint32_t first = FirstOwnedComputeTile(tileBegin, block, kAic);
                if (first < count2) {
                    const uint32_t completed = (count2 - 1U - first) / kAic + 1U;
                    AtomicAdd(ComputeFlag(args.gmm2Done, index), static_cast<int32_t>(completed));
                }
            }
        }
    }
}

__aicore__ inline ComputeBatchRange LoadComputeBatch(const Arguments& args, uint32_t index)
{
    const auto* gm = reinterpret_cast<__gm__ ComputeBatch*>(args.computeBatches) + index;
    const ComputeBatch batch{gm->fragmentBegin, gm->fragmentCount};
    return ComputeBatchFragments(batch);
}

__aicore__ inline ComputeBatchTask LoadComputeTask(const Arguments& args, uint32_t index)
{
    const auto* gm = reinterpret_cast<__gm__ ComputeBatchTask*>(args.computeTasks) + index;
    return {gm->stage, gm->batchIndex};
}

// Only an AIC and its paired AIV0 enter this routine. They use the identical
// host-planned expert schedule and global tile ownership. SwiGLU remains
// independent; the GMM1 M-group window covers earlier activation latency.
__aicore__ inline void RunCubeAndPrologue(const Arguments& args, uint32_t block, Timer& timer)
{
    // Only one stage owns the shared L1/L0/prologue buffers at a time. The
    // objects keep their views; each expert batch acquires/releases events.
    UrmaGmm1 gmm1;
    UrmaGmm2 gmm2;
    for (uint32_t step = 0U; step < args.config.computeTaskCount; ++step) {
        const auto task = LoadComputeTask(args, step);
        const auto range = LoadComputeBatch(args, task.batchIndex);
        if (range.begin == range.end) continue;
        const auto first = LoadFragment(args, range.begin);
        const bool isGmm1 = task.stage == ComputeBatchStage::Gmm1;
        const uint64_t tileBegin = isGmm1 ? first.gmm1TileBegin : first.gmm2TileBegin;
        const uint32_t tilesPerGroup = isGmm1 ? FragmentGmm1Tiles(args, first) : FragmentGmm2Tiles(args, first);
        const uint32_t lastOwned =
            LastOwnedComputeGroup(tileBegin, range.end - range.begin, tilesPerGroup, block, kAic);
        if (lastOwned == range.end - range.begin) continue;
        const uint32_t lastFragment = range.begin + lastOwned;
        const auto event = g_coreType == AIC ? (isGmm1 ? TraceEvent::Gmm1Batch : TraceEvent::Gmm2Batch)
                                             : (isGmm1 ? TraceEvent::Gmm1DequantBatch : TraceEvent::Gmm2DequantBatch);
        ComputeBatchTimer batchTimer(timer, isGmm1 ? Phase::Gmm1 : Phase::Gmm2, event, task.batchIndex, first.expert);
        // A core can skip groups between its own tiles. Both paired engines
        // close on their last owned group, not necessarily the expert's last.
        for (uint32_t fragmentIndex = range.begin; fragmentIndex <= lastFragment; ++fragmentIndex) {
            if (task.stage == ComputeBatchStage::Gmm1) {
                RunGmm1Fragment(args, block, fragmentIndex, gmm1, fragmentIndex == lastFragment, batchTimer);
            } else {
                RunGmm2Fragment(args, block, fragmentIndex, gmm2, range.begin, fragmentIndex == lastFragment,
                                batchTimer);
            }
        }
        batchTimer.End();
    }
}

__aicore__ inline bool AreGroupInputsReady(const Arguments& args, const Fragment& fragment)
{
    for (uint32_t index = 0U; index < fragment.segmentCount; ++index) {
        const auto segment = LoadRowSegment(args, fragment.segmentBegin + index);
        auto* ready = reinterpret_cast<__gm__ uint64_t*>(args.symmetric + args.config.dispatchReadyOffset +
                                                         segment.inputReadyIndex * kFlagBytes);
        if (ReadGmByPassDCache(ready) != args.config.epoch) return false;
    }
    return true;
}

// A single worker owns a whole group's scales. Arbitrary segment row offsets
// cannot be represented safely by the MX zZ L1 copy, so gather just this small
// plane in aligned UB rows, then store compact GM rows for the original loader.
__aicore__ inline void GatherInputScale(const Arguments& args, const Fragment& fragment, uint32_t index, Timer& timer)
{
    GlobalTensor<uint8_t> source, destination;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.symmetric + args.config.receiveScaleOffset));
    source.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.gatheredInputScale));
    LocalTensor<uint8_t> scratch(TPosition::VECCALC, 0U, kInputScaleUbBytes);
    const uint64_t start = timer.TraceBegin();
    for (uint32_t segmentIndex = 0U; segmentIndex < fragment.segmentCount; ++segmentIndex) {
        const auto segment = LoadRowSegment(args, fragment.segmentBegin + segmentIndex);
        DataCopyPad(scratch[segment.tileRow * kInputScaleUbStride], source[segment.receiveRow * kInputScaleBytes],
                    {static_cast<uint16_t>(segment.rows), kInputScaleBytes, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
    }
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    DataCopyPad(destination[fragment.rowBegin * kInputScaleBytes], scratch,
                {static_cast<uint16_t>(fragment.rows), kInputScaleBytes, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WriteGmByPassDCache(ComputeFlag(args.inputScaleDone, index), 1);
    auto meta = FragmentTraceMeta(fragment, index);
    meta.bytes = static_cast<uint64_t>(fragment.rows) * kInputScaleBytes;
    timer.TraceEnd(TraceEvent::InputScaleGather, start, meta);
}

__aicore__ inline void RunReadyActivationTile(const Arguments& args, uint32_t worker, const Fragment& fragment,
                                              uint32_t fragmentIndex, uint32_t tile, UrmaActivation& epilogue,
                                              Timer& timer)
{
    const MegaMoeImpl::BlockJobContext job{worker, kActivationWorkers};
    const auto fragmentMeta = FragmentTraceMeta(fragment, fragmentIndex);
    auto addr = Gmm1Addresses(args, fragment);
    addr.activationToGmm2Flag = nullptr;
    epilogue.UpdateGlobalAddr({static_cast<int64_t>(fragment.rowBegin * args.config.intermediate),
                               static_cast<int64_t>(fragment.rowBegin * (args.config.intermediate / 32U)), 0, 0, 0, 0});
    auto config = UrmaGmmConfig::BuildGmm1ProblemConfig(FragmentShape(args, fragment), job, kComputeTileM);
    const auto layouts = UrmaGmmConfig::BuildLayouts(config);
    auto gmC = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ bfloat16_t*>(addr.gmm1OutGlobal)), layouts.c);
    MegaMoeImpl::GmmKernel::BlockScheduler scheduler(
        {config.m, config.outputN, config.k},
        MegaMoeImpl::GmmKernel::BlockScheduler::Params{
            Te::MakeCoord(static_cast<int64_t>(kComputeTileM), static_cast<int64_t>(kComputeTileN))});
    epilogue.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    const auto coord = scheduler.GetBlockCoord(tile);
    const auto tileShape = scheduler.GetBlockShape(coord);
    const uint32_t m = Get<MegaMoeImpl::M_VALUE>(coord);
    const uint32_t n = Get<MegaMoeImpl::N_VALUE>(coord);
    const uint32_t rows = Get<MegaMoeImpl::M_VALUE>(tileShape);
    const uint32_t columns = Get<MegaMoeImpl::N_VALUE>(tileShape);
    // Only the staging copy changes to M128; ready counters stay M256.
    config.tileM = kActivationTileM;
    timer.Begin(Phase::Activation);
    for (uint32_t offset = 0U; offset < rows; offset += kActivationTileM) {
        const auto slice = ActivationSlice(m, rows, offset);
        const MegaMoeImpl::GmmKernel::BlockScheduler::BlockShape sliceShape{slice.rows, columns, config.k};
        const auto sliceMeta = ComputeTileTraceMeta(fragmentMeta, slice.rowBegin, n, slice.rows, columns);
        const uint64_t sliceStart = timer.TraceBegin();
        MegaMoeImpl::GmmKernel::Gmm1Aiv1EpilogueTileA8W4<bfloat16_t, UrmaGmmConfig::MakeLayoutC, true>(
            epilogue, gmC, addr, config, sliceShape, slice.rowBegin, n);
        timer.TraceEnd(TraceEvent::ActivationSlice, sliceStart, sliceMeta);
    }
    // The epilogue has drained all output DMA before the gather task can reuse UB.
    AtomicAdd(ComputeFlag(args.activationDone, fragmentIndex), 1);
    timer.End(Phase::Activation);
}

// Each AIV1 owns independent cursors for input-scale groups and activation
// tiles. Never block one cursor on a producer: GMM1 needs future scale gathers
// even while earlier activations are pending on the very same worker.
__aicore__ inline void RunActivation(const Arguments& args, uint32_t worker, Timer& timer)
{
    UrmaActivation epilogue;
    epilogue.Init({.yGmAddr = reinterpret_cast<GM_ADDR>(args.activation),
                   .yScaleGmAddr = reinterpret_cast<GM_ADDR>(args.activationScale)});
    uint32_t nextGather = worker;
    uint32_t nextActivation = 0U;
    uint32_t activationTile = 0U;
    bool activationPrepared = false;
    bool idle = false;
    uint64_t idleStart = 0U;
    TraceEvent idleEvent = TraceEvent::WaitGmm1;
    TraceMeta idleMeta;
    while (nextGather < args.config.fragmentCount || nextActivation < args.config.fragmentCount) {
        bool progress = false;
        if (nextGather < args.config.fragmentCount) {
            const auto fragment = LoadFragment(args, nextGather);
            if (AreGroupInputsReady(args, fragment)) {
                if (idle) {
                    timer.TraceEnd(idleEvent, idleStart, idleMeta);
                    idle = false;
                }
                GatherInputScale(args, fragment, nextGather, timer);
                nextGather += kActivationWorkers;
                progress = true;
            }
        }
        // Skip groups without this worker's activation tile, but never skip a
        // scale gather merely because its owner has no activation in that group.
        while (nextActivation < args.config.fragmentCount) {
            const auto fragment = LoadFragment(args, nextActivation);
            const uint32_t tileCount = FragmentGmm1Tiles(args, fragment);
            if (!activationPrepared) {
                activationTile = FirstOwnedComputeTile(fragment.gmm1TileBegin, worker, kActivationWorkers);
                activationPrepared = true;
            }
            if (activationTile >= tileCount) {
                ++nextActivation;
                activationPrepared = false;
                continue;
            }
            if (ReadGmByPassDCache(ComputeFlag(args.gmm1Done, fragment.gmm1TileBegin + activationTile)) != 0) {
                if (idle) {
                    timer.TraceEnd(idleEvent, idleStart, idleMeta);
                    idle = false;
                }
                RunReadyActivationTile(args, worker, fragment, nextActivation, activationTile, epilogue, timer);
                activationTile += kActivationWorkers;
                progress = true;
            }
            break;
        }
        if (!progress && (nextGather < args.config.fragmentCount || nextActivation < args.config.fragmentCount)) {
            if (!idle) {
                idleStart = timer.TraceBegin();
                idle = true;
                const bool waitingActivation = nextActivation < args.config.fragmentCount;
                const uint32_t index = waitingActivation ? nextActivation : nextGather;
                idleEvent = waitingActivation ? TraceEvent::WaitGmm1 : TraceEvent::WaitDispatch;
                idleMeta = FragmentTraceMeta(LoadFragment(args, index), index);
            }
            const uint64_t backoff = GetSystemCycle();
            while (GetSystemCycle() - backoff < 64U) {
            }
        }
    }
    if (idle) timer.TraceEnd(idleEvent, idleStart, idleMeta);
}

}  // namespace MegaMoeUrma
