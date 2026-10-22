// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"
#include "compute.hpp"
#include "pack_reduce.hpp"
#include "pack_round.hpp"
#include "tail_completion.hpp"
#include "timer.hpp"
#include "transport.hpp"

namespace MegaMoeUrma {
using namespace AscendC;

__aicore__ inline void TraceLocalSync(Timer& timer, uint32_t stage)
{
    TraceMeta meta;
    meta.cv = stage;  // 0=two-round startup pack, 3=local invocation completion.
    const uint64_t start = timer.TraceBegin();
    SyncAll<false>();
    timer.TraceEnd(TraceEvent::LocalSync, start, meta);
}

// Publish before packing so remote startup can overlap local DMA. Do not
// wait here: all workers must reach the two-round startup join. Previous-call
// receive/return consumers finished at the prior local join, and this call's
// counter resets precede the kernel on the same stream. Self pack and remote
// PUTs address disjoint source intervals in receive storage.
__aicore__ inline void PublishPeerStart(const Arguments& args, uint32_t owner, Timer& timer)
{
    const uint32_t peer = PeerForOwner(args.config.rank, owner);
    LocalTensor<uint64_t> signal(TPosition::VECCALC, 0U, 4U);
    TraceMeta meta;
    meta.peer = peer;
    timer.Begin(Phase::Barrier);
    const uint64_t start = timer.TraceBegin();
    PublishDone(reinterpret_cast<__gm__ uint8_t*>(args.symmetric), PeerPhaseSlot(args, 0U, args.config.rank), peer,
                args.config.epoch, signal, EVENT_ID0);
    timer.TraceEnd(TraceEvent::PublishStart, start, meta);
    timer.End(Phase::Barrier);
}

__aicore__ inline void WaitChunkComputed(const Arguments& args, const Chunk& chunk, Timer& timer)
{
    for (uint32_t j = 0U; j < chunk.fragmentCount; ++j) {
        // A communication CV intersects arbitrary expert-major compute groups.
        // Host deduplicates this adjacency list; each group counter covers its
        // complete GMM2 N direction after every segmented FIX store completes.
        const uint32_t index = reinterpret_cast<__gm__ uint32_t*>(args.chunkGroupIndices)[chunk.fragmentBegin + j];
        const Fragment fragment = LoadFragment(args, index);
        TraceMeta meta = ChunkTraceMeta(chunk);
        meta.fragment = index;
        meta.expert = fragment.expert;
        meta.row = fragment.rowBegin;
        meta.rows = fragment.rows;
        const uint64_t start = timer.TraceBegin();
        WaitComputeFlag(args.gmm2Done, index, FragmentGmm2Tiles(args, fragment));
        timer.TraceEnd(TraceEvent::WaitCombineReady, start, meta);
    }
}

// Original owners keep QP0, all Dispatch, and the Combine suffix for cross
// peers (all CVs for intra peers). Helpers own disjoint CV prefixes on QP1;
// queue tails and CQ heads still have one writer each.
__aicore__ inline void CommunicatePeer(const Arguments& args, uint32_t owner, Timer& timer)
{
    const uint32_t peer = PeerForOwner(args.config.rank, owner);
    auto* symmetric = reinterpret_cast<__gm__ uint8_t*>(args.symmetric);
    LocalTensor<uint64_t> signal(TPosition::VECCALC, 0U, 4U);
    TraceMeta peerMeta;
    peerMeta.peer = peer;
    bool packFinished = args.config.packRoundCount <= kPackStartupRounds;
    if (packFinished) MarkPackWorkerComplete(args);
    timer.Begin(Phase::Barrier);
    // Start was published before the startup Pack. Wait only now, immediately
    // before this owner's first possible payload write to the peer.
    uint64_t start = timer.TraceBegin();
    WaitStartAtLeast(symmetric + PeerPhaseSlot(args, 0U, peer), args.config.epoch);
    timer.TraceEnd(TraceEvent::WaitStartPeer, start, peerMeta);
    timer.End(Phase::Barrier);

    const uint64_t h = args.config.hidden;
    const uint64_t scales = h / 32U;
    for (uint32_t round = 0U; round < args.config.packRoundCount; ++round) {
        const Chunk chunk = LoadPackChunk(args, round, peer);
        // An empty peer must still produce its fixed share of every Pack
        // round, including data for other peers and self. Never exit early.
        if (chunk.rows == 0U) {
            PackOneRound(args, round + kPackStartupRounds, owner, kAic, timer);
            if (!packFinished && round + kPackStartupRounds + 1U >= args.config.packRoundCount) {
                MarkPackWorkerComplete(args);
                packFinished = true;
            }
            continue;
        }
        TraceMeta meta = ChunkTraceMeta(chunk);
        start = timer.TraceBegin();
        while (ReadGmByPassDCache(PackReadyFlag(args, peer, round)) < static_cast<int32_t>(chunk.rows)) {
            const uint64_t backoff = GetSystemCycle();
            while (GetSystemCycle() - backoff < 64U) {
            }
        }
        timer.TraceEnd(TraceEvent::WaitDispatch, start, meta);
        meta.bytes = chunk.rows * h;
        timer.Begin(Phase::Dispatch);
        start = timer.TraceBegin();
        Put(symmetric, args.config.receiveXOffset + chunk.remoteRow * h,
            reinterpret_cast<__gm__ uint8_t*>(args.expandedX + chunk.localRow * h),
            static_cast<uint32_t>(chunk.rows * h), peer);
        timer.TraceEnd(TraceEvent::DispatchPutX, start, meta);
        meta.bytes = chunk.rows * scales;
        start = timer.TraceBegin();
        Put(symmetric, args.config.receiveScaleOffset + chunk.remoteRow * scales,
            reinterpret_cast<__gm__ uint8_t*>(args.expandedScale + chunk.localRow * scales),
            static_cast<uint32_t>(chunk.rows * scales), peer);
        timer.TraceEnd(TraceEvent::DispatchPutScale, start, meta);
        // The current URMA write progresses while MTE prepares this worker's
        // fixed quota two CVs ahead. No claims, extra work stealing or phase join.
        PackOneRound(args, round + kPackStartupRounds, owner, kAic, timer);
        // Publish immediately after this worker's final Pack quota, before
        // quiet and the final Dispatch rounds, so QP1 can overlap that tail.
        if (!packFinished && round + kPackStartupRounds + 1U >= args.config.packRoundCount) {
            MarkPackWorkerComplete(args);
            packFinished = true;
        }
        meta.bytes = 0U;
        start = timer.TraceBegin();
        Quiet(peer);
        timer.TraceEnd(TraceEvent::DispatchQuiet, start, meta);
        const uint64_t readyIndex = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + chunk.cv;
        start = timer.TraceBegin();
        PublishDone(symmetric, args.config.dispatchReadyOffset + readyIndex * kFlagBytes, peer, args.config.epoch,
                    signal, EVENT_ID0);
        timer.TraceEnd(TraceEvent::DispatchPublish, start, meta);
        timer.End(Phase::Dispatch, chunk.rows * (h + scales), peer);
    }
    // Empty peers must still join the local dispatch phase. Owner 0 accounts
    // self only when every steady Pack worker has finished, including helpers.
    AtomicAdd(ComputeFlag(args.localFlags, kLocalFlagDispatchDone), 1);
    start = timer.TraceBegin();
    if (owner == 0U) {
        WaitComputeFlag(args.localFlags, kLocalFlagPackDone, kAic);
        AtomicAdd(ComputeFlag(args.localFlags, kLocalFlagDispatchDone), 1);
    }
    WaitComputeFlag(args.localFlags, kLocalFlagDispatchDone, kWorld);
    timer.TraceEnd(TraceEvent::WaitLocalDispatch, start, peerMeta);

    const uint64_t rowBytes = h * sizeof(bfloat16_t);
    uint64_t combineBytes = 0U;
    bool combineStarted = false;
    bool peerCounted = false;
    bool selfCounted = false;
    const uint32_t helper = CrossHelperForPeer(args.config.rank, peer);
    const auto range = FindCombinePeerRange(args, peer);
    const uint32_t begin = helper < kCombineHelpers ? range.middle : range.begin;
    for (uint32_t index = begin; index < range.end; ++index) {
        const Chunk chunk = LoadChunk(args.recvChunks, index);
        WaitChunkComputed(args, chunk, timer);
        if (!combineStarted) {
            timer.Begin(Phase::Combine);
            combineStarted = true;
        }
        TraceMeta meta = ChunkTraceMeta(chunk);
        meta.bytes = chunk.rows * rowBytes;
        start = timer.TraceBegin();
        Put(symmetric, args.config.returnYOffset + chunk.remoteRow * rowBytes,
            reinterpret_cast<__gm__ uint8_t*>(args.gmm2Output + chunk.localRow * rowBytes),
            static_cast<uint32_t>(chunk.rows * rowBytes), peer);
        timer.TraceEnd(TraceEvent::CombinePut, start, meta);
        // Opportunistically account this peer's incoming completion without
        // waiting. The PUT is already submitted and can progress meanwhile.
        ObserveIncomingCompletion(args, owner, peerCounted, selfCounted);
        meta.bytes = 0U;
        start = timer.TraceBegin();
        Quiet(peer);
        timer.TraceEnd(TraceEvent::CombineQuiet, start, meta);
        const uint64_t readyIndex = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + chunk.cv;
        start = timer.TraceBegin();
        PublishDone(symmetric, args.config.combineReadyOffset + readyIndex * kFlagBytes, peer, args.config.epoch,
                    signal, EVENT_ID0);
        timer.TraceEnd(TraceEvent::CombinePublish, start, meta);
        combineBytes += chunk.rows * rowBytes;
    }
    // Every CV now has its own completion flag, so the source can reduce
    // completed tokens while other peers are still computing/returning rows.
    // The aggregate envelope includes producer gaps between CVs.
    if (combineStarted) timer.End(Phase::Combine, combineBytes, peer);
    if (helper < kCombineHelpers) {
        start = timer.TraceBegin();
        while (ReadGmByPassDCache(LocalCompletionFlag(args, kLocalFlagHelperDoneBase + helper)) == 0) {
            ObserveIncomingCompletion(args, owner, peerCounted, selfCounted);
            const uint64_t backoff = GetSystemCycle();
            while (GetSystemCycle() - backoff < 200U) {
            }
        }
        timer.TraceEnd(TraceEvent::WaitCombineReady, start, peerMeta);
    }
    // The helper prefix and our suffix both completed quiet and CV-ready.
    // Only this original owner publishes whole-peer finished and counts inbound.
    // Once all sends finish, this owner only monitors its remaining inbound
    // contribution (plus self on owner 0). No per-token monitor or rank barrier.
    FinishCombinePeer(args, owner, signal, peerCounted, selfCounted);
}

__aicore__ inline void Process(const Arguments& args)
{
    Timer timer;
    timer.Init(args.timer, args.trace, args.traceCapacity);
    const uint32_t block = GetBlockIdx() / GetTaskRation();
    const int64_t originalOverflow = GetCtrlSpr<MegaMoeImpl::OVERFLOW_MODE_CTRL, MegaMoeImpl::OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<MegaMoeImpl::OVERFLOW_MODE_CTRL, MegaMoeImpl::OVERFLOW_MODE_CTRL>(0);

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U && block < kCommunicationWorkers) {
            PublishPeerStart(args, block, timer);
        }
        const uint32_t worker = LocalWorkerIndex(block, GetSubBlockIdx());
        for (uint32_t round = 0U; round < kPackStartupRounds; ++round) {
            PackOneRound(args, round, worker, kLocalWorkers, timer);
        }
    }
    // No CV loop contains a rank-wide or local full-core synchronization.
    TraceLocalSync(timer, 0U);
    if constexpr (g_coreType == AIC) {
        RunCubeAndPrologue(args, block, timer);
        MarkCubeComplete(args);
    } else if (GetSubBlockIdx() == 0U) {
        RunCubeAndPrologue(args, block, timer);
    } else if (block < kCommunicationWorkers) {
        CommunicatePeer(args, block, timer);
    } else {
        RunActivation(args, block - kCommunicationWorkers, timer);
    }
    if constexpr (g_coreType == AIV) {
        // Only this engine releases its phase-private UB/events. Other
        // compute and communication engines may still be producing rows.
        PipeBarrier<PIPE_ALL>();
        const uint32_t worker = ReduceWorkerIndex(block, GetSubBlockIdx());
        if (worker < kReduceWorkers) {
            timer.Begin(Phase::ReducePhase);
            Reduce(args, worker, timer);
            const uint64_t owned = ReduceOwnedRows(args.config.bs, worker);
            timer.End(Phase::ReducePhase, owned * args.config.hidden * sizeof(bfloat16_t));
            PipeBarrier<PIPE_ALL>();
        }
    }
    // A local MIX join, not a rank barrier: every reducer, peer owner and AIC
    // must finish before returning and reusing this invocation's workspaces.
    TraceLocalSync(timer, 3U);
    timer.FinishTrace();
    SetCtrlSpr<MegaMoeImpl::OVERFLOW_MODE_CTRL, MegaMoeImpl::OVERFLOW_MODE_CTRL>(originalOverflow);
}

}  // namespace MegaMoeUrma
