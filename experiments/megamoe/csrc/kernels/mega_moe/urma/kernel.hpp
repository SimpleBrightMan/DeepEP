// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"
#include "compute.hpp"
#include "pack_reduce.hpp"
#include "timer.hpp"
#include "transport.hpp"

namespace MegaMoeUrma {
using namespace AscendC;

__aicore__ inline Chunk LoadChunk(uint64_t address, uint32_t index)
{
    const auto* gm = reinterpret_cast<__gm__ Chunk*>(address) + index;
    return {gm->peer,          gm->cv,      gm->localRow, gm->remoteRow, gm->rows, gm->fragmentBegin,
            gm->fragmentCount, gm->reserved};
}

__aicore__ inline TraceMeta ChunkTraceMeta(const Chunk& chunk)
{
    TraceMeta meta;
    meta.peer = chunk.peer;
    meta.cv = chunk.cv;
    meta.row = chunk.localRow;
    meta.rows = chunk.rows;
    return meta;
}

__aicore__ inline void TraceLocalSync(Timer& timer, uint32_t stage)
{
    TraceMeta meta;
    meta.cv = stage;  // 0=pack, 3=local invocation completion; 1/2 reserved.
    const uint64_t start = timer.TraceBegin();
    SyncAll<false>();
    timer.TraceEnd(TraceEvent::LocalSync, start, meta);
}

// Pack writes self rows directly into their final receive interval. It drains
// all AIV DMA before the local pack barrier; only then can this single publisher
// expose every self CV. This also supplies the one self dispatch contribution,
// including when this rank has no self routes.
__aicore__ inline void PublishSelfDispatchReady(const Arguments& args)
{
    for (uint64_t begin = 0U, cv = 0U; begin < args.config.selfRows; begin += args.config.cvRows, ++cv) {
        auto* ready = reinterpret_cast<__gm__ uint64_t*>(
            args.symmetric + args.config.dispatchReadyOffset +
            (static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + cv) * kFlagBytes);
        WriteGmByPassDCache(ready, args.config.epoch);
    }
    AtomicAdd(ComputeFlag(args.localFlags, 0U), 1);
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

__aicore__ inline uint64_t PeerPhaseSlot(const Arguments& args, uint32_t phase, uint32_t source)
{
    return args.config.barrierOffset + (static_cast<uint64_t>(phase) * kWorld + source) * kFlagBytes;
}

// One and only one AIV1 owns all PUT/quiet operations for this peer for the
// entire invocation. Dispatch and Combine share that owner's QP serially;
// no queue tail or CQ head has multiple writers.
__aicore__ inline void CommunicatePeer(const Arguments& args, uint32_t owner, Timer& timer)
{
    const uint32_t peer = PeerForOwner(args.config.rank, owner);
    auto* symmetric = reinterpret_cast<__gm__ uint8_t*>(args.symmetric);
    LocalTensor<uint64_t> signal(TPosition::VECCALC, 0U, 4U);
    TraceMeta peerMeta;
    peerMeta.peer = peer;
    timer.Begin(Phase::Barrier);
    // Start is sent only after all local Expand work and the preceding call's
    // Reduce have completed. A peer must acknowledge this epoch before any
    // payload overwrites its symmetric receive/return storage.
    uint64_t start = timer.TraceBegin();
    PublishDone(symmetric, PeerPhaseSlot(args, 0U, args.config.rank), peer, args.config.epoch, signal, EVENT_ID0);
    timer.TraceEnd(TraceEvent::PublishStart, start, peerMeta);
    start = timer.TraceBegin();
    WaitStartAtLeast(symmetric + PeerPhaseSlot(args, 0U, peer), args.config.epoch);
    timer.TraceEnd(TraceEvent::WaitStartPeer, start, peerMeta);
    timer.End(Phase::Barrier);

    const uint64_t h = args.config.hidden;
    const uint64_t scales = h / 32U;
    for (uint32_t index = 0U; index < args.config.sendChunkCount; ++index) {
        const Chunk chunk = LoadChunk(args.sendChunks, index);
        if (chunk.peer != peer) continue;
        TraceMeta meta = ChunkTraceMeta(chunk);
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
        meta.bytes = 0U;
        start = timer.TraceBegin();
        Quiet(peer);
        timer.TraceEnd(TraceEvent::DispatchQuiet, start, meta);
        const uint64_t readyIndex = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + chunk.cv;
        start = timer.TraceBegin();
        PublishDone(symmetric, args.config.dispatchReadyOffset + readyIndex * kFlagBytes, peer, args.config.epoch,
                    signal, EVENT_ID0);
        timer.TraceEnd(TraceEvent::DispatchPublish, start, meta);
        timer.End(Phase::Dispatch, chunk.rows * (h + scales));
    }
    // Empty peers must still join the local dispatch phase.
    AtomicAdd(ComputeFlag(args.localFlags, 0U), 1);
    start = timer.TraceBegin();
    WaitComputeFlag(args.localFlags, 0U, kWorld);
    timer.TraceEnd(TraceEvent::WaitLocalDispatch, start, peerMeta);

    const uint64_t rowBytes = h * sizeof(bfloat16_t);
    uint64_t combineBytes = 0U;
    bool combineStarted = false;
    for (uint32_t index = 0U; index < args.config.recvChunkCount; ++index) {
        const Chunk chunk = LoadChunk(args.recvChunks, index);
        if (chunk.peer != peer) continue;
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
    if (combineStarted) timer.End(Phase::Combine, combineBytes);
}

__aicore__ inline void Process(const Arguments& args)
{
    Timer timer;
    timer.Init(args.timer, args.trace, args.traceCapacity);
    const uint32_t block = GetBlockIdx() / GetTaskRation();
    const int64_t originalOverflow = GetCtrlSpr<MegaMoeImpl::OVERFLOW_MODE_CTRL, MegaMoeImpl::OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<MegaMoeImpl::OVERFLOW_MODE_CTRL, MegaMoeImpl::OVERFLOW_MODE_CTRL>(0);

    if constexpr (g_coreType == AIV) {
        const uint32_t worker = LocalWorkerIndex(block, GetSubBlockIdx());
        timer.Begin(Phase::Pack);
        const uint64_t start = timer.TraceBegin();
        Expand(args, worker);
        const uint64_t routes = static_cast<uint64_t>(args.config.bs) * args.config.topk;
        const uint64_t owned = LocalOwnedRows(routes, worker);
        TraceMeta meta;
        meta.rows = owned;
        meta.bytes = owned * (args.config.hidden + args.config.hidden / 32U);
        timer.TraceEnd(TraceEvent::Pack, start, meta);
        timer.End(Phase::Pack, owned * (args.config.hidden + args.config.hidden / 32U));
        PipeBarrier<PIPE_ALL>();
    }
    // No CV loop contains a rank-wide or local full-core synchronization.
    TraceLocalSync(timer, 0U);
    if constexpr (g_coreType == AIC) {
        RunCubeAndPrologue(args, block, timer);
    } else if (GetSubBlockIdx() == 0U) {
        RunCubeAndPrologue(args, block, timer);
    } else if (block < kCommunicationWorkers) {
        CommunicatePeer(args, block, timer);
    } else {
        if (block == kCommunicationWorkers) PublishSelfDispatchReady(args);
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
