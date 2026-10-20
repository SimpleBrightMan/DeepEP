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
    meta.cv = stage;  // 0=pack, 1=compute/return, 2=rank barrier, 3=reduce.
    const uint64_t start = timer.TraceBegin();
    SyncAll<false>();
    timer.TraceEnd(TraceEvent::LocalSync, start, meta);
}

// Self traffic never enters a remote queue. This UB is used only by AIV1
// block15 before/after its activation work, with complete DMA drains at the
// transition; it cannot alias the activation epilogue while live.
__aicore__ inline void CopyLocalPlane(uint64_t destination, uint64_t source, uint64_t bytes)
{
    constexpr uint32_t kCopyBytes = 16U * 1024U;
    LocalTensor<uint8_t> ub(TPosition::VECCALC, 0U, kCopyBytes);
    GlobalTensor<uint8_t> input, output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(source));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(destination));
    for (uint64_t offset = 0U; offset < bytes;) {
        const uint64_t remaining = bytes - offset;
        const uint32_t count = remaining < kCopyBytes ? static_cast<uint32_t>(remaining) : kCopyBytes;
        DataCopyPad(ub, input[offset], {1U, count, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopyPad(output[offset], ub, {1U, count, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        offset += count;
    }
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void SelfDispatch(const Arguments& args, Timer& timer)
{
    const uint64_t h = args.config.hidden;
    const uint64_t scales = h / 32U;
    for (uint32_t index = 0U; index < args.config.sendChunkCount; ++index) {
        const Chunk chunk = LoadChunk(args.sendChunks, index);
        if (chunk.peer != args.config.rank) continue;
        TraceMeta meta = ChunkTraceMeta(chunk);
        meta.bytes = chunk.rows * (h + scales);
        timer.Begin(Phase::Dispatch);
        const uint64_t start = timer.TraceBegin();
        CopyLocalPlane(args.symmetric + args.config.receiveXOffset + chunk.remoteRow * h,
                       args.expandedX + chunk.localRow * h, chunk.rows * h);
        CopyLocalPlane(args.symmetric + args.config.receiveScaleOffset + chunk.remoteRow * scales,
                       args.expandedScale + chunk.localRow * scales, chunk.rows * scales);
        auto* ready = reinterpret_cast<__gm__ uint64_t*>(
            args.symmetric + args.config.dispatchReadyOffset +
            (static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + chunk.cv) * kFlagBytes);
        WriteGmByPassDCache(ready, args.config.epoch);
        timer.TraceEnd(TraceEvent::SelfDispatch, start, meta);
        timer.End(Phase::Dispatch, chunk.rows * (h + scales));
    }
    AtomicAdd(ComputeFlag(args.localFlags, 0U), 1);
}

__aicore__ inline void WaitChunkComputed(const Arguments& args, const Chunk& chunk, Timer& timer)
{
    for (uint32_t j = 0U; j < chunk.fragmentCount; ++j) {
        const uint32_t index = chunk.fragmentBegin + j;
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

__aicore__ inline void SelfCombine(const Arguments& args, Timer& timer)
{
    // The self owner follows the same dispatch-all-before-combine contract.
    uint64_t start = timer.TraceBegin();
    WaitComputeFlag(args.localFlags, 0U, kWorld);
    timer.TraceEnd(TraceEvent::WaitLocalDispatch, start);
    const uint64_t rowBytes = static_cast<uint64_t>(args.config.hidden) * sizeof(bfloat16_t);
    for (uint32_t index = 0U; index < args.config.recvChunkCount; ++index) {
        const Chunk chunk = LoadChunk(args.recvChunks, index);
        if (chunk.peer != args.config.rank) continue;
        WaitChunkComputed(args, chunk, timer);
        TraceMeta meta = ChunkTraceMeta(chunk);
        meta.bytes = chunk.rows * rowBytes;
        timer.Begin(Phase::Combine);
        start = timer.TraceBegin();
        CopyLocalPlane(args.symmetric + args.config.returnYOffset + chunk.remoteRow * rowBytes,
                       args.gmm2Output + chunk.localRow * rowBytes, chunk.rows * rowBytes);
        timer.TraceEnd(TraceEvent::SelfCombine, start, meta);
        timer.End(Phase::Combine, chunk.rows * rowBytes);
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
    WaitLocalDone(symmetric + PeerPhaseSlot(args, 0U, peer), args.config.epoch);
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
    // Empty peers must still join the dispatch phase and final handshake.
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
        combineBytes += chunk.rows * rowBytes;
    }
    // GMM2 rows are never reused within this invocation. Keep the peer's PUT
    // queue in flight across chunks; only the final quiet releases the source
    // lifetime and certifies all returned rows. The recorded envelope includes
    // producer gaps between the first submission and this final completion.
    start = timer.TraceBegin();
    Quiet(peer);
    timer.TraceEnd(TraceEvent::CombineQuiet, start, peerMeta);
    if (combineStarted) timer.End(Phase::Combine, combineBytes);
    // Each source's completion is independent. The following local MIX
    // barrier joins all 15 incoming peers before any core runs Reduce.
    timer.Begin(Phase::Barrier);
    start = timer.TraceBegin();
    PublishDone(symmetric, PeerPhaseSlot(args, 1U, args.config.rank), peer, args.config.epoch, signal, EVENT_ID0);
    WaitLocalDone(symmetric + PeerPhaseSlot(args, 1U, peer), args.config.epoch);
    timer.TraceEnd(TraceEvent::FinalPeerHandshake, start, peerMeta);
    timer.End(Phase::Barrier);
}

__aicore__ inline void Process(const Arguments& args)
{
    Timer timer;
    timer.Init(args.timer, args.trace, args.traceCapacity);
    const uint32_t block = GetBlockIdx() / GetTaskRation();
    const int64_t originalOverflow = GetCtrlSpr<MegaMoeImpl::OVERFLOW_MODE_CTRL, MegaMoeImpl::OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<MegaMoeImpl::OVERFLOW_MODE_CTRL, MegaMoeImpl::OVERFLOW_MODE_CTRL>(0);

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            timer.Begin(Phase::Pack);
            const uint64_t start = timer.TraceBegin();
            Expand(args, block);
            const uint64_t routes = static_cast<uint64_t>(args.config.bs) * args.config.topk;
            const uint64_t owned = block < routes ? (routes - 1U - block) / kAic + 1U : 0U;
            TraceMeta meta;
            meta.rows = owned;
            meta.bytes = owned * (args.config.hidden + args.config.hidden / 32U);
            timer.TraceEnd(TraceEvent::Pack, start, meta);
            timer.End(Phase::Pack, owned * (args.config.hidden + args.config.hidden / 32U));
        }
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
        if (block == kCommunicationWorkers) SelfDispatch(args, timer);
        RunActivation(args, block - kCommunicationWorkers, timer);
        if (block == kCommunicationWorkers) SelfCombine(args, timer);
    }
    if constexpr (g_coreType == AIV) PipeBarrier<PIPE_ALL>();
    TraceLocalSync(timer, 1U);
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U && block == 0U) {
            // The peer handshakes certify every local send/receive. Join all
            // ranks as well, so Reduce starts only after the entire operation's
            // return traffic has completed, including traffic between others.
            LocalTensor<uint64_t> signal(TPosition::VECCALC, 0U, 4U);
            timer.Begin(Phase::Barrier);
            const uint64_t start = timer.TraceBegin();
            RankBarrier(reinterpret_cast<__gm__ uint8_t*>(args.symmetric), PeerPhaseSlot(args, 2U, 0U),
                        args.config.rank, args.config.epoch, signal, EVENT_ID0);
            timer.TraceEnd(TraceEvent::RankBarrier, start);
            timer.End(Phase::Barrier);
        }
        PipeBarrier<PIPE_ALL>();
    }
    TraceLocalSync(timer, 2U);
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            timer.Begin(Phase::ReducePhase);
            const uint64_t start = timer.TraceBegin();
            Reduce(args, block);
            const uint64_t owned = block < args.config.bs ? (args.config.bs - 1U - block) / kAic + 1U : 0U;
            TraceMeta meta;
            meta.rows = owned;
            meta.bytes = owned * args.config.hidden * sizeof(bfloat16_t);
            timer.TraceEnd(TraceEvent::Reduce, start, meta);
            timer.End(Phase::ReducePhase, owned * args.config.hidden * sizeof(bfloat16_t));
        }
        PipeBarrier<PIPE_ALL>();
    }
    TraceLocalSync(timer, 3U);
    timer.FinishTrace();
    SetCtrlSpr<MegaMoeImpl::OVERFLOW_MODE_CTRL, MegaMoeImpl::OVERFLOW_MODE_CTRL>(originalOverflow);
}

}  // namespace MegaMoeUrma
