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
#include "communication_schedule.hpp"

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

// Host descriptors are peer-major and CV-major within each peer. Locate the
// actual ranges once without scanning unrelated peers in the payload loop.
__aicore__ inline uint32_t ChunkPeerLowerBound(uint64_t address, uint32_t count, uint32_t peer)
{
    const auto* chunks = reinterpret_cast<__gm__ Chunk*>(address);
    uint32_t begin = 0U;
    uint32_t end = count;
    while (begin < end) {
        const uint32_t middle = begin + (end - begin) / 2U;
        if (chunks[middle].peer < peer) {
            begin = middle + 1U;
        } else {
            end = middle;
        }
    }
    return begin;
}

__aicore__ inline uint32_t PeerDispatchCvCount(const Arguments& args, uint32_t peer)
{
    const uint32_t end = ChunkPeerLowerBound(args.sendChunks, args.config.sendChunkCount, peer + 1U);
    if (end == 0U) return 0U;
    const auto* last = reinterpret_cast<__gm__ Chunk*>(args.sendChunks) + end - 1U;
    return last->peer == peer ? last->cv + 1U : 0U;
}

__aicore__ inline void TraceLocalSync(Timer& timer, uint32_t stage)
{
    TraceMeta meta;
    meta.cv = stage;  // 0=two-round startup pack, 3=local invocation completion.
    const uint64_t start = timer.TraceBegin();
    const auto coarsePhase = stage == 0U ? CoarsePhase::SyncStart : CoarsePhase::SyncEnd;
    timer.CoarseBegin(coarsePhase);
    SyncAll<false>();
    timer.CoarseEnd(coarsePhase);
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

// Neither speculative Combine nor a full scan waits on an unready CV.
// Each dependency covers the complete GMM2 N direction and all FIX stores.
// Inspect once and return at the first incomplete compute group.
__aicore__ inline bool IsChunkComputed(const Arguments& args, const Chunk& chunk)
{
    for (uint32_t j = 0U; j < chunk.fragmentCount; ++j) {
        const uint32_t index = reinterpret_cast<__gm__ uint32_t*>(args.chunkGroupIndices)[chunk.fragmentBegin + j];
        const uint32_t rows = reinterpret_cast<__gm__ Fragment*>(args.fragments)[index].rows;
        if (ReadGmByPassDCache(ComputeFlag(args.gmm2Done, index)) <
            static_cast<int32_t>(ComputeTileCount(rows, args.config.hidden))) {
            return false;
        }
    }
    return true;
}

// One owner per peer, QP0 only. Dispatch finishes before any Combine PUT.
__aicore__ inline void CommunicatePeer(const Arguments& args, uint32_t owner, Timer& timer)
{
    const uint32_t peer = PeerForOwner(args.config.rank, owner);
    auto* symmetric = reinterpret_cast<__gm__ uint8_t*>(args.symmetric);
    LocalTensor<uint64_t> signal(TPosition::VECCALC, 0U, 4U);
    AsyncPublishBuffer publisher(signal, EVENT_ID0);
    publisher.Init(args.config.epoch);
    TraceMeta peerMeta; peerMeta.peer = peer;
    timer.Begin(Phase::Barrier);
    uint64_t start = timer.TraceBegin();
    WaitStartAtLeast(symmetric + PeerPhaseSlot(args, 0U, peer), args.config.epoch);
    timer.TraceEnd(TraceEvent::WaitStartPeer, start, peerMeta);
    timer.End(Phase::Barrier);
    const uint64_t h = args.config.hidden;
    const uint64_t scales = h / 32U;
    const uint64_t rowBytes = h * sizeof(bfloat16_t);
    const uint32_t dispatchCvs = PeerDispatchCvCount(args, peer);
    uint32_t nextPackRound = kPackStartupRounds;
    uint32_t nextDispatch = 0U, step = 0U;
    uint64_t dispatchBytes = 0U;
    bool dispatchStarted = false;
    while (nextDispatch < dispatchCvs) {
        const uint32_t target = DispatchBatchLimit(step, dispatchCvs - nextDispatch,
                                                   nextPackRound >= args.config.packRoundCount);
        const Chunk first = LoadPackChunk(args, nextDispatch, peer);
        uint32_t count = 1U;
        uint64_t rows = first.rows;
        while (count < target) {
            const Chunk next = LoadPackChunk(args, nextDispatch + count, peer);
            if (!AdjacentChunks(first, rows, count, next) || (rows + next.rows) * h > kMaxPutBytes) break;
            rows += next.rows;
            ++count;
        }
        const uint32_t end = nextDispatch + count;
        // Growing a send from 2 to 4/8 CVs must not wait for this owner's
        // own still-unexecuted Pack quota. Produce exactly once, by cursor.
        while (nextPackRound < end && nextPackRound < args.config.packRoundCount)
            PackOneRound(args, nextPackRound++, owner, kAic, timer);
        TraceMeta meta = ChunkTraceMeta(first); meta.rows = rows;
        start = timer.TraceBegin();
        for (uint32_t cv = nextDispatch; cv < end; ++cv) {
            const Chunk chunk = LoadPackChunk(args, cv, peer);
            while (ReadGmByPassDCache(PackReadyFlag(args, peer, cv)) < static_cast<int32_t>(chunk.rows)) {
                const uint64_t backoff = GetSystemCycle();
                while (GetSystemCycle() - backoff < 64U) {}
            }
        }
        timer.TraceEnd(TraceEvent::WaitDispatch, start, meta);
        if (!dispatchStarted) { timer.CoarseBegin(CoarsePhase::Dispatch); timer.Begin(Phase::Dispatch); dispatchStarted = true; }
        meta.bytes = rows * h;
        start = timer.TraceBegin();
        Put(symmetric, args.config.receiveXOffset + first.remoteRow * h,
            reinterpret_cast<__gm__ uint8_t*>(args.expandedX + first.localRow * h),
            static_cast<uint32_t>(meta.bytes), peer);
        timer.TraceEnd(TraceEvent::DispatchPutX, start, meta);
        meta.bytes = rows * scales;
        start = timer.TraceBegin();
        Put(symmetric, args.config.receiveScaleOffset + first.remoteRow * scales,
            reinterpret_cast<__gm__ uint8_t*>(args.expandedScale + first.localRow * scales),
            static_cast<uint32_t>(meta.bytes), peer);
        timer.TraceEnd(TraceEvent::DispatchPutScale, start, meta);
        // Preserve two original Pack rounds of lookahead; all original quotas
        // still execute once, including those moved before this larger PUT.
        while (nextPackRound < end + kPackStartupRounds && nextPackRound < args.config.packRoundCount)
            PackOneRound(args, nextPackRound++, owner, kAic, timer);
        meta.bytes = 0U;
        start = timer.TraceBegin();
        publisher.Drain();  // previous publication, AFTER this batch's PUTs
        Quiet(peer);
        timer.TraceEnd(TraceEvent::DispatchQuiet, start, meta);
        start = timer.TraceBegin();
        for (uint32_t cv = nextDispatch; cv < end; ++cv) {
            const uint64_t index = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + cv;
            publisher.Enqueue(symmetric, args.config.dispatchReadyOffset + index * kFlagBytes, peer);
        }
        timer.TraceEnd(TraceEvent::DispatchPublish, start, meta);
        dispatchBytes += rows * (h + scales);
        nextDispatch = end;
        ++step;
    }
    start = timer.TraceBegin();
    publisher.Drain();
    timer.CoarseEnd(CoarsePhase::Dispatch);
    timer.TraceEnd(TraceEvent::DispatchSendComplete, start, peerMeta);
    if (dispatchStarted) timer.End(Phase::Dispatch, dispatchBytes);
    // A short/empty peer still owns quotas belonging to other destinations.
    while (nextPackRound < args.config.packRoundCount)
        PackOneRound(args, nextPackRound++, owner, kAic, timer);
    MarkPackWorkerComplete(args);
    AtomicAdd(ComputeFlag(args.localFlags, kLocalFlagDispatchDone), 1);
    start = timer.TraceBegin();
    if (owner == 0U) {
        WaitComputeFlag(args.localFlags, kLocalFlagPackDone, kAic);
        AtomicAdd(ComputeFlag(args.localFlags, kLocalFlagDispatchDone), 1);
    }
    WaitComputeFlag(args.localFlags, kLocalFlagDispatchDone, args.config.world);
    timer.TraceEnd(TraceEvent::WaitLocalDispatch, start, peerMeta);

    constexpr uint32_t kSubmittedOffset = 32U;
    static_assert(kSubmittedOffset + kMaxPeerChunks * sizeof(uint32_t) <= kRequiredUbBytes);
    LocalTensor<uint32_t> submitted(TPosition::VECCALC, kSubmittedOffset, kMaxPeerChunks);
    const uint32_t begin = ChunkPeerLowerBound(args.recvChunks, args.config.recvChunkCount, peer);
    const uint32_t end = ChunkPeerLowerBound(args.recvChunks, args.config.recvChunkCount, peer + 1U);
    const uint32_t total = end - begin;
    for (uint32_t i = 0; i < total; ++i) submitted.SetValue(i, 0U);
    uint32_t remaining = total;
    uint64_t combineBytes = 0U, waitStart = 0U;
    bool combineStarted = false, waiting = false;
    bool peerCounted = false, selfCounted = false;
    while (remaining != 0U) {
        bool progress = false;
        const uint64_t scanStart = timer.TraceBegin();
        for (uint32_t i = 0; i < total;) {
            if (submitted.GetValue(i)) { ++i; continue; }
            const Chunk first = LoadChunk(args.recvChunks, begin + i);
            if (!IsChunkComputed(args, first)) { ++i; continue; }
            uint32_t count = 1U;
            uint64_t rows = first.rows;
            // Never absorb the final actual unsent CV into another PUT.
            const uint32_t limit = remaining > 1U ? remaining - 1U : 1U;
            while (count < limit && i + count < total && CombineRangeFitsWord(first.cv, count + 1U)) {
                if (submitted.GetValue(i + count)) break;
                const Chunk next = LoadChunk(args.recvChunks, begin + i + count);
                if (!AdjacentChunks(first, rows, count, next) ||
                    (rows + next.rows) * rowBytes > kMaxPutBytes || !IsChunkComputed(args, next)) break;
                rows += next.rows;
                ++count;
            }
            if (waiting) {
                timer.TraceEnd(TraceEvent::WaitCombineReady, waitStart, peerMeta);
                waiting = false;
            }
            if (!combineStarted) { timer.CoarseBegin(CoarsePhase::Combine); timer.Begin(Phase::Combine); combineStarted = true; }
            TraceMeta meta = ChunkTraceMeta(first); meta.rows = rows; meta.bytes = rows * rowBytes;
            start = timer.TraceBegin();
            Put(symmetric, args.config.returnYOffset + first.remoteRow * rowBytes,
                reinterpret_cast<__gm__ uint8_t*>(args.gmm2Output + first.localRow * rowBytes),
                static_cast<uint32_t>(meta.bytes), peer);
            timer.TraceEnd(TraceEvent::CombinePut, start, meta);
            start = timer.TraceBegin();
            PublishCombineRange(symmetric, args.config.combineReadyOffset, args.config.maxPeerChunks,
                                args.config.rank, peer, first.cv, count);
            meta.bytes = 0U;
            timer.TraceEnd(TraceEvent::CombinePublish, start, meta);
            for (uint32_t j = 0; j < count; ++j) submitted.SetValue(i + j, 1U);
            combineBytes += rows * rowBytes;
            remaining -= count;
            i += count;
            progress = true;
        }
        ObserveIncomingCompletion(args, owner, peerCounted, selfCounted);
        if (!progress && remaining != 0U) {
            if (!waiting) { waitStart = scanStart; waiting = true; }
            const uint64_t backoff = GetSystemCycle();
            while (GetSystemCycle() - backoff < 200U) {}
        }
    }
    // SO finished follows all ready masks on QP0, including an empty stream.
    PublishOrdered(symmetric, PeerPhaseSlot(args, 1U, args.config.rank), peer, args.config.epoch);
    start = timer.TraceBegin();
    Quiet(peer); // Always required, even with timer disabled: protect next invocation.
    timer.CoarseEnd(CoarsePhase::Combine);
    timer.TraceEnd(TraceEvent::CombineSendComplete, start, peerMeta);
    if (combineStarted) timer.End(Phase::Combine, combineBytes);
    while (!ObserveIncomingCompletion(args, owner, peerCounted, selfCounted)) {
        const uint64_t backoff = GetSystemCycle();
        while (GetSystemCycle() - backoff < 200U) {}
    }
}

__aicore__ inline void RunProfiledCompute(const Arguments& args, uint32_t block, Timer& timer)
{
    // Select once, outside the loop. NoopTimer has no state, clocks, stores,
    // cache operations or fences; both coarse and formal use this instantiation.
    if (args.coarseTrace != 0U || (args.trace == 0U && args.timer == 0U)) {
        const auto phase = g_coreType == AIC ? CoarsePhase::Gmm : CoarsePhase::Dequant;
        timer.CoarseBegin(phase);
        NoopTimer disabled;
        RunCubeAndPrologue(args, block, disabled);
        timer.CoarseEnd(phase);
    } else {
        RunCubeAndPrologue(args, block, timer);
    }
}

__aicore__ inline void Process(const Arguments& args)
{
    Timer timer;
    timer.Init(args.timer, args.trace, args.traceCapacity, args.coarseTrace != 0U);
    const uint32_t block = GetBlockIdx() / GetTaskRation();
    const uint32_t communicationWorkers = CommunicationWorkers(args.config.world);
    const int64_t originalOverflow = GetCtrlSpr<MegaMoeImpl::OVERFLOW_MODE_CTRL, MegaMoeImpl::OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<MegaMoeImpl::OVERFLOW_MODE_CTRL, MegaMoeImpl::OVERFLOW_MODE_CTRL>(0);

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U && block < communicationWorkers) {
            PublishPeerStart(args, block, timer);
        }
        timer.CoarseBegin(CoarsePhase::Pack);
        const uint32_t worker = LocalWorkerIndex(block, GetSubBlockIdx());
        for (uint32_t round = 0U; round < kPackStartupRounds; ++round) {
            PackOneRound(args, round, worker, kLocalWorkers, timer);
        }
    }
    if constexpr (g_coreType == AIV) timer.CoarseEnd(CoarsePhase::Pack);
    // No CV loop contains a rank-wide or local full-core synchronization.
    TraceLocalSync(timer, 0U);
    if constexpr (g_coreType == AIC) {
        RunProfiledCompute(args, block, timer);
        MarkCubeComplete(args);
    } else if (GetSubBlockIdx() == 0U) {
        RunProfiledCompute(args, block, timer);
    } else if (block < communicationWorkers) {
        CommunicatePeer(args, block, timer);
    } else {
        timer.CoarseBegin(CoarsePhase::Activation);
        RunActivation(args, block - communicationWorkers, timer);
        timer.CoarseEnd(CoarsePhase::Activation);
    }
    if constexpr (g_coreType == AIV) {
        // Only this engine releases its phase-private UB/events. Other
        // compute and communication engines may still be producing rows.
        PipeBarrier<PIPE_ALL>();
        const uint32_t worker = ReduceWorkerIndex(block, GetSubBlockIdx(), args.config.world);
        if (worker < ReduceWorkers(args.config.world)) {
            timer.Begin(Phase::ReducePhase);
            timer.CoarseBegin(CoarsePhase::Reduce);
            Reduce(args, worker, timer);
            timer.CoarseEnd(CoarsePhase::Reduce);
            const uint64_t owned = ReduceOwnedRows(args.config.bs, worker, args.config.world);
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
