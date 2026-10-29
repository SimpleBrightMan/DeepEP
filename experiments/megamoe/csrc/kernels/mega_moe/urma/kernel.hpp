// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"
#include "compute.hpp"
#include "pack_reduce.hpp"
#include "pack_round.hpp"
#include "peer_reorder_progress.hpp"
#include "tail_completion.hpp"
#include "timer.hpp"
#include "transport.hpp"

namespace MegaMoeUrma {
using namespace AscendC;

// Reorder owns low 128KiB and Pack owns the following 58,880B. Keep the
// communication owner's persistent signal/CV metadata above BOTH regions:
// reorder can execute with submitted Combine PUTs still in flight.
constexpr uint32_t kCommunicationSignalOffset = kPackUbBase + kPackUbBytes;
constexpr uint32_t kCommunicationSubmittedOffset = kCommunicationSignalOffset + kSignalCopyBytes;
constexpr uint32_t kCommunicationPendingOffset = kCommunicationSubmittedOffset + kMaxPeerChunks * sizeof(uint32_t);
static_assert(kCommunicationSignalOffset % 32U == 0U);
static_assert(kCommunicationPendingOffset + kMaxPeerChunks * sizeof(uint32_t) <= kRequiredUbBytes);
// After the local Dispatch join, this owner's Pack UB is dead. Reuse 16KiB
// of it for remote/self span bits, away from both DMA slots and CV counters.
constexpr uint32_t kRemoteReorderBitmapOffset = kPackUbBase;
constexpr uint32_t kSelfReorderBitmapOffset = kRemoteReorderBitmapOffset + kPeerReorderBitmapBytes;
static_assert(kSelfReorderBitmapOffset + kPeerReorderBitmapBytes <= kCommunicationSignalOffset);

template <typename T>
__aicore__ inline void WaitProgressFlag(__gm__ T* flag, T expected, Timer& timer, TraceEvent event,
                                        const TraceMeta& meta)
{
    const uint64_t start = timer.TraceBegin();
    while (ReadGmByPassDCache(flag) < expected) {
        const uint64_t backoff = GetSystemCycle();
        while (GetSystemCycle() - backoff < 64U) {
        }
    }
    timer.TraceEnd(event, start, meta);
}

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
    LocalTensor<uint64_t> signal(TPosition::VECCALC, kCommunicationSignalOffset, 4U);
    TraceMeta meta;
    meta.peer = peer;
    timer.Begin(Phase::Barrier);
    const uint64_t start = timer.TraceBegin();
    PublishDone(reinterpret_cast<__gm__ uint8_t*>(args.symmetric), PeerPhaseSlot(args, 0U, args.config.rank), peer,
                args.config.epoch, signal, EVENT_ID0);
    timer.TraceEnd(TraceEvent::PublishStart, start, meta);
    timer.End(Phase::Barrier);
}

// One owner, one QP. Data and SO notifications remain in flight while this
// worker prepares other disjoint GM rows. No Combine is submitted in Dispatch.
__aicore__ inline void CommunicatePeer(const Arguments& args, uint32_t owner, Timer& timer)
{
    const uint32_t peer = PeerForOwner(args.config.rank, owner);
    const uint32_t workers = CommunicationWorkers(args.config.world);
    auto* symmetric = reinterpret_cast<__gm__ uint8_t*>(args.symmetric);
    TraceMeta peerMeta;
    peerMeta.peer = peer;
    timer.Begin(Phase::Barrier);
    auto* startFlag = reinterpret_cast<__gm__ uint64_t*>(symmetric + PeerPhaseSlot(args, 0U, peer));
    WaitProgressFlag(startFlag, args.config.epoch, timer, TraceEvent::WaitStartPeer, peerMeta);
    timer.End(Phase::Barrier);

    const uint64_t h = args.config.hidden;
    const uint64_t scales = h / 32U;
    const uint64_t rowBytes = h * sizeof(bfloat16_t);
    for (uint32_t round = 0U; round < args.config.packRoundCount; ++round) {
        const Chunk chunk = LoadPackChunk(args, round, peer);
        if (chunk.rows != 0U) {
            auto meta = ChunkTraceMeta(chunk);
            WaitProgressFlag(PackReadyFlag(args, peer, round), static_cast<int32_t>(chunk.rows), timer,
                             TraceEvent::WaitDispatch, meta);
            timer.Begin(Phase::Dispatch);
            meta.bytes = chunk.rows * h;
            uint64_t start = timer.TraceBegin();
            Put(symmetric, args.config.receiveXOffset + chunk.remoteRow * h,
                reinterpret_cast<__gm__ uint8_t*>(args.expandedX + chunk.localRow * h),
                static_cast<uint32_t>(meta.bytes), peer);
            timer.TraceEnd(TraceEvent::DispatchPutX, start, meta);
            meta.bytes = chunk.rows * scales;
            start = timer.TraceBegin();
            Put(symmetric, args.config.receiveScaleOffset + chunk.remoteRow * scales,
                reinterpret_cast<__gm__ uint8_t*>(args.expandedScale + chunk.localRow * scales),
                static_cast<uint32_t>(meta.bytes), peer);
            timer.TraceEnd(TraceEvent::DispatchPutScale, start, meta);
            meta.bytes = 0U;
            start = timer.TraceBegin();
            const uint64_t ready = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + chunk.cv;
            PublishOrdered(symmetric, args.config.dispatchReadyOffset + ready * kFlagBytes, peer, args.config.epoch);
            timer.TraceEnd(TraceEvent::DispatchPublish, start, meta);
            timer.End(Phase::Dispatch, chunk.rows * (h + scales), true);
        }
        // Even an empty peer keeps its static quota for other peers/self.
        // Pack's GM writes are complete before it increments packReady.
        PackOneRound(args, round + kPackStartupRounds, owner, kAic, timer);
        if (chunk.rows != 0U) {
            // Prepare future rows while this CV is in flight, but retire its
            // X/scale/SO operations before submitting the next CV to this peer.
            const auto meta = ChunkTraceMeta(chunk);
            const uint64_t start = timer.TraceBegin();
            Quiet(peer);
            timer.TraceEnd(TraceEvent::DispatchQuiet, start, meta);
            timer.MarkSendComplete(Phase::Dispatch);
        }
    }
    MarkPackWorkerComplete(args);
    AtomicAdd(LocalCompletionFlag(args, kLocalFlagDispatchDone), 1);
    if (owner == 0U) {
        WaitProgressFlag(LocalCompletionFlag(args, kLocalFlagPackDone), static_cast<int32_t>(kAic), timer,
                         TraceEvent::WaitLocalDispatch, peerMeta);
        AtomicAdd(LocalCompletionFlag(args, kLocalFlagDispatchDone), 1);
    }
    // Every nonempty CV has completed its peer quiet before this local join.
    // This is not a global rank barrier.
    WaitProgressFlag(LocalCompletionFlag(args, kLocalFlagDispatchDone), static_cast<int32_t>(args.config.world), timer,
                     TraceEvent::WaitLocalDispatch, peerMeta);

    const uint32_t begin = ChunkPeerLowerBound(args.recvChunks, args.config.recvChunkCount, peer);
    const uint32_t end = ChunkPeerLowerBound(args.recvChunks, args.config.recvChunkCount, peer + 1U);
    const uint32_t selfBegin = ChunkPeerLowerBound(args.recvChunks, args.config.recvChunkCount, args.config.rank);
    const uint32_t selfEnd = ChunkPeerLowerBound(args.recvChunks, args.config.recvChunkCount, args.config.rank + 1U);
    // Remote CVs hold submitted ROW counts; self CVs hold copied SPAN counts.
    // Both become the submitted sentinel only after the full CV ready flag.
    // Each row is written once, with no GM reuse before final quiet.
    LocalTensor<uint32_t> submitted(TPosition::VECCALC, kCommunicationSubmittedOffset, kMaxPeerChunks);
    LocalTensor<uint32_t> selfCopied(TPosition::VECCALC, kCommunicationPendingOffset, kMaxPeerChunks);
    LocalTensor<uint32_t> remoteSpanCopied(TPosition::VECCALC, kRemoteReorderBitmapOffset, kPeerReorderBitmapWords);
    LocalTensor<uint32_t> selfSpanCopied(TPosition::VECCALC, kSelfReorderBitmapOffset, kPeerReorderBitmapWords);
    const auto remoteSpans = PeerReorderSpanRange(args, begin, end);
    const auto selfSpans = PeerReorderSpanRange(args, selfBegin, selfEnd);
    ClearReorderBitmap(remoteSpanCopied, remoteSpans.spanCount);
    ClearReorderBitmap(selfSpanCopied, selfSpans.spanCount);
    uint32_t remaining = end - begin;
    uint32_t selfRemaining = 0U;
    for (uint32_t j = 0U; j < remaining; ++j) submitted.SetValue(j, 0U);
    for (uint32_t j = owner; j < selfEnd - selfBegin; j += workers) {
        selfCopied.SetValue(j, 0U);
        ++selfRemaining;
    }
    bool sentComplete = false;
    bool finishedPublished = false;
    bool selfComplete = false;
    bool peerCounted = false;
    bool selfCounted = false;
    bool waiting = false;
    uint64_t waitStart = 0U;
    while (!sentComplete || !selfComplete) {
        bool progress = false;
        for (uint32_t j = 0U; j < end - begin; ++j) {
            uint32_t sentRows = submitted.GetValue(j);
            if (sentRows == kReorderCvSubmitted) continue;
            const Chunk chunk = LoadChunk(args.recvChunks, begin + j);
            const auto range = LoadPeerReorderRange(args, begin + j);
            const uint32_t advanced = ReorderReadyPeerSpans(args, range, chunk, remoteSpans.spanBegin,
                                                           remoteSpanCopied, timer, waiting, waitStart, peerMeta);
            if (advanced == 0U) continue;
            progress = true;
            const uint32_t readyRows = ReorderedPeerPrefixRows(args, range, remoteSpans.spanBegin, remoteSpanCopied);
            while (sentRows < chunk.rows) {
                const uint32_t left = chunk.rows - sentRows;
                const uint32_t putRows = left < kCombinePutRows ? left : kCombinePutRows;
                if (sentRows + putRows > readyRows) break;
                auto meta = ChunkTraceMeta(chunk);
                meta.row = chunk.localRow + sentRows;
                meta.rows = putRows;
                meta.bytes = putRows * rowBytes;
                timer.Begin(Phase::Combine);
                const uint64_t start = timer.TraceBegin();
                Put(symmetric, args.config.returnYOffset + (chunk.remoteRow + sentRows) * rowBytes,
                    reinterpret_cast<__gm__ uint8_t*>(args.gmm2Output + (chunk.localRow + sentRows) * rowBytes),
                    static_cast<uint32_t>(meta.bytes), peer);
                timer.TraceEnd(TraceEvent::CombinePut, start, meta);
                timer.End(Phase::Combine, meta.bytes, true);
                sentRows += putRows;
            }
            submitted.SetValue(j, sentRows);
            if (sentRows != chunk.rows) continue;
            // Payloads may leave in 128-row pieces, but Reduce still sees one
            // ready flag per original CV (256 rows by default). Same-QP SO
            // orders the flag after EVERY preceding data PUT, without quiet.
            auto meta = ChunkTraceMeta(chunk);
            meta.bytes = 0U;
            timer.Begin(Phase::Combine);
            const uint64_t start = timer.TraceBegin();
            const uint64_t ready = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + chunk.cv;
            PublishOrdered(symmetric, args.config.combineReadyOffset + ready * kFlagBytes, peer, args.config.epoch);
            timer.TraceEnd(TraceEvent::CombinePublish, start, meta);
            timer.End(Phase::Combine, 0U, true);
            submitted.SetValue(j, kReorderCvSubmitted);
            --remaining;
            progress = true;
        }
        if (remaining == 0U && !finishedPublished) {
            if (waiting) {
                timer.TraceEnd(TraceEvent::WaitCombineReady, waitStart, peerMeta);
                waiting = false;
            }
            // Publish remote finished promptly, including empty edges. Defer
            // the blocking quiet until this owner's self copies are complete.
            const uint64_t publishStart = timer.TraceBegin();
            PublishOrdered(symmetric, PeerPhaseSlot(args, 1U, args.config.rank), peer, args.config.epoch);
            timer.TraceEnd(TraceEvent::CombinePublish, publishStart, peerMeta);
            finishedPublished = true;
            progress = true;
        }
        for (uint32_t j = owner; j < selfEnd - selfBegin; j += workers) {
            const uint32_t copied = selfCopied.GetValue(j);
            if (copied == kReorderCvSubmitted) continue;
            const Chunk chunk = LoadChunk(args.recvChunks, selfBegin + j);
            const auto range = LoadPeerReorderRange(args, selfBegin + j);
            const uint32_t advanced = ReorderReadyPeerSpans(args, range, chunk, selfSpans.spanBegin,
                                                           selfSpanCopied, timer, waiting, waitStart, peerMeta);
            if (advanced == 0U) continue;
            progress = true;
            selfCopied.SetValue(j, copied + advanced);
            if (copied + advanced != range.spanCount) continue;
            const uint64_t ready = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + chunk.cv;
            // Reorder drained local stores. Self requires no URMA operation.
            WriteGmByPassDCache(
                reinterpret_cast<__gm__ uint64_t*>(symmetric + args.config.combineReadyOffset + ready * kFlagBytes),
                args.config.epoch);
            selfCopied.SetValue(j, kReorderCvSubmitted);
            --selfRemaining;
            progress = true;
        }
        if (selfRemaining == 0U && !selfComplete) {
            MarkReorderComplete(args);
            selfComplete = true;
            progress = true;
        }
        if (finishedPublished && selfComplete && !sentComplete) {
            if (waiting) {
                timer.TraceEnd(TraceEvent::WaitCombineReady, waitStart, peerMeta);
                waiting = false;
            }
            // Self ready must never depend on a remote final quiet. Unready
            // self CVs are rescanned above while remote operations progress.
            // Final quiet is mandatory even when timer is disabled.
            const uint64_t completeStart = timer.TraceBegin();
            Quiet(peer);
            timer.TraceEnd(TraceEvent::CombineSendComplete, completeStart, peerMeta);
            if (end != begin) timer.MarkSendComplete(Phase::Combine);
            sentComplete = true;
            progress = true;
        }
        ObserveIncomingCompletion(args, owner, peerCounted, selfCounted);
        if (!progress) {
            if (!waiting) {
                waitStart = timer.TraceBegin();
                waiting = true;
            }
            const uint64_t start = GetSystemCycle();
            while (GetSystemCycle() - start < 200U) {
            }
        }
    }
    if (waiting) timer.TraceEnd(TraceEvent::WaitCombineReady, waitStart, peerMeta);
    FinishCombinePeer(args, owner, peerCounted, selfCounted);
}

__aicore__ inline void Process(const Arguments& args)
{
    Timer timer;
    timer.Init(args.timer, args.trace, args.traceCapacity);
    const uint32_t block = GetBlockIdx() / GetTaskRation();
    const uint32_t communicationWorkers = CommunicationWorkers(args.config.world);
    const int64_t originalOverflow = GetCtrlSpr<MegaMoeImpl::OVERFLOW_MODE_CTRL, MegaMoeImpl::OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<MegaMoeImpl::OVERFLOW_MODE_CTRL, MegaMoeImpl::OVERFLOW_MODE_CTRL>(0);

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U && block < communicationWorkers) {
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
    } else if (block < communicationWorkers) {
        CommunicatePeer(args, block, timer);
    } else {
        RunActivation(args, block - communicationWorkers, timer);
    }
    if constexpr (g_coreType == AIV) {
        // Only this engine releases its phase-private UB/events. Other
        // compute and communication engines may still be producing rows.
        PipeBarrier<PIPE_ALL>();
        const uint32_t worker = ReduceWorkerIndex(block, GetSubBlockIdx(), args.config.world);
        if (worker < ReduceWorkers(args.config.world)) {
            timer.Begin(Phase::ReducePhase);
            Reduce(args, worker, timer);
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
