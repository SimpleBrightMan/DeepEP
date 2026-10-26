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

// Reorder owns low 128KiB and Pack owns the following 58,880B. Keep the
// communication owner's persistent signal/CV metadata above BOTH regions:
// reorder can execute with submitted Combine PUTs still in flight.
constexpr uint32_t kCommunicationSignalOffset = kPackUbBase + kPackUbBytes;
constexpr uint32_t kCommunicationSubmittedOffset = kCommunicationSignalOffset + kSignalCopyBytes;
constexpr uint32_t kCommunicationPendingOffset = kCommunicationSubmittedOffset + kMaxPeerChunks * sizeof(uint32_t);
static_assert(kCommunicationSignalOffset % 32U == 0U);
static_assert(kCommunicationPendingOffset + kMaxPeerChunks * sizeof(uint32_t) <= kRequiredUbBytes);

// Waiting for a peer/local phase cannot stop the paired AIC's reorder work.
// Split a wait trace only when actual work is available, so exported waits
// never overlap Gmm2Reorder records and empty polling creates no extra rows.
template <typename T>
__aicore__ inline void WaitWithReorder(__gm__ T* flag, T expected, OwnedReorder& reorder, Timer& timer,
                                       TraceEvent event, const TraceMeta& meta)
{
    uint64_t start = timer.TraceBegin();
    while (ReadGmByPassDCache(flag) < expected) {
        if (reorder.Ready()) {
            timer.TraceEnd(event, start, meta);
            reorder.DrainReady();
            start = timer.TraceBegin();
        } else {
            const uint64_t backoff = GetSystemCycle();
            while (GetSystemCycle() - backoff < 64U) {
            }
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

// Retire exactly the submitted CVs, which need not form a contiguous prefix.
// Their disjoint remote rows and independent ready slots permit out-of-order
// submission. Only the original owner touches this peer's QP1 and flags.
__aicore__ inline void RetireCombineBatch(const Arguments& args, uint32_t owner, uint32_t combineBegin,
                                          LocalTensor<uint32_t>& pending, uint32_t& pendingCount, uint64_t& pendingRows,
                                          LocalTensor<uint64_t>& signal, Timer& timer, bool& peerCounted,
                                          bool& selfCounted, OwnedReorder& reorder)
{
    if (pendingCount == 0U) return;
    const uint32_t peer = PeerForOwner(args.config.rank, owner);
    auto* symmetric = reinterpret_cast<__gm__ uint8_t*>(args.symmetric);
    TraceMeta meta;
    meta.peer = peer;
    meta.rows = pendingRows;
    if (pendingCount == 1U) meta = ChunkTraceMeta(LoadChunk(args.recvChunks, combineBegin + pending.GetValue(0U)));
    // Multiple CVs may have holes: retain unknown CV/row, and sum actual rows.
    // Finish every currently ready tile owned by the paired AIC before quiet.
    // DrainReady never waits for an unfinished producer. Keep its own trace
    // outside the CombineQuiet event and preserve the pending CV list in UB.
    reorder.DrainReady();
    ObserveIncomingCompletion(args, owner, peerCounted, selfCounted);
    uint64_t start = timer.TraceBegin();
    Quiet(peer, kCombineQp);
    timer.TraceEnd(TraceEvent::CombineQuiet, start, meta);
    for (uint32_t i = 0U; i < pendingCount; ++i) {
        const Chunk completed = LoadChunk(args.recvChunks, combineBegin + pending.GetValue(i));
        meta = ChunkTraceMeta(completed);
        const uint64_t readyIndex = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + completed.cv;
        start = timer.TraceBegin();
        PublishDone(symmetric, args.config.combineReadyOffset + readyIndex * kFlagBytes, peer, args.config.epoch,
                    signal, EVENT_ID0);
        timer.TraceEnd(TraceEvent::CombinePublish, start, meta);
    }
    pendingCount = 0U;
    pendingRows = 0U;
}

// One and only one AIV1 owns all PUT/quiet operations for this peer for the
// entire invocation. Dispatch uses QP0 and Combine uses QP1, so a Dispatch
// quiet cannot drain early Combine. Both queues retain the same single owner.
__aicore__ inline void CommunicatePeer(const Arguments& args, uint32_t owner, Timer& timer)
{
    const uint32_t peer = PeerForOwner(args.config.rank, owner);
    auto* symmetric = reinterpret_cast<__gm__ uint8_t*>(args.symmetric);
    LocalTensor<uint64_t> signal(TPosition::VECCALC, kCommunicationSignalOffset, 4U);
    OwnedReorder reorder(args, owner, timer);
    TraceMeta peerMeta;
    peerMeta.peer = peer;
    timer.Begin(Phase::Barrier);
    // Start was published before the startup Pack. Wait only now, immediately
    // before this owner's first possible payload write to the peer.
    auto* startFlag = reinterpret_cast<__gm__ uint64_t*>(symmetric + PeerPhaseSlot(args, 0U, peer));
    WaitWithReorder(startFlag, args.config.epoch, reorder, timer, TraceEvent::WaitStartPeer, peerMeta);
    timer.End(Phase::Barrier);

    const uint64_t h = args.config.hidden;
    const uint64_t scales = h / 32U;
    const uint64_t rowBytes = h * sizeof(bfloat16_t);
    const uint32_t dispatchCvs = PeerDispatchCvCount(args, peer);
    const uint32_t combineBegin = ChunkPeerLowerBound(args.recvChunks, args.config.recvChunkCount, peer);
    const uint32_t combineEnd = ChunkPeerLowerBound(args.recvChunks, args.config.recvChunkCount, peer + 1U);
    uint32_t nextCombine = combineBegin;
    uint64_t combineBytes = 0U;
    uint64_t start = 0U;
    bool combineStarted = false;
    for (uint32_t round = 0U; round < args.config.packRoundCount; ++round) {
        const Chunk chunk = LoadPackChunk(args, round, peer);
        // An empty peer must still produce its fixed share of every Pack
        // round, including data for other peers and self. Never exit early.
        if (chunk.rows == 0U) {
            PackOneRound(args, round + kPackStartupRounds, owner, kAic, timer);
            reorder.DrainReady();
            continue;
        }
        TraceMeta meta = ChunkTraceMeta(chunk);
        WaitWithReorder(PackReadyFlag(args, peer, round), static_cast<int32_t>(chunk.rows), reorder, timer,
                        TraceEvent::WaitDispatch, meta);
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
        // Pack retains priority. Only after its assigned quota is complete
        // does this AIV1 reorder ready GMM2 tiles from its paired AIC.
        reorder.DrainReady();
        // Only the last two actual outgoing CVs may submit one ready Combine
        // each. A shorter peer must not use the rank-wide Pack round count.
        // Reserve one CV for the final standalone post-Dispatch submission.
        // No Combine quiet or ready publication occurs during Dispatch.
        if (chunk.cv + 2U >= dispatchCvs && nextCombine + 1U < combineEnd) {
            const Chunk early = LoadChunk(args.recvChunks, nextCombine);
            if (IsChunkComputed(args, early)) {
                if (!combineStarted) {
                    timer.Begin(Phase::Combine);
                    combineStarted = true;
                }
                auto earlyMeta = ChunkTraceMeta(early);
                earlyMeta.bytes = early.rows * rowBytes;
                const uint64_t earlyStart = timer.TraceBegin();
                Put(symmetric, args.config.returnYOffset + early.remoteRow * rowBytes,
                    reinterpret_cast<__gm__ uint8_t*>(args.gmm2Output + early.localRow * rowBytes),
                    static_cast<uint32_t>(earlyMeta.bytes), peer, kCombineQp);
                timer.TraceEnd(TraceEvent::CombinePut, earlyStart, earlyMeta);
                combineBytes += earlyMeta.bytes;
                ++nextCombine;
            }
        }
        meta.bytes = 0U;
        start = timer.TraceBegin();
        Quiet(peer, kDispatchQp);
        timer.TraceEnd(TraceEvent::DispatchQuiet, start, meta);
        const uint64_t readyIndex = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + chunk.cv;
        start = timer.TraceBegin();
        PublishDone(symmetric, args.config.dispatchReadyOffset + readyIndex * kFlagBytes, peer, args.config.epoch,
                    signal, EVENT_ID0);
        timer.TraceEnd(TraceEvent::DispatchPublish, start, meta);
        timer.End(Phase::Dispatch, chunk.rows * (h + scales));
    }
    MarkPackWorkerComplete(args);
    // Empty peers must still join the local dispatch phase. Owner 0 accounts
    // self only when every steady Pack worker has finished, including helpers.
    AtomicAdd(ComputeFlag(args.localFlags, kLocalFlagDispatchDone), 1);
    if (owner == 0U) {
        WaitWithReorder(ComputeFlag(args.localFlags, kLocalFlagPackDone), static_cast<int32_t>(kAic), reorder, timer,
                        TraceEvent::WaitLocalDispatch, peerMeta);
        AtomicAdd(ComputeFlag(args.localFlags, kLocalFlagDispatchDone), 1);
    }
    WaitWithReorder(ComputeFlag(args.localFlags, kLocalFlagDispatchDone), static_cast<int32_t>(args.config.world),
                    reorder, timer, TraceEvent::WaitLocalDispatch, peerMeta);

    bool peerCounted = false;
    bool selfCounted = false;
    // Scalar-only owner-private metadata must survive every Reorder call.
    // The host planner enforces the same kMaxPeerChunks bound.
    LocalTensor<uint32_t> submitted(TPosition::VECCALC, kCommunicationSubmittedOffset, kMaxPeerChunks);
    LocalTensor<uint32_t> pending(TPosition::VECCALC, kCommunicationPendingOffset, kMaxPeerChunks);
    const uint32_t combineCount = combineEnd - combineBegin;
    const uint32_t earlyCount = nextCombine - combineBegin;
    uint32_t pendingCount = 0U;
    uint64_t pendingRows = 0U;
    for (uint32_t i = 0U; i < combineCount; ++i) {
        submitted.SetValue(i, i < earlyCount ? 1U : 0U);
        if (i < earlyCount) {
            pending.SetValue(pendingCount++, i);
            pendingRows += LoadChunk(args.recvChunks, combineBegin + i).rows;
        }
    }
    // Retire the early prefix before scanning any new producers.
    RetireCombineBatch(args, owner, combineBegin, pending, pendingCount, pendingRows, signal, timer, peerCounted,
                       selfCounted, reorder);
    uint32_t remaining = combineCount - earlyCount;
    bool waiting = false;
    uint64_t waitStart = 0U;
    while (remaining != 0U) {
        // Readiness may depend on THIS communication owner's reorder tiles.
        // Progress even when no CV can be submitted and hence no quiet occurs.
        if (reorder.Ready()) {
            if (waiting) {
                timer.TraceEnd(TraceEvent::WaitCombineReady, waitStart, peerMeta);
                waiting = false;
            }
            reorder.DrainReady();
        }
        // Always scan this peer's CVs from beginning to end. Already submitted
        // or unready CVs are skipped; readiness does not imply a ready prefix.
        const uint64_t scanStart = timer.TraceBegin();
        for (uint32_t i = 0U; i < combineCount; ++i) {
            if (submitted.GetValue(i) != 0U) continue;
            const Chunk chunk = LoadChunk(args.recvChunks, combineBegin + i);
            if (!IsChunkComputed(args, chunk)) continue;
            if (waiting) {
                // Close idle scope BEFORE emitting any PUT/quiet/flag trace.
                timer.TraceEnd(TraceEvent::WaitCombineReady, waitStart, peerMeta);
                waiting = false;
            }
            if (remaining == 1U) {
                // Last ACTUAL unsent CV, not necessarily the highest index.
                // Publish all earlier results before submitting it alone.
                RetireCombineBatch(args, owner, combineBegin, pending, pendingCount, pendingRows, signal, timer,
                                   peerCounted, selfCounted, reorder);
            }
            if (!combineStarted) {
                timer.Begin(Phase::Combine);
                combineStarted = true;
            }
            TraceMeta meta = ChunkTraceMeta(chunk);
            meta.bytes = chunk.rows * rowBytes;
            start = timer.TraceBegin();
            Put(symmetric, args.config.returnYOffset + chunk.remoteRow * rowBytes,
                reinterpret_cast<__gm__ uint8_t*>(args.gmm2Output + chunk.localRow * rowBytes),
                static_cast<uint32_t>(meta.bytes), peer, kCombineQp);
            timer.TraceEnd(TraceEvent::CombinePut, start, meta);
            combineBytes += meta.bytes;
            submitted.SetValue(i, 1U);
            pending.SetValue(pendingCount++, i);
            pendingRows += chunk.rows;
            --remaining;
            if (remaining == 0U) break;
        }
        if (pendingCount != 0U) {
            // One quiet for everything found ready this sweep. A final CV is
            // the only pending entry here after the preceding batch flush.
            RetireCombineBatch(args, owner, combineBegin, pending, pendingCount, pendingRows, signal, timer,
                               peerCounted, selfCounted, reorder);
        } else if (remaining != 0U) {
            if (!waiting) {
                waitStart = scanStart;
                waiting = true;
            }
            ObserveIncomingCompletion(args, owner, peerCounted, selfCounted);
            const uint64_t backoff = GetSystemCycle();
            while (GetSystemCycle() - backoff < 200U) {
            }
        }
    }
    // Every CV now has its own completion flag, so the source can reduce
    // completed tokens while other peers are still computing/returning rows.
    // The aggregate envelope includes producer gaps between CVs.
    if (combineStarted) timer.End(Phase::Combine, combineBytes);
    // Publish finished promptly, then keep reordering the paired AIC's tiles
    // while observing inbound completion. Empty peer edges also take this path;
    // stopping at outbound completion could strand another peer's Combine.
    FinishCombinePeer(args, owner, signal, peerCounted, selfCounted, reorder);
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
