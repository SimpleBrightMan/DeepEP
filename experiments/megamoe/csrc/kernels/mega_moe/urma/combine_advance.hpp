// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"
#include "compute_contract.hpp"
#include "tail_completion.hpp"
#include "timer.hpp"
#include "transport.hpp"

namespace MegaMoeUrma {

constexpr uint32_t kCombineHelperQp = 1U;
constexpr uint32_t kCombineSignalUbOffset = ActivationUbContract::inputEnd;
static_assert(kCombineSignalUbOffset + kSignalCopyBytes <= kRequiredUbBytes);

__aicore__ inline uint32_t CrossPeerForHelper(uint32_t rank, uint32_t helper)
{
    const uint32_t ownGroupBegin = rank / kGroupSize * kGroupSize;
    return helper < ownGroupBegin ? helper : helper + kGroupSize;
}

// kCombineHelpers means that this peer is in our own group and has no helper.
__aicore__ inline uint32_t CrossHelperForPeer(uint32_t rank, uint32_t peer)
{
    const uint32_t ownGroupBegin = rank / kGroupSize * kGroupSize;
    if (peer >= ownGroupBegin && peer < ownGroupBegin + kGroupSize) return kCombineHelpers;
    return peer < ownGroupBegin ? peer : peer - kGroupSize;
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

struct CombinePeerRange {
    uint32_t begin;
    uint32_t middle;
    uint32_t end;
};

__aicore__ inline uint32_t CombinePeerLowerBound(const Arguments& args, uint32_t peer)
{
    const auto* chunks = reinterpret_cast<__gm__ Chunk*>(args.recvChunks);
    uint32_t begin = 0U;
    uint32_t end = args.config.recvChunkCount;
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

// The host already emits contiguous peer-major CV descriptors. Split actual
// nonempty CVs, not maxPeerChunks or a balanced-route approximation. A partial
// final CV stays whole, and an empty peer has an empty prefix and suffix.
__aicore__ inline CombinePeerRange FindCombinePeerRange(const Arguments& args, uint32_t peer)
{
    // Locate boundaries once, without scalar-scanning every other peer's CVs.
    const uint32_t begin = CombinePeerLowerBound(args, peer);
    const uint32_t end = CombinePeerLowerBound(args, peer + 1U);
    return {begin, begin + (end - begin + 1U) / 2U, end};
}

// One activation worker owns the first half of one cross-group peer on QP1.
// The scheduler submits at most one CV, drains ready compute tasks, then calls
// CompletePending. No readiness probe blocks activation or scale gathering.
class CombineHelper {
public:
    __aicore__ inline void Init(const Arguments& args, uint32_t worker)
    {
        helper_ = worker;
        if (helper_ >= kCombineHelpers) return;  // Thirteenth activation worker.
        peer_ = CrossPeerForHelper(args.config.rank, helper_);
        const auto range = FindCombinePeerRange(args, peer_);
        next_ = range.begin;
        end_ = range.middle;
        if (HasWork()) {
            chunk_ = LoadChunk(args.recvChunks, next_);
        } else {
            MarkDone(args);
        }
    }

    __aicore__ inline bool HasWork() const
    {
        return next_ < end_;
    }
    __aicore__ inline bool Pending() const
    {
        return pending_;
    }
    __aicore__ inline TraceMeta CurrentMeta() const
    {
        return ChunkTraceMeta(chunk_);
    }

    __aicore__ inline bool ReadyToSubmit(const Arguments& args)
    {
        using namespace AscendC;
        if (!HasWork() || pending_) return false;
        if (!packDone_) {
            packDone_ = ReadGmByPassDCache(LocalCompletionFlag(args, kLocalFlagPackDone)) == static_cast<int32_t>(kAic);
            if (!packDone_) return false;
        }
        // QP1 can start before the original owner exits Dispatch. Retain the
        // same next-invocation safety as its QP0 startup handshake.
        if (!peerStarted_) {
            auto* start = reinterpret_cast<__gm__ uint64_t*>(args.symmetric + PeerPhaseSlot(args, 0U, peer_));
            peerStarted_ = ReadGmByPassDCache(start) >= args.config.epoch;
            if (!peerStarted_) return false;
        }
        // Completion is monotone within an invocation: remember groups already
        // observed ready instead of rereading the entire CV adjacency each poll.
        while (readyGroup_ < chunk_.fragmentCount) {
            const uint32_t index =
                reinterpret_cast<__gm__ uint32_t*>(args.chunkGroupIndices)[chunk_.fragmentBegin + readyGroup_];
            const uint32_t rows = reinterpret_cast<__gm__ Fragment*>(args.fragments)[index].rows;
            auto* flag = reinterpret_cast<__gm__ int32_t*>(args.gmm2Done + static_cast<uint64_t>(index) * kFlagBytes);
            if (ReadGmByPassDCache(flag) < static_cast<int32_t>(ComputeTileCount(rows, args.config.hidden))) {
                return false;
            }
            ++readyGroup_;
        }
        return true;
    }

    __aicore__ inline void Submit(const Arguments& args, Timer& timer)
    {
        if (!started_) {
            timer.Begin(Phase::Combine);
            started_ = true;
        }
        const uint64_t rowBytes = args.config.hidden * sizeof(uint16_t);
        auto meta = CurrentMeta();
        meta.bytes = chunk_.rows * rowBytes;
        const uint64_t start = timer.TraceBegin();
        Put(reinterpret_cast<__gm__ uint8_t*>(args.symmetric), args.config.returnYOffset + chunk_.remoteRow * rowBytes,
            reinterpret_cast<__gm__ uint8_t*>(args.gmm2Output + chunk_.localRow * rowBytes),
            static_cast<uint32_t>(meta.bytes), peer_, kCombineHelperQp);
        timer.TraceEnd(TraceEvent::CombinePut, start, meta);
        bytes_ += meta.bytes;
        pending_ = true;
    }

    __aicore__ inline void CompletePending(const Arguments& args, Timer& timer)
    {
        using namespace AscendC;
        if (!pending_) return;
        auto meta = CurrentMeta();
        uint64_t start = timer.TraceBegin();
        Quiet(peer_, kCombineHelperQp);
        timer.TraceEnd(TraceEvent::CombineQuiet, start, meta);
        // Activation/gather fully close their MTE events before returning.
        // Signal UB lies above both compute scratch regions, with its own event.
        LocalTensor<uint64_t> signal(TPosition::VECCALC, kCombineSignalUbOffset, 4U);
        const uint64_t readyIndex = static_cast<uint64_t>(args.config.rank) * args.config.maxPeerChunks + chunk_.cv;
        start = timer.TraceBegin();
        PublishDone(reinterpret_cast<__gm__ uint8_t*>(args.symmetric),
                    args.config.combineReadyOffset + readyIndex * kFlagBytes, peer_, args.config.epoch, signal,
                    EVENT_ID7);
        timer.TraceEnd(TraceEvent::CombinePublish, start, meta);
        pending_ = false;
        ++next_;
        readyGroup_ = 0U;
        if (HasWork()) {
            chunk_ = LoadChunk(args.recvChunks, next_);
        } else {
            timer.End(Phase::Combine, bytes_, peer_);
            MarkDone(args);
        }
    }

private:
    __aicore__ inline void MarkDone(const Arguments& args)
    {
        // The original peer owner alone sends peer-finished and observes the
        // inbound counter. This flag certifies our last quiet AND CV publication.
        AscendC::WriteGmByPassDCache(LocalCompletionFlag(args, kLocalFlagHelperDoneBase + helper_), int32_t{1});
    }

    uint32_t helper_ = kCombineHelpers;
    uint32_t peer_ = kWorld;
    uint32_t next_ = 0U;
    uint32_t end_ = 0U;
    uint32_t readyGroup_ = 0U;
    Chunk chunk_{};
    uint64_t bytes_ = 0U;
    bool packDone_ = false;
    bool peerStarted_ = false;
    bool pending_ = false;
    bool started_ = false;
};

}  // namespace MegaMoeUrma
