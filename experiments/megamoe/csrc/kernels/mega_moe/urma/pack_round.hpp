// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "pack_pipeline.hpp"
#include "runtime_comm_routes.hpp"
#include "timer.hpp"

namespace MegaMoeUrma {

constexpr uint32_t kPackStartupRounds = 2U;

__aicore__ inline __gm__ int32_t* PackReadyFlag(const Arguments& args, uint32_t peer, uint32_t cv)
{
    return reinterpret_cast<__gm__ int32_t*>(
        args.packReady + (static_cast<uint64_t>(peer) * args.config.maxPeerChunks + cv) * kFlagBytes);
}

__aicore__ inline Chunk LoadPackChunk(const Arguments& args, uint32_t round, uint32_t peer)
{
    return RuntimeSendChunk(args, peer, round);
}

__aicore__ inline uint64_t PackPartitionBegin(uint64_t rows, uint32_t worker, uint32_t workers)
{
    const uint64_t remainder = rows % workers;
    return rows / workers * worker + (worker < remainder ? worker : remainder);
}

// One captured MTE3 fence and ready quota. Capture at the old segment's end;
// never recapture it after submitting the next segment's payload.
class PackBatch {
public:
    __aicore__ inline PackBatch(const Arguments& args, Timer& timer) : args_(args), timer_(timer) {}
    __aicore__ inline void RetirePending()
    {
        if (!pending_) return;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(static_cast<AscendC::TEventID>(0U));
        AscendC::AtomicAdd(PackReadyFlag(args_, meta_.peer, meta_.cv), static_cast<int32_t>(meta_.rows));
        pending_ = false;
    }
    __aicore__ inline void Submit(uint32_t peer, uint32_t cv, uint64_t row, uint64_t rows)
    {
        if (rows == 0U) return;
        timer_.Begin(Phase::Pack);
        const uint64_t start = timer_.TraceBegin();
        pipeline_.Submit(args_, row, rows, *this);
        // Submit retires an older fence at min(5,rows), so event0 is free here.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(static_cast<AscendC::TEventID>(0U));
        meta_ = TraceMeta{};
        meta_.peer = peer; meta_.cv = cv; meta_.row = row; meta_.rows = rows;
        meta_.bytes = rows * (args_.config.hidden + args_.config.hidden / 32U);
        pending_ = true;
        // Non-nested trace: this bar contains submission and the previous
        // quota's retirement. Its own ready may be published in a later bar.
        timer_.TraceEnd(TraceEvent::Pack, start, meta_);
        timer_.End(Phase::Pack, meta_.bytes);
    }
    __aicore__ inline void FlushReady()
    {
        if (!pending_) return;
        const uint64_t start = timer_.TraceBegin();
        RetirePending();
        TraceMeta meta = meta_; meta.bytes = 0U;
        timer_.TraceEnd(TraceEvent::PackRetire, start, meta);
    }
    __aicore__ inline void Close()
    {
        if (!pipeline_.Active()) return;
        const uint64_t start = timer_.TraceBegin();
        RetirePending();
        pipeline_.Close();
        TraceMeta meta = meta_; meta.bytes = 0U;
        timer_.TraceEnd(TraceEvent::PackRetire, start, meta);
    }
private:
    const Arguments& args_;
    Timer& timer_;
    PackPipeline pipeline_;
    TraceMeta meta_{};
    bool pending_ = false;
};

// A virtual round concatenates one CV per target, including self. Splitting
// that virtual interval balances rows across 56 startup / 28 steady workers.
// A CV may cross worker boundaries; its local atomic counter counts completed
// rows, never claimed rows. Each static row is copied and counted exactly once.
__aicore__ inline void PackOneRound(const Arguments& args, uint32_t round, uint32_t worker, uint32_t workers,
                                    PackBatch& pack)
{
    if (round >= args.config.packRoundCount) return;
    const uint64_t rows = RuntimePackRoundRows(args, round);
    const uint64_t begin = PackPartitionBegin(rows, worker, workers);
    const uint64_t end = PackPartitionBegin(rows, worker + 1U, workers);
    if (begin == end) return;
    uint64_t cursor = 0U;
    for (uint32_t peer = 0U; peer < args.config.world && cursor < end; ++peer) {
        const Chunk chunk = LoadPackChunk(args, round, peer);
        const uint64_t next = cursor + chunk.rows;
        if (next > begin && cursor < end) {
            const uint64_t first = begin > cursor ? begin : cursor;
            const uint64_t last = end < next ? end : next;
            if (first < last) {
                pack.Submit(peer, round, chunk.localRow + first - cursor, last - first);
            }
        }
        cursor = next;
    }
}

// Compatibility wrapper: callers that cannot retain events close this round.
__aicore__ inline void PackOneRound(const Arguments& args, uint32_t round, uint32_t worker,
                                    uint32_t workers, Timer& timer)
{
    PackBatch pack(args, timer);
    PackOneRound(args, round, worker, workers, pack);
    pack.Close();
}

__aicore__ inline void MarkPackWorkerComplete(const Arguments& args)
{
    auto* done = reinterpret_cast<__gm__ int32_t*>(args.localFlags + kLocalFlagPackDone * kFlagBytes);
    AscendC::AtomicAdd(done, 1);
}

}  // namespace MegaMoeUrma
