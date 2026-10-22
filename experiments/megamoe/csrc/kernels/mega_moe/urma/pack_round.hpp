// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "pack_pipeline.hpp"
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
    const auto* gm = &reinterpret_cast<const __gm__ PackRound*>(args.packRounds)[round].chunks[peer];
    return {gm->peer,          gm->cv,      gm->localRow, gm->remoteRow, gm->rows, gm->fragmentBegin,
            gm->fragmentCount, gm->reserved};
}

__aicore__ inline uint64_t PackPartitionBegin(uint64_t rows, uint32_t worker, uint32_t workers)
{
    const uint64_t remainder = rows % workers;
    return rows / workers * worker + (worker < remainder ? worker : remainder);
}

// A virtual round concatenates one CV per target, including self. Splitting
// that virtual interval balances rows across 56 startup / 28 steady workers.
// A CV may cross worker boundaries; its local atomic counter counts completed
// rows, never claimed rows. Each static row is copied and counted exactly once.
__aicore__ inline void PackOneRound(const Arguments& args, uint32_t round, uint32_t worker, uint32_t workers,
                                    Timer& timer)
{
    if (round >= args.config.packRoundCount) return;
    const auto* gm = reinterpret_cast<const __gm__ PackRound*>(args.packRounds) + round;
    const uint64_t rows = gm->rows;
    const uint64_t begin = PackPartitionBegin(rows, worker, workers);
    const uint64_t end = PackPartitionBegin(rows, worker + 1U, workers);
    if (begin == end) return;
    uint64_t cursor = 0U;
    for (uint32_t peer = 0U; peer < kWorld && cursor < end; ++peer) {
        const Chunk chunk = LoadPackChunk(args, round, peer);
        const uint64_t next = cursor + chunk.rows;
        if (next > begin && cursor < end) {
            const uint64_t first = begin > cursor ? begin : cursor;
            const uint64_t last = end < next ? end : next;
            if (first < last) {
                TraceMeta meta;
                meta.peer = peer;
                meta.cv = round;
                meta.row = chunk.localRow + first - cursor;
                meta.rows = last - first;
                meta.bytes = meta.rows * (args.config.hidden + args.config.hidden / 32U);
                timer.Begin(Phase::Pack);
                const uint64_t start = timer.TraceBegin();
                ExpandRows(args, meta.row, meta.rows);
                AscendC::AtomicAdd(PackReadyFlag(args, peer, round), static_cast<int32_t>(meta.rows));
                timer.TraceEnd(TraceEvent::Pack, start, meta);
                timer.End(Phase::Pack, meta.bytes);
            }
        }
        cursor = next;
    }
}

__aicore__ inline void MarkPackWorkerComplete(const Arguments& args)
{
    auto* done = reinterpret_cast<__gm__ int32_t*>(args.localFlags + kLocalFlagPackDone * kFlagBytes);
    AscendC::AtomicAdd(done, 1);
}

}  // namespace MegaMoeUrma
