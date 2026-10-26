// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"
#include "compute_contract.hpp"
#include "transport.hpp"

namespace MegaMoeUrma {

__aicore__ inline __gm__ int32_t* LocalCompletionFlag(const Arguments& args, uint32_t index)
{
    return reinterpret_cast<__gm__ int32_t*>(args.localFlags + static_cast<uint64_t>(index) * kFlagBytes);
}

__aicore__ inline uint64_t PeerPhaseSlot(const Arguments& args, uint32_t phase, uint32_t source)
{
    return args.config.barrierOffset + (static_cast<uint64_t>(phase) * args.config.world + source) * kFlagBytes;
}

// AIC completion certifies contiguous scratch, not self returnY. The helper
// team must also finish all final-layout stores before the aggregate bypass.
__aicore__ inline void MarkCubeComplete(const Arguments& args)
{
    AscendC::AtomicAdd(LocalCompletionFlag(args, kLocalFlagAicDone), 1);
}

__aicore__ inline void MarkReorderComplete(const Arguments& args)
{
    AscendC::AtomicAdd(LocalCompletionFlag(args, kLocalFlagReorderDone), 1);
}

// Nonblocking: call once before each Combine quiet, then poll at the tail.
// Each peer has exactly one local observer. Once counted, it never reads that
// flag again in this invocation. Outbound completion alone is NOT incoming
// completion: inspect the peer-owned slot in this rank's symmetric window.
__aicore__ inline bool ObserveIncomingCompletion(const Arguments& args, uint32_t owner, bool& peerCounted,
                                                 bool& selfCounted)
{
    using namespace AscendC;
    if (!peerCounted) {
        const uint32_t peer = PeerForOwner(args.config.rank, owner);
        auto* finished = reinterpret_cast<__gm__ uint64_t*>(args.symmetric + PeerPhaseSlot(args, 1U, peer));
        // Empty peer streams can finish a later invocation before this owner
        // observes the prior one. A later finished epoch also certifies the
        // current return data; per-CV payload flags still require exact epochs.
        if (ReadGmByPassDCache(finished) >= args.config.epoch) {
            AtomicAdd(LocalCompletionFlag(args, kLocalFlagIncomingDone), 1);
            peerCounted = true;
        }
    }
    // Only owner 0 contributes self. Empty AIC/helper workloads still report.
    // All remote finished flags plus AIC28 alone must NOT bypass self reads:
    // scratch can be complete while the last helper's self scatter is pending.
    if (owner == 0U && !selfCounted &&
        ReadGmByPassDCache(LocalCompletionFlag(args, kLocalFlagAicDone)) == static_cast<int32_t>(kAic) &&
        ReadGmByPassDCache(LocalCompletionFlag(args, kLocalFlagReorderDone)) ==
            static_cast<int32_t>(ActivationWorkers(args.config.world))) {
        AtomicAdd(LocalCompletionFlag(args, kLocalFlagIncomingDone), 1);
        selfCounted = true;
    }
    return peerCounted && (owner != 0U || selfCounted);
}

// All outbound CVs have already completed quiet and their per-CV publication.
// Empty edges also publish once. Publish BEFORE waiting for the corresponding
// incoming notification, so peers cannot deadlock while waiting on each other.
// Phase 1 has separate source-owned slots, with monotonically increasing epochs.
__aicore__ inline void FinishCombinePeer(const Arguments& args, uint32_t owner, AscendC::LocalTensor<uint64_t> signal,
                                         bool& peerCounted, bool& selfCounted)
{
    using namespace AscendC;
    const uint32_t peer = PeerForOwner(args.config.rank, owner);
    PublishDone(reinterpret_cast<__gm__ uint8_t*>(args.symmetric), PeerPhaseSlot(args, 1U, args.config.rank), peer,
                args.config.epoch, signal, EVENT_ID0);
    while (!ObserveIncomingCompletion(args, owner, peerCounted, selfCounted)) {
        const uint64_t start = GetSystemCycle();
        while (GetSystemCycle() - start < 200U) {
        }
    }
}

}  // namespace MegaMoeUrma
