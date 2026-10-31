// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once

#include <cstdint>
#include "combine_ready.hpp"
#include "kernel_operator.h"
#include "shmem.h"

namespace MegaMoeUrma {

// Defaults/capacity only. Kernel calls pass the active Config::world.
constexpr uint32_t kWorldSize = 16U;
constexpr uint32_t kSignalSlotBytes = 128U;
constexpr uint32_t kSignalCopyBytes = 32U;
constexpr uint32_t kMaxPutBytes = 256U * 1024U * 1024U;
constexpr uint32_t kDispatchQp = 0U;
constexpr uint32_t kCombineQp = 0U;

// Dispatch completes before Combine. Both use the same peer-owned QP0 because
// the SO atomic API publishes on QP0. No second worker may touch that SQ/CQ.
__aicore__ inline uint32_t PeerForOwner(uint32_t rank, uint32_t owner)
{
    return owner < rank ? owner : owner + 1U;
}

// dstOffset names a LOCAL symmetric-window address; SHMEM translates it for
// peer. Do not pass an already translated MTE peer pointer to this function.
// The caller completes local GM producers before submitting and retains src
// unchanged until Quiet(peer, qp) returns. This transport deliberately has no GET.
__aicore__ inline void Put(__gm__ uint8_t* symmetricBase, uint64_t dstOffset, __gm__ uint8_t* src, uint32_t bytes,
                           uint32_t peer, uint32_t qp = kDispatchQp)
{
    if (bytes == 0U) return;
    if (bytes > kMaxPutBytes) {
        aclshmemi_kernel_abort("MegaMoE URMA PUT exceeds 256 MiB: bytes=%u peer=%u\n", bytes, peer);
        return;
    }
    if (qp != kDispatchQp) {
        aclshmemi_kernel_abort("MegaMoE commfirst SO requires QP0: peer=%u qp=%u\n", peer, qp);
        return;
    }
    aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_S, ACLSHMEMX_UDMA_OP_CONFIG_RO>(
        symmetricBase + dstOffset, src, nullptr, bytes, peer, qp, 0U);
}

// Blocking completion of only this peer/QP, including all prior SO markers.
// The one-owner rule also applies here: no second worker may poll the same CQ.
__aicore__ inline void Quiet(uint32_t peer, uint32_t qp = kDispatchQp)
{
    if (qp != kDispatchQp) {
        aclshmemi_kernel_abort("MegaMoE commfirst SO requires QP0 quiet: peer=%u qp=%u\n", peer, qp);
        return;
    }
    aclshmemx_udma_qp_quiet(peer, qp);
}

// One SO operation can mark a contiguous merged run of up to 64 CVs. The
// sender's submitted bitmap must guarantee that no bit is ever added twice:
// addition of DISJOINT masks is exactly OR, with no carry between bits.
// Host clears this receiver-owned area before start publication; every sender
// waits for that start, and final quiet retires all flags before reuse.
__aicore__ inline void PublishCombineRange(__gm__ uint8_t* symmetricBase, uint64_t readyOffset,
                                           uint32_t maxPeerChunks, uint32_t source, uint32_t peer,
                                           uint32_t firstCv, uint32_t count)
{
    if (!CombineRangeFitsWord(firstCv, count) || firstCv >= maxPeerChunks || count > maxPeerChunks - firstCv) {
        aclshmemi_kernel_abort("Invalid Combine SO range: cv=%u count=%u capacity=%u\n", firstCv, count, maxPeerChunks);
        return;
    }
    auto* flag = reinterpret_cast<__gm__ uint64_t*>(
        symmetricBase + readyOffset + CombineReadyWordIndex(maxPeerChunks, source, firstCv) * kSignalSlotBytes);
    aclshmemx_udma_atomic_add<uint64_t, ACLSHMEMX_UDMA_OP_CONFIG_SO>(flag, CombineReadyMask(firstCv, count), peer);
}

// One marker per slot/invocation (Dispatch CV or peer-finished). The receiver
// resets these slots before PublishStart; adding epoch writes it exactly once.
// Same-QP SO makes each flag visible only after preceding payload writes.
__aicore__ inline void PublishOrdered(__gm__ uint8_t* symmetricBase, uint64_t offset,
                                      uint32_t peer, uint64_t epoch)
{
    aclshmemx_udma_atomic_add<uint64_t, ACLSHMEMX_UDMA_OP_CONFIG_SO>(
        reinterpret_cast<__gm__ uint64_t*>(symmetricBase + offset), epoch, peer);
}

// Immutable 32B epoch signal throughout the invocation. Enqueue does not wait
// for MTE3: the next payload may be submitted before Drain waits for previous
// publications. Only after Drain may the caller overwrite/reuse this UB.
class AsyncPublishBuffer {
public:
    __aicore__ inline AsyncPublishBuffer(AscendC::LocalTensor<uint64_t> signal, AscendC::TEventID event)
        : signal_(signal), event_(event) {}

    __aicore__ inline void Init(uint64_t epoch)
    {
        Drain();
        signal_.SetValue(0, epoch);
        signal_.SetValue(1, 0U);
        signal_.SetValue(2, 0U);
        signal_.SetValue(3, 0U);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(event_);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(event_);
    }

    __aicore__ inline void Enqueue(__gm__ uint8_t* symmetricBase, uint64_t offset, uint32_t peer)
    {
        AscendC::GlobalTensor<uint64_t> destination;
        destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(aclshmem_ptr(symmetricBase + offset, peer)));
        AscendC::DataCopy(destination, signal_, kSignalCopyBytes / sizeof(uint64_t));
        pending_ = true;
    }

    __aicore__ inline void Drain()
    {
        if (!pending_) return;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(event_);
        pending_ = false;
    }

private:
    AscendC::LocalTensor<uint64_t> signal_;
    AscendC::TEventID event_;
    bool pending_ = false;
};

// Publish only AFTER Quiet(peer, phaseQp) covered every payload PUT for this chunk.
// Each (source peer, CV chunk, phase) owns a separate 128B destination slot.
// The caller supplies 32B of UB and a free event ID; no hidden TPipe/UB state
// is introduced. MTE3_S protects both publication and immediate UB reuse.
__aicore__ inline void PublishDone(__gm__ uint8_t* symmetricBase, uint64_t signalOffset, uint32_t peer,
                                   uint64_t sequence, AscendC::LocalTensor<uint64_t> signalUb,
                                   AscendC::TEventID eventId)
{
    signalUb.SetValue(0, sequence);
    signalUb.SetValue(1, 0U);
    signalUb.SetValue(2, 0U);
    signalUb.SetValue(3, 0U);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
    auto* remoteSlot = reinterpret_cast<__gm__ uint64_t*>(aclshmem_ptr(symmetricBase + signalOffset, peer));
    AscendC::GlobalTensor<uint64_t> done;
    done.SetGlobalBuffer(remoteSlot);
    AscendC::DataCopy(done, signalUb, kSignalCopyBytes / sizeof(uint64_t));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
}

// Local polling only: this is not a remote read/GET and may be used by AIC or
// AIV. ReadGmByPassDCache avoids a stale locally cached signal. Payload consumers
// must likewise use the stage's coherent/cache-bypassing input access policy.
__aicore__ inline void WaitLocalDone(__gm__ uint8_t* localSlot, uint64_t sequence)
{
    auto* value = reinterpret_cast<__gm__ uint64_t*>(localSlot);
    while (AscendC::ReadGmByPassDCache(value) != sequence) {
    }
}

// Empty peer edges may advance to the next invocation before the slow owner
// observes the old start. Accept a later start only here; Dispatch flags retain
// exact epochs and Combine bitmaps were cleared before this start. Any nonempty edge prevents such advancement
// until its old payload has been consumed and locally reduced.
__aicore__ inline void WaitStartAtLeast(__gm__ uint8_t* localSlot, uint64_t sequence)
{
    auto* value = reinterpret_cast<__gm__ uint64_t*>(localSlot);
    while (AscendC::ReadGmByPassDCache(value) < sequence) {
    }
}

// Only one local leader enters this function AFTER joining all local workers
// and their per-peer Quiet calls. All remote reads are of this rank's own flag
// slots. Each source rank owns one cache line in every destination rank.
// sequence must increase across phases and invocations; >= tolerates a fast
// peer reaching the next barrier before a slow peer observes this barrier.
// Protect returnY reuse after reduce with a final barrier or a next-invocation
// peer handshake whose publication follows completion of every local reducer.
__aicore__ inline void RankBarrier(__gm__ uint8_t* symmetricBase, uint64_t barrierOffset, uint32_t rank,
                                   uint64_t sequence, AscendC::LocalTensor<uint64_t> signalUb,
                                   AscendC::TEventID eventId, uint32_t world = kWorldSize)
{
    for (uint32_t peer = 0; peer < world; ++peer) {
        PublishDone(symmetricBase, barrierOffset + static_cast<uint64_t>(rank) * kSignalSlotBytes, peer, sequence,
                    signalUb, eventId);
    }
    for (uint32_t peer = 0; peer < world; ++peer) {
        auto* localSlot = reinterpret_cast<__gm__ uint64_t*>(symmetricBase + barrierOffset +
                                                             static_cast<uint64_t>(peer) * kSignalSlotBytes);
        while (AscendC::ReadGmByPassDCache(localSlot) < sequence) {
        }
    }
}

}  // namespace MegaMoeUrma
