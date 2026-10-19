// Copyright (c) 2026, SimpleBrightMan
//
// Minimal All-to-All payload mover for the UBX MTE path.
//
// Each rank owns `world * tokens * token_bytes` bytes of the shared symmetric
// heap. Source rank s writes its per-destination block to the destination's
// region s, so the destination address offset is independent of the peer and
// only the mapped base pointer changes. Remote stores use the same
// aclshmem_ptr + MTE3 DataCopy path as MegaMoE's token communication; the
// local leg uses the identical GM->UB->GM path without a remote mapping.

#include "kernel_operator.h"
#include "shmem.h"
#include "launch.hpp"

using namespace AscendC;

namespace {

constexpr uint32_t kTileBytes = 16U * 1024U;
constexpr uint32_t kBuffers = 4U;
constexpr uint32_t kUbBytes = kTileBytes * kBuffers;
constexpr uint32_t kDrainUbBytes = 2U * kTileBytes;
constexpr uint32_t kMaxWorld = 128U;

// mode: 0 = self only, 1 = same physical 4-card group (excluding self),
//       2 = outside the group, 3 = every rank including self.
__aicore__ inline bool SelectedPeer(uint32_t peer, uint32_t rank, uint32_t group, uint32_t mode)
{
    const bool same_group = (peer / group) == (rank / group);
    if (mode == 0U) {
        return peer == rank;
    }
    if (mode == 1U) {
        return same_group && peer != rank;
    }
    if (mode == 2U) {
        return !same_group;
    }
    return true;
}

__aicore__ inline void LoadTile(LocalTensor<uint8_t> bank, __gm__ uint8_t* source, uint32_t bytes)
{
    GlobalTensor<uint8_t> src;
    src.SetGlobalBuffer(source);
    DataCopyPad(bank, src, DataCopyExtParams(1U, bytes, 0U, 0U, 0U),
                DataCopyPadExtParams<uint8_t>{false, 0U, 0U, 0U});
}

__aicore__ inline void StoreTile(__gm__ uint8_t* destination, LocalTensor<uint8_t> bank, uint32_t bytes)
{
    GlobalTensor<uint8_t> dst;
    dst.SetGlobalBuffer(destination);
    DataCopyPad(dst, bank, DataCopyExtParams(1U, bytes, 0U, 0U, 0U));
}

}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void alltoall_mte_kernel(
    GM_ADDR send, GM_ADDR workspace, GM_ADDR timing, uint32_t rank, uint32_t world, uint32_t group,
    uint32_t token_bytes, uint32_t tokens, uint32_t mode, uint32_t num_cores)
{
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
    util_set_ffts_config(0U);
    if (world == 0U || world > kMaxWorld || group == 0U || rank >= world || mode > 3U || token_bytes == 0U ||
        token_bytes % 32U != 0U || tokens == 0U || num_cores == 0U) {
        return;
    }
    const uint32_t core = GetBlockIdx();
    if (core >= num_cores) {
        return;
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> ub_buffer;
    pipe.InitBuffer(ub_buffer, kUbBytes);
    auto ub = ub_buffer.Get<uint8_t>();

    const uint64_t region = uint64_t(token_bytes) * uint64_t(tokens);
    auto* local_base = reinterpret_cast<__gm__ uint8_t*>(workspace) + uint64_t(rank) * region;

    uint32_t peers[kMaxWorld];
    GM_ADDR bases[kMaxWorld];
    uint32_t peer_count = 0U;
    for (uint32_t peer = 0U; peer < world; ++peer) {
        if (!SelectedPeer(peer, rank, group, mode)) {
            continue;
        }
        peers[peer_count] = peer;
        bases[peer_count] = peer == rank
                                ? reinterpret_cast<GM_ADDR>(local_base)
                                : reinterpret_cast<GM_ADDR>(aclshmem_ptr(local_base, static_cast<int>(peer)));
        ++peer_count;
    }
    if (peer_count == 0U) {
        return;
    }

    for (uint32_t slot = 0U; slot < kBuffers; ++slot) {
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slot));
    }

    const uint64_t start_cycle = GetSystemCycle();
    uint64_t bytes_done = 0U;
    const uint64_t chunks = (region + kTileBytes - 1U) / kTileBytes;
    const uint64_t total = uint64_t(peer_count) * chunks;
    uint32_t slot = 0U;
    for (uint64_t task = core; task < total; task += num_cores) {
        const uint32_t peer_index = static_cast<uint32_t>(task % peer_count);
        const uint64_t chunk = task / peer_count;
        const uint64_t offset = chunk * kTileBytes;
        const uint64_t remaining = region - offset;
        const uint32_t bytes = static_cast<uint32_t>(remaining < kTileBytes ? remaining : kTileBytes);
        const auto event = static_cast<TEventID>(slot);
        auto bank = ub[slot * kTileBytes];

        WaitFlag<HardEvent::MTE3_MTE2>(event);
        LoadTile(bank, send + uint64_t(peers[peer_index]) * region + offset, bytes);
        SetFlag<HardEvent::MTE2_MTE3>(event);
        WaitFlag<HardEvent::MTE2_MTE3>(event);
        StoreTile(bases[peer_index] + offset, bank, bytes);
        SetFlag<HardEvent::MTE3_MTE2>(event);
        bytes_done += bytes;
        slot = (slot + 1U) % kBuffers;
    }
    for (uint32_t pending = 0U; pending < kBuffers; ++pending) {
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(pending));
    }
    aclshmemx_mte_quiet();
    const uint64_t end_cycle = GetSystemCycle();
    if (timing != nullptr) {
        auto record = ub.ReinterpretCast<uint64_t>();
        record.SetValue(0U, start_cycle);
        record.SetValue(1U, end_cycle);
        record.SetValue(2U, bytes_done);
        record.SetValue(3U, 0U);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
        GlobalTensor<uint64_t> timing_gm;
        timing_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(timing) + uint64_t(core) * 4U);
        DataCopyPad(timing_gm, record, DataCopyExtParams(1U, 32U, 0U, 0U, 0U));
        SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
    }
#endif
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void alltoall_mte_drain_kernel(
    GM_ADDR workspace, GM_ADDR out, uint64_t total_bytes, uint32_t num_cores)
{
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
    util_set_ffts_config(0U);
    if (total_bytes == 0U || num_cores == 0U) {
        return;
    }
    const uint32_t core = GetBlockIdx();
    if (core >= num_cores) {
        return;
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> ub_buffer;
    pipe.InitBuffer(ub_buffer, kDrainUbBytes);
    auto ub = ub_buffer.Get<uint8_t>();
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

    const uint64_t chunks = (uint64_t(total_bytes) + kTileBytes - 1U) / kTileBytes;
    uint32_t slot = 0U;
    for (uint64_t chunk = core; chunk < chunks; chunk += num_cores) {
        const uint64_t offset = chunk * kTileBytes;
        const uint64_t remaining = uint64_t(total_bytes) - offset;
        const uint32_t bytes = static_cast<uint32_t>(remaining < kTileBytes ? remaining : kTileBytes);
        const auto event = slot == 0U ? EVENT_ID0 : EVENT_ID1;
        auto bank = ub[slot * kTileBytes];
        WaitFlag<HardEvent::MTE3_MTE2>(event);
        LoadTile(bank, workspace + offset, bytes);
        SetFlag<HardEvent::MTE2_MTE3>(event);
        WaitFlag<HardEvent::MTE2_MTE3>(event);
        StoreTile(out + offset, bank, bytes);
        SetFlag<HardEvent::MTE3_MTE2>(event);
        slot ^= 1U;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
#endif
}

extern "C" ASCEND_DEEPEP_EXPORT void alltoall_mte_kernel_do(void* stream, uint8_t* send, uint8_t* workspace,
                                                            uint8_t* timing, uint32_t rank, uint32_t world,
                                                            uint32_t group, uint32_t token_bytes, uint32_t tokens,
                                                            uint32_t mode, uint32_t num_cores)
{
    alltoall_mte_kernel<<<num_cores, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kUbBytes), stream>>>(
        send, workspace, timing, rank, world, group, token_bytes, tokens, mode, num_cores);
}

extern "C" ASCEND_DEEPEP_EXPORT void alltoall_mte_drain_kernel_do(void* stream, uint8_t* workspace, uint8_t* out,
                                                                 uint64_t total_bytes, uint32_t num_cores)
{
    alltoall_mte_drain_kernel<<<num_cores, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(kDrainUbBytes), stream>>>(workspace, out,
                                                                                                   total_bytes,
                                                                                                   num_cores);
}
