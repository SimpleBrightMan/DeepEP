#include "kernel_operator.h"
#include "shmem.h"
#include "simt_api/device_functions.h"
#include "tiling.hpp"
#include "launch.hpp"
using namespace AscendC;
namespace {
template <typename T>
__aicore__ inline void NativeLoad(LocalTensor<T> ub, __gm__ T* source, uint32_t count)
{
    if (!count) {
        return;
    }
    GlobalTensor<T> gm;
    gm.SetGlobalBuffer(source);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
    DataCopyPad(ub, gm, DataCopyExtParams(1, count * sizeof(T), 0, 0, 0), DataCopyPadExtParams<T>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
}
template <typename T>
__aicore__ inline void NativeStore(__gm__ T* destination, LocalTensor<T> ub, uint32_t count)
{
    if (!count) {
        return;
    }
    GlobalTensor<T> gm;
    gm.SetGlobalBuffer(destination);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
    DataCopyPad(gm, ub, DataCopyExtParams(1, count * sizeof(T), 0, 0, 0));
    SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
}
}  // namespace

namespace {
// Payload sources and destinations are offsets in this Session SHMEM heap.
// Keep peer ownership,
// group credits, dual-QP SO completion and local output-copy scheduling together.
#if (defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)) && CATLASS_ARCH == 3510
// Preserve cross-server group/credit and dual-QP 3:1. Secondary is in dispatch.cpp.
// Adaptation: stable Notify rows and symmetric workspace offsets, no expert migration.
namespace DispatchMoonep {
__aicore__ inline __gm__ uint8_t* LivePeerWindow(GM_ADDR workspace, __gm__ uint64_t* book, int32_t peer,
                                                 uint32_t book_words = 2U)
{
    // SHMEM translates the local registered offset exactly once.
    // Dispatch invalidates this call's address table after the publication
    // barrier. This is scalar AICore code; asc_ldcg is a SIMT VF intrinsic.
    GlobalTensor<uint64_t> addresses;
    addresses.SetGlobalBuffer(book);
    return workspace + addresses.GetValue(uint64_t(peer) * book_words);
}

constexpr uint32_t kUdmaWqeBatchSize = 128;
constexpr uint32_t kUdmaWqeBytes = 64;
// Payload uses QP0/1; control values occupy separate 512B workspace slots.
constexpr uint32_t kMaxCreatedUdmaQpCount = 6;
constexpr uint32_t kMaxActiveUdmaQpCount = 2;
constexpr uint32_t kActiveUdmaQpIndices[kMaxActiveUdmaQpCount] = {0, 1};
constexpr uint32_t kPeerCompletionValueSlot = 3;
constexpr uint32_t kPeerCreditFlagSlot = 5;
static_assert(kPeerCompletionValueSlot != kActiveUdmaQpIndices[0] &&
              kPeerCompletionValueSlot != kActiveUdmaQpIndices[1]);
static_assert(kPeerCompletionValueSlot != kPeerCreditFlagSlot);
static_assert(kActiveUdmaQpIndices[kMaxActiveUdmaQpCount - 1] < kMaxCreatedUdmaQpCount);
static_assert(kPeerCreditFlagSlot < kMaxCreatedUdmaQpCount);
static_assert(kPeerCreditFlagSlot > kActiveUdmaQpIndices[kMaxActiveUdmaQpCount - 1]);

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
inline constexpr aclshmemx_udma_op_config_t kUdmaDataRoNoCqeConfig{0U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t kUdmaDataRoCqeConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
inline constexpr aclshmemx_udma_op_config_t kUdmaSyncSoCqeConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_SO};
#endif

constexpr uint32_t kUdmaWqeScratchBytes = kUdmaWqeBatchSize * kUdmaWqeBytes;
constexpr uint32_t kWeightGmConflictGroupBytes = 512;
constexpr uint32_t kWeightsPerGmConflictGroup = kWeightGmConflictGroupBytes / sizeof(float);
constexpr uint32_t kWeightSimtThreadCount = kWeightsPerGmConflictGroup;
constexpr uint32_t kMteUbBytes = 16 * 1024;
constexpr uint32_t kLocalCopyTileBytes = 16 * 1024;
constexpr uint32_t slotChunkCount = 8192;
constexpr uint32_t kSlotTensorBytes = slotChunkCount * sizeof(int32_t);
constexpr uint32_t kSlotCompareMaskBytes = ((slotChunkCount / 8 + 31) / 32) * 32;
constexpr uint32_t kSlotSelectUbBytes = 4 * kSlotTensorBytes + kSlotCompareMaskBytes;
constexpr uint32_t kDuplicateHiddenStageBytes = 65504;
constexpr uint32_t kDuplicateMaxPrimaryRows = 4;
constexpr uint32_t kDuplicateHiddenStagingBytes = 2 * kDuplicateHiddenStageBytes;
constexpr uint32_t kAivUbCapacityBytes = 192 * 1024;
constexpr uint32_t kMaxDispatchGroupWidth = 16;
// 128P production topology: two 64P frames, each laid out as 8 direct-link
// rows x 8 columns.  A red-box dispatch group is one 4-row x 4-column tile,
// so each frame contains two row blocks x two left/right sides = four groups.
constexpr uint32_t kDispatchFrameCount = 2;
constexpr uint32_t kDispatchRanksPerFrame = 64;
constexpr uint32_t kDispatchRanksPerDirectRow = 8;
constexpr uint32_t kDispatchRowsPerFrame = 8;
constexpr uint32_t kDispatchGroupRows = 4;
constexpr uint32_t kDispatchGroupColumns = 4;
constexpr uint32_t kDispatchGroupRowBlocks = kDispatchRowsPerFrame / kDispatchGroupRows;
constexpr uint32_t kDispatchGroupSides = kDispatchRanksPerDirectRow / kDispatchGroupColumns;
constexpr uint32_t kDispatchGroupsPerFrame = kDispatchGroupRowBlocks * kDispatchGroupSides;
constexpr uint32_t kTwoFrameDispatchWorldSize = kDispatchFrameCount * kDispatchRanksPerFrame;

// 128P uses the frozen staged, two-frame schedule.
constexpr int64_t kPeerSyncFlagBytes = 512;
constexpr int64_t kPeerSyncSlotBytes = kMaxCreatedUdmaQpCount * kPeerSyncFlagBytes;
constexpr int64_t kRankSyncSlotBytes = kMaxDispatchGroupWidth * kPeerSyncSlotBytes;
constexpr uint32_t kMaxDispatchAivCount = 32;
constexpr int64_t kGroupSyncFlagBytes = 512;
constexpr uint32_t kGroupSyncRoundCount = 4;
constexpr uint32_t kGroupSyncPhasesPerGroup = 2;
constexpr uint32_t kGroupSyncCompletionPhase = 0;
constexpr uint32_t kGroupSyncDonePhase = 1;
constexpr int32_t kStrictGroupSyncMaxWorldSize = 64;
static_assert(kPeerSyncFlagBytes >= sizeof(uint64_t));
static_assert(kPeerSyncFlagBytes % sizeof(uint64_t) == 0);
static_assert(kGroupSyncFlagBytes >= sizeof(uint64_t));
static_assert(kGroupSyncFlagBytes % sizeof(uint64_t) == 0);
static_assert(kMaxDispatchGroupWidth % 2 == 0);
static_assert(kMaxDispatchAivCount > kMaxDispatchGroupWidth);
static_assert((1U << kGroupSyncRoundCount) >= kMaxDispatchGroupWidth);
static_assert(kDispatchGroupRows * kDispatchGroupColumns == kMaxDispatchGroupWidth);
static_assert(kDispatchRowsPerFrame * kDispatchRanksPerDirectRow == kDispatchRanksPerFrame);
static_assert(kDispatchGroupsPerFrame * kMaxDispatchGroupWidth == kDispatchRanksPerFrame);
static_assert(kDispatchGroupRowBlocks == 2);
static_assert(kDispatchGroupSides == 2);
static_assert(kDispatchGroupsPerFrame == 4);
static_assert(kTwoFrameDispatchWorldSize == 128);
static_assert(kWeightGmConflictGroupBytes % sizeof(float) == 0);
static_assert(kWeightSimtThreadCount == 128);
static_assert(kDuplicateHiddenStagingBytes <= kSlotSelectUbBytes);
static_assert(kUdmaWqeScratchBytes + kMteUbBytes + kLocalCopyTileBytes + kSlotSelectUbBytes <=
              kAivUbCapacityBytes - 8 * 1024);

constexpr uint32_t kVectorSlotSelectCountAlign = 64;
// Local MESH QPs remain peer-private in the shared-Clos SHMEM backend.
// Cross QPs are physical shared SQs, so each sending AIV owns one pair.
__aicore__ inline uint32_t DispatchPhysicalQp(uint32_t rank, uint32_t peer, uint32_t slot)
{
    return rank / 8U == peer / 8U ? slot : 2U * (GetBlockIdx() - 8U) + slot;
}
__aicore__ inline uint32_t GetChunkQpSlot(int64_t chunk_index, uint32_t active_qp_count)
{
    if (active_qp_count == 2) {
        return chunk_index % 4 == 3 ? 1U : 0U;
    }
    return static_cast<uint32_t>(chunk_index % active_qp_count);
}

__aicore__ inline __gm__ uint64_t* GetPeerCompletionFlag(int32_t src_rank, uint32_t peer_rank_index, uint32_t qp_idx,
                                                         __gm__ uint8_t* sync_buffer)
{
    return reinterpret_cast<__gm__ uint64_t*>(sync_buffer + static_cast<int64_t>(src_rank) * kRankSyncSlotBytes +
                                              static_cast<int64_t>(peer_rank_index) * kPeerSyncSlotBytes +
                                              qp_idx * kPeerSyncFlagBytes);
}

__aicore__ inline __gm__ uint64_t* GetPeerCreditFlag(int32_t src_rank, uint32_t peer_rank_index,
                                                     __gm__ uint8_t* sync_buffer)
{
    return reinterpret_cast<__gm__ uint64_t*>(sync_buffer + static_cast<int64_t>(src_rank) * kRankSyncSlotBytes +
                                              static_cast<int64_t>(peer_rank_index) * kPeerSyncSlotBytes +
                                              kPeerCreditFlagSlot * kPeerSyncFlagBytes);
}

__aicore__ inline uint64_t GetPeerCreditToken(uint64_t magic, uint32_t group, uint32_t group_count)
{
    return magic * (static_cast<uint64_t>(group_count) + 1U) + group + 1U;
}

__aicore__ inline __gm__ uint64_t* GetGroupSyncFlag(uint32_t aiv_index, uint32_t round,
                                                    __gm__ uint8_t* group_sync_buffer)
{
    const int64_t flag_index = static_cast<int64_t>(aiv_index) * kGroupSyncRoundCount + round;
    return reinterpret_cast<__gm__ uint64_t*>(group_sync_buffer + flag_index * kGroupSyncFlagBytes);
}

__aicore__ inline uint64_t GetGroupSyncInitToken(uint64_t magic, uint32_t group_count)
{
    const uint64_t token_count = static_cast<uint64_t>(group_count) * kGroupSyncPhasesPerGroup + 1U;
    return magic * token_count + 1U;
}

__aicore__ inline uint64_t GetGroupSyncPhaseToken(uint64_t magic, uint32_t group, uint32_t group_count, uint32_t phase)
{
    const uint64_t token_count = static_cast<uint64_t>(group_count) * kGroupSyncPhasesPerGroup + 1U;
    const uint64_t group_token_offset = static_cast<uint64_t>(group) * kGroupSyncPhasesPerGroup + phase + 2U;
    return magic * token_count + group_token_offset;
}

// Local dissemination is used ONLY by AIV8..15; FM/local-copy never join it.
__aicore__ inline void SyncActiveAivs(uint32_t aiv_index, uint32_t active_aiv_count, uint64_t sync_token,
                                      __gm__ uint8_t* group_sync_buffer)
{
    for (uint32_t distance = 1U, round = 0U; distance < active_aiv_count; distance <<= 1U, ++round) {
        const uint32_t next_aiv = (aiv_index + distance) % active_aiv_count;
        auto* next = GetGroupSyncFlag(next_aiv, round, group_sync_buffer);
        *next = sync_token;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(next));
        aclshmem_uint64_wait_until(GetGroupSyncFlag(aiv_index, round, group_sync_buffer), ACLSHMEM_CMP_GE, sync_token);
    }
}

__aicore__ inline uint32_t GetDispatchGroupWidth(int32_t world_size)
{
    return world_size < static_cast<int32_t>(kMaxDispatchGroupWidth) ? static_cast<uint32_t>(world_size)
                                                                     : kMaxDispatchGroupWidth;
}

__aicore__ inline bool UseTwoFrameDispatchSchedule(int32_t world_size)
{
    return world_size == static_cast<int32_t>(kTwoFrameDispatchWorldSize);
}

// Four rounds per frame, with the original per-edge completion/credit dependency.
__aicore__ inline uint32_t GetDispatchFrameToggle(uint32_t group)
{
    return group / kDispatchGroupsPerFrame;
}

__aicore__ inline uint32_t GetDispatchFrameSubround(uint32_t group)
{
    return group % kDispatchGroupsPerFrame;
}

__aicore__ inline int32_t BuildTwoFrameDispatchRank(uint32_t frame, uint32_t group_row, uint32_t group_side,
                                                    uint32_t group_rank_index)
{
    const uint32_t lane_row = group_rank_index / kDispatchGroupColumns;
    const uint32_t lane_column = group_rank_index % kDispatchGroupColumns;
    const uint32_t row = group_row * kDispatchGroupRows + lane_row;
    const uint32_t column = group_side * kDispatchGroupColumns + lane_column;
    return static_cast<int32_t>(frame * kDispatchRanksPerFrame + row * kDispatchRanksPerDirectRow + column);
}

__aicore__ inline uint32_t GetGroupRankIndex(int32_t rank, int32_t world_size)
{
    if (UseTwoFrameDispatchSchedule(world_size)) {
        const uint32_t local_rank = static_cast<uint32_t>(rank) % kDispatchRanksPerFrame;
        const uint32_t row = local_rank / kDispatchRanksPerDirectRow;
        const uint32_t column = local_rank % kDispatchRanksPerDirectRow;
        return (row % kDispatchGroupRows) * kDispatchGroupColumns + column % kDispatchGroupColumns;
    }

    const int32_t half_world_size = world_size / 2;
    const uint32_t half_group_width = GetDispatchGroupWidth(world_size) / 2;
    return static_cast<uint32_t>(rank / half_world_size) * half_group_width +
           static_cast<uint32_t>(rank % static_cast<int32_t>(half_group_width));
}

// 128P staged policy: left/right groups start in opposite frames, then switch
// frames after four rounds. Preserve independent two-sequence credit progress.
// For every round the transform is bijective, so every destination red-box
// group receives from exactly one source red-box group.
__aicore__ inline int32_t GetGroupDstRank(int32_t rank, int32_t world_size, uint32_t group, uint32_t group_rank_index)
{
    if (UseTwoFrameDispatchSchedule(world_size)) {
        const uint32_t source_frame = static_cast<uint32_t>(rank) / kDispatchRanksPerFrame;
        const uint32_t local_rank = static_cast<uint32_t>(rank) % kDispatchRanksPerFrame;
        const uint32_t source_row = local_rank / kDispatchRanksPerDirectRow;
        const uint32_t source_column = local_rank % kDispatchRanksPerDirectRow;
        const uint32_t source_group_row = source_row / kDispatchGroupRows;
        const uint32_t source_group_side = source_column / kDispatchGroupColumns;

        const uint32_t frame_toggle = GetDispatchFrameToggle(group);
        const uint32_t subround = GetDispatchFrameSubround(group);

        const uint32_t group_row_delta = subround / 2U;
        const uint32_t group_side_delta = subround % 2U;

        const uint32_t destination_frame = source_frame ^ source_group_side ^ frame_toggle;
        const uint32_t destination_group_row = (source_group_row + group_row_delta) % kDispatchGroupRowBlocks;
        const uint32_t destination_group_side = source_group_side ^ group_side_delta;

        return BuildTwoFrameDispatchRank(destination_frame, destination_group_row, destination_group_side,
                                         group_rank_index);
    }

    const int32_t half_world_size = world_size / 2;
    const uint32_t group_width = GetDispatchGroupWidth(world_size);
    const uint32_t half_group_width = group_width / 2;
    const uint32_t group_count = static_cast<uint32_t>(world_size) / group_width;
    const uint32_t rank_group = static_cast<uint32_t>(rank % half_world_size) / half_group_width;
    const uint32_t peer_group = (rank_group + group) % group_count;
    return static_cast<int32_t>(group_rank_index / half_group_width) * half_world_size +
           static_cast<int32_t>(peer_group * half_group_width + group_rank_index % half_group_width);
}

__aicore__ inline int32_t GetGroupIncomingPeer(int32_t rank, int32_t world_size, uint32_t group,
                                               uint32_t group_rank_index)
{
    if (UseTwoFrameDispatchSchedule(world_size)) {
        const uint32_t destination_frame = static_cast<uint32_t>(rank) / kDispatchRanksPerFrame;
        const uint32_t local_rank = static_cast<uint32_t>(rank) % kDispatchRanksPerFrame;
        const uint32_t destination_row = local_rank / kDispatchRanksPerDirectRow;
        const uint32_t destination_column = local_rank % kDispatchRanksPerDirectRow;
        const uint32_t destination_group_row = destination_row / kDispatchGroupRows;
        const uint32_t destination_group_side = destination_column / kDispatchGroupColumns;

        const uint32_t frame_toggle = GetDispatchFrameToggle(group);
        const uint32_t subround = GetDispatchFrameSubround(group);
        const uint32_t group_row_delta = subround / 2U;
        const uint32_t group_side_delta = subround % 2U;
        const uint32_t source_group_side = destination_group_side ^ group_side_delta;
        const uint32_t source_group_row =
            (destination_group_row + kDispatchGroupRowBlocks - group_row_delta) % kDispatchGroupRowBlocks;
        const uint32_t source_frame = destination_frame ^ source_group_side ^ frame_toggle;

        return BuildTwoFrameDispatchRank(source_frame, source_group_row, source_group_side, group_rank_index);
    }

    const int32_t half_world_size = world_size / 2;
    const uint32_t group_width = GetDispatchGroupWidth(world_size);
    const uint32_t half_group_width = group_width / 2;
    const uint32_t group_count = static_cast<uint32_t>(world_size) / group_width;

    const uint32_t rank_group = static_cast<uint32_t>(rank % half_world_size) / half_group_width;
    const uint32_t incoming_group = (rank_group + group_count - group % group_count) % group_count;

    return static_cast<int32_t>(group_rank_index / half_group_width) * half_world_size +
           static_cast<int32_t>(incoming_group * half_group_width + group_rank_index % half_group_width);
}

__aicore__ inline int32_t GetNextGroupCreditPeer(int32_t rank, int32_t world_size, uint32_t group,
                                                 uint32_t group_rank_index)
{
    return GetGroupIncomingPeer(rank, world_size, group + 1U, group_rank_index);
}

__aicore__ inline void DispatchSplitRange(int64_t total, int64_t part_count, int64_t part_index, int64_t& range_begin,
                                          int64_t& range_end)
{
    if (total <= 0 || part_count <= 0 || part_index < 0 || part_index >= part_count) {
        range_begin = 0;
        range_end = 0;
        return;
    }
    const int64_t base_count = total / part_count;
    const int64_t remainder = total - base_count * part_count;
    range_begin = part_index * base_count + (part_index < remainder ? part_index : remainder);
    range_end = range_begin + base_count + (part_index < remainder ? 1 : 0);
}

__aicore__ inline void DispatchSplitWeightRange(int64_t total, int64_t part_count, int64_t part_index,
                                                int64_t& range_begin, int64_t& range_end)
{
    const int64_t group_size = static_cast<int64_t>(kWeightsPerGmConflictGroup);
    const int64_t group_count = total > 0 ? total / group_size + (total % group_size != 0 ? 1 : 0) : 0;
    int64_t group_begin = 0;
    int64_t group_end = 0;
    DispatchSplitRange(group_count, part_count, part_index, group_begin, group_end);
    range_begin = group_begin * group_size;
    range_end = group_end * group_size;
    if (range_begin > total) {
        range_begin = total;
    }
    if (range_end > total) {
        range_end = total;
    }
}

__aicore__ inline void IssueGmBytesToUb(AscendC::LocalTensor<uint8_t> destination_ub, GM_ADDR source_gm, uint32_t bytes)
{
    AscendC::GlobalTensor<uint8_t> source_tensor;
    source_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(source_gm), bytes);
    AscendC::DataCopyPadExtParams<uint8_t> pad_parameters{false, 0, 0, 0};
    AscendC::DataCopyExtParams copy_parameters{1, bytes, 0, 0, 0};
    AscendC::DataCopyPad(destination_ub, source_tensor, copy_parameters, pad_parameters);
}

__aicore__ inline void IssueUbBytesToGm(GM_ADDR destination_gm, AscendC::LocalTensor<uint8_t> source_ub, uint32_t bytes)
{
    AscendC::GlobalTensor<uint8_t> destination_tensor;
    destination_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(destination_gm), bytes);
    AscendC::DataCopyExtParams copy_parameters{1, bytes, 0, 0, 0};
    AscendC::DataCopyPad(destination_tensor, source_ub, copy_parameters);
}

// Logical slots are initialized by AIV16..31 before the address barrier.
__aicore__ inline void InitializeTokenScatterPeerSignals(int32_t current_rank, uint32_t group_rank_index,
                                                         uint64_t magic, uint32_t active_qp_count,
                                                         __gm__ uint8_t* sync_buffer)
{
    for (uint32_t i = 0; i < active_qp_count; ++i) {
        const uint32_t qp_idx = kActiveUdmaQpIndices[i];
        __gm__ uint64_t* signal_flag = GetPeerCompletionFlag(current_rank, group_rank_index, qp_idx, sync_buffer);
        *signal_flag = 0;
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(signal_flag));
    }
    __gm__ uint64_t* completion_value =
        GetPeerCompletionFlag(current_rank, group_rank_index, kPeerCompletionValueSlot, sync_buffer);
    *completion_value = magic;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(completion_value));
    __gm__ uint64_t* credit_flag = GetPeerCreditFlag(current_rank, group_rank_index, sync_buffer);
    *credit_flag = 0;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(credit_flag));
}

__aicore__ inline bool IsSourceReadyForForwarding(int32_t incoming_peer, uint32_t current_rank_index, uint64_t magic,
                                                  uint32_t active_qp_count, __gm__ uint8_t* sync_buffer)
{
    AscendC::GlobalTensor<uint64_t> flag_tensor;
    const uint32_t used_qps = incoming_peer / 8 == aclshmem_my_pe() / 8 ? 1U : active_qp_count;
    for (uint32_t i = 0; i < used_qps; ++i) {
        const uint32_t qp_idx = kActiveUdmaQpIndices[i];
        __gm__ uint64_t* flag = GetPeerCompletionFlag(incoming_peer, current_rank_index, qp_idx, sync_buffer);
        flag_tensor.SetGlobalBuffer(flag);
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(flag));
        if (flag_tensor.GetValue(0) != magic) {
            return false;
        }
    }
    return true;
}

__aicore__ inline void PublishPeerCompletionSignal(int32_t dst_rank, int32_t current_rank, uint32_t group_rank_index,
                                                   uint32_t active_qp_count, __gm__ uint8_t* sync_buffer,
                                                   __ubuf__ uint8_t* udma_wqe_buffer)
{
    __gm__ uint64_t* completion_value =
        GetPeerCompletionFlag(current_rank, group_rank_index, kPeerCompletionValueSlot, sync_buffer);
    if (dst_rank == current_rank) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }
    for (uint32_t i = 0; i < active_qp_count; ++i) {
        const uint32_t qp_idx = kActiveUdmaQpIndices[i];
        __gm__ uint64_t* signal_flag = GetPeerCompletionFlag(current_rank, group_rank_index, qp_idx, sync_buffer);

        if (dst_rank == current_rank) {
            *signal_flag = *completion_value;
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(signal_flag));
            continue;
        }
        aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kUdmaSyncSoCqeConfig>(
            signal_flag, completion_value, reinterpret_cast<__ubuf__ uint64_t*>(udma_wqe_buffer), 1, dst_rank,
            DispatchPhysicalQp(current_rank, dst_rank, qp_idx), EVENT_ID0);
    }
}

__aicore__ inline void PublishNextPeerCredit(uint32_t group, int32_t current_rank, int32_t world_size,
                                             uint32_t group_rank_index, uint64_t magic, uint32_t group_count,
                                             __gm__ uint8_t* sync_buffer,
                                             AscendC::TBuf<AscendC::QuePosition::VECCALC>& local_copy_buffer)
{
    const int32_t next_peer = GetNextGroupCreditPeer(current_rank, world_size, group, group_rank_index);
    if (next_peer == current_rank) {
        return;
    }

    constexpr uint32_t credit_word_count = kPeerSyncFlagBytes / sizeof(uint64_t);
    const uint64_t next_credit_token = GetPeerCreditToken(magic, group + 1U, group_count);
    AscendC::LocalTensor<uint64_t> credit_local = local_copy_buffer.Get<uint64_t>();
    for (uint32_t i = 0; i < credit_word_count; ++i) {
        credit_local.SetValue(i, i == 0U ? next_credit_token : 0U);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

    __gm__ uint64_t* local_credit = GetPeerCreditFlag(current_rank, group_rank_index, sync_buffer);
    __gm__ uint64_t* remote_credit = reinterpret_cast<__gm__ uint64_t*>(aclshmem_ptr(local_credit, next_peer));
    AscendC::GlobalTensor<uint64_t> remote_credit_tensor;
    remote_credit_tensor.SetGlobalBuffer(remote_credit, credit_word_count);
    AscendC::DataCopy(remote_credit_tensor, credit_local, credit_word_count);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

// One permanent owner per physical peer, including BOTH primary QPs.
// An FM owner never joins a Clos group. Self-copy has no QP.
__aicore__ inline uint32_t MoonClosOwners(uint32_t world)
{
    return world == 128U ? 16U : 8U;
}
__aicore__ inline uint32_t MoonLocalFirst(uint32_t world)
{
    return 8U + MoonClosOwners(world);
}
__aicore__ inline uint32_t MoonLocalWorkers(uint32_t world)
{
    return 33U - MoonLocalFirst(world);
}
__aicore__ inline uint32_t MoonLocalWorker(uint32_t world, uint32_t core)
{
    // Keep the original local worker0 as the self-ready publisher. AIV7 is last.
    return core == 7U ? MoonLocalWorkers(world) - 1U : core - MoonLocalFirst(world);
}

struct MoonSecondaryState {
    // GM_ADDR is a pointer macro, so each member needs its own declaration.
    GM_ADDR hidden;
    GM_ADDR workspace;
    __gm__ uint64_t* book;
    __gm__ uint8_t* sync;
    uint32_t rank, world, core, lanes, row_bytes;
    uint64_t magic;
};

__aicore__ inline void CopyMoonRows(GM_ADDR source, GM_ADDR destination, uint32_t row_bytes,
                                    AscendC::LocalTensor<int32_t> rows, uint32_t count,
                                    AscendC::TBuf<AscendC::QuePosition::VECCALC>& slot_buffer)
{
    const uint32_t stride = ((row_bytes + 31U) / 32U) * 32U;
    const uint32_t fit = stride <= kDuplicateHiddenStageBytes ? kDuplicateHiddenStageBytes / stride : 1U;
    const uint32_t rows_per_tile = fit < kDuplicateMaxPrimaryRows ? fit : kDuplicateMaxPrimaryRows;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    uint32_t stage = 0;
    for (uint32_t row = 0; row < count; row += rows_per_tile) {
        const uint32_t n = count - row < rows_per_tile ? count - row : rows_per_tile;
        for (uint32_t offset = 0; offset < row_bytes; offset += kDuplicateHiddenStageBytes) {
            const uint32_t bytes =
                row_bytes - offset < kDuplicateHiddenStageBytes ? row_bytes - offset : kDuplicateHiddenStageBytes;
            const AscendC::TEventID event = stage == 0U ? EVENT_ID0 : EVENT_ID1;
            auto hidden =
                slot_buffer.GetWithOffset<uint8_t>(kDuplicateHiddenStageBytes, stage * kDuplicateHiddenStageBytes);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            for (uint32_t i = 0; i < n; ++i) {
                IssueGmBytesToUb(hidden[i * stride],
                                 source + uint64_t(rows.GetValue((row + i) * 2U)) * row_bytes + offset, bytes);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            for (uint32_t i = 0; i < n; ++i) {
                IssueUbBytesToGm(destination + uint64_t(rows.GetValue((row + i) * 2U + 1U)) * row_bytes + offset,
                                 hidden[i * stride], bytes);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            stage ^= 1U;
        }
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

struct MoonPrimaryState {
    GM_ADDR input;
    GM_ADDR destinations;
    MoonSecondaryState* secondary;
    uint32_t tokens, topk, address_stride;
};

// One complete selection tile per visit. All deferred WQEs are submitted before
// switching peers. QP selection remains based on the ORIGINAL route-tile index.
__aicore__ inline void IssueMoonPrimaryTile(MoonPrimaryState& p, int32_t dst_rank, int64_t chunk_start,
                                            __ubuf__ uint8_t* udma_wqe_buffer,
                                            AscendC::TBuf<AscendC::QuePosition::VECCALC>& slot_select_buffer)
{
    auto& s = *p.secondary;
    auto* hidden = p.input;
    auto* encoded_destinations = reinterpret_cast<__gm__ int32_t*>(p.destinations);
    auto* workspace = s.workspace;
    auto* book = s.book;
    const uint32_t current_rank = s.rank, active_qp_count = 2U;
    const int64_t num_topk = p.topk, route_count = int64_t(p.tokens) * p.topk;
    const int64_t hidden_row_bytes = s.row_bytes, address_stride = p.address_stride;

    const uint32_t tensor_bytes = slotChunkCount * sizeof(int32_t);
    const uint32_t compare_mask_bytes = ((slotChunkCount / 8 + 31) / 32) * 32;
    AscendC::LocalTensor<int32_t> dst_local = slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, 0);
    AscendC::LocalTensor<int32_t> dst_rank_tensor =
        slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, tensor_bytes);
    AscendC::LocalTensor<int32_t> route_indices =
        slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, tensor_bytes * 2);
    AscendC::LocalTensor<int32_t> selected_route_indices =
        slot_select_buffer.GetWithOffset<int32_t>(slotChunkCount, tensor_bytes * 3);
    AscendC::LocalTensor<uint8_t> compare_mask =
        slot_select_buffer.GetWithOffset<uint8_t>(compare_mask_bytes, tensor_bytes * 4);

    AscendC::GlobalTensor<int32_t> dst_global;
    dst_global.SetGlobalBuffer(encoded_destinations, route_count);
    uint32_t slot_shift = 0;
    // Address encoding follows receive capacity, NOT the source route count.
    for (int64_t value = address_stride; value > 1; value >>= 1) {
        ++slot_shift;
    }
    const uint32_t route_tile = dst_rank / 8 == current_rank / 8 ? slotChunkCount : 1024U;
    const int64_t chunk_index = chunk_start / route_tile;
    const uint32_t qp_slot = GetChunkQpSlot(chunk_index, dst_rank / 8 == current_rank / 8 ? 1U : active_qp_count);
    const uint32_t qp_idx = DispatchPhysicalQp(current_rank, dst_rank, kActiveUdmaQpIndices[qp_slot]);

    const int64_t remaining_count = route_count - chunk_start;
    const uint32_t chunk_count =
        static_cast<uint32_t>(remaining_count > static_cast<int64_t>(route_tile) ? route_tile : remaining_count);
    const uint32_t padded_count =
        ((chunk_count + kVectorSlotSelectCountAlign - 1) / kVectorSlotSelectCountAlign) * kVectorSlotSelectCountAlign;

    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
    AscendC::DataCopyExtParams dst_copy_parameters{1, static_cast<uint32_t>(chunk_count * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> dst_pad_parameters{false, 0, 0, 0};
    AscendC::DataCopyPad(dst_local, dst_global[chunk_start], dst_copy_parameters, dst_pad_parameters);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

    AscendC::ShiftRight(dst_rank_tensor, dst_local, static_cast<int32_t>(slot_shift), padded_count);
    AscendC::CreateVecIndex(route_indices, static_cast<int32_t>(chunk_start), padded_count);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);

    AscendC::Compares(compare_mask, dst_rank_tensor, dst_rank, AscendC::CMPMODE::EQ, padded_count);
    AscendC::PipeBarrier<PIPE_V>();
    uint64_t selected_route_count = 0;
    AscendC::GatherMask(selected_route_indices, route_indices, compare_mask.ReinterpretCast<uint32_t>(), true,
                        chunk_count, {1U, 1U, 0U, 0U}, selected_route_count);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);

    const uint64_t local_selected_count = selected_route_count;

    aclshmemx_submit_state_t udma_submit_state{};
    aclshmemx_defer_t defer_action(udma_submit_state);
    aclshmemx_submit_t submit_action(udma_submit_state);

    constexpr uint32_t batch_rows = kUdmaWqeBatchSize;
    auto* target = LivePeerWindow(workspace, book, dst_rank);

    for (uint64_t batch_start = 0; batch_start < local_selected_count; batch_start += batch_rows) {
        const uint64_t batch_end =
            local_selected_count - batch_start > batch_rows ? batch_start + batch_rows : local_selected_count;

        for (uint64_t selected = batch_start; selected + 1 < batch_end; ++selected) {
            const int64_t route_index = selected_route_indices.GetValue(selected);
            const int64_t token_index = route_index / num_topk;
            const int64_t output_offset = dst_local.GetValue(route_index - chunk_start) % address_stride;
            __gm__ uint8_t* hidden_destination = target + output_offset * hidden_row_bytes;
            __gm__ uint8_t* hidden_source = hidden + token_index * hidden_row_bytes;
            aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kUdmaDataRoNoCqeConfig>(
                hidden_destination, hidden_source, udma_wqe_buffer, static_cast<uint32_t>(hidden_row_bytes), dst_rank,
                qp_idx, EVENT_ID0, defer_action);
        }

        const uint64_t selected = batch_end - 1;
        const int64_t route_index = selected_route_indices.GetValue(selected);
        const int64_t token_index = route_index / num_topk;
        const int64_t output_offset = dst_local.GetValue(route_index - chunk_start) % address_stride;
        __gm__ uint8_t* hidden_destination = target + output_offset * hidden_row_bytes;
        __gm__ uint8_t* hidden_source = hidden + token_index * hidden_row_bytes;
        aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kUdmaDataRoCqeConfig>(
            hidden_destination, hidden_source, udma_wqe_buffer, static_cast<uint32_t>(hidden_row_bytes), dst_rank,
            qp_idx, EVENT_ID0, submit_action);
    }
}

__aicore__ inline bool MoonFlagEquals(__gm__ uint64_t* flag, uint64_t expected)
{
    AscendC::GlobalTensor<uint64_t> value;
    value.SetGlobalBuffer(flag);
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(flag));
    return value.GetValue(0) == expected;
}

// All calls for a physical cross-server peer (BOTH QPs) belong to 8 + logical%8.
// Its two logical peers progress independently until both have submitted final SO.
__aicore__ inline void DispatchMoonCross(MoonPrimaryState& p, __ubuf__ uint8_t* wqe,
                                         AscendC::TBuf<AscendC::QuePosition::VECCALC>& copy,
                                         AscendC::TBuf<AscendC::QuePosition::VECCALC>& slots)
{
    auto& s = *p.secondary;
    if (s.world <= 8U) {
        return;
    }
    const uint32_t owner = s.core - 8U, width = GetDispatchGroupWidth(s.world);
    const uint32_t owners = MoonClosOwners(s.world), sequences = 16U / owners;
    const uint32_t groups = s.world / width, receiver_index = GetGroupRankIndex(s.rank, s.world);
    auto* group_sync = s.sync + uint64_t(s.world) * kRankSyncSlotBytes;
    const bool strict = s.world <= kStrictGroupSyncMaxWorldSize;
    SyncActiveAivs(owner, owners, GetGroupSyncInitToken(s.magic, groups), group_sync);
    const uint32_t logical[2]{owner, owner + owners};
    uint32_t group[2]{0U, sequences == 2U ? 0U : groups};
    int32_t peers[2]{}, incoming[2]{};
    uint64_t cursor[2]{};
    bool initialize[2]{true, true}, submitted[2]{}, credit[2]{}, ready[2]{};
    while (group[0] < groups || group[1] < groups) {
        for (uint32_t j = 0; j < sequences; ++j) {
            if (group[j] >= groups) {
                continue;
            }
            if (initialize[j]) {
                peers[j] = GetGroupDstRank(s.rank, s.world, group[j], logical[j]);
                incoming[j] = GetGroupIncomingPeer(s.rank, s.world, group[j], logical[j]);
                cursor[j] = 0;
                // Local FM runs on 0..6; it NEVER participates in Clos waits.
                submitted[j] = peers[j] / 8 == s.rank / 8;
                credit[j] = group[j] == 0U;
                ready[j] = false;
                initialize[j] = false;
            }
            if (!submitted[j]) {
                if (!credit[j]) {
                    credit[j] = MoonFlagEquals(GetPeerCreditFlag(peers[j], receiver_index, s.sync),
                                               GetPeerCreditToken(s.magic, group[j], groups));
                    if (!credit[j]) {
                        continue;
                    }
                }
                // One complete tile per visit; never switch with deferred WQEs.
                if (cursor[j] < uint64_t(p.tokens) * p.topk) {
                    IssueMoonPrimaryTile(p, peers[j], cursor[j], wqe, slots);
                    cursor[j] += 1024U;
                }
                if (cursor[j] >= uint64_t(p.tokens) * p.topk) {
                    PublishPeerCompletionSignal(peers[j], s.rank, logical[j], 2U, s.sync, wqe);
                    submitted[j] = true;
                }
            }
            if (submitted[j] && !ready[j]) {
                // Poll, don't block the other logical sequence on incoming SO.
                ready[j] = incoming[j] / 8 == s.rank / 8 ||
                           IsSourceReadyForForwarding(incoming[j], receiver_index, s.magic, 2U, s.sync);
            }
        }
        if (strict && (!ready[0] || !ready[1])) {
            continue;
        }
        const uint32_t strict_group = group[0];
        if (strict) {
            SyncActiveAivs(owner, 8U, GetGroupSyncPhaseToken(s.magic, strict_group, groups, kGroupSyncCompletionPhase),
                           group_sync);
        }
        // Only grant the ORIGINAL next incoming edge, after this edge is ready.
        // In strict mode grant both credits before either blocking quiet, as before.
        for (uint32_t j = 0; j < sequences; ++j) {
            if (ready[j] && group[j] + 1U < groups) {
                // A predecessor may be a skipped local edge. Still publish the
                // NEXT cross edge's credit; otherwise that sender can wait forever.
                const int32_t next = GetNextGroupCreditPeer(s.rank, s.world, group[j], logical[j]);
                if (next / 8 != s.rank / 8) {
                    PublishNextPeerCredit(group[j], s.rank, s.world, logical[j], s.magic, groups, s.sync, copy);
                }
            }
        }
        // SDK quiet can still block this AIV; this is not asynchronous CQ polling.
        // Both QPs are retired by their permanent owner before its sequence advances.
        for (uint32_t j = 0; j < sequences; ++j)
            if (ready[j] && group[j] < groups && peers[j] / 8 != s.rank / 8) {
                aclshmemx_udma_qp_quiet(peers[j], DispatchPhysicalQp(s.rank, peers[j], kActiveUdmaQpIndices[0]));
                aclshmemx_udma_qp_quiet(peers[j], DispatchPhysicalQp(s.rank, peers[j], kActiveUdmaQpIndices[1]));
            }
        if (strict) {
            SyncActiveAivs(owner, 8U, GetGroupSyncPhaseToken(s.magic, strict_group, groups, kGroupSyncDonePhase),
                           group_sync);
        }
        for (uint32_t j = 0; j < sequences; ++j)
            if (ready[j] && group[j] < groups) {
                ++group[j];
                initialize[j] = true;
                ready[j] = false;
            }
    }
}

__aicore__ inline void DispatchMoonLocalPeer(MoonPrimaryState& p, __ubuf__ uint8_t* wqe,
                                             AscendC::TBuf<AscendC::QuePosition::VECCALC>& slots)
{
    auto& s = *p.secondary;
    if (s.core + 1U >= s.lanes) {
        return;
    }
    const uint32_t lane = (s.rank % 8U + s.core + 1U) % s.lanes;
    const int32_t peer = s.rank / 8U * 8U + lane;
    for (uint64_t route = 0; route < uint64_t(p.tokens) * p.topk; route += slotChunkCount) {
        IssueMoonPrimaryTile(p, peer, route, wqe, slots);
    }
    PublishPeerCompletionSignal(peer, s.rank, GetGroupRankIndex(peer, s.world), 1U, s.sync, wqe);
    aclshmemx_udma_qp_quiet(peer, kActiveUdmaQpIndices[0]);
}

// <=64P: 17 local writers use the unused upper group region.
// 128P: 16 Clos owners need the full group region; nine local flags follow FM signals.
__aicore__ inline __gm__ uint64_t* MoonLocalPrimaryDone(MoonSecondaryState& s, uint32_t worker)
{
    return reinterpret_cast<__gm__ uint64_t*>(s.sync + uint64_t(s.world) * kRankSyncSlotBytes +
                                              (s.world == 128U ? 40960U : 16384U) + uint64_t(worker) * 512U);
}

__aicore__ inline void DispatchMoonLocalCopy(MoonPrimaryState& p, __ubuf__ uint8_t* wqe,
                                             AscendC::TBuf<AscendC::QuePosition::VECCALC>& copy,
                                             AscendC::TBuf<AscendC::QuePosition::VECCALC>& slots)
{
    auto& s = *p.secondary;
    const uint32_t worker = MoonLocalWorker(s.world, s.core);
    int64_t begin = 0, end = 0;
    DispatchSplitRange(int64_t(p.tokens) * p.topk, MoonLocalWorkers(s.world), worker, begin, end);
    auto routes = copy.Get<int32_t>();
    auto rows = routes[kUdmaWqeBatchSize];

    for (int64_t first = begin; first < end; first += kUdmaWqeBatchSize) {
        const uint32_t n = end - first < kUdmaWqeBatchSize ? end - first : kUdmaWqeBatchSize;
        NativeLoad(routes, reinterpret_cast<__gm__ int32_t*>(p.destinations) + first, n);
        uint32_t count = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const int32_t encoded = routes.GetValue(i);
            if (encoded < 0 || uint32_t(encoded) / p.address_stride != s.rank) {
                continue;
            }
            rows.SetValue(count * 2U, (first + i) / p.topk);
            rows.SetValue(count * 2U + 1U, uint32_t(encoded) % p.address_stride);
            ++count;
        }
        if (!count) {
            continue;
        }
        CopyMoonRows(p.input, s.hidden, s.row_bytes, rows, count, slots);
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    auto* done = MoonLocalPrimaryDone(s, worker);
    *done = s.magic;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(done));

    // Other writers need not wait for worker0; each has drained its own copy.
    // Self expansion starts only after every active writer has published done.
    if (worker == 0U) {
        for (uint32_t i = 0; i < MoonLocalWorkers(s.world); ++i) {
            aclshmem_uint64_wait_until(MoonLocalPrimaryDone(s, i), ACLSHMEM_CMP_EQ, s.magic);
        }
        PublishPeerCompletionSignal(s.rank, s.rank, GetGroupRankIndex(s.rank, s.world), 1U, s.sync, wqe);
    }
}

__simt_vf__ __launch_bounds__(kWeightSimtThreadCount) inline void DispatchScatterWeightsSimtVf(
    __gm__ float* weights, __gm__ int32_t* encoded_destinations, GM_ADDR workspace, __gm__ uint64_t* book,
    uint32_t address_stride, uint32_t route_begin, uint32_t local_route_count, uint32_t route_shift,
    uint32_t first_group, uint32_t book_words)
{
    const uint32_t thread_index = static_cast<uint32_t>(threadIdx.x);
    const uint32_t local_group_count =
        (local_route_count + kWeightsPerGmConflictGroup - 1U) / kWeightsPerGmConflictGroup;
    for (uint32_t group_order = 0; group_order < local_group_count; ++group_order) {
        uint32_t group_index = first_group + group_order;
        if (group_index >= local_group_count) {
            group_index -= local_group_count;
        }
        const uint32_t local_route_index = group_index * kWeightsPerGmConflictGroup + thread_index;
        if (local_route_index >= local_route_count) {
            continue;
        }

        const uint32_t route_index = route_begin + local_route_index;
        const uint32_t encoded_bits = static_cast<uint32_t>(encoded_destinations[route_index]);
        if (encoded_bits == 0x80000000U) {
            continue;
        }  // invalid route, NOT duplicate destination zero
        const uint32_t sign_mask = 0U - (encoded_bits >> 31U);
        const uint32_t flat_destination = encoded_bits ^ sign_mask;
        // dispatch_dst is repacked with a power-of-two stride.
        const uint32_t destination_rank = flat_destination >> route_shift;
        const uint32_t destination_local_offset = flat_destination & (address_stride - 1U);

        const uint64_t base = uint64_t(destination_rank) * book_words;
        // Weight addresses refer to this invocation's registered output.
        const uint64_t weight_offset = asc_ldcg(book + base + 1U);
        auto* target = reinterpret_cast<__gm__ float*>(workspace + weight_offset);
        ::simt::aclshmem_float_p(target + destination_local_offset, weights[route_index],
                                 static_cast<int32_t>(destination_rank));
    }
    asc_threadfence();
}

__aicore__ inline void DispatchScatterWeightsSimt(GM_ADDR weights_gm, GM_ADDR dst_gm, GM_ADDR workspace,
                                                  __gm__ uint64_t* book, uint32_t address_stride, int64_t route_count,
                                                  int64_t aiv_index, int64_t aiv_count, uint32_t book_words)
{
    if (route_count <= 0 || aiv_count <= 0 || aiv_index < 0 || aiv_index >= aiv_count) {
        return;
    }

    int64_t route_begin = 0;
    int64_t route_end = 0;
    DispatchSplitWeightRange(route_count, aiv_count, aiv_index, route_begin, route_end);
    if (route_begin >= route_end) {
        return;
    }

    uint32_t route_shift = 0;
    for (uint32_t value = address_stride; value > 1U; value >>= 1U) {
        ++route_shift;
    }

    __gm__ float* weights = reinterpret_cast<__gm__ float*>(weights_gm);
    __gm__ int32_t* encoded_destinations = reinterpret_cast<__gm__ int32_t*>(dst_gm);
    const int32_t current_rank = aclshmem_my_pe();
    const uint32_t route_begin_u32 = static_cast<uint32_t>(route_begin);
    const uint32_t local_route_count = static_cast<uint32_t>(route_end - route_begin);
    const uint32_t local_group_count =
        (local_route_count + kWeightsPerGmConflictGroup - 1U) / kWeightsPerGmConflictGroup;
    const uint32_t first_group =
        (static_cast<uint32_t>(current_rank) + static_cast<uint32_t>(aiv_index)) % local_group_count;

    asc_vf_call<DispatchScatterWeightsSimtVf>(dim3(kWeightSimtThreadCount), weights, encoded_destinations, workspace,
                                              book, address_stride, route_begin_u32, local_route_count, route_shift,
                                              first_group, book_words);

    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    return;
}

}  // namespace DispatchMoonep
#endif
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
using namespace DispatchMoonep;
using ascend_deepep::DispatchTiling;

// Notify assigns exactly one final row to each valid expert occurrence. Each
// secondary record has one owner: floor(count*w/workers)..floor(count*(w+1)/workers).
// The record's primary row is read-only once the original source's SO arrives.
// Secondary never uses a QP; every primary physical peer retains its sole owner.
__aicore__ inline void Forward(MoonSecondaryState& s, GM_ADDR forward, GM_ADDR counts, uint32_t capacity,
                               TBuf<QuePosition::VECCALC>& metadata, TBuf<QuePosition::VECCALC>& staging)
{
    const uint32_t first = MoonLocalFirst(s.world);
    const uint32_t workers = 40U - first;
    const uint32_t worker = s.core < 8U ? s.core : s.core - first + 8U;
    uint32_t cursor[128]{}, end[128]{};
    bool ready[128]{};
    auto records = metadata.Get<int32_t>();
    NativeLoad(records, reinterpret_cast<__gm__ int32_t*>(counts), s.world);
    uint32_t pending = 0;
    for (uint32_t source = 0; source < s.world; ++source) {
        const uint32_t n = records.GetValue(source);
        cursor[source] = uint64_t(n) * worker / workers;
        end[source] = uint64_t(n) * (worker + 1U) / workers;
        pending += cursor[source] != end[source];
    }
    while (pending) {
        for (uint32_t source = 0; source < s.world; ++source) {
            if (cursor[source] == end[source]) {
                continue;
            }
            if (!ready[source]) {
                ready[source] =
                    IsSourceReadyForForwarding(source, GetGroupRankIndex(s.rank, s.world), s.magic, 2U, s.sync);
                if (!ready[source]) {
                    continue;
                }
            }
            const uint32_t n = end[source] - cursor[source] < 64U ? end[source] - cursor[source] : 64U;
            NativeLoad(records,
                       reinterpret_cast<__gm__ int32_t*>(forward) + (uint64_t(source) * capacity + cursor[source]) * 6U,
                       n * 6U);
            // Two hidden staging slots, each at a 32-byte aligned UB offset.
            // MTE3->MTE2 credits cover reuse, including the final drain.
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            uint32_t stage = 0;
            for (uint32_t i = 0; i < n;) {
                const int32_t peer = records.GetValue(i * 6U + 3U);
                if (peer < 0) {
                    ++i;
                    continue;
                }  // singleton marker has no secondary payload
                const uint32_t primary = records.GetValue(i * 6U + 2U);
                const TEventID event = stage == 0 ? EVENT_ID0 : EVENT_ID1;
                auto ub = staging.GetWithOffset<uint8_t>(16384, stage * 16384);
                WaitFlag<HardEvent::MTE3_MTE2>(event);
                IssueGmBytesToUb(ub, s.hidden + uint64_t(primary) * s.row_bytes, s.row_bytes);
                SetFlag<HardEvent::MTE2_MTE3>(event);
                WaitFlag<HardEvent::MTE2_MTE3>(event);
                // Consecutive fan-out records reuse this immutable hidden UB
                // row. Only task/tile boundaries can cause an extra primary read.
                do {
                    const int32_t target = records.GetValue(i * 6U + 3U);
                    const uint32_t row = records.GetValue(i * 6U + 4U);
                    if (target >= 0) {
                        auto* destination = s.hidden + uint64_t(row) * s.row_bytes;
                        // Same-rank duplicates have a distinct final row too.
                        if (uint32_t(target) != s.rank) {
                            destination = reinterpret_cast<__gm__ uint8_t*>(aclshmem_ptr(destination, target));
                        }
                        IssueUbBytesToGm(destination, ub, s.row_bytes);
                    }
                    ++i;
                } while (i < n && uint32_t(records.GetValue(i * 6U + 2U)) == primary);
                SetFlag<HardEvent::MTE3_MTE2>(event);
                stage ^= 1U;
            }
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            cursor[source] += n;
            pending -= cursor[source] == end[source];
        }
    }
    // Follow Notify's mapped-MTE protocol: drain instructions and flush the
    // data cache before this worker announces completion through the collective.
    aclshmemx_mte_quiet();
}

// Independent aggregate scales: 32 AIVs own disjoint peers/sources. Each
// source->destination pair owns a complete inbox (no credits/receiver progress
// needed for sending). A record is [row:int32, padding:28B, scale:224B]. Hidden
// forwarding never carries scales. All normal MTE segments are 32B aligned.
__aicore__ inline void Scales(GM_ADDR workspace, GM_ADDR input, GM_ADDR destinations, const DispatchTiling& t,
                              TBuf<QuePosition::VECCALC>& metadata, TBuf<QuePosition::VECCALC>& staging)
{
    const uint32_t owner = GetBlockIdx() - 32U, routes = t.tokens * t.topk;
    auto dst = metadata.Get<int32_t>();
    auto pack = staging.Get<uint8_t>();
    for (uint32_t index = owner; index < t.world; index += 32U) {
        const uint32_t peer = (index + t.rank) % t.world;
        auto* inbox = workspace + t.inbox + uint64_t(t.rank) * t.inbox_stride;
        auto* remote = reinterpret_cast<__gm__ uint8_t*>(aclshmem_ptr(inbox, peer));
        uint32_t total = 0, packed = 0;
        for (uint32_t first = 0; first < routes; first += 256U) {
            const uint32_t n = routes - first < 256U ? routes - first : 256U;
            NativeLoad(dst, reinterpret_cast<__gm__ int32_t*>(destinations) + first, n);
            for (uint32_t i = 0; i < n; ++i) {
                const int32_t encoded = dst.GetValue(i);
                if (uint32_t(encoded) == 0x80000000U) {
                    continue;
                }
                const uint32_t flat = encoded < 0 ? uint32_t(~encoded) : uint32_t(encoded);
                if (flat / t.address_stride != peer) {
                    continue;
                }
                pack.ReinterpretCast<int32_t>().SetValue(packed * 64U, flat % t.address_stride);
                NativeLoad(pack[packed * 256U + 32U], input + uint64_t((first + i) / t.topk) * 224U, 224U);
                ++packed;
                if (packed == 32U) {
                    NativeStore(remote + 512U + uint64_t(total) * 256U, pack, packed * 256U);
                    total += packed;
                    packed = 0;
                }
            }
        }
        if (packed) {
            NativeStore(remote + 512U + uint64_t(total) * 256U, pack, packed * 256U);
            total += packed;
        }
        aclshmemx_mte_quiet();  // payload visibility must precede publication
        // Previous NativeStore drained MTE3 before overwriting the pack buffer.
        pack.ReinterpretCast<uint64_t>().SetValue(0, uint64_t(total) + 1U);
        NativeStore(reinterpret_cast<__gm__ uint64_t*>(remote), pack.ReinterpretCast<uint64_t>(), 1U);
        aclshmemx_mte_quiet();
    }
    for (uint32_t source = owner; source < t.world; source += 32U) {
        auto* inbox = workspace + t.inbox + uint64_t(source) * t.inbox_stride;
        auto* flag = reinterpret_cast<__gm__ uint64_t*>(inbox);
        aclshmem_uint64_wait_until(flag, ACLSHMEM_CMP_NE, 0U);
        GlobalTensor<uint64_t> header;
        header.SetGlobalBuffer(flag);
        dcci_cacheline(inbox);
        const uint32_t total = header.GetValue(0) - 1U;
        for (uint32_t first = 0; first < total; first += 32U) {
            const uint32_t n = total - first < 32U ? total - first : 32U;
            NativeLoad(pack, inbox + 512U + uint64_t(first) * 256U, n * 256U);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t row = pack.ReinterpretCast<int32_t>().GetValue(i * 64U);
                NativeStore(workspace + t.scales + uint64_t(row) * 224U, pack[i * 256U + 32U], 224U);
            }
        }
    }
}

__aicore__ inline void Dispatch(GM_ADDR workspace, GM_ADDR destinations, GM_ADDR forward, GM_ADDR counts,
                                GM_ADDR weights, GM_ADDR scales, const DispatchTiling& t)
{
    const uint32_t core = GetBlockIdx(), cores = t.fp8 ? 64U : 32U;
    TPipe pipe;
    TBuf<TPosition::VECOUT> wqe_buffer;
    TBuf<QuePosition::VECCALC> copy, slots;
    // UB ledger: 8192 WQE + 16384 metadata + 132096 select/ping-pong =
    // 156672 bytes. The remaining launch budget includes SIMT's 8KiB reserve.
    pipe.InitBuffer(wqe_buffer, kUdmaWqeScratchBytes);
    pipe.InitBuffer(copy, kLocalCopyTileBytes);
    pipe.InitBuffer(slots, kSlotSelectUbBytes);
    auto* wqe = reinterpret_cast<__ubuf__ uint8_t*>(wqe_buffer.Get<uint8_t>().GetPhyAddr());
    auto* sync = workspace + t.sync;
    auto* book = reinterpret_cast<__gm__ uint64_t*>(workspace + t.book);
    // Every invocation clears control lines, even when its shape/layout changed.
    // This kernel's entry barrier follows the prior call's output copies on
    // each rank's stream. Native code drains the prior collective before reuse.
    auto zeros = copy.Get<uint64_t>();
    for (uint32_t i = 0; i < 64U; ++i) {
        zeros.SetValue(i, 0U);
    }
    const uint64_t sync_bytes = uint64_t(t.world) * kRankSyncSlotBytes + 65536U;
    for (uint64_t offset = uint64_t(core) * 512U; offset < sync_bytes; offset += uint64_t(cores) * 512U) {
        NativeStore(reinterpret_cast<__gm__ uint64_t*>(sync + offset), zeros, 64U);
    }
    if (t.fp8) {
        for (uint32_t source = core; source < t.world; source += cores) {
            NativeStore(reinterpret_cast<__gm__ uint64_t*>(workspace + t.inbox + uint64_t(source) * t.inbox_stride),
                        zeros, 1U);
        }
    }
    if (core == 0U) {
        auto entries = copy.Get<uint64_t>();
        for (uint32_t peer = 0; peer < t.world; ++peer) {
            entries.SetValue(peer * 2U, t.output);
            entries.SetValue(peer * 2U + 1U, t.weights);
        }
        NativeStore(book, entries, t.world * 2U);
    }
    // All control stores complete before flags become readable by any sender.
    aclshmemx_barrier_all_vec();
    if (core >= 16U && core - 16U < GetDispatchGroupWidth(t.world)) {
        InitializeTokenScatterPeerSignals(t.rank, core - 16U, 1U, 2U, sync);
    }
    aclshmemx_barrier_all_vec();
    for (uint32_t offset = 0; offset < t.world * 16U; offset += 64U) {
        dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(book) + offset);
    }
    if (core >= 32U) {
        Scales(workspace, scales, destinations, t, copy, slots);
    } else {
        MoonSecondaryState s{};
        s.hidden = workspace + t.output;
        s.workspace = workspace;
        s.book = book;
        s.sync = sync;
        s.rank = t.rank;
        s.world = t.world;
        s.core = core;
        s.lanes = t.world < 8U ? t.world : 8U;
        s.row_bytes = t.fp8 ? 7168U : 14336U;
        s.magic = 1U;
        MoonPrimaryState p{};
        p.input = workspace + t.input;
        p.destinations = destinations;
        p.secondary = &s;
        p.tokens = t.tokens;
        p.topk = t.topk;
        p.address_stride = t.address_stride;
        const uint32_t local_first = MoonLocalFirst(t.world);
        if (core < 7U) {
            DispatchMoonLocalPeer(p, wqe, slots);
        } else if (core >= 8U && core < local_first) {
            DispatchMoonCross(p, wqe, copy, slots);
        } else {
            DispatchMoonLocalCopy(p, wqe, copy, slots);
        }
        if (core < 8U || core >= local_first) {
            Forward(s, forward, counts, t.tokens * t.topk, copy, slots);
        }
        if (weights != nullptr) {
            DispatchScatterWeightsSimt(weights, destinations, workspace, book, t.address_stride,
                                       int64_t(t.tokens) * t.topk, core, 32U, 2U);
        }
    }
    // Fixed participation: all 32/64 AIVs join, including empty work owners.
    // Primary has quieted its QPs; secondary/scale UB slots have been drained.
    aclshmemx_barrier_all_vec();
}
#endif
}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dispatch_kernel(GM_ADDR workspace,
                                                                                    GM_ADDR destinations,
                                                                                    GM_ADDR forward, GM_ADDR counts,
                                                                                    GM_ADDR weights, GM_ADDR scales,
                                                                                    ascend_deepep::DispatchTiling t)
{
    util_set_ffts_config(0U);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
    Dispatch(workspace, destinations, forward, counts, weights, scales, t);
#endif
}

extern "C" ASCEND_DEEPEP_EXPORT void dispatch_kernel_do(void* stream, uint8_t* workspace, uint8_t* destinations,
                                                        uint8_t* forward, uint8_t* counts, uint8_t* weights,
                                                        uint8_t* scales, uint8_t* tiling)
{
    const auto t = *reinterpret_cast<ascend_deepep::DispatchTiling*>(tiling);
    dispatch_kernel<<<t.fp8 ? 64 : 32, ASCEND_DEEPEP_KERNEL_LAUNCH_UB(192 * 1024), stream>>>(
        workspace, destinations, forward, counts, weights, scales, t);
}
