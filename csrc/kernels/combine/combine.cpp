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
#include "../common.hpp"
#include "tiling.hpp"

namespace {
using CombineTiling = ascend_deepep::PutCombineTiling;
namespace combine_control = ascend_deepep::combine_control;
namespace combine_config = ascend_deepep::combine_config;
// xasc 也会解析主机启动代码；UDMA 配置和 MicroAPI 仅在设备编译阶段可见。
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
// Enumerate remote sources in rank order, then sources in this gateway's server.
// Group membership is filtered by the caller; source/chunk IDs stay unchanged.
__aicore__ inline constexpr uint32_t SourceRankAtOrder(uint32_t rank, uint32_t world, uint32_t order)
{
    const uint32_t local_begin = rank / 8U * 8U;
    const uint32_t local_num = world < 8U ? world : 8U;
    const uint32_t remote_num = world - local_num;
    if (order >= remote_num) {
        return local_begin + order - remote_num;
    }
    return order < local_begin ? order : order + local_num;
}
__aicore__ inline uint32_t SourceChunkCount(const CombineTiling& tiling)
{
    return tiling.rank_num * tiling.chunks_per_group;
}
constexpr uint32_t kReduceTileElements = 8192;
constexpr uint32_t kReduceOutputSlots = 2;
constexpr uint32_t kLocalCopyTileBytes = 16384;
constexpr uint32_t kLocalCopySlots = 2;
constexpr uint32_t kLocalCopyBufferBytes = kLocalCopySlots * kLocalCopyTileBytes;
constexpr uint32_t kWqeBytes = 64;
constexpr uint32_t kWqeBatchSize = 128;
constexpr uint32_t kWqeBatchScratchBytes = kWqeBatchSize * kWqeBytes;
// Remote SendPartial first counts token PUTs per QP and then reuses the same
// UB area to decode metadata while issuing the payload WQEs.
constexpr uint32_t kForwardListFields = 6;
constexpr uint32_t kForwardMetaBatchRows = 128;
// Cross PUT can keep one aggregate batch per QP because the token schedule
// interleaves QP0 and QP1. Gather uses the first batch-sized half.
constexpr uint32_t kWqeScratchBytes = 2U * kWqeBatchScratchBytes;
constexpr uint32_t kGatherBatchRows = 128;
constexpr uint32_t kGatherTaskFields = 3;
constexpr uint32_t kSignalUbBytes = 512;
constexpr uint32_t kCombineUbBudgetBytes = 192U * 1024U;
constexpr uint32_t kReservedUbBytes = 8U * 1024U;
// 输入 BF16、转换临时 FP32、累加器 FP32，加上两个互不重叠的 BF16 输出槽。
constexpr uint32_t kReduceUbBytes =
    kReduceTileElements * ((1U + kReduceOutputSlots) * sizeof(bfloat16_t) + 2U * sizeof(float));
constexpr uint32_t kMainUbBytes = kReduceUbBytes + kLocalCopyBufferBytes + kWqeScratchBytes + kSignalUbBytes;
static_assert(2U * kReduceTileElements * sizeof(bfloat16_t) <= kLocalCopyBufferBytes,
              "source dual-input bank must fit copy UB");
static_assert(kReduceTileElements % 128U == 0U, "source register vectors need padded tiles");
static_assert(kReduceTileElements % 16U == 0U, "BF16 and FP32 reduction tiles must be 32-byte aligned");
// 使用 192 KiB launch 预算，为通信/同步内部开销额外留出 8 KiB。
static_assert(kMainUbBytes + kReservedUbBytes <= kCombineUbBudgetBytes, "Combine UB budget exceeded");
static_assert(combine_control::kGatherDoneLanes * combine_control::kSlotBytes <= kLocalCopyTileBytes,
              "gather flags must fit byte-copy UB");
static_assert(kGatherBatchRows * kWqeBytes <= kWqeBatchScratchBytes, "gather WQE batch must fit scratch UB");
static_assert(kGatherBatchRows * kGatherTaskFields * sizeof(int64_t) <= kLocalCopyTileBytes,
              "gather metadata batch must fit byte-copy UB");
static_assert(kForwardMetaBatchRows * kForwardListFields * sizeof(int32_t) <= kLocalCopyTileBytes,
              "forward metadata batch must fit byte-copy UB");
constexpr aclshmemx_udma_op_config_t kGatherPayloadRoNoCqeConfig{0U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
constexpr aclshmemx_udma_op_config_t kPayloadRoNoCqeConfig{0U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
constexpr aclshmemx_udma_op_config_t kPayloadRoConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
constexpr aclshmemx_udma_op_config_t kPayloadRoBatchedConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_RO};
constexpr aclshmemx_udma_op_config_t kCompletionSoConfig{1U, 0U, 0U, ACLSHMEMX_UDMA_ODR_SO};
enum class PartialClass : uint32_t { All, DirectOnly, ReducedOnly };

__aicore__ inline void CountRemoteToken(const CombineTiling& tiling, uint32_t destination_rank,
                                        PartialClass partial_class, uint32_t token_id, bool needs_reduce,
                                        uint32_t remaining_qp_rows[2])
{
    if ((partial_class == PartialClass::DirectOnly && needs_reduce) ||
        (partial_class == PartialClass::ReducedOnly && !needs_reduce)) {
        return;
    }
    const uint32_t qp_id = destination_rank / 8U == tiling.rank_id / 8U ? 0U : ((token_id / 512U) % 4U == 3U ? 1U : 0U);
    ++remaining_qp_rows[qp_id];
}

__aicore__ inline void SendRemoteToken(GM_ADDR workspace, const CombineTiling& tiling, uint32_t destination_rank,
                                       uint32_t server_num, uint32_t token_id, uint32_t primary_row, bool needs_reduce,
                                       __ubuf__ uint8_t* wqe_scratch_ub, uint32_t* remaining_qp_rows,
                                       uint32_t* pending_qp_wqes, aclshmemx_defer_t** udma_defer,
                                       aclshmemx_submit_t** udma_submit)
{
    auto* source_address = workspace +
                           (needs_reduce ? tiling.server_partial_offset_bytes : tiling.expert_input_offset_bytes) +
                           uint64_t(primary_row) * tiling.hidden_size * 2U;
    auto* destination_address = workspace + tiling.returned_partial_offset_bytes +
                                (uint64_t(token_id) * server_num + tiling.rank_id / 8U) * tiling.hidden_size * 2U;
    const uint32_t qp_id = destination_rank / 8U == tiling.rank_id / 8U ? 0U : ((token_id / 512U) % 4U == 3U ? 1U : 0U);
    const uint32_t remaining = --remaining_qp_rows[qp_id];
    const bool close_batch = remaining == 0U || pending_qp_wqes[qp_id] + 1U == kWqeBatchSize;
    auto* batch_wqe_scratch = wqe_scratch_ub + uint64_t(qp_id) * kWqeBatchScratchBytes;
    if (close_batch) {
        aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kPayloadRoBatchedConfig>(
            destination_address, source_address, batch_wqe_scratch, tiling.hidden_size * 2U, destination_rank, qp_id,
            EVENT_ID0, *udma_submit[qp_id]);
        pending_qp_wqes[qp_id] = 0U;
    } else {
        aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kPayloadRoNoCqeConfig>(
            destination_address, source_address, batch_wqe_scratch, tiling.hidden_size * 2U, destination_rank, qp_id,
            EVENT_ID0, *udma_defer[qp_id]);
        ++pending_qp_wqes[qp_id];
    }
}
// 控制区：SO 完成标志 [W,2]、credit 许可 [W]、不可变信号源 [32]、self-return 完成槽 [64]，
// 以及 server 内汇聚完成标志 [global_chunk,8]、本卡归约完成计数 [global_chunk]。
// 每个汇聚标志只由本 server 的一个 expert rank 写入；归约计数只由本卡归约核原子累加。
// 每个标志独占 512 字节。数据区偏移由 host 使用带溢出检查的 uint64 运算计算。
// 发送源数据在所属 QP 的 quiet 完成前保持有效；本地输出在 kernel 完成前保持有效。
// 回传阶段的 barrier 结束所有控制槽读写，之后同一 stream 的下一次 Prepare 才能重置这些槽位。
// UB 分配：BF16 输入 16 KiB、FP32 临时区 32 KiB、累加器 32 KiB、双槽 BF16 输出 32 KiB，
// 双缓冲字节拷贝区 32 KiB、WQE 暂存区 16 KiB、标量信号区 512 字节。
// 共 160.5 KiB。四个归约缓冲区独立，不复用地址。
// 归约核使用字节拷贝区的前 4 KiB 轮询汇聚标志。本地回传核 0..7 使用各自的拷贝 UB，
// 不执行归约核的标志轮询；不会在异步操作尚未结束时复用 UB。
// 核 0..7 在 gather 阶段先用拷贝区的前 3072 字节缓存 backward 元数据，回传阶段才用于数据拷贝。
__aicore__ inline __gm__ uint64_t* Flag(GM_ADDR workspace, uint32_t flag_slot)
{
    return reinterpret_cast<__gm__ uint64_t*>(workspace + uint64_t(flag_slot) * combine_control::kSlotBytes);
}
constexpr AscendC::MicroAPI::CastTrait kReduceBf16ToFp32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
constexpr AscendC::MicroAPI::CastTrait kReduceFp32ToBf16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

template <bool HasAccumulator, bool HasSecond, bool Final>
__simd_vf__ void SourceReducePairVf(__ubuf__ bfloat16_t* first, __ubuf__ bfloat16_t* second,
                                    __ubuf__ float* accumulator, __ubuf__ bfloat16_t* output, uint32_t element_num)
{
    // 两条独立的 64 元素计算链；每个元素仍按 server 顺序累加。
    uint32_t offset = 0;
    for (; offset + 128U <= element_num; offset += 128U) {
        uint32_t valid = 64U;
        auto mask = AscendC::MicroAPI::UpdateMask<float>(valid);
        AscendC::MicroAPI::RegTensor<bfloat16_t> bf16_value0, bf16_value1;
        AscendC::MicroAPI::RegTensor<float> sum0, sum1, value0, value1;
        if constexpr (HasAccumulator) {
            AscendC::MicroAPI::LoadAlign(sum0, accumulator + offset);
            AscendC::MicroAPI::LoadAlign(sum1, accumulator + offset + 64U);
        }
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value0,
                                                                                               first + offset);
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value1,
                                                                                               first + offset + 64U);
        if constexpr (HasAccumulator) {
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value0, bf16_value0, mask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value1, bf16_value1, mask);
            AscendC::MicroAPI::Add(sum0, sum0, value0, mask);
            AscendC::MicroAPI::Add(sum1, sum1, value1, mask);
        } else {
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(sum0, bf16_value0, mask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(sum1, bf16_value1, mask);
        }
        if constexpr (HasSecond) {
            AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value0,
                                                                                                   second + offset);
            AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                bf16_value1, second + offset + 64U);
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value0, bf16_value0, mask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value1, bf16_value1, mask);
            AscendC::MicroAPI::Add(sum0, sum0, value0, mask);
            AscendC::MicroAPI::Add(sum1, sum1, value1, mask);
        }
        if constexpr (Final) {
            AscendC::MicroAPI::Cast<bfloat16_t, float, kReduceFp32ToBf16>(bf16_value0, sum0, mask);
            AscendC::MicroAPI::Cast<bfloat16_t, float, kReduceFp32ToBf16>(bf16_value1, sum1, mask);
            AscendC::MicroAPI::StoreAlign<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(output + offset,
                                                                                                   bf16_value0, mask);
            AscendC::MicroAPI::StoreAlign<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                output + offset + 64U, bf16_value1, mask);
        } else {
            AscendC::MicroAPI::StoreAlign(accumulator + offset, sum0, mask);
            AscendC::MicroAPI::StoreAlign(accumulator + offset + 64U, sum1, mask);
        }
    }
    // 尾部只读实际存在的向量；UB 按完整 tile 分配，写回由有效元素 mask 限定。
    for (; offset < element_num; offset += 64U) {
        uint32_t valid = element_num - offset < 64U ? element_num - offset : 64U;
        auto mask = AscendC::MicroAPI::UpdateMask<float>(valid);
        AscendC::MicroAPI::RegTensor<bfloat16_t> bf16_value;
        AscendC::MicroAPI::RegTensor<float> sum, value;
        if constexpr (HasAccumulator) {
            AscendC::MicroAPI::LoadAlign(sum, accumulator + offset);
        }
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value,
                                                                                               first + offset);
        if constexpr (HasAccumulator) {
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value, bf16_value, mask);
            AscendC::MicroAPI::Add(sum, sum, value, mask);
        } else {
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(sum, bf16_value, mask);
        }
        if constexpr (HasSecond) {
            AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value,
                                                                                                   second + offset);
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value, bf16_value, mask);
            AscendC::MicroAPI::Add(sum, sum, value, mask);
        }
        if constexpr (Final) {
            AscendC::MicroAPI::Cast<bfloat16_t, float, kReduceFp32ToBf16>(bf16_value, sum, mask);
            AscendC::MicroAPI::StoreAlign<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(output + offset,
                                                                                                   bf16_value, mask);
        } else {
            AscendC::MicroAPI::StoreAlign(accumulator + offset, sum, mask);
        }
    }
}

template <bool HasFourth>
__simd_vf__ void SourceReduceFourVf(__ubuf__ bfloat16_t* first, __ubuf__ bfloat16_t* second, __ubuf__ bfloat16_t* third,
                                    __ubuf__ bfloat16_t* fourth, __ubuf__ bfloat16_t* output, uint32_t element_num)
{
    // 只在寄存器保存 FP32 累加值；第三/第四路不能预先相加，以保持原有舍入顺序。
    // 两条独立的 64 元素计算链；每个元素仍按 server 顺序累加。
    uint32_t offset = 0;
    for (; offset + 128U <= element_num; offset += 128U) {
        uint32_t valid = 64U;
        auto mask = AscendC::MicroAPI::UpdateMask<float>(valid);
        AscendC::MicroAPI::RegTensor<bfloat16_t> bf16_value0, bf16_value1;
        AscendC::MicroAPI::RegTensor<float> sum0, sum1, value0, value1;
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value0,
                                                                                               first + offset);
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value1,
                                                                                               first + offset + 64U);
        AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(sum0, bf16_value0, mask);
        AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(sum1, bf16_value1, mask);
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value0,
                                                                                               second + offset);
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value1,
                                                                                               second + offset + 64U);
        AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value0, bf16_value0, mask);
        AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value1, bf16_value1, mask);
        AscendC::MicroAPI::Add(sum0, sum0, value0, mask);
        AscendC::MicroAPI::Add(sum1, sum1, value1, mask);
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value0,
                                                                                               third + offset);
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value1,
                                                                                               third + offset + 64U);
        AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value0, bf16_value0, mask);
        AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value1, bf16_value1, mask);
        AscendC::MicroAPI::Add(sum0, sum0, value0, mask);
        AscendC::MicroAPI::Add(sum1, sum1, value1, mask);
        if constexpr (HasFourth) {
            AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value0,
                                                                                                   fourth + offset);
            AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                bf16_value1, fourth + offset + 64U);
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value0, bf16_value0, mask);
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value1, bf16_value1, mask);
            AscendC::MicroAPI::Add(sum0, sum0, value0, mask);
            AscendC::MicroAPI::Add(sum1, sum1, value1, mask);
        }
        AscendC::MicroAPI::Cast<bfloat16_t, float, kReduceFp32ToBf16>(bf16_value0, sum0, mask);
        AscendC::MicroAPI::Cast<bfloat16_t, float, kReduceFp32ToBf16>(bf16_value1, sum1, mask);
        AscendC::MicroAPI::StoreAlign<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(output + offset,
                                                                                               bf16_value0, mask);
        AscendC::MicroAPI::StoreAlign<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(output + offset + 64U,
                                                                                               bf16_value1, mask);
    }
    // 尾部只读实际存在的向量；UB 按完整 tile 分配，写回由有效元素 mask 限定。
    for (; offset < element_num; offset += 64U) {
        uint32_t valid = element_num - offset < 64U ? element_num - offset : 64U;
        auto mask = AscendC::MicroAPI::UpdateMask<float>(valid);
        AscendC::MicroAPI::RegTensor<bfloat16_t> bf16_value;
        AscendC::MicroAPI::RegTensor<float> sum, value;
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value,
                                                                                               first + offset);
        AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(sum, bf16_value, mask);
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value,
                                                                                               second + offset);
        AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value, bf16_value, mask);
        AscendC::MicroAPI::Add(sum, sum, value, mask);
        AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value,
                                                                                               third + offset);
        AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value, bf16_value, mask);
        AscendC::MicroAPI::Add(sum, sum, value, mask);
        if constexpr (HasFourth) {
            AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(bf16_value,
                                                                                                   fourth + offset);
            AscendC::MicroAPI::Cast<float, bfloat16_t, kReduceBf16ToFp32>(value, bf16_value, mask);
            AscendC::MicroAPI::Add(sum, sum, value, mask);
        }
        AscendC::MicroAPI::Cast<bfloat16_t, float, kReduceFp32ToBf16>(bf16_value, sum, mask);
        AscendC::MicroAPI::StoreAlign<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(output + offset,
                                                                                               bf16_value, mask);
    }
}

__aicore__ inline bool Ready(__gm__ uint64_t* flag_address, uint64_t expected_value)
{
    GlobalTensor<uint64_t> flag_gm;
    flag_gm.SetGlobalBuffer(flag_address);
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(flag_address));
    return flag_gm.GetValue(0) >= expected_value;
}
__aicore__ inline void StoreFlag(__gm__ uint64_t* flag_address, LocalTensor<uint64_t> flag_ub, uint64_t flag_value)
{
    flag_ub.SetValue(0, flag_value);
    NativeStore(flag_address, flag_ub, 1U);
}
__aicore__ inline void Prepare(GM_ADDR workspace, uint32_t rank_num, uint32_t source_chunk_num,
                               uint32_t control_slot_num)
{
    const uint32_t core_id = GetBlockIdx();
    TPipe pipe;
    TBuf<QuePosition::VECCALC> signal_buffer;
    pipe.InitBuffer(signal_buffer, kSignalUbBytes);
    auto signal_ub = signal_buffer.Get<uint64_t>();

    // 排在同一 stream 的上一次 Combine 之后，本卡所有核均已结束且所有 QP 已 quiet。
    // 上轮尾部还确认了空任务的 SO，避免迟到的旧信号污染下一轮清零后的控制槽。
    // 这里仅重置控制槽，不修改数据区，因此无需在重置前再增加一次集合同步。
    for (uint64_t flag_slot = core_id; flag_slot < control_slot_num; flag_slot += combine_config::kAivCores) {
        // 每个 512 字节槽位只有一个写入核，仅首个 uint64 用作信号。
        // 所有 QP 完成前，SO 使用的信号源值始终保持为 1。
        // PUT SO reads this fixed source bank; gather_done is a RECEIVE bank
        // and must start at zero, just like incoming SO and reduce counters.
        const bool completion_source = flag_slot >= 3U * rank_num && flag_slot < 3U * rank_num + 32U;
        StoreFlag(Flag(workspace, uint32_t(flag_slot)), signal_ub, completion_source ? 1U : 0U);
    }
    // NativeStore 通过 MTE3_S 等待写回。所有接收槽重置完成前，任何 rank 都不能发送新的
    // SO/credit。这里不清零数据区。
    aclshmemx_barrier_all_vec();
}
__aicore__ inline void WaitGatherDone(GM_ADDR workspace, const CombineTiling& tiling, uint32_t source_rank,
                                      uint32_t chunk_index, uint32_t contributor_mask,
                                      LocalTensor<uint64_t> gather_flags_ub)
{
    // 没有远端贡献时无需读取 flag；本卡贡献已由同一 stream 的输入拷贝准备好。
    if (!contributor_mask) {
        return;
    }
    constexpr uint32_t words_per_slot = combine_control::kSlotBytes / sizeof(uint64_t);
    GlobalTensor<uint64_t> gather_flags_gm;
    gather_flags_gm.SetGlobalBuffer(
        Flag(workspace, 3U * tiling.rank_num + combine_control::kGatherDoneBase +
                            (source_rank * tiling.chunks_per_group + chunk_index) * combine_control::kGatherDoneLanes));
    gather_flags_gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    while (true) {
        // 每次轮询进行一次对齐的 GM->UB 搬运（8 个 rank 共 4 KiB）。
        // 只检查每个独立槽位的首个信号值，不读取其填充区作为标志。
        SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
        DataCopy(gather_flags_ub, gather_flags_gm, combine_control::kGatherDoneLanes * words_per_slot);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
        bool all_ready = true;
        for (uint32_t local_rank_id = 0; local_rank_id < combine_control::kGatherDoneLanes; ++local_rank_id) {
            if (contributor_mask & (1U << local_rank_id)) {
                all_ready = (gather_flags_ub.GetValue(local_rank_id * words_per_slot) != 0U) && all_ready;
            }
        }
        if (all_ready) {
            return;
        }
    }
}
__aicore__ inline void WaitReduceDone(GM_ADDR workspace, const CombineTiling& tiling, uint32_t source_rank,
                                      uint32_t chunk_index, uint32_t reduce_core_num,
                                      LocalTensor<int32_t> reduce_done_ub)
{
    GlobalTensor<int32_t> reduce_done_gm;
    reduce_done_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
        Flag(workspace, 3U * tiling.rank_num + combine_control::kGatherDoneBase +
                            SourceChunkCount(tiling) * combine_control::kGatherDoneLanes +
                            source_rank * tiling.chunks_per_group + chunk_index)));
    reduce_done_gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    do {
        SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
        DataCopyPad(reduce_done_ub, reduce_done_gm, DataCopyExtParams(1, sizeof(int32_t), 0, 0, 0),
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
    } while (reduce_done_ub.GetValue(0) != static_cast<int32_t>(reduce_core_num));
}
__aicore__ inline bool ReduceDoneReady(GM_ADDR workspace, const CombineTiling& tiling, uint32_t source_rank,
                                       uint32_t chunk_index, uint32_t reduce_core_num)
{
    GlobalTensor<int32_t> reduce_done_gm;
    auto* reduce_done_address = reinterpret_cast<__gm__ int32_t*>(
        Flag(workspace, 3U * tiling.rank_num + combine_control::kGatherDoneBase +
                            SourceChunkCount(tiling) * combine_control::kGatherDoneLanes +
                            source_rank * tiling.chunks_per_group + chunk_index));
    reduce_done_gm.SetGlobalBuffer(reduce_done_address);
    reduce_done_gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(reduce_done_address));
    return reduce_done_gm.GetValue(0) >= static_cast<int32_t>(reduce_core_num);
}
__aicore__ inline bool ChunkNeedsGatewayReduce(GlobalTensor<int32_t>& forward_list_gm,
                                               GlobalTensor<int32_t>& chunk_ranges_gm,
                                               GlobalTensor<int32_t>& chunk_masks_gm, const CombineTiling& tiling,
                                               uint32_t source_rank, uint32_t chunk_index)
{
    const uint64_t source_chunk = uint64_t(source_rank) * tiling.chunks_per_group + chunk_index;
    if (chunk_masks_gm.GetValue(source_chunk) != 0) {
        return true;
    }
    // The gather mask excludes this rank. Local extra contributions still
    // produce server_partial and must finish before any return path reads it.
    const uint32_t begin = chunk_ranges_gm.GetValue(source_chunk * 2U);
    const uint32_t end = chunk_ranges_gm.GetValue(source_chunk * 2U + 1U);
    const uint64_t source_base = uint64_t(source_rank) * tiling.forward_row_capacity;
    for (uint32_t task = begin; task < end; ++task) {
        if (forward_list_gm.GetValue((source_base + task) * 6U + 3U) >= 0) {
            return true;
        }
    }
    return false;
}
__aicore__ inline bool SourceNeedsGatewayReduce(GlobalTensor<int32_t>& forward_list_gm,
                                                GlobalTensor<int32_t>& chunk_ranges_gm,
                                                GlobalTensor<int32_t>& chunk_masks_gm, const CombineTiling& tiling,
                                                uint32_t source_rank)
{
    for (uint32_t chunk_index = 0; chunk_index < tiling.chunks_per_group; ++chunk_index) {
        if (ChunkNeedsGatewayReduce(forward_list_gm, chunk_ranges_gm, chunk_masks_gm, tiling, source_rank,
                                    chunk_index)) {
            return true;
        }
    }
    return false;
}
__aicore__ inline void AddBf16(__gm__ bfloat16_t* source_address, uint32_t element_num,
                               LocalTensor<bfloat16_t> bf16_input_ub, LocalTensor<float> fp32_input_ub,
                               LocalTensor<float> fp32_accumulator_ub)
{
    GlobalTensor<bfloat16_t> input_gm;
    input_gm.SetGlobalBuffer(source_address);
    DataCopy(bf16_input_ub, input_gm, element_num);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(fp32_input_ub, bf16_input_ub, RoundMode::CAST_NONE, element_num);
    PipeBarrier<PIPE_V>();
    Add(fp32_accumulator_ub, fp32_accumulator_ub, fp32_input_ub, element_num);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
}
__aicore__ inline bool TokenNeedsGatewayReduce(GlobalTensor<int32_t>& forward_list_gm, uint64_t source_task_base,
                                               uint32_t task_begin)
{
    // The caller has already found the complete token run and task_begin is
    // its first row.  The planner stores the primary row first: -1 in column
    // 3 means this token is a singleton, while any non-negative contributor
    // rank means the gateway must combine at least one extra contribution.
    return forward_list_gm.GetValue((source_task_base + task_begin) * 6U + 3U) >= 0;
}
__aicore__ inline uint32_t AlignTaskToTokenBoundary(GlobalTensor<int32_t>& forward_list_gm, uint64_t source_task_base,
                                                    uint32_t task_count, uint32_t task_index)
{
    if (task_index == 0U || task_index >= task_count) {
        return task_index < task_count ? task_index : task_count;
    }
    const int32_t previous_token = forward_list_gm.GetValue((source_task_base + task_index - 1U) * 6U + 1U);
    while (task_index < task_count &&
           forward_list_gm.GetValue((source_task_base + task_index) * 6U + 1U) == previous_token) {
        ++task_index;
    }
    return task_index;
}
__aicore__ inline uint32_t SelfCopySplitTask(GlobalTensor<int32_t>& forward_list_gm, const CombineTiling& tiling,
                                             uint32_t task_count)
{
    return AlignTaskToTokenBoundary(forward_list_gm, uint64_t(tiling.rank_id) * tiling.forward_row_capacity, task_count,
                                    task_count / 2U);
}
__aicore__ inline void ReduceGateway(GM_ADDR workspace, GlobalTensor<int32_t>& forward_list_gm,
                                     GlobalTensor<int32_t>& chunk_ranges_gm, const CombineTiling& tiling,
                                     uint32_t source_rank_id, uint32_t chunk_index, uint32_t reduce_core_index,
                                     uint32_t reduce_core_num, LocalTensor<bfloat16_t> bf16_input_ub,
                                     LocalTensor<float> fp32_input_ub, LocalTensor<float> fp32_accumulator_ub,
                                     LocalTensor<bfloat16_t> bf16_output_ub)
{
    // 槽 A/B 分别使用 EVENT_ID1/2；仅在实际写回后产生 MTE3_V 事件。
    uint32_t output_slot = 0;
    bool output_store_pending[kReduceOutputSlots]{};
    // Reduce one source/chunk run. The caller walks remote sources first,
    // then same-server sources, and chunks second.
    {
        const uint64_t range_index = (uint64_t(source_rank_id) * tiling.chunks_per_group + chunk_index) * 2U;
        const uint32_t chunk_task_begin = chunk_ranges_gm.GetValue(range_index);
        const uint32_t chunk_task_end = chunk_ranges_gm.GetValue(range_index + 1U);
        const uint32_t task_num = chunk_task_end - chunk_task_begin;
        const uint64_t source_task_base = uint64_t(source_rank_id) * tiling.forward_row_capacity;
        // Rotate the statically sliced partition across source ranks. This
        // keeps the existing disjoint ranges and token-boundary handling, but
        // spreads the q/q+1 slices caused by integer division over all reduce
        // cores instead of assigning the same remainder pattern to each one.
        const uint32_t partition_index = (reduce_core_index + source_rank_id) % reduce_core_num;
        uint32_t task_index = chunk_task_begin + uint64_t(task_num) * partition_index / reduce_core_num;
        const uint32_t worker_task_end =
            chunk_task_begin + uint64_t(task_num) * (partition_index + 1U) / reduce_core_num;
        // 仅当 token 的首条任务落在本核原始分工区间内时，才由本核负责该 token。
        while (task_index < worker_task_end && task_index > chunk_task_begin &&
               forward_list_gm.GetValue((source_task_base + task_index) * 6U + 1U) ==
                   forward_list_gm.GetValue((source_task_base + task_index - 1U) * 6U + 1U)) {
            ++task_index;
        }
        while (task_index < worker_task_end) {
            uint32_t token_task_end = task_index + 1;
            const int32_t token_id = forward_list_gm.GetValue((source_task_base + task_index) * 6U + 1U);
            const uint32_t primary_row = forward_list_gm.GetValue((source_task_base + task_index) * 6U + 2U);
            while (token_task_end < chunk_task_end &&
                   forward_list_gm.GetValue((source_task_base + token_task_end) * 6U + 1U) == token_id) {
                ++token_task_end;
            }
            if (!TokenNeedsGatewayReduce(forward_list_gm, source_task_base, task_index)) {
                task_index = token_task_end;
                continue;
            }
            for (uint32_t hidden_offset = 0; hidden_offset < tiling.hidden_size; hidden_offset += kReduceTileElements) {
                const uint32_t element_num = tiling.hidden_size - hidden_offset < kReduceTileElements
                                                 ? tiling.hidden_size - hidden_offset
                                                 : kReduceTileElements;
                // 首份贡献直接初始化累加器，省去清零和首次加法。
                GlobalTensor<bfloat16_t> primary_input_gm;
                primary_input_gm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ bfloat16_t*>(workspace + tiling.expert_input_offset_bytes) +
                    uint64_t(primary_row) * tiling.hidden_size + hidden_offset);
                DataCopy(bf16_input_ub, primary_input_gm, element_num);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Cast(fp32_accumulator_ub, bf16_input_ub, RoundMode::CAST_NONE, element_num);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                for (uint32_t contributor_task_index = task_index; contributor_task_index < token_task_end;
                     ++contributor_task_index) {
                    const int32_t contributor_rank =
                        forward_list_gm.GetValue((source_task_base + contributor_task_index) * 6U + 3U);
                    if (contributor_rank < 0) {
                        continue;
                    }

                    const bool is_local_contribution = uint32_t(contributor_rank) == tiling.rank_id;
                    const uint32_t contributor_row = forward_list_gm.GetValue(
                        (source_task_base + contributor_task_index) * 6U + (is_local_contribution ? 4U : 5U));
                    AddBf16(reinterpret_cast<__gm__ bfloat16_t*>(workspace + (is_local_contribution
                                                                                  ? tiling.expert_input_offset_bytes
                                                                                  : tiling.gather_input_offset_bytes)) +
                                uint64_t(contributor_row) * tiling.hidden_size + hidden_offset,
                            element_num, bf16_input_ub, fp32_input_ub, fp32_accumulator_ub);
                }
                // server 内先用 FP32 完成累加，再将 partial 舍入为 BF16 后写回并发送。
                // 只等待即将复用的槽，另一个槽可继续写回；按实际 token/tile 写出次数轮转。
                const auto output_event_id = output_slot == 0U ? EVENT_ID1 : EVENT_ID2;
                auto bf16_output_slot_ub = bf16_output_ub[output_slot * kReduceTileElements];
                if (output_store_pending[output_slot]) {
                    WaitFlag<HardEvent::MTE3_V>(output_event_id);
                    output_store_pending[output_slot] = false;
                }
                Cast(bf16_output_slot_ub, fp32_accumulator_ub, RoundMode::CAST_RINT, element_num);
                // 当前 Cast 读完累加器后，下一条 token/tile 才能重新初始化它。
                PipeBarrier<PIPE_V>();
                GlobalTensor<bfloat16_t> partial_output_gm;
                partial_output_gm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ bfloat16_t*>(workspace + tiling.server_partial_offset_bytes) +
                    uint64_t(primary_row) * tiling.hidden_size + hidden_offset);
                SetFlag<HardEvent::V_MTE3>(output_event_id);
                WaitFlag<HardEvent::V_MTE3>(output_event_id);
                DataCopy(partial_output_gm, bf16_output_slot_ub, element_num);
                SetFlag<HardEvent::MTE3_V>(output_event_id);
                output_store_pending[output_slot] = true;
                output_slot ^= 1U;
            }
            task_index = token_task_end;
        }
    }
    // 返回后会发布本 chunk 的 reduce_done；两个输出槽都必须排空。
    // 空任务核、只写出一条的核，不等待未产生的事件。
    for (uint32_t slot = 0; slot < kReduceOutputSlots; ++slot) {
        if (output_store_pending[slot]) {
            const auto output_event_id = slot == 0U ? EVENT_ID1 : EVENT_ID2;
            WaitFlag<HardEvent::MTE3_V>(output_event_id);
        }
    }
}
__aicore__ inline void SendPartial(GM_ADDR workspace, GM_ADDR forward_list, GlobalTensor<int32_t>& forward_list_gm,
                                   const CombineTiling& tiling, uint32_t destination_rank, uint32_t task_begin,
                                   uint32_t task_end, LocalTensor<uint8_t> copy_ub, __ubuf__ uint8_t* wqe_scratch_ub,
                                   PartialClass partial_class = PartialClass::All)
{
    const uint32_t server_num = (tiling.rank_num + 7U) / 8U;
    const uint64_t source_task_base = uint64_t(destination_rank) * tiling.forward_row_capacity;
    uint32_t local_copy_rows = 0;
    const bool remote_destination = destination_rank != tiling.rank_id;
    uint32_t remaining_qp_rows[2]{};
    uint32_t pending_qp_wqes[2]{};
    aclshmemx_submit_state_t udma_submit_state[2]{};
    aclshmemx_defer_t udma_defer_qp0(udma_submit_state[0]);
    aclshmemx_defer_t udma_defer_qp1(udma_submit_state[1]);
    aclshmemx_submit_t udma_submit_qp0(udma_submit_state[0]);
    aclshmemx_submit_t udma_submit_qp1(udma_submit_state[1]);
    aclshmemx_defer_t* udma_defer[2] = {&udma_defer_qp0, &udma_defer_qp1};
    aclshmemx_submit_t* udma_submit[2] = {&udma_submit_qp0, &udma_submit_qp1};
    if (remote_destination) {
        // Count one WQE per token group first. This lets the last WQE of each
        // QP close its deferred batch even though the schedule interleaves QP0
        // and QP1 token ranges.
        auto forward_meta_ub = copy_ub.ReinterpretCast<int32_t>();
        uint32_t scan_index = task_begin;
        while (scan_index < task_end) {
            const uint32_t batch_begin = scan_index;
            const uint32_t batch_end =
                task_end - batch_begin < kForwardMetaBatchRows ? task_end : batch_begin + kForwardMetaBatchRows;
            const uint32_t loaded_rows = batch_end - batch_begin;
            NativeLoad(forward_meta_ub,
                       reinterpret_cast<__gm__ int32_t*>(forward_list) +
                           uint64_t(source_task_base + batch_begin) * kForwardListFields,
                       loaded_rows * kForwardListFields);

            // A batch may end in the middle of a token run. Defer that final
            // run and reload its rows with the next batch so every processed
            // token remains complete. This also keeps the planner's token
            // boundary invariant independent of the chosen batch size.
            uint32_t complete_rows = loaded_rows;
            if (batch_end < task_end) {
                const int32_t last_token = forward_meta_ub.GetValue((loaded_rows - 1U) * kForwardListFields + 1U);
                while (complete_rows > 0U &&
                       forward_meta_ub.GetValue((complete_rows - 1U) * kForwardListFields + 1U) == last_token) {
                    --complete_rows;
                }
            }
            uint32_t local_index = 0U;
            while (local_index < complete_rows) {
                const uint32_t token_begin = local_index;
                const uint32_t token_id =
                    static_cast<uint32_t>(forward_meta_ub.GetValue(token_begin * kForwardListFields + 1U));
                uint32_t token_end = token_begin + 1U;
                while (token_end < complete_rows && static_cast<uint32_t>(forward_meta_ub.GetValue(
                                                        token_end * kForwardListFields + 1U)) == token_id) {
                    ++token_end;
                }
                const bool needs_reduce = forward_meta_ub.GetValue(token_begin * kForwardListFields + 3U) >= 0;
                CountRemoteToken(tiling, destination_rank, partial_class, token_id, needs_reduce, remaining_qp_rows);
                local_index = token_end;
            }
            if (complete_rows != 0U) {
                scan_index = batch_begin + complete_rows;
                continue;
            }

            // The fallback is only reachable if one token run is at least as
            // large as the metadata batch. It guarantees forward progress for
            // unusual top-k settings without imposing a hidden run-length
            // limit on the planner.
            const uint32_t token_id =
                forward_list_gm.GetValue((source_task_base + scan_index) * kForwardListFields + 1U);
            uint32_t token_end = scan_index + 1U;
            while (token_end < task_end && static_cast<uint32_t>(forward_list_gm.GetValue(
                                               (source_task_base + token_end) * kForwardListFields + 1U)) == token_id) {
                ++token_end;
            }
            CountRemoteToken(tiling, destination_rank, partial_class, token_id,
                             TokenNeedsGatewayReduce(forward_list_gm, source_task_base, scan_index), remaining_qp_rows);
            scan_index = token_end;
        }
    }
    // Remote payloads use the batched metadata path above. The local-return
    // path below keeps the existing tiled byte-copy implementation; neither
    if (remote_destination) {
        // The payload path uses the same six forward-list fields as the
        // planning pass above. Keep metadata in UB while walking complete
        // token runs, avoiding scalar GM reads for every remote token.
        auto forward_meta_ub = copy_ub.ReinterpretCast<int32_t>();
        uint32_t send_index = task_begin;
        while (send_index < task_end) {
            const uint32_t batch_begin = send_index;
            const uint32_t batch_end =
                task_end - batch_begin < kForwardMetaBatchRows ? task_end : batch_begin + kForwardMetaBatchRows;
            const uint32_t loaded_rows = batch_end - batch_begin;
            NativeLoad(forward_meta_ub,
                       reinterpret_cast<__gm__ int32_t*>(forward_list) +
                           uint64_t(source_task_base + batch_begin) * kForwardListFields,
                       loaded_rows * kForwardListFields);
            uint32_t complete_rows = loaded_rows;
            if (batch_end < task_end) {
                const int32_t last_token = forward_meta_ub.GetValue((loaded_rows - 1U) * kForwardListFields + 1U);
                while (complete_rows > 0U &&
                       forward_meta_ub.GetValue((complete_rows - 1U) * kForwardListFields + 1U) == last_token) {
                    --complete_rows;
                }
            }
            if (complete_rows != 0U) {
                uint32_t local_index = 0U;
                while (local_index < complete_rows) {
                    const uint32_t token_begin = local_index;
                    const uint32_t token_id =
                        static_cast<uint32_t>(forward_meta_ub.GetValue(token_begin * kForwardListFields + 1U));
                    uint32_t token_end = token_begin + 1U;
                    while (token_end < complete_rows && static_cast<uint32_t>(forward_meta_ub.GetValue(
                                                            token_end * kForwardListFields + 1U)) == token_id) {
                        ++token_end;
                    }
                    const bool needs_reduce = forward_meta_ub.GetValue(token_begin * kForwardListFields + 3U) >= 0;
                    if (!((partial_class == PartialClass::DirectOnly && needs_reduce) ||
                          (partial_class == PartialClass::ReducedOnly && !needs_reduce))) {
                        SendRemoteToken(
                            workspace, tiling, destination_rank, server_num, token_id,
                            static_cast<uint32_t>(forward_meta_ub.GetValue(token_begin * kForwardListFields + 2U)),
                            needs_reduce, wqe_scratch_ub, remaining_qp_rows, pending_qp_wqes, udma_defer, udma_submit);
                    }
                    local_index = token_end;
                }
                send_index = batch_begin + complete_rows;
                continue;
            }

            // A run larger than the metadata batch is unusual, but this
            // fallback preserves progress without imposing a top-k limit.
            const uint32_t token_id =
                forward_list_gm.GetValue((source_task_base + send_index) * kForwardListFields + 1U);
            const uint32_t primary_row =
                forward_list_gm.GetValue((source_task_base + send_index) * kForwardListFields + 2U);
            uint32_t token_end = send_index + 1U;
            while (token_end < task_end && static_cast<uint32_t>(forward_list_gm.GetValue(
                                               (source_task_base + token_end) * kForwardListFields + 1U)) == token_id) {
                ++token_end;
            }
            const bool needs_reduce = TokenNeedsGatewayReduce(forward_list_gm, source_task_base, send_index);
            if (!((partial_class == PartialClass::DirectOnly && needs_reduce) ||
                  (partial_class == PartialClass::ReducedOnly && !needs_reduce))) {
                SendRemoteToken(workspace, tiling, destination_rank, server_num, token_id, primary_row, needs_reduce,
                                wqe_scratch_ub, remaining_qp_rows, pending_qp_wqes, udma_defer, udma_submit);
            }
            send_index = token_end;
        }
    }
    if (!remote_destination) {
        for (uint32_t task_index = task_begin; task_index < task_end;) {
            const uint32_t token_id = forward_list_gm.GetValue((source_task_base + task_index) * 6U + 1U);
            const uint32_t primary_row = forward_list_gm.GetValue((source_task_base + task_index) * 6U + 2U);
            uint32_t token_end = task_index + 1U;
            while (token_end < task_end &&
                   uint32_t(forward_list_gm.GetValue((source_task_base + token_end) * 6U + 1U)) == token_id) {
                ++token_end;
            }
            const bool needs_reduce = TokenNeedsGatewayReduce(forward_list_gm, source_task_base, task_index);
            if ((partial_class == PartialClass::DirectOnly && needs_reduce) ||
                (partial_class == PartialClass::ReducedOnly && !needs_reduce)) {
                task_index = token_end;
                continue;
            }
            auto* source_address =
                workspace + (needs_reduce ? tiling.server_partial_offset_bytes : tiling.expert_input_offset_bytes) +
                uint64_t(primary_row) * tiling.hidden_size * 2U;
            auto* destination_address =
                workspace + tiling.returned_partial_offset_bytes +
                (uint64_t(token_id) * server_num + tiling.rank_id / 8U) * tiling.hidden_size * 2U;
            if (destination_rank == tiling.rank_id) {
                // 仅与本 rank 的 server 内编号对应的核写入这些本地回传行。
                // 前面的归约计数等待已保证 gateway 的部分归约结果可读。
                GlobalTensor<uint8_t> local_source_gm;
                GlobalTensor<uint8_t> local_destination_gm;
                local_source_gm.SetGlobalBuffer(source_address);
                local_destination_gm.SetGlobalBuffer(destination_address);
                const uint32_t row_bytes = tiling.hidden_size * 2U;
                // Keep two independent byte-copy tiles in flight.  MTE2 can load
                // the next tile while MTE3 writes the previous one back to GM.
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
                uint32_t copy_slot = 0U;
                for (uint32_t byte_offset = 0; byte_offset < row_bytes; byte_offset += kLocalCopyTileBytes) {
                    const uint32_t copy_bytes =
                        row_bytes - byte_offset < kLocalCopyTileBytes ? row_bytes - byte_offset : kLocalCopyTileBytes;
                    const DataCopyExtParams copy_params(1, copy_bytes, 0, 0, 0);
                    const auto copy_event_id = copy_slot == 0U ? EVENT_ID2 : EVENT_ID4;
                    auto copy_slot_ub = copy_ub[copy_slot * kLocalCopyTileBytes];
                    WaitFlag<HardEvent::MTE3_MTE2>(copy_event_id);
                    DataCopyPad(copy_slot_ub, local_source_gm[byte_offset], copy_params,
                                DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
                    SetFlag<HardEvent::MTE2_MTE3>(copy_event_id);
                    WaitFlag<HardEvent::MTE2_MTE3>(copy_event_id);
                    DataCopyPad(local_destination_gm[byte_offset], copy_slot_ub, copy_params);
                    SetFlag<HardEvent::MTE3_MTE2>(copy_event_id);
                    copy_slot ^= 1U;
                }
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);

            } else {
                const uint32_t qp_id =
                    destination_rank / 8U == tiling.rank_id / 8U ? 0U : ((token_id / 512U) % 4U == 3U ? 1U : 0U);
                const uint32_t remaining = --remaining_qp_rows[qp_id];
                const bool close_batch = remaining == 0U || pending_qp_wqes[qp_id] + 1U == kWqeBatchSize;
                auto* batch_wqe_scratch = wqe_scratch_ub + uint64_t(qp_id) * kWqeBatchScratchBytes;
                if (close_batch) {
                    aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kPayloadRoConfig>(
                        destination_address, source_address, batch_wqe_scratch, tiling.hidden_size * 2U,
                        destination_rank, qp_id, EVENT_ID0, *udma_submit[qp_id]);
                    pending_qp_wqes[qp_id] = 0U;
                } else {
                    aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kPayloadRoNoCqeConfig>(
                        destination_address, source_address, batch_wqe_scratch, tiling.hidden_size * 2U,
                        destination_rank, qp_id, EVENT_ID0, *udma_defer[qp_id]);
                    ++pending_qp_wqes[qp_id];
                }
            }
            task_index = token_end;
        }
    }
}
__aicore__ inline void WaitReturnedFlag(GM_ADDR workspace, uint32_t flag_slot)
{
    while (!Ready(Flag(workspace, flag_slot), 1U)) {
    }
}
// 每个核独立确认 incoming group；这里的拓扑方向必须和 CPU schedule 一致。
// self 回传由多个核分片，每个写入者拥有一个独立完成槽，不能由单个核代发完成。
__aicore__ inline void WaitReturnedGroup(GM_ADDR workspace, const CombineTiling& tiling, uint32_t group,
                                         uint32_t reduce_core_begin)
{
    // group 在此表示最终归约子阶段；展平后还原通信 group 和其中的逻辑 peer。
    const uint32_t topology_width = first_hit_schedule::Width(tiling.rank_num);
    const uint32_t first_peer = group * tiling.final_reduce_peers_per_stage;
    for (uint32_t index = first_peer; index < first_peer + tiling.final_reduce_peers_per_stage; ++index) {
        const uint32_t gateway_rank = first_hit_schedule::Peer(tiling.rank_id, tiling.rank_num, index / topology_width,
                                                               index % topology_width, true);
        if (gateway_rank == tiling.rank_id) {
            WaitReturnedFlag(workspace, 3U * tiling.rank_num + combine_control::kSelfReturnBase + tiling.rank_id % 8U);
            for (uint32_t writer = reduce_core_begin; writer < combine_config::kAivCores; ++writer) {
                WaitReturnedFlag(workspace, 3U * tiling.rank_num + combine_control::kSelfReturnBase + writer);
            }
            continue;
        }
        const bool gateway_local = gateway_rank / 8U == tiling.rank_id / 8U;
        // 同一轮 flag 单调从 0 变为 1。QP0 观测到后不必在 QP1 轮询中重复读取。
        WaitReturnedFlag(workspace, gateway_rank * 2U);
        if (!gateway_local) {
            WaitReturnedFlag(workspace, gateway_rank * 2U + 1U);
        }
    }
}
__aicore__ inline void QuietCrossServerQps(const CombineTiling& tiling, uint32_t core_id)
{
    const uint32_t sender_index = core_id - 8U;
    const uint32_t sender_num = 16U / tiling.cross_peers_per_core;
    const uint32_t sequence_num = 16U / sender_num;
    const uint32_t group_num = tiling.rank_num / 16U;
    for (uint32_t group_index = 0; group_index < group_num; ++group_index) {
        for (uint32_t sequence_index = 0; sequence_index < sequence_num; ++sequence_index) {
            const uint32_t outgoing_rank = first_hit_schedule::Peer(tiling.rank_id, tiling.rank_num, group_index,
                                                                    sender_index + sequence_index * sender_num);
            if (outgoing_rank / 8U != tiling.rank_id / 8U) {
                aclshmemx_udma_qp_quiet(outgoing_rank, 0U);
                aclshmemx_udma_qp_quiet(outgoing_rank, 1U);
            }
        }
    }
}
__aicore__ inline void CrossSendIndependent(GM_ADDR workspace, GM_ADDR forward_list,
                                            GlobalTensor<int32_t>& forward_list_gm,
                                            GlobalTensor<int32_t>& chunk_ranges_gm,
                                            GlobalTensor<int32_t>& chunk_masks_gm, const CombineTiling& tiling,
                                            uint32_t core_id, uint32_t reduce_core_num, LocalTensor<uint8_t> copy_ub,
                                            LocalTensor<uint64_t> signal_ub, LocalTensor<int32_t> reduce_done_ub,
                                            __ubuf__ uint8_t* wqe_scratch_ub)
{
    (void)reduce_done_ub;
    const uint32_t sender_index = core_id - 8U;
    const uint32_t sender_num = 16U / tiling.cross_peers_per_core;
    const uint32_t sequence_num = 16U / sender_num;
    const uint32_t group_num = tiling.rank_num / 16U;
    const bool wait_credit = tiling.wait_credit != 0U;

    // A sender owns at most two source ranks.  Process one topology group at
    // a time, prioritize remote sources then rank id, and finish all chunks
    // of one source before selecting the next source.  This preserves
    // the credit/incoming-SO dependency between groups while making the
    // remote-first source-major order explicit in both the device pipeline and
    for (uint32_t group_index = 0; group_index < group_num; ++group_index) {
        bool sequence_done[2]{};
        for (uint32_t source_order = 0; source_order < sequence_num; ++source_order) {
            uint32_t sequence_index = 0;
            uint32_t selected_priority = 2U * tiling.rank_num;
            for (uint32_t candidate = 0; candidate < sequence_num; ++candidate) {
                if (sequence_done[candidate]) {
                    continue;
                }
                const uint32_t candidate_rank = first_hit_schedule::Peer(tiling.rank_id, tiling.rank_num, group_index,
                                                                         sender_index + candidate * sender_num);
                const uint32_t candidate_priority =
                    candidate_rank + (candidate_rank / 8U == tiling.rank_id / 8U ? tiling.rank_num : 0U);
                if (candidate_priority < selected_priority) {
                    selected_priority = candidate_priority;
                    sequence_index = candidate;
                }
            }
            sequence_done[sequence_index] = true;

            const uint32_t outgoing_rank = first_hit_schedule::Peer(tiling.rank_id, tiling.rank_num, group_index,
                                                                    sender_index + sequence_index * sender_num);
            const uint32_t incoming_rank = first_hit_schedule::Peer(tiling.rank_id, tiling.rank_num, group_index,
                                                                    sender_index + sequence_index * sender_num, true);
            const bool outgoing_local = outgoing_rank / 8U == tiling.rank_id / 8U;
            const bool incoming_local = incoming_rank / 8U == tiling.rank_id / 8U;
            bool peer_credit_ready = outgoing_local || group_index == 0U || !wait_credit;
            bool peer_direct_sent = false;

            if (outgoing_local) {
                // The local source rank has no cross-server payload.  Its
                // incoming SO is still part of the group hand-off below.
            } else {
                for (uint32_t chunk_index = 0; chunk_index < tiling.chunks_per_group; ++chunk_index) {
                    const uint64_t range_index = (uint64_t(outgoing_rank) * tiling.chunks_per_group + chunk_index) * 2U;
                    const bool chunk_needs_reduce = ChunkNeedsGatewayReduce(
                        forward_list_gm, chunk_ranges_gm, chunk_masks_gm, tiling, outgoing_rank, chunk_index);
                    bool chunk_sent = false;
                    while (!chunk_sent) {
                        if (!peer_credit_ready && wait_credit) {
                            if (Ready(Flag(workspace, 2U * tiling.rank_num + outgoing_rank), group_index + 1U)) {
                                peer_credit_ready = true;
                            }
                        }
                        if (peer_credit_ready && !peer_direct_sent) {
                            // A token with only the gateway's primary contribution is
                            // already valid in expert_input. Submit it before waiting
                            // for the chunk-wide gateway reduction counter.
                            SendPartial(workspace, forward_list, forward_list_gm, tiling, outgoing_rank,
                                        chunk_ranges_gm.GetValue(range_index),
                                        chunk_ranges_gm.GetValue(range_index + 1U), copy_ub, wqe_scratch_ub,
                                        PartialClass::DirectOnly);
                            peer_direct_sent = true;
                        }
                        if (!peer_credit_ready ||
                            (chunk_needs_reduce &&
                             !ReduceDoneReady(workspace, tiling, outgoing_rank, chunk_index, reduce_core_num))) {
                            continue;
                        }
                        SendPartial(workspace, forward_list, forward_list_gm, tiling, outgoing_rank,
                                    chunk_ranges_gm.GetValue(range_index), chunk_ranges_gm.GetValue(range_index + 1U),
                                    copy_ub, wqe_scratch_ub, PartialClass::ReducedOnly);
                        if (chunk_index + 1U == tiling.chunks_per_group) {
                            for (uint32_t qp_id = 0; qp_id < 2U; ++qp_id) {
                                aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kCompletionSoConfig>(
                                    Flag(workspace, tiling.rank_id * 2U + qp_id),
                                    Flag(workspace, 3U * tiling.rank_num + core_id),
                                    reinterpret_cast<__ubuf__ uint64_t*>(wqe_scratch_ub), 1U, outgoing_rank, qp_id,
                                    EVENT_ID0);
                            }
                        }
                        peer_direct_sent = false;
                        chunk_sent = true;
                    }
                }
                // The peer issue envelope covers all chunks of this source
                // rank.  Source-major scheduling moved the old per-chunk
                // End mark out of the inner loop, so publish it once here.
            }

            // 关闭 credit 时不在发送路径等待接收；最终归约和 core0 尾部排空仍检查所有 SO。
            // 开启 credit 时保留原有接收完成 -> 发布下一组 credit 的握手。
            if (wait_credit && !incoming_local) {
                while (!(Ready(Flag(workspace, incoming_rank * 2U), 1U) &&
                         Ready(Flag(workspace, incoming_rank * 2U + 1U), 1U))) {
                }
            }
            if (group_index + 1U < group_num) {
                const uint32_t next_incoming_rank =
                    first_hit_schedule::Peer(tiling.rank_id, tiling.rank_num, group_index + 1U,
                                             sender_index + sequence_index * sender_num, true);
                if (wait_credit && next_incoming_rank / 8U != tiling.rank_id / 8U) {
                    StoreFlag(reinterpret_cast<__gm__ uint64_t*>(aclshmem_ptr(
                                  Flag(workspace, 2U * tiling.rank_num + tiling.rank_id), next_incoming_rank)),
                              signal_ub, group_index + 2U);
                }
            } else {
            }
        }
    }
}
__aicore__ inline void CrossSend(GM_ADDR workspace, GM_ADDR forward_list, GlobalTensor<int32_t>& forward_list_gm,
                                 GlobalTensor<int32_t>& chunk_ranges_gm, GlobalTensor<int32_t>& chunk_masks_gm,
                                 const CombineTiling& tiling, uint32_t core_id, uint32_t reduce_core_num,
                                 LocalTensor<uint8_t> copy_ub, LocalTensor<uint64_t> signal_ub,
                                 LocalTensor<int32_t> reduce_done_ub, __ubuf__ uint8_t* wqe_scratch_ub)
{
    // 独立推进每个 peer 的 chunk，保持 peer/QP 的发送所有权。
    CrossSendIndependent(workspace, forward_list, forward_list_gm, chunk_ranges_gm, chunk_masks_gm, tiling, core_id,
                         reduce_core_num, copy_ub, signal_ub, reduce_done_ub, wqe_scratch_ub);
}
__aicore__ inline void Combine(GM_ADDR workspace, GM_ADDR forward_list, GM_ADDR forward_counts, GM_ADDR backward_list,
                               GM_ADDR backward_counts, GM_ADDR token_server_mask, GM_ADDR chunk_ranges,
                               GM_ADDR chunk_masks, GM_ADDR reduce_schedule, const CombineTiling& tiling)
{
    const uint32_t core_id = GetBlockIdx(), local_rank_num = tiling.rank_num < 8U ? tiling.rank_num : 8U;
    const uint32_t server_num = (tiling.rank_num + 7U) / 8U;
    const uint32_t group_num = tiling.rank_num / first_hit_schedule::Width(tiling.rank_num);
    // Keep one owner for both QPs of each logical peer.  The EP32+ policy
    // selects eight senders/two sequences or sixteen senders/one sequence;
    // remaining AIVs are dedicated to gateway reduction.
    const uint32_t cross_sender_num = tiling.rank_num <= 8U ? 0U : 16U / tiling.cross_peers_per_core;
    const uint32_t reduce_core_begin = 8U + cross_sender_num;
    const uint32_t reduce_core_num = combine_config::kAivCores - reduce_core_begin;
    const bool is_reduce_core = core_id >= reduce_core_begin && core_id < combine_config::kAivCores;
    const bool is_cross_sender = core_id >= 8U && core_id < reduce_core_begin;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> bf16_input_buffer, fp32_input_buffer, fp32_accumulator_buffer, bf16_output_buffer,
        copy_buffer, signal_buffer;
    TBuf<TPosition::VECOUT> wqe_buffer;
    pipe.InitBuffer(bf16_input_buffer, kReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(fp32_input_buffer, kReduceTileElements * sizeof(float));
    pipe.InitBuffer(fp32_accumulator_buffer, kReduceTileElements * sizeof(float));
    pipe.InitBuffer(bf16_output_buffer, kReduceOutputSlots * kReduceTileElements * sizeof(bfloat16_t));
    pipe.InitBuffer(copy_buffer, kLocalCopyBufferBytes);
    pipe.InitBuffer(signal_buffer, kSignalUbBytes);
    pipe.InitBuffer(wqe_buffer, kWqeScratchBytes);
    auto bf16_input_ub = bf16_input_buffer.Get<bfloat16_t>();
    auto fp32_input_ub = fp32_input_buffer.Get<float>();
    auto fp32_accumulator_ub = fp32_accumulator_buffer.Get<float>();
    auto bf16_output_ub = bf16_output_buffer.Get<bfloat16_t>();
    auto copy_ub = copy_buffer.Get<uint8_t>();
    auto signal_ub = signal_buffer.Get<uint64_t>();
    auto* wqe_scratch_ub = reinterpret_cast<__ubuf__ uint8_t*>(wqe_buffer.Get<uint8_t>().GetPhyAddr());
    GlobalTensor<int32_t> forward_list_gm, forward_counts_gm, backward_counts_gm, chunk_ranges_gm, chunk_masks_gm;
    GlobalTensor<int64_t> token_server_mask_gm;
    GlobalTensor<int32_t> reduce_schedule_gm;
    if (reduce_schedule != nullptr) {
        reduce_schedule_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(reduce_schedule));
    }
    forward_list_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(forward_list));
    forward_counts_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(forward_counts));
    backward_counts_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(backward_counts));
    chunk_ranges_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(chunk_ranges));
    chunk_masks_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(chunk_masks));
    token_server_mask_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(token_server_mask));
    // Prepare 已在所有 rank 上完成控制槽重置和同步发布。它是同一 stream 上的独立 kernel，
    // 专家卡向本 server 的 gateway 汇聚贡献，之后完成对应 QP 的 quiet。
    // 固定的 server 内对端对应唯一的 QP0 控制核，不发送本 rank 自己的贡献数据。
    if (core_id < local_rank_num) {
        const uint32_t peer_rank = tiling.rank_id / 8U * 8U + core_id;
        const uint32_t task_num = backward_counts_gm.GetValue(core_id);
        auto gather_tasks_ub = copy_buffer.Get<int64_t>();
        const uint64_t peer_task_word_offset = uint64_t(core_id) * tiling.backward_row_capacity * kGatherTaskFields;
        // 此状态及 WQE 暂存区只由当前核用于 peer_rank 的 QP0；未提交时不能切换对端或 QP。
        aclshmemx_submit_state_t gather_submit_state{};
        aclshmemx_defer_t gather_defer(gather_submit_state);
        aclshmemx_submit_t gather_submit(gather_submit_state);
        uint32_t batch_start = 0;
        // The planner lays the backward rows out as
        // [remote groups][same-server groups]. Walk the same two phases here.
        for (uint32_t locality_phase = 0; locality_phase < 2U; ++locality_phase) {
            const bool local_phase = locality_phase != 0U;
            for (uint32_t group_index = 0; group_index < group_num; ++group_index) {
                while (batch_start < task_num) {
                    const uint32_t loaded_rows =
                        task_num - batch_start < kGatherBatchRows ? task_num - batch_start : kGatherBatchRows;
                    // 从紧凑 GM 表加载到对齐的 UB 起点；尾批只读有效行，不读取下一 peer 的任务。
                    NativeLoad(gather_tasks_ub,
                               reinterpret_cast<__gm__ int64_t*>(backward_list) + peer_task_word_offset +
                                   uint64_t(batch_start) * kGatherTaskFields,
                               loaded_rows * kGatherTaskFields);
                    // backward 按目的 gateway 的 (本地性,group,source,chunk) 排列，跨 server source 优先。
                    // expert 只读有贡献的行；不遍历 chunk 数，也不为缺失 chunk 发信号。
                    const uint32_t first_source = static_cast<uint32_t>(gather_tasks_ub.GetValue(0));
                    if (first_hit_schedule::Group(peer_rank, tiling.rank_num, first_source) != group_index ||
                        (first_source / 8U == peer_rank / 8U) != local_phase) {
                        break;
                    }
                    uint32_t batch_rows = 0;
                    uint32_t chunk_end_id = 0;
                    // 边发送边判断 chunk 尾标记，每行只解码一次偏移，不再预扫描。
                    for (uint32_t task_index = 0; task_index < loaded_rows; ++task_index) {
                        const uint32_t task_word_offset = task_index * kGatherTaskFields;
                        const uint64_t packed_offset =
                            static_cast<uint64_t>(gather_tasks_ub.GetValue(task_word_offset + 2U));
                        chunk_end_id = static_cast<uint32_t>(packed_offset >> 32U);
                        // low32 是纯行号；high32 仅用于 chunk 标记，不能参与 payload 地址计算。
                        const uint32_t reduce_row = static_cast<uint32_t>(packed_offset & 0xffffffffULL);
                        auto* destination_address = workspace + tiling.gather_input_offset_bytes +
                                                    uint64_t(reduce_row) * tiling.hidden_size * 2U;
                        auto* source_address =
                            workspace + tiling.expert_input_offset_bytes +
                            uint64_t(gather_tasks_ub.GetValue(task_word_offset + 1U)) * tiling.hidden_size * 2U;
                        const bool need_submit = chunk_end_id != 0U || task_index + 1U == loaded_rows;
                        if (need_submit) {
                            // chunk 尾或 UB 批次尾：当前行提交整批 RO，并请求一个 CQE。
                            aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kPayloadRoConfig>(
                                destination_address, source_address, wqe_scratch_ub, tiling.hidden_size * 2U, peer_rank,
                                0U, EVENT_ID0, gather_submit);
                        } else {
                            aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_MTE3, kGatherPayloadRoNoCqeConfig>(
                                destination_address, source_address, wqe_scratch_ub, tiling.hidden_size * 2U, peer_rank,
                                0U, EVENT_ID0, gather_defer);
                        }
                        ++batch_rows;
                        if (chunk_end_id) {
                            // 已提交当前 chunk 的全部 RO，退出后补 SO；下一 chunk 留给下一批。
                            break;
                        }
                    }
                    // submit 返回前已完成 UB WQE 到 SQ 的拷贝；payload 仍可异步传输，最终由 SO/quiet 收尾。
                    // 只推进实际发送的行数；UB 批次尾若没有 chunk 标记，不发布 SO。
                    batch_start += batch_rows;
                    if (chunk_end_id) {
                        const uint32_t source_chunk = chunk_end_id - 1U;
                        const uint32_t done_slot = 3U * tiling.rank_num + combine_control::kGatherDoneBase +
                                                   source_chunk * combine_control::kGatherDoneLanes +
                                                   tiling.rank_id % 8U;
                        // 源值固定为 1，直到最后 quiet 都不改写。每个 chunk/lane 独占目的槽。
                        aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kCompletionSoConfig>(
                            Flag(workspace, done_slot), Flag(workspace, 3U * tiling.rank_num + core_id),
                            reinterpret_cast<__ubuf__ uint64_t*>(wqe_scratch_ub), 1U, peer_rank, 0U, EVENT_ID0);
                    }
                }
            }
        }
        if (task_num) {
            // 各组之间不 quiet；最后统一排空，随后本核才可复用此 QP0 做 local return。
            aclshmemx_udma_qp_quiet(peer_rank, 0U);
        }
    }
    auto reduce_done_ub = signal_buffer.Get<int32_t>();
    if (is_reduce_core) {
        // Keep the reducer's source/group walk identical to the backward-list
        // walk: remote groups first, then same-server groups.
        for (uint32_t locality_phase = 0; locality_phase < 2U; ++locality_phase) {
            const bool local_phase = locality_phase != 0U;
            for (uint32_t group_index = 0; group_index < group_num; ++group_index) {
                for (uint32_t source_order = 0; source_order < tiling.rank_num; ++source_order) {
                    const uint32_t source_rank = SourceRankAtOrder(tiling.rank_id, tiling.rank_num, source_order);
                    if (first_hit_schedule::Group(tiling.rank_id, tiling.rank_num, source_rank) != group_index ||
                        (source_rank / 8U == tiling.rank_id / 8U) != local_phase) {
                        continue;
                    }
                    for (uint32_t chunk_index = 0; chunk_index < tiling.chunks_per_group; ++chunk_index) {
                        const uint32_t source_chunk = source_rank * tiling.chunks_per_group + chunk_index;
                        const uint32_t contributor_mask = chunk_masks_gm.GetValue(source_chunk);
                        // gateway 等本 chunk 的有效专家卡 SO 完成标志。
                        WaitGatherDone(workspace, tiling, source_rank, chunk_index, contributor_mask,
                                       copy_buffer.Get<uint64_t>());
                        // ReduceGateway 累加、BF16 写回，然后发布 reduce_done。
                        ReduceGateway(workspace, forward_list_gm, chunk_ranges_gm, tiling, source_rank, chunk_index,
                                      core_id - reduce_core_begin, reduce_core_num, bf16_input_ub, fp32_input_ub,
                                      fp32_accumulator_ub, bf16_output_ub);
                        // 每个 chunk 有独立计数，不能用累计次数推算完成（快核可能已推进下一 chunk）。
                        // 本核 BF16 partial 全部写回后再加 1；没有任务的归约核也必须发布完成。
                        SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
                        WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
                        reduce_done_ub.SetValue(0, 1);
                        SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
                        WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
                        SetAtomicAdd<int32_t>();
                        GlobalTensor<int32_t> reduce_done_gm;
                        reduce_done_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                            Flag(workspace, 3U * tiling.rank_num + combine_control::kGatherDoneBase +
                                                SourceChunkCount(tiling) * combine_control::kGatherDoneLanes +
                                                source_chunk)));
                        DataCopyPad(reduce_done_gm, reduce_done_ub, DataCopyExtParams(1, sizeof(int32_t), 0, 0, 0));
                        SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
                        WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
                        SetAtomicNone();
                    }
                }
            }
        }
    }
    if (core_id < local_rank_num) {
        // 不能在 gather 中途等待 local return，否则可能挡住后续组所需的输入。
        // 此时本核已提交并排空全部 gather，只需等目的源卡所属组的归约完成。
        const uint32_t peer_rank = tiling.rank_id / 8U * 8U + core_id;
        const bool peer_needs_reduce =
            SourceNeedsGatewayReduce(forward_list_gm, chunk_ranges_gm, chunk_masks_gm, tiling, peer_rank);
        if (peer_needs_reduce) {
            // LocalReturnReduceWait：本 server 回传核等待其目的源卡所属组的最后一个
            // chunk 归约完成。没有任何归约任务时，直接进入回传，避免空轮询占用 AIV。
            // 归约核按 chunk 顺序发布，最后一个 chunk 全核完成也保证之前各 chunk 完成。
            WaitReduceDone(workspace, tiling, peer_rank, tiling.chunks_per_group - 1U, reduce_core_num, reduce_done_ub);
        }
    }
    if (core_id < local_rank_num) {
        // LocalReturnBegin/End：gateway 回传同 server 的源卡；回传自己时为本地拷贝。
        // Remote local-return QP0 is drained after source reduction below.
        const uint32_t peer_rank = tiling.rank_id / 8U * 8U + core_id;
        const uint32_t task_count = forward_counts_gm.GetValue(peer_rank);
        // Self-return is split at a complete token boundary; remote local
        // returns keep their original full-range submission.
        const uint32_t task_end =
            peer_rank == tiling.rank_id ? SelfCopySplitTask(forward_list_gm, tiling, task_count) : task_count;
        SendPartial(workspace, forward_list, forward_list_gm, tiling, peer_rank, 0U, task_end, copy_ub, wqe_scratch_ub);
        if (peer_rank != tiling.rank_id) {
            // The completion SO below is the receiver-side ordering boundary.
            // Keep the QP outstanding until after source reduction, alongside
            // the cross-server QPs.
            aclshmemx_udma_qp_put_nbi<uint64_t, PIPE_MTE3, kCompletionSoConfig>(
                Flag(workspace, tiling.rank_id * 2U), Flag(workspace, 3U * tiling.rank_num + core_id),
                reinterpret_cast<__ubuf__ uint64_t*>(wqe_scratch_ub), 1U, peer_rank, 0U, EVENT_ID0);
        }
    } else if (is_cross_sender) {
        // CrossReturnTotalBegin/End：跨 server 回传从 CrossSend 开始，结束于 source reduce 后的最终 quiet。
        CrossSend(workspace, forward_list, forward_list_gm, chunk_ranges_gm, chunk_masks_gm, tiling, core_id,
                  reduce_core_num, copy_ub, signal_ub, reduce_done_ub, wqe_scratch_ub);
    }
    if (is_reduce_core) {
        // Keep half of self-return on its original local core.  The remaining
        // token-group range is disjointly partitioned across reducers after
        // their gateway work, so no per-token metadata or extra barrier is needed.
        const uint32_t self_task_count = forward_counts_gm.GetValue(tiling.rank_id);
        const uint32_t self_copy_split = SelfCopySplitTask(forward_list_gm, tiling, self_task_count);
        const uint32_t helper_index = core_id - reduce_core_begin;
        const uint32_t helper_count = reduce_core_num;
        const uint32_t remaining_tasks = self_task_count - self_copy_split;
        const uint32_t raw_begin = self_copy_split + uint64_t(remaining_tasks) * helper_index / helper_count;
        const uint32_t raw_end = self_copy_split + uint64_t(remaining_tasks) * (helper_index + 1U) / helper_count;
        const uint64_t self_task_base = uint64_t(tiling.rank_id) * tiling.forward_row_capacity;
        const uint32_t helper_begin =
            AlignTaskToTokenBoundary(forward_list_gm, self_task_base, self_task_count, raw_begin);
        const uint32_t helper_end = AlignTaskToTokenBoundary(forward_list_gm, self_task_base, self_task_count, raw_end);
        if (helper_begin < helper_end) {
            const bool self_needs_reduce =
                SourceNeedsGatewayReduce(forward_list_gm, chunk_ranges_gm, chunk_masks_gm, tiling, tiling.rank_id);
            if (self_needs_reduce) {
                WaitReduceDone(workspace, tiling, tiling.rank_id, tiling.chunks_per_group - 1U, reduce_core_num,
                               reduce_done_ub);
            }
            SendPartial(workspace, forward_list, forward_list_gm, tiling, tiling.rank_id, helper_begin, helper_end,
                        copy_ub, wqe_scratch_ub);
        }
    }
    // self copy 的每个参与核独占一个完成槽；空分片也发布，再进入任何接收等待。
    if (is_reduce_core || core_id == tiling.rank_id % 8U) {
        SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
        StoreFlag(Flag(workspace, 3U * tiling.rank_num + combine_control::kSelfReturnBase + core_id), signal_ub, 1U);
    }
    // SourceReduceBegin/End：源卡最终累加并写回；结束点在最后一笔输出写回完成之后。
    // gateway 阶段已排空所有输出事件，此处从两个空闲槽开始复用同一块 UB。
    // 本核前序工作已结束，无需其他核同步即可复用私有 UB。
    // 每个 bank 装两行 BF16，复用 FP32 临时区和 local-copy 区，各 32 KiB。
    auto source_bank0 = fp32_input_buffer.Get<bfloat16_t>();
    auto source_bank1 = copy_buffer.Get<bfloat16_t>();
    const uint32_t final_stage_num = tiling.rank_num / tiling.final_reduce_peers_per_stage;
    uint32_t next_ready_group = 0;
    // 批内按核静态分片，快核不用等待其他核进入最终归约。
    for (uint32_t group = 0; group < final_stage_num; ++group) {
        // 空 schedule：前面的批次为空，最后一批覆盖所有 token。
        // 沿用下方补查所有前序阶段的逻辑，确保全部回传到齐后才读取 partial。
        const uint32_t begin = reduce_schedule != nullptr ? reduce_schedule_gm.GetValue(group) : 0U;
        const uint32_t end = reduce_schedule != nullptr ? reduce_schedule_gm.GetValue(group + 1U)
                                                        : (group + 1U == final_stage_num ? tiling.token_num : 0U);
        const uint32_t local_tokens =
            begin + core_id < end ? (end - begin - core_id + combine_config::kAivCores - 1U) / combine_config::kAivCores
                                  : 0U;
        if (local_tokens) {
            // 跳过空批次后，仍要补查前面尚未确认的 group。
            while (next_ready_group <= group) {
                WaitReturnedGroup(workspace, tiling, next_ready_group, reduce_core_begin);
                ++next_ready_group;
            }
        }
        bool source_input_read_pending[2]{};
        uint32_t source_output_slot = 0;
        bool source_output_store_pending[kReduceOutputSlots]{};
        for (uint32_t index = begin + core_id; index < end; index += combine_config::kAivCores) {
            const uint32_t token_id =
                reduce_schedule != nullptr ? reduce_schedule_gm.GetValue(final_stage_num + 1U + index) : index;
            const uint64_t token_server_bits = token_server_mask_gm.GetValue(token_id);
            uint32_t source_servers[16];
            uint32_t source_count = 0;
            for (uint32_t server_id = 0; server_id < server_num; ++server_id) {
                if (token_server_bits & (1ULL << server_id)) {
                    source_servers[source_count++] = server_id;
                }
            }
            for (uint32_t hidden_offset = 0; hidden_offset < tiling.hidden_size; hidden_offset += kReduceTileElements) {
                const uint32_t element_num = tiling.hidden_size - hidden_offset < kReduceTileElements
                                                 ? tiling.hidden_size - hidden_offset
                                                 : kReduceTileElements;
                const auto output_event_id = source_output_slot == 0U ? EVENT_ID1 : EVENT_ID2;
                auto bf16_output_slot_ub = bf16_output_ub[source_output_slot * kReduceTileElements];
                // 空 mask 也必须写零，不能沿用上一 token 的累加器。
                if (!source_count) {
                    if (source_output_store_pending[source_output_slot]) {
                        WaitFlag<HardEvent::MTE3_V>(output_event_id);
                        source_output_store_pending[source_output_slot] = false;
                    }
                    Duplicate(bf16_output_slot_ub, static_cast<bfloat16_t>(0.0F), element_num);
                    PipeBarrier<PIPE_V>();
                } else if (server_num <= 4U && source_count >= 3U) {
                    // EP32：两个已有的 32 KiB bank 各容纳两行 BF16 partial。
                    // 三/四路在一次 VF 内顺序累加，避免中间 FP32 结果写回再读 UB。
                    for (uint32_t bank = 0; bank < 2U; ++bank) {
                        const auto input_event_id = bank == 0U ? EVENT_ID0 : EVENT_ID4;
                        auto input_bank = bank == 0U ? source_bank0 : source_bank1;
                        if (source_input_read_pending[bank]) {
                            WaitFlag<HardEvent::V_MTE2>(input_event_id);
                            source_input_read_pending[bank] = false;
                        }
                        for (uint32_t input = 0; input < 2U && bank * 2U + input < source_count; ++input) {
                            const uint32_t server_id = source_servers[bank * 2U + input];
                            GlobalTensor<bfloat16_t> partial_gm;
                            partial_gm.SetGlobalBuffer(
                                reinterpret_cast<__gm__ bfloat16_t*>(workspace + tiling.returned_partial_offset_bytes) +
                                (uint64_t(token_id) * server_num + server_id) * tiling.hidden_size + hidden_offset);
                            DataCopy(input_bank[input * kReduceTileElements], partial_gm, element_num);
                        }
                        SetFlag<HardEvent::MTE2_V>(input_event_id);
                    }
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
                    if (source_output_store_pending[source_output_slot]) {
                        WaitFlag<HardEvent::MTE3_V>(output_event_id);
                        source_output_store_pending[source_output_slot] = false;
                    }
                    auto* first = reinterpret_cast<__ubuf__ bfloat16_t*>(source_bank0.GetPhyAddr());
                    auto* third = reinterpret_cast<__ubuf__ bfloat16_t*>(source_bank1.GetPhyAddr());
                    auto* output = reinterpret_cast<__ubuf__ bfloat16_t*>(bf16_output_slot_ub.GetPhyAddr());
                    if (source_count == 4U) {
                        asc_vf_call<SourceReduceFourVf<true>>(first, first + kReduceTileElements, third,
                                                              third + kReduceTileElements, output, element_num);
                    } else {
                        asc_vf_call<SourceReduceFourVf<false>>(first, first + kReduceTileElements, third,
                                                               third + kReduceTileElements, output, element_num);
                    }
                    for (uint32_t bank = 0; bank < 2U; ++bank) {
                        SetFlag<HardEvent::V_MTE2>(bank == 0U ? EVENT_ID0 : EVENT_ID4);
                        source_input_read_pending[bank] = true;
                    }
                } else {
                    const uint32_t pair_count = (source_count + 1U) / 2U;
                    for (uint32_t pair = 0; pair < pair_count; ++pair) {
                        const uint32_t bank_index = pair % 2U;
                        const auto input_event_id = bank_index == 0U ? EVENT_ID0 : EVENT_ID4;
                        // 首轮搬当前和下一 bank；之后当前输入已经由上轮预取。
                        const uint32_t prefetch_begin = pair == 0U ? 0U : pair + 1U;
                        const uint32_t prefetch_end = pair + 2U < pair_count ? pair + 2U : pair_count;
                        for (uint32_t prefetch = prefetch_begin; prefetch < prefetch_end; ++prefetch) {
                            const uint32_t next_bank_index = prefetch % 2U;
                            const auto next_event_id = next_bank_index == 0U ? EVENT_ID0 : EVENT_ID4;
                            auto next_bank = next_bank_index == 0U ? source_bank0 : source_bank1;
                            if (source_input_read_pending[next_bank_index]) {
                                WaitFlag<HardEvent::V_MTE2>(next_event_id);
                                source_input_read_pending[next_bank_index] = false;
                            }
                            for (uint32_t input = 0; input < 2U && prefetch * 2U + input < source_count; ++input) {
                                const uint32_t server_id = source_servers[prefetch * 2U + input];
                                GlobalTensor<bfloat16_t> partial_gm;
                                partial_gm.SetGlobalBuffer(
                                    reinterpret_cast<__gm__ bfloat16_t*>(workspace +
                                                                         tiling.returned_partial_offset_bytes) +
                                    (uint64_t(token_id) * server_num + server_id) * tiling.hidden_size + hidden_offset);
                                DataCopy(next_bank[input * kReduceTileElements], partial_gm, element_num);
                            }
                            SetFlag<HardEvent::MTE2_V>(next_event_id);
                        }
                        WaitFlag<HardEvent::MTE2_V>(input_event_id);
                        const bool final_pair = pair + 1U == pair_count;
                        // 仅最后一组 VF 写输出；其他组仍能与上一 token 的 MTE3 重叠。
                        if (final_pair && source_output_store_pending[source_output_slot]) {
                            WaitFlag<HardEvent::MTE3_V>(output_event_id);
                            source_output_store_pending[source_output_slot] = false;
                        }
                        auto input_bank = bank_index == 0U ? source_bank0 : source_bank1;
                        auto* first_input = reinterpret_cast<__ubuf__ bfloat16_t*>(input_bank.GetPhyAddr());
                        auto* second_input = first_input + kReduceTileElements;
                        auto* accumulator = reinterpret_cast<__ubuf__ float*>(fp32_accumulator_ub.GetPhyAddr());
                        auto* output = reinterpret_cast<__ubuf__ bfloat16_t*>(bf16_output_slot_ub.GetPhyAddr());
                        const bool has_second = pair * 2U + 1U < source_count;
                        if (pair == 0U) {
                            if (!has_second) {
                                asc_vf_call<SourceReducePairVf<false, false, true>>(first_input, second_input,
                                                                                    accumulator, output, element_num);
                            } else if (final_pair) {
                                asc_vf_call<SourceReducePairVf<false, true, true>>(first_input, second_input,
                                                                                   accumulator, output, element_num);
                            } else {
                                asc_vf_call<SourceReducePairVf<false, true, false>>(first_input, second_input,
                                                                                    accumulator, output, element_num);
                            }
                        } else if (!has_second) {
                            asc_vf_call<SourceReducePairVf<true, false, true>>(first_input, second_input, accumulator,
                                                                               output, element_num);
                        } else if (final_pair) {
                            asc_vf_call<SourceReducePairVf<true, true, true>>(first_input, second_input, accumulator,
                                                                              output, element_num);
                        } else {
                            asc_vf_call<SourceReducePairVf<true, true, false>>(first_input, second_input, accumulator,
                                                                               output, element_num);
                        }
                        // 下一 VF 读取本次 accumulator；bank 释放不能代替这个 V 依赖。
                        PipeBarrier<PIPE_V>();
                        SetFlag<HardEvent::V_MTE2>(input_event_id);
                        source_input_read_pending[bank_index] = true;
                    }
                }
                SetFlag<HardEvent::V_MTE3>(output_event_id);
                WaitFlag<HardEvent::V_MTE3>(output_event_id);
                GlobalTensor<bfloat16_t> output_gm;
                output_gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspace + tiling.output_offset_bytes) +
                                          uint64_t(token_id) * tiling.hidden_size + hidden_offset);
                DataCopy(output_gm, bf16_output_slot_ub, element_num);
                SetFlag<HardEvent::MTE3_V>(output_event_id);
                source_output_store_pending[source_output_slot] = true;
                source_output_slot ^= 1U;
            }
        }
        // 批次结束只排空实际使用的 bank，空批次不等待未产生的事件。
        for (uint32_t bank = 0; bank < 2U; ++bank) {
            if (source_input_read_pending[bank]) {
                WaitFlag<HardEvent::V_MTE2>(bank == 0U ? EVENT_ID0 : EVENT_ID4);
            }
        }
        // 返回前排空两个输出槽；没有 token 的核不等待。
        for (uint32_t slot = 0; slot < kReduceOutputSlots; ++slot) {
            if (source_output_store_pending[slot]) {
                const auto output_event_id = slot == 0U ? EVENT_ID1 : EVENT_ID2;
                WaitFlag<HardEvent::MTE3_V>(output_event_id);
            }
        }
    }
    // 空批次也可能有迟到的 SO。只让 core0 在计算结束后排空剩余 group，
    // 保证下一次 Prepare 不被旧信号污染；不阻塞其他核的提前归约，不做 SyncAll。
    if (core_id == 0U) {
        while (next_ready_group < final_stage_num) {
            WaitReturnedGroup(workspace, tiling, next_ready_group, reduce_core_begin);
            ++next_ready_group;
        }
    }
    if (core_id < local_rank_num && tiling.rank_id / 8U * 8U + core_id != tiling.rank_id) {
        const uint32_t peer_rank = tiling.rank_id / 8U * 8U + core_id;
        aclshmemx_udma_qp_quiet(peer_rank, 0U);
    }
    if (is_cross_sender) {
        // Incoming SO is the data-ordering boundary for the remote partials.  Keep
        // the local QP drain after source reduction while still draining before
        // this kernel returns.

        QuietCrossServerQps(tiling, core_id);
    }
}
#endif
}  // namespace

// Session-level entry/exit barriers are separate from the arithmetic kernel.
// All 64 AIVs participate, including empty experts and zero-token workers.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void combine_prepare_kernel(
    GM_ADDR workspace, ascend_deepep::PutCombineTiling tiling)
{
    util_set_ffts_config(0U);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
    Prepare(workspace, tiling.rank_num, tiling.rank_num * tiling.chunks_per_group, tiling.control_slot_num);
#endif
}
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void combine_kernel(
    GM_ADDR workspace, GM_ADDR forward, GM_ADDR counts, GM_ADDR backward, GM_ADDR backward_counts, GM_ADDR mask,
    GM_ADDR ranges, GM_ADDR masks, ascend_deepep::PutCombineTiling tiling)
{
    util_set_ffts_config(0U);
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
    Combine(workspace, forward, counts, backward, backward_counts, mask, ranges, masks, nullptr, tiling);
#endif
}
extern "C" ASCEND_DEEPEP_EXPORT void combine_prepare_kernel_do(void* stream, uint8_t* workspace, uint8_t* tiling)
{
    combine_prepare_kernel<<<64, 192U * 1024U, stream>>>(workspace,
                                                         *reinterpret_cast<ascend_deepep::PutCombineTiling*>(tiling));
}
extern "C" ASCEND_DEEPEP_EXPORT void combine_kernel_do(void* stream, uint8_t* workspace, uint8_t* forward,
                                                       uint8_t* counts, uint8_t* backward, uint8_t* backward_counts,
                                                       uint8_t* mask, uint8_t* ranges, uint8_t* masks, uint8_t* tiling)
{
    combine_kernel<<<64, 192U * 1024U, stream>>>(workspace, forward, counts, backward, backward_counts, mask, ranges,
                                                 masks, *reinterpret_cast<ascend_deepep::PutCombineTiling*>(tiling));
}
