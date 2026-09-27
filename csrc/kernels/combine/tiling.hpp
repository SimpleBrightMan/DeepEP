#pragma once
#include <cstdint>

namespace ascend_deepep {
namespace combine_config {
constexpr uint32_t kAivCores = 64;
}  // namespace combine_config
namespace combine_control {
constexpr uint32_t kSlotBytes = 512;
// 每个 chunk 独立的汇聚标志和归约计数；控制区从 3*world+96 之后开始。
constexpr uint32_t kMaxGroups = 8;
constexpr uint32_t kSelfReturnBase = 32;
constexpr uint32_t kGatherDoneBase = 96;
static_assert(kSelfReturnBase + combine_config::kAivCores == kGatherDoneBase);
constexpr uint32_t kGatherDoneLanes = 8;
}  // namespace combine_control
struct PutCombineTiling {
    uint32_t rank_id, rank_num, token_num, hidden_size, expert_row_capacity, gather_row_capacity;
    uint32_t forward_row_capacity, backward_row_capacity;
    uint32_t chunk_token_num, chunks_per_group, total_chunk_num, control_slot_num;
    uint64_t expert_input_offset_bytes, gather_input_offset_bytes, server_partial_offset_bytes,
        returned_partial_offset_bytes, output_offset_bytes;
    uint32_t wait_credit;
    // EP32+ cross-server sender policy: one or two group logical indices per AIV.
    // Host and device must be rebuilt together when the tiling ABI changes.
    uint32_t cross_peers_per_core;
    // 最终归约子阶段宽度；不改变通信 group、控制槽和发送核分工。
    uint32_t final_reduce_peers_per_stage;
};

// Caller validates public bounds before narrowing. All region arithmetic is uint64.
inline PutCombineTiling MakeCombineTiling(uint32_t rank, uint32_t world, uint32_t tokens, uint32_t topk, uint32_t rows,
                                          uint32_t gather_rows, uint32_t chunk_tokens = 256U)
{
    PutCombineTiling t{};
    t.rank_id = rank;
    t.rank_num = world;
    t.token_num = tokens;
    t.hidden_size = 7168U;
    t.expert_row_capacity = rows;
    t.gather_row_capacity = gather_rows;
    t.forward_row_capacity = tokens * topk;
    t.backward_row_capacity = world * t.forward_row_capacity;
    t.chunk_token_num = chunk_tokens;
    t.chunks_per_group = (tokens + chunk_tokens - 1U) / chunk_tokens;
    t.total_chunk_num = world / (world < 16U ? world : 16U) * t.chunks_per_group;
    t.control_slot_num = 3U * world + combine_control::kGatherDoneBase +
                         world * t.chunks_per_group * (combine_control::kGatherDoneLanes + 1U);
    t.wait_credit = 1U;
    t.cross_peers_per_core = world >= 32U ? 1U : 2U;
    t.final_reduce_peers_per_stage = world < 16U ? world : 16U;
    uint64_t cursor = uint64_t(t.control_slot_num) * combine_control::kSlotBytes;
    auto reserve = [&](uint64_t bytes) {
        const uint64_t offset = cursor;
        cursor += (bytes + 511U) / 512U * 512U;
        return offset;
    };
    t.expert_input_offset_bytes = reserve(uint64_t(rows) * 7168U * 2U);
    t.gather_input_offset_bytes = reserve(uint64_t(gather_rows) * 7168U * 2U);
    t.server_partial_offset_bytes = reserve(uint64_t(rows) * 7168U * 2U);
    t.returned_partial_offset_bytes = reserve(uint64_t(tokens) * ((world + 7U) / 8U) * 7168U * 2U);
    t.output_offset_bytes = reserve(uint64_t(tokens) * 7168U * 2U);
    return t;
}

inline uint64_t CombineWorkspaceBytes(const PutCombineTiling& t)
{
    return t.output_offset_bytes + uint64_t(t.token_num) * t.hidden_size * 2U;
}
}  // namespace ascend_deepep
