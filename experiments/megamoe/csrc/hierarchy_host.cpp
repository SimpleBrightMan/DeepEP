// Copyright (c) 2026, SimpleBright_Man
#include "hierarchy_host.hpp"
#include "hierarchy_plan.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>
#include <acl/acl.h>
#include "check.hpp"
#include "shmem.h"
#include "kernels/mega_moe/common/mega_moe_constants.hpp"
#include "kernels/mega_moe/common/mega_moe_workspace.hpp"

namespace deepep::megamoe {
namespace {
using Clock = std::chrono::steady_clock;
double Milliseconds(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
uint64_t Align(uint64_t n)
{
    return (n + 511U) / 512U * 512U;
}
struct Layout {
    uint64_t weights = 0, proxy, partial, group, bytes;
    Layout(uint64_t batch, uint64_t topk, uint64_t hidden)
    {
        const uint64_t row = (hidden + 255U) / 256U * 256U + ((hidden + 31U) / 32U + 31U) / 32U * 32U;
        proxy = Align(batch * topk * sizeof(float));
        partial = proxy + Align(4U * batch * row);
        group = partial + Align(16U * batch * hidden * sizeof(float));
        bytes = group + Align(4U * batch * hidden * sizeof(float));
    }
};
template <typename T>
torch::Tensor Upload(const std::vector<T>& values, const torch::TensorOptions& options)
{
    const size_t bytes = values.size() * sizeof(T);
    auto result = torch::empty({static_cast<int64_t>(std::max<size_t>(bytes, 32U))}, options.dtype(torch::kUInt8));
    if (bytes)
        ASCEND_DEEPEP_CHECK_ACL(aclrtMemcpy(result.data_ptr(), bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    return result;
}
void Copies(std::vector<HierarchyCopyTask>& tasks, uint64_t source, uint64_t destination, uint64_t bytes)
{
    while (bytes) {
        uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(bytes, 16384));
        tasks.push_back({source, destination, chunk, 0});
        source += chunk;
        destination += chunk;
        bytes -= chunk;
    }
}
}  // namespace

int64_t hierarchy_buffer_size(int64_t world, int64_t experts, int64_t tokens, int64_t topk, int64_t hidden)
{
    TORCH_CHECK(world == 16 && tokens > 256 && tokens <= 8192 && tokens % 128 == 0 && hidden % 256 == 0,
                "hierarchy requires EP16, BS/max_tokens multiple of 128 in [384,8192], H multiple of 256");
    const auto original = buffer_size(world, experts, tokens, topk, hidden);
    const int64_t total = original + static_cast<int64_t>(Align(Layout(tokens, topk, hidden).bytes));
    const int64_t aligned = (total + (2LL << 20) - 1) / (2LL << 20) * (2LL << 20);
    TORCH_CHECK(aligned <= (8LL << 30), "hierarchy symmetric buffer exceeds the supported 8 GiB limit");
    return aligned;
}

void Plan::configure_hierarchy(torch::Tensor global_topk_ids)
{
    auto start = Clock::now();
    TORCH_CHECK(tiling_.groupedMatmulMode == MegaMoeImpl::GROUPED_MATMUL_MODE_A8W4 && tiling_.sharedExpertNum == 0 &&
                    tiling_.epWorldSize == 16 && tiling_.bs > 256 && tiling_.bs % 128 == 0 && tiling_.h % 256 == 0,
                "hierarchy v1 requires W4A8, EP16, no shared experts, BS>256 multiple of 128, H%256=0");
    TORCH_CHECK(global_topk_ids.device().is_cpu() && global_topk_ids.scalar_type() == torch::kInt32 &&
                    global_topk_ids.is_contiguous() &&
                    global_topk_ids.sizes() == torch::IntArrayRef({16, tiling_.bs, tiling_.topK}),
                "global_topk_ids must be CPU contiguous int32 [16, BS, K]");
    const auto required =
        hierarchy_buffer_size(16, tiling_.moeExpertPerRank, tiling_.numMaxTokensPerRank, tiling_.topK, tiling_.h);
    TORCH_CHECK(symmetric_bytes_ >= required,
                "hierarchy needs SymmBuffer(hierarchical=True), symmetric bytes required=", required);
    auto plan =
        hierarchy::Build(global_topk_ids.data_ptr<int32_t>(), tiling_.bs, tiling_.topK, tiling_.moeExpertPerRank);
    const uint32_t rank = static_cast<uint32_t>(aclshmem_my_pe());
    const uint32_t own = rank / 4, batch = tiling_.bs, topk = tiling_.topK, hidden = tiling_.h;
    const uint64_t packed = (hidden + 255U) / 256U * 256U + ((hidden + 31U) / 32U + 31U) / 32U * 32U;
    const uint64_t bf16 = hidden * sizeof(uint16_t), fp32 = hidden * sizeof(float);
    const Layout layout(tiling_.numMaxTokensPerRank, topk, hidden);
    const uint64_t extra = buffer_size(16, tiling_.moeExpertPerRank, tiling_.numMaxTokensPerRank, topk, hidden);
    uint64_t peers[16]{};
    for (uint32_t peer = 0; peer < 16; ++peer) {
        peers[peer] = reinterpret_cast<uint64_t>(aclshmem_ptr(reinterpret_cast<void*>(symmetric_address_), peer));
        TORCH_CHECK(peers[peer] != 0, "cannot resolve hierarchy peer ", peer);
    }
    uint64_t quant_offset = MegaMoeImpl::EXCEPTION_DUMP_REGION_SIZE + MegaMoeImpl::PEERMEM_DATA_OFFSET +
                            MegaMoeImpl::CalcRouteIndexRecvSize(MegaMoeImpl::CalcDispatchRouteIndexAlignSize(&tiling_),
                                                                tiling_.moeExpertPerRank, 16) +
                            MegaMoeImpl::CalcExpertCountRecvSize(tiling_.moeExpertPerRank, 16);
    auto state = std::make_shared<HierarchyState>();
    const auto options = workspace_.options();
    const auto& mine = plan.ranks[rank];
    const uint64_t token_count = mine.token_keys.size(), edge_count = mine.edge_keys.size();
    state->cache = torch::empty({static_cast<int64_t>(std::max<uint64_t>(1, token_count) * packed)}, options);
    state->edge_outputs = torch::empty({static_cast<int64_t>(std::max<uint64_t>(1, edge_count) * bf16)}, options);
    state->weights = torch::empty({static_cast<int64_t>(16ULL * batch * topk * sizeof(float))}, options);
    state->token_slots = Upload(mine.token_slots, options);
    std::vector<int32_t> edge_slots(16U * batch * topk, -1);
    for (uint32_t slot = 0; slot < edge_count; ++slot) edge_slots[mine.edge_keys[slot]] = slot;
    state->edge_slots = Upload(edge_slots, options);
    state->expected_ids = global_topk_ids[rank].clone();
    state->config = {reinterpret_cast<uint64_t>(state->cache.data_ptr()),
                     reinterpret_cast<uint64_t>(state->token_slots.data_ptr()),
                     reinterpret_cast<uint64_t>(state->edge_slots.data_ptr()),
                     reinterpret_cast<uint64_t>(state->edge_outputs.data_ptr()),
                     2,
                     batch};
    state->weight_source = peers[rank] + extra + layout.weights;
    std::vector<HierarchyCopyTask> cross, intra, metadata;
    uint64_t local_bytes = 0, intra_bytes = 0;
    for (uint32_t offset = 1; offset <= 3; ++offset) {
        uint32_t source = ((own + 4 - offset) % 4) * 4 + rank % 4;
        for (uint32_t token = 0; token < batch; ++token) {
            uint32_t key = source * batch + token;
            if (((plan.consumers[key] >> (own * 4)) & 15U) == 0) continue;
            Copies(cross, peers[source] + quant_offset + token * packed,
                   peers[rank] + extra + layout.proxy + hierarchy::ProxySlot(key, batch) * packed, packed);
        }
    }
    for (uint32_t slot = 0; slot < token_count; ++slot) {
        uint32_t key = mine.token_keys[slot], source = key / batch;
        uint32_t upstream = source / 4 == own ? source : hierarchy::Proxy(own, source);
        uint64_t address = source / 4 == own
                               ? peers[source] + quant_offset + (key % batch) * packed
                               : peers[upstream] + extra + layout.proxy + hierarchy::ProxySlot(key, batch) * packed;
        Copies(intra, address, state->config.local_cache + slot * packed, packed);
        (upstream == rank ? local_bytes : intra_bytes) += packed;
    }
    const uint64_t weight_bytes = static_cast<uint64_t>(batch) * topk * sizeof(float);
    for (uint32_t source = 0; source < 16; ++source)
        Copies(metadata, peers[source] + extra + layout.weights,
               reinterpret_cast<uint64_t>(state->weights.data_ptr()) + source * weight_bytes, weight_bytes);

    std::vector<HierarchyReduceRow> local_rows, group_rows, source_rows;
    std::vector<HierarchyReduceInput> local_inputs, group_inputs, source_inputs;
    uint32_t edge_cursor = 0;
    for (uint32_t slot = 0; slot < token_count; ++slot) {
        const uint32_t key = mine.token_keys[slot], begin = static_cast<uint32_t>(local_inputs.size());
        while (edge_cursor < edge_count && mine.edge_keys[edge_cursor] / topk == key) {
            uint32_t edge = mine.edge_keys[edge_cursor];
            local_inputs.push_back({state->config.edge_outputs + edge_cursor * bf16,
                                    reinterpret_cast<uint64_t>(state->weights.data_ptr()) + edge * sizeof(float)});
            ++edge_cursor;
        }
        local_rows.push_back({peers[rank] + extra + layout.partial + slot * fp32, begin,
                              static_cast<uint32_t>(local_inputs.size()) - begin});
    }
    uint64_t combine_intra = 0, combine_cross = 0;
    for (uint32_t source = rank % 4; source < 16; source += 4) {
        for (uint32_t token = 0; token < batch; ++token) {
            uint32_t key = source * batch + token, begin = static_cast<uint32_t>(group_inputs.size());
            for (uint32_t consumer = own * 4; consumer < own * 4 + 4; ++consumer) {
                int32_t slot = plan.ranks[consumer].token_slots[key];
                if (slot < 0) continue;
                group_inputs.push_back({peers[consumer] + extra + layout.partial + slot * fp32, 0});
                if (consumer != rank) combine_intra += fp32;
            }
            if (group_inputs.size() != begin)
                group_rows.push_back({peers[rank] + extra + layout.group + hierarchy::ProxySlot(key, batch) * fp32,
                                      begin, static_cast<uint32_t>(group_inputs.size()) - begin});
        }
    }
    for (uint32_t token = 0; token < batch; ++token) {
        uint32_t key = rank * batch + token, begin = static_cast<uint32_t>(source_inputs.size());
        for (uint32_t group = 0; group < 4; ++group) {
            if (((plan.consumers[key] >> (group * 4)) & 15U) == 0) continue;
            uint32_t proxy = hierarchy::Proxy(group, rank);
            source_inputs.push_back({peers[proxy] + extra + layout.group + hierarchy::ProxySlot(key, batch) * fp32, 0});
            if (group != own) combine_cross += fp32;
        }
        source_rows.push_back({token * bf16, begin, static_cast<uint32_t>(source_inputs.size()) - begin});
    }
    state->cross_count = cross.size();
    state->intra_count = intra.size();
    state->metadata_count = metadata.size();
    state->local_count = local_rows.size();
    state->group_count = group_rows.size();
    state->source_count = source_rows.size();
    state->copy_cross = Upload(cross, options);
    state->copy_intra = Upload(intra, options);
    state->copy_metadata = Upload(metadata, options);
    state->local_rows = Upload(local_rows, options);
    state->local_inputs = Upload(local_inputs, options);
    state->group_rows = Upload(group_rows, options);
    state->group_inputs = Upload(group_inputs, options);
    state->source_rows = Upload(source_rows, options);
    state->source_inputs = Upload(source_inputs, options);
    state->metrics = {{"plan_ms", Milliseconds(start)},
                      {"extra_symmetric_bytes", static_cast<double>(layout.bytes)},
                      {"local_cache_bytes", static_cast<double>(token_count * packed)},
                      {"edge_output_bytes", static_cast<double>(edge_count * bf16)},
                      {"dispatch_cross_bytes", static_cast<double>(cross.size() * packed)},
                      {"dispatch_intra_bytes", static_cast<double>(intra_bytes)},
                      {"dispatch_local_bytes", static_cast<double>(local_bytes)},
                      {"combine_intra_bytes", static_cast<double>(combine_intra)},
                      {"combine_cross_bytes", static_cast<double>(combine_cross)},
                      {"weight_metadata_remote_bytes", static_cast<double>(15 * weight_bytes)}};
    hierarchy_ = std::move(state);
}

std::map<std::string, double> Plan::hierarchy_metrics() const
{
    return hierarchy_ ? hierarchy_->metrics : std::map<std::string, double>{};
}

void ExecuteHierarchy(HierarchyState& state, torch::Tensor ids, torch::Tensor weights, torch::Tensor output,
                      const PlatformInfo& platform, void* stream, uint64_t ffts,
                      const std::function<void(MegaMoeHierarchyConfig)>& launch)
{
    const auto total_start = Clock::now();
    if (state.route_identity == nullptr) {
        TORCH_CHECK(torch::equal(ids.cpu(), state.expected_ids), "hierarchy routes differ from configured plan");
        state.route_owner = ids;
        state.route_identity = ids.data_ptr();
        state.route_version = ids._version();
    }
    TORCH_CHECK(state.route_identity == ids.data_ptr() && state.route_version == ids._version(),
                "hierarchy route tensor changed; collectively configure_hierarchy again");
    auto phase = [&](const char* name, const std::function<void()>& work, bool barrier) {
        auto start = Clock::now();
        work();
        ASCEND_DEEPEP_CHECK_ACL(aclrtSynchronizeStream(stream));
        if (barrier) aclshmem_barrier_all();
        state.metrics[name] = Milliseconds(start);
    };
    auto copy = [&](torch::Tensor& tasks, uint32_t count) {
        mega_moe_hierarchy_copy_do(platform.block_dim, platform.ub, stream, static_cast<uint8_t*>(tasks.data_ptr()),
                                   count, ffts);
    };
    auto reduce = [&](torch::Tensor& rows, torch::Tensor& inputs, uint32_t count, uint32_t mode) {
        mega_moe_hierarchy_reduce_do(platform.block_dim, platform.ub, stream, static_cast<uint8_t*>(rows.data_ptr()),
                                     static_cast<uint8_t*>(inputs.data_ptr()), count, output.size(1), mode,
                                     static_cast<uint8_t*>(output.data_ptr()), ffts);
    };
    phase(
        "last_prepare_ms",
        [&] {
            const auto bytes = weights.numel() * weights.element_size();
            ASCEND_DEEPEP_CHECK_ACL(aclrtMemcpyAsync(reinterpret_cast<void*>(state.weight_source), bytes,
                                                     weights.data_ptr(), bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
            auto config = state.config;
            config.mode = 1;
            launch(config);
        },
        true);
    phase("last_metadata_ms", [&] { copy(state.copy_metadata, state.metadata_count); }, false);
    phase("last_dispatch_cross_ms", [&] { copy(state.copy_cross, state.cross_count); }, true);
    phase("last_dispatch_intra_ms", [&] { copy(state.copy_intra, state.intra_count); }, false);
    phase("last_compute_ms", [&] { launch(state.config); }, true);
    phase("last_local_reduce_ms", [&] { reduce(state.local_rows, state.local_inputs, state.local_count, 1); }, true);
    phase("last_group_reduce_ms", [&] { reduce(state.group_rows, state.group_inputs, state.group_count, 2); }, true);
    phase(
        "last_source_reduce_ms", [&] { reduce(state.source_rows, state.source_inputs, state.source_count, 3); }, true);
    state.metrics["last_forward_ms"] = Milliseconds(total_start);
}
}  // namespace deepep::megamoe
