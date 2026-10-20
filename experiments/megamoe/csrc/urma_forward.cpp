// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#include "urma_forward.hpp"
#include "kernels/mega_moe/urma/compute_contract.hpp"
#include "kernels/mega_moe/urma/expert_plan.hpp"

#include <algorithm>
#include <cstring>
#include <vector>
#include <acl/acl.h>
#include "check.hpp"
#include "shmem.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tiling/platform/platform_ascendc.h"

namespace deepep::megamoe {
namespace {
using MegaMoeUrma::Config;
constexpr uint64_t kAlignment = 512;
uint64_t Align(uint64_t value, uint64_t alignment = kAlignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

Config Layout(int64_t world, int64_t max_tokens, int64_t topk, int64_t hidden, int64_t max_recv_rows, int64_t cv_rows)
{
    TORCH_CHECK(world == 16 && max_tokens > 0 && max_tokens <= 8192 && topk == 8 && hidden == 3584,
                "MegaMoE URMA v1 requires EP16, H3584, K8 and max_tokens in [1,8192]");
    TORCH_CHECK(cv_rows >= 128 && cv_rows <= 4096 && cv_rows % 128 == 0,
                "URMA CV rows must be a multiple of 128 in [128,4096]");
    TORCH_CHECK(max_recv_rows > 0 && max_recv_rows <= world * max_tokens * topk,
                "max_recv_rows must be positive and no larger than global expanded rows");
    Config config{};
    config.hidden = hidden;
    config.topk = topk;
    config.cvRows = cv_rows;
    config.maxPeerChunks = (max_tokens * topk + cv_rows - 1) / cv_rows;
    config.receiveXOffset = 0;
    config.receiveScaleOffset = Align(static_cast<uint64_t>(max_recv_rows) * hidden);
    config.returnYOffset = Align(config.receiveScaleOffset + static_cast<uint64_t>(max_recv_rows) * (hidden / 32));
    config.dispatchReadyOffset = Align(config.returnYOffset + static_cast<uint64_t>(max_tokens) * topk * hidden * 2);
    config.combineReadyOffset =
        Align(config.dispatchReadyOffset + 16ULL * config.maxPeerChunks * MegaMoeUrma::kFlagBytes);
    config.barrierOffset = Align(config.combineReadyOffset + 16ULL * config.maxPeerChunks * MegaMoeUrma::kFlagBytes);
    config.symmetricBytes = Align(config.barrierOffset + 4ULL * 16 * MegaMoeUrma::kFlagBytes, 2ULL << 20);
    return config;
}

void CheckTensor(const torch::Tensor& tensor, int64_t dims, const char* name, const c10::Device& device)
{
    TORCH_CHECK(tensor.defined() && tensor.device() == device && tensor.is_contiguous() && tensor.dim() == dims, name,
                " must be a contiguous ", dims, "D tensor on ", device);
}

uint64_t Address(const torch::Tensor& tensor)
{
    return reinterpret_cast<uint64_t>(tensor.data_ptr());
}

template <class T>
torch::Tensor Upload(const std::vector<T>& records, const torch::TensorOptions& options)
{
    auto cpu = torch::zeros({static_cast<int64_t>(std::max<size_t>(1, records.size()) * sizeof(T))},
                            torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
    if (!records.empty()) std::memcpy(cpu.data_ptr(), records.data(), records.size() * sizeof(T));
    return cpu.to(options.device());
}
}  // namespace

int64_t urma_buffer_size(int64_t world, int64_t max_tokens, int64_t topk, int64_t hidden, int64_t max_recv_rows,
                         int64_t cv_rows)
{
    return static_cast<int64_t>(Layout(world, max_tokens, topk, hidden, max_recv_rows, cv_rows).symmetricBytes);
}

UrmaPlan::UrmaPlan(torch::Tensor example, torch::Tensor counts_cpu, int64_t intermediate, int64_t topk, int64_t cv_rows,
                   int64_t max_recv_rows, int64_t max_tokens, int64_t rank, uint64_t ffts, uint64_t symmetric_address,
                   int64_t symmetric_bytes)
{
    TORCH_CHECK(example.device().type() == c10::DeviceType::PrivateUse1 && example.dim() == 2 &&
                    example.is_contiguous() && example.size(0) > 0 && example.size(0) <= max_tokens,
                "URMA example must be contiguous NPU [BS,H], with 0<BS<=max_tokens");
    TORCH_CHECK(intermediate == 1280 && rank >= 0 && rank < 16, "URMA v1 requires I1280 and rank in [0,16)");
    TORCH_CHECK(counts_cpu.device().is_cpu() && counts_cpu.scalar_type() == torch::kInt64 &&
                    counts_cpu.is_contiguous() && counts_cpu.sizes() == torch::IntArrayRef({16, 384}),
                "URMA counts must be contiguous CPU int64 [16,384]");
    args_.config = Layout(16, max_tokens, topk, example.size(1), max_recv_rows, cv_rows);
    auto& config = args_.config;
    config.bs = example.size(0);
    config.intermediate = intermediate;
    config.localExperts = 24;
    config.rank = rank;
    TORCH_CHECK(symmetric_address != 0 && symmetric_bytes >= static_cast<int64_t>(config.symmetricBytes),
                "URMA symmetric workspace is too small: required ", config.symmetricBytes);
    args_.symmetric = symmetric_address;
    args_.ffts = ffts;

    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "Cannot query Ascend platform");
    uint64_t ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const uint32_t aic = platform->GetCoreNumAic();
    const uint32_t aiv = platform->GetCoreNumAiv();
    TORCH_CHECK(aic >= MegaMoeUrma::kAic && aiv >= 2U * MegaMoeUrma::kAic,
                "URMA v1 needs at least 28 AIC / 56 AIV; detected AIC=", aic, ", AIV=", aiv, ", UB=", ub,
                " bytes. It always launches 28 AIC blocks");
    TORCH_CHECK(ub >= MegaMoeUrma::kRequiredUbBytes, "URMA v1 requires ", MegaMoeUrma::kRequiredUbBytes,
                " bytes of UB for the W4 prologue and 128-row SwiGLU slices; detected UB=", ub, " bytes, AIC=", aic,
                ", AIV=", aiv);
    // Physical devices may have more cores. The kernel's 28/15/13 partitions
    // and timer storage are fixed: never launch the extra physical blocks.
    constexpr uint32_t launch_aic = MegaMoeUrma::kAic;
    constexpr uint32_t launch_aiv = 2U * launch_aic;
    platform_ = {launch_aic, launch_aiv, static_cast<uint32_t>(ub),
                 platform->CalcTschBlockDim(launch_aiv, launch_aic, launch_aiv), platform->GetLibApiWorkSpaceSize()};

    TORCH_CHECK(aclshmem_my_pe() == rank && aclshmem_n_pes() == 16, "URMA plan does not match its SHMEM runtime");
    for (int peer = 0; peer < 16; ++peer) {
        TORCH_CHECK(aclshmem_ptr(reinterpret_cast<void*>(symmetric_address), peer) != nullptr,
                    "URMA completion signals require an MTE-accessible symmetric window for every peer; rank=", rank,
                    " cannot map peer=", peer,
                    ". Multi-host testing requires one mutually accessible UB memory domain.");
    }

    const auto plan =
        MegaMoeUrma::BuildExpertPlan(counts_cpu.data_ptr<int64_t>(), config, static_cast<uint64_t>(max_recv_rows));
    config = plan.config;
    const auto options = example.options().dtype(torch::kUInt8);
    const uint64_t expanded_rows = static_cast<uint64_t>(config.bs) * topk;
    auto allocate = [&](uint64_t bytes) {
        return torch::empty({static_cast<int64_t>(std::max<uint64_t>(bytes, 512))}, options);
    };
    expanded_x_ = allocate(expanded_rows * config.hidden);
    expanded_scale_ = allocate(expanded_rows * (config.hidden / 32));
    gathered_input_scale_ = allocate(config.receiveRows * (config.hidden / 32));
    gmm1_ = allocate(config.receiveRows * intermediate * 4);
    activation_ = allocate(config.receiveRows * intermediate);
    activation_scale_ = allocate(config.receiveRows * (intermediate / 32));
    gmm2_ = allocate(config.receiveRows * config.hidden * 2);
    fragments_ = Upload(plan.fragments, options);
    segments_ = Upload(plan.segments, options);
    compute_batches_ = Upload(plan.computeBatches, options);
    chunk_group_indices_ = Upload(plan.chunkGroupIndices, options);
    auto dependency_cpu = torch::empty({static_cast<int64_t>(expanded_rows)},
                                       torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));
    std::memcpy(dependency_cpu.data_ptr(), plan.reduceDependencies.data(), expanded_rows * sizeof(uint64_t));
    expanded_reduce_dependencies_ = dependency_cpu.to(options.device());
    send_chunks_ = Upload(plan.sendChunks, options);
    recv_chunks_ = Upload(plan.recvChunks, options);
    gmm1_done_ = allocate(config.gmm1Tiles * MegaMoeUrma::kFlagBytes);
    input_scale_done_ = allocate(config.fragmentCount * MegaMoeUrma::kFlagBytes);
    activation_done_ = allocate(config.fragmentCount * MegaMoeUrma::kFlagBytes);
    gmm2_done_ = allocate(config.fragmentCount * MegaMoeUrma::kFlagBytes);
    local_flags_ = allocate(16 * MegaMoeUrma::kFlagBytes);
    system_workspace_ = allocate(platform_.system_workspace + (2ULL << 20));

    args_.expandedX = Address(expanded_x_);
    args_.expandedScale = Address(expanded_scale_);
    args_.gatheredInputScale = Address(gathered_input_scale_);
    args_.gmm1Output = Address(gmm1_);
    args_.activation = Address(activation_);
    args_.activationScale = Address(activation_scale_);
    args_.gmm2Output = Address(gmm2_);
    args_.fragments = Address(fragments_);
    args_.segments = Address(segments_);
    args_.computeBatches = Address(compute_batches_);
    args_.chunkGroupIndices = Address(chunk_group_indices_);
    args_.sendChunks = Address(send_chunks_);
    args_.recvChunks = Address(recv_chunks_);
    args_.gmm1Done = Address(gmm1_done_);
    args_.inputScaleDone = Address(input_scale_done_);
    args_.activationDone = Address(activation_done_);
    args_.gmm2Done = Address(gmm2_done_);
    args_.localFlags = Address(local_flags_);
    args_.systemWorkspace = Address(system_workspace_);
}

void UrmaPlan::bind_reduce_dependencies(torch::Tensor reduce_index)
{
    TORCH_CHECK(!reduce_dependencies_.defined(), "URMA Reduce dependencies have already been bound");
    CheckTensor(reduce_index, 2, "reduce_index", expanded_x_.device());
    const auto& config = args_.config;
    TORCH_CHECK(reduce_index.scalar_type() == torch::kInt64 &&
                    reduce_index.sizes() == torch::IntArrayRef({config.bs, config.topk}),
                "reduce_index must be int64 [BS,topk]");
    // One NPU gather per route preparation avoids a second per-route GM lookup
    // in the Reduce loop. Values are opaque int64 bit patterns: the high bit
    // distinguishes direct-self GMM2 completion from remote producer/CV flags.
    reduce_dependencies_ =
        expanded_reduce_dependencies_.index_select(0, reduce_index.reshape({-1})).reshape({config.bs, config.topk});
    bound_reduce_index_ = reduce_index;
    bound_reduce_index_version_ = reduce_index._version();
    args_.reduceDependencies = Address(reduce_dependencies_);
    const auto stream = c10_npu::getCurrentNPUStream();
    for (const auto* tensor : {&expanded_reduce_dependencies_, &reduce_dependencies_, &bound_reduce_index_}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    }
}

torch::Tensor UrmaPlan::forward(torch::Tensor x, torch::Tensor x_scales, torch::Tensor order,
                                torch::Tensor reduce_index, torch::Tensor topk_weights, torch::Tensor weight1,
                                torch::Tensor weight2, torch::Tensor weight_scales1, torch::Tensor weight_scales2,
                                c10::optional<torch::Tensor> timer, c10::optional<torch::Tensor> out, int64_t epoch,
                                c10::optional<torch::Tensor> trace)
{
    const auto& c = args_.config;
    const auto device = expanded_x_.device();
    TORCH_CHECK(reduce_dependencies_.defined() && bound_reduce_index_.defined(),
                "URMA Reduce dependencies must be bound before forward");
    TORCH_CHECK(reduce_index.defined() && reduce_index.data_ptr() == bound_reduce_index_.data_ptr() &&
                    reduce_index._version() == bound_reduce_index_version_ &&
                    bound_reduce_index_._version() == bound_reduce_index_version_,
                "URMA reduce_index differs from its bound inverse or was modified; prepare a new route plan");
    CheckTensor(x, 2, "x", device);
    CheckTensor(x_scales, 3, "x_scales", device);
    CheckTensor(order, 1, "order", device);
    CheckTensor(reduce_index, 2, "reduce_index", device);
    CheckTensor(topk_weights, 2, "topk_weights", device);
    CheckTensor(weight1, 3, "weight1", device);
    CheckTensor(weight2, 3, "weight2", device);
    CheckTensor(weight_scales1, 4, "weight_scales1", device);
    CheckTensor(weight_scales2, 4, "weight_scales2", device);
    TORCH_CHECK(x.scalar_type() == c10::ScalarType::Float8_e4m3fn && x.sizes() == torch::IntArrayRef({c.bs, c.hidden}),
                "x must be prepared FP8 E4M3 [BS,H]");
    TORCH_CHECK(x_scales.scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                    x_scales.sizes() == torch::IntArrayRef({c.bs, c.hidden / 64, 2}),
                "x_scales must be FP8 E8M0 [BS,H/64,2]");
    TORCH_CHECK(order.scalar_type() == torch::kInt64 && order.numel() == static_cast<int64_t>(c.bs) * c.topk &&
                    reduce_index.scalar_type() == torch::kInt64 &&
                    reduce_index.sizes() == torch::IntArrayRef({c.bs, c.topk}),
                "Invalid route permutation shape/type");
    TORCH_CHECK(
        topk_weights.scalar_type() == torch::kFloat32 && topk_weights.sizes() == torch::IntArrayRef({c.bs, c.topk}),
        "weights must be FP32 [BS,K]");
    TORCH_CHECK(weight1.scalar_type() == torch::kUInt8 && weight2.scalar_type() == torch::kUInt8 &&
                    weight1.sizes() == torch::IntArrayRef({24, c.intermediate * 2, c.hidden / 2}) &&
                    weight2.sizes() == torch::IntArrayRef({24, c.hidden, c.intermediate / 2}),
                "URMA v1 requires main MegaMoE packed FP4 NZ weight layouts");
    TORCH_CHECK(weight_scales1.scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                    weight_scales2.scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                    weight_scales1.sizes() == torch::IntArrayRef({24, c.intermediate * 2, c.hidden / 64, 2}) &&
                    weight_scales2.sizes() == torch::IntArrayRef({24, c.hidden, c.intermediate / 64, 2}),
                "Invalid FP4 weight scales");
    TORCH_CHECK(epoch > last_epoch_ && epoch < (1LL << 60), "URMA epoch must increase across calls");
    if (timer) {
#ifndef DEEPEP_MEGAMOE_TIMER
        TORCH_CHECK(false, "Rebuild with DEEPEP_MEGAMOE_TIMER=ON for URMA timers");
#endif
        CheckTensor(*timer, 1, "timer", device);
        TORCH_CHECK(timer->scalar_type() == torch::kInt64 && timer->numel() == MegaMoeUrma::kTimerElements,
                    "Invalid URMA timer allocation");
    }
    uint64_t trace_capacity = 0;
    if (trace) {
#ifndef DEEPEP_MEGAMOE_TIMER
        TORCH_CHECK(false, "Rebuild with DEEPEP_MEGAMOE_TIMER=ON for URMA traces");
#endif
        CheckTensor(*trace, 1, "trace", device);
        TORCH_CHECK(Address(*trace) % 128U == 0,
                    "trace storage must be 128-byte aligned; use UrmaBuffer.allocate_trace()");
        static_assert(MegaMoeUrma::kTraceHeaderElements == MegaMoeUrma::kTraceRecordElements);
        constexpr int64_t trace_row_elements =
            static_cast<int64_t>(MegaMoeUrma::kTimerCores) * MegaMoeUrma::kTraceRecordElements;
        TORCH_CHECK(trace->scalar_type() == torch::kInt64 && trace->numel() % trace_row_elements == 0 &&
                        trace->numel() >= 2 * trace_row_elements &&
                        trace->numel() <= (int64_t{MegaMoeUrma::kTraceMaxCapacity} + 1) * trace_row_elements,
                    "trace must contain 84 * (16 + capacity * 16) int64 elements; capacity must be in [1,",
                    MegaMoeUrma::kTraceMaxCapacity, "]");
        trace_capacity = static_cast<uint64_t>(trace->numel() / trace_row_elements - 1);
    }
    auto output = out ? *out : torch::empty({c.bs, c.hidden}, x.options().dtype(torch::kBFloat16));
    CheckTensor(output, 2, "out", device);
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16 && output.sizes() == x.sizes(), "out must be BF16 [BS,H]");
    auto stream = c10_npu::getCurrentNPUStream();
    for (auto* tensor : {&gmm1_done_, &input_scale_done_, &activation_done_, &gmm2_done_, &local_flags_}) {
        ASCEND_DEEPEP_CHECK_ACL(
            aclrtMemsetAsync(tensor->data_ptr(), tensor->numel(), 0, tensor->numel(), stream.stream()));
    }
    if (timer) {
        ASCEND_DEEPEP_CHECK_ACL(
            aclrtMemsetAsync(timer->data_ptr(), timer->nbytes(), 0, timer->nbytes(), stream.stream()));
    }
    if (trace) {
        ASCEND_DEEPEP_CHECK_ACL(
            aclrtMemsetAsync(trace->data_ptr(), trace->nbytes(), 0, trace->nbytes(), stream.stream()));
    }
    args_.config.epoch = epoch;
    args_.x = Address(x);
    args_.xScale = Address(x_scales);
    args_.order = Address(order);
    args_.reduceIndex = Address(reduce_index);
    args_.weights = Address(topk_weights);
    args_.weight1 = Address(weight1);
    args_.weight2 = Address(weight2);
    args_.weightScale1 = Address(weight_scales1);
    args_.weightScale2 = Address(weight_scales2);
    args_.output = Address(output);
    args_.timer = timer ? Address(*timer) : 0;
    args_.trace = trace ? Address(*trace) : 0;
    args_.traceCapacity = trace_capacity;
    mega_moe_urma_kernel_do(platform_.block_dim, platform_.ub, stream.stream(), &args_);
    last_epoch_ = epoch;
    for (const auto* tensor : {&x,
                               &x_scales,
                               &order,
                               &reduce_index,
                               &topk_weights,
                               &weight1,
                               &weight2,
                               &weight_scales1,
                               &weight_scales2,
                               &output,
                               &expanded_x_,
                               &expanded_scale_,
                               &gathered_input_scale_,
                               &gmm1_,
                               &activation_,
                               &activation_scale_,
                               &gmm2_,
                               &fragments_,
                               &segments_,
                               &compute_batches_,
                               &chunk_group_indices_,
                               &expanded_reduce_dependencies_,
                               &reduce_dependencies_,
                               &bound_reduce_index_,
                               &send_chunks_,
                               &recv_chunks_,
                               &gmm1_done_,
                               &input_scale_done_,
                               &activation_done_,
                               &gmm2_done_,
                               &local_flags_,
                               &system_workspace_}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    }
    if (timer) c10_npu::NPUCachingAllocator::recordStream(timer->storage().data_ptr(), stream);
    if (trace) c10_npu::NPUCachingAllocator::recordStream(trace->storage().data_ptr(), stream);
    return output;
}

int64_t UrmaPlan::workspace_bytes() const
{
    int64_t bytes = 0;
    for (const auto* tensor : {&expanded_x_,
                               &expanded_scale_,
                               &gathered_input_scale_,
                               &gmm1_,
                               &activation_,
                               &activation_scale_,
                               &gmm2_,
                               &fragments_,
                               &segments_,
                               &compute_batches_,
                               &chunk_group_indices_,
                               &expanded_reduce_dependencies_,
                               &reduce_dependencies_,
                               &send_chunks_,
                               &recv_chunks_,
                               &gmm1_done_,
                               &input_scale_done_,
                               &activation_done_,
                               &gmm2_done_,
                               &local_flags_,
                               &system_workspace_})
        if (tensor->defined()) bytes += tensor->nbytes();
    return bytes;
}
}  // namespace deepep::megamoe
