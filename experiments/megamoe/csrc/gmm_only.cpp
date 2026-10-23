// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#include "gmm_only.hpp"
#include "kernels/mega_moe/urma/compute_contract.hpp"
#include "kernels/mega_moe/urma/expert_plan.hpp"
#include <algorithm>
#include <cstring>
#include <type_traits>
#include "check.hpp"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tiling/platform/platform_ascendc.h"

extern "C" void mega_moe_gmm_only_kernel_do(uint32_t, uint32_t, void*, const MegaMoeUrma::Arguments*, uint32_t);

namespace deepep::megamoe {
GmmOnlyPlan::GmmOnlyPlan(torch::Tensor counts, torch::Tensor w1, torch::Tensor w2, torch::Tensor s1, torch::Tensor s2,
                         int64_t rank, int64_t cv_rows, int64_t ahead)
{
    TORCH_CHECK(counts.device().is_cpu() && counts.scalar_type() == torch::kInt64 && counts.is_contiguous() &&
                    counts.sizes() == torch::IntArrayRef({16, 384}),
                "counts must be CPU int64 [16,384]");
    TORCH_CHECK(rank >= 0 && rank < 16 && cv_rows >= 128 && cv_rows <= 4096 && cv_rows % 128 == 0 && ahead >= 0 &&
                    ahead <= 65536,
                "Invalid rank/CV/lookahead");
    TORCH_CHECK(w1.device().type() == c10::DeviceType::PrivateUse1, "NPU weights required");
    device_ = w1.device().index();
    int current = -1;
    ASCEND_DEEPEP_CHECK_ACL(aclrtGetDevice(&current));
    TORCH_CHECK(current == device_, "Select the weights' device before creating the plan");
    for (const auto* tensor : {&w1, &w2, &s1, &s2}) {
        TORCH_CHECK(tensor->device() == w1.device() && tensor->is_contiguous(),
                    "Weights must be contiguous on one NPU");
    }
    TORCH_CHECK(w1.scalar_type() == torch::kUInt8 && w2.scalar_type() == torch::kUInt8 &&
                    w1.sizes() == torch::IntArrayRef({24, 2560, 1792}) &&
                    w2.sizes() == torch::IntArrayRef({24, 3584, 640}),
                "Expected prepared FP4 NZ weights for H3584/I1280/E384/EP16");
    TORCH_CHECK(
        s1.scalar_type() == c10::ScalarType::Float8_e8m0fnu && s2.scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
            s1.sizes() == torch::IntArrayRef({24, 2560, 56, 2}) && s2.sizes() == torch::IntArrayRef({24, 3584, 20, 2}),
        "Invalid E8M0 weight scales");
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "Cannot query Ascend platform");
    uint64_t ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    TORCH_CHECK(
        platform->GetCoreNumAic() >= 28 && platform->GetCoreNumAiv() >= 56 && ub >= MegaMoeUrma::kRequiredUbBytes,
        "GMM-only requires 28 AIC / 56 AIV and 192KiB UB");
    platform_ = {28, 56, static_cast<uint32_t>(ub), platform->CalcTschBlockDim(56, 28, 56),
                 platform->GetLibApiWorkSpaceSize()};
    // This target is built exclusively for Ascend950 / CATLASS_ARCH=3510.
    // Like SHMEM's C310 path, it needs no legacy FFTS address: the device
    // util_set_ffts_config is a no-op. Internal CATLASS cross-core events remain.
    // rtGetC2cCtrlAddr is unsupported here and must not gate plan creation.
    args_.ffts = 0;
    auto& config = args_.config;
    config.bs = 2048;
    config.hidden = 3584;
    config.intermediate = 1280;
    config.topk = 8;
    config.localExperts = 24;
    config.rank = rank;
    config.cvRows = cv_rows;
    config.maxPeerChunks = (2048 * 8 + cv_rows - 1) / cv_rows;
    config.gmm1AheadMGroups = ahead;
    auto plan = MegaMoeUrma::BuildExpertPlan(counts.data_ptr<int64_t>(), config, 16ULL * 2048 * 8);
    config = plan.config;
    TORCH_CHECK(config.receiveRows > 0, "Empty destination cannot measure GMM MFU");
    auto align = [](uint64_t bytes) { return (bytes + 511U) / 512U * 512U; };
    config.receiveXOffset = 0;
    config.receiveScaleOffset = align(config.receiveRows * config.hidden);
    config.returnYOffset = align(config.receiveScaleOffset + config.receiveRows * (config.hidden / 32));
    config.symmetricBytes = align(config.returnYOffset + 2048ULL * 8 * config.hidden * 2);
    const auto options = w1.options().dtype(torch::kUInt8);
    auto keep = [&](torch::Tensor tensor) {
        storage_.push_back(tensor);
        return reinterpret_cast<uint64_t>(tensor.data_ptr());
    };
    auto allocate = [&](uint64_t bytes, int fill) {
        return keep(torch::full({static_cast<int64_t>(std::max<uint64_t>(bytes, 512))}, fill, options));
    };
    auto upload = [&](const auto& records) {
        using Item = typename std::decay_t<decltype(records)>::value_type;
        const auto bytes = std::max<size_t>(records.size(), 1) * sizeof(Item);
        auto cpu = torch::zeros({static_cast<int64_t>(bytes)}, torch::TensorOptions().dtype(torch::kUInt8));
        if (!records.empty()) std::memcpy(cpu.data_ptr(), records.data(), records.size() * sizeof(Item));
        return keep(cpu.to(w1.device()));
    };
    // Plain LOCAL HBM. No SHMEM init, registration, rendezvous or real peer access.
    args_.symmetric = allocate(config.symmetricBytes, 0x38);                              // FP8 +1 in source-major X.
    args_.gatheredInputScale = allocate(config.receiveRows * (config.hidden / 32), 127);  // E8M0 1.
    args_.activation = allocate(config.receiveRows * config.intermediate, 0x38);          // Independent GMM2 input.
    args_.activationScale = allocate(config.receiveRows * (config.intermediate / 32), 127);
    args_.gmm1Output = allocate(config.receiveRows * config.intermediate * 4, 0);
    args_.gmm2Output = allocate(config.receiveRows * config.hidden * 2, 0);
    args_.systemWorkspace = allocate(platform_.system_workspace + (2ULL << 20), 0);
    args_.fragments = upload(plan.fragments);
    args_.segments = upload(plan.segments);
    args_.computeBatches = upload(plan.computeBatches);
    args_.computeTasks = upload(plan.computeTasks);
    args_.weight1 = keep(w1);
    args_.weight2 = keep(w2);
    args_.weightScale1 = keep(s1);
    args_.weightScale2 = keep(s2);
    for (const auto& task : plan.computeTasks) {
        if (task.stage == MegaMoeUrma::ComputeBatchStage::Gmm2) break;
        startup_groups_ += plan.computeBatches[task.batchIndex].fragmentCount;
    }
}

void GmmOnlyPlan::forward(int64_t mode)
{
    TORCH_CHECK(mode >= 0 && mode <= 2, "mode: 0=both, 1=gmm1, 2=gmm2");
    int device = -1;
    ASCEND_DEEPEP_CHECK_ACL(aclrtGetDevice(&device));
    TORCH_CHECK(device == device_, "Select this plan's NPU before forward");
    const auto stream = c10_npu::getCurrentNPUStream();
    mega_moe_gmm_only_kernel_do(platform_.block_dim, platform_.ub, stream.stream(), &args_, mode);
    for (const auto& tensor : storage_) {
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }
}

std::map<std::string, int64_t> GmmOnlyPlan::describe() const
{
    const auto& c = args_.config;
    return {{"bs", c.bs},
            {"logical_rank", c.rank},
            {"ep", 16},
            {"receive_rows", c.receiveRows},
            {"m_groups", c.fragmentCount},
            {"gmm1_tiles", c.gmm1Tiles},
            {"gmm2_tiles", c.gmm2Tiles},
            {"startup_mgroups", startup_groups_},
            {"compute_tasks", c.computeTaskCount},
            {"aic", 28},
            {"gmm1_flops", 4LL * c.receiveRows * c.hidden * c.intermediate},
            {"gmm2_flops", 2LL * c.receiveRows * c.hidden * c.intermediate}};
}
}  // namespace deepep::megamoe
