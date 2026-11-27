// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once

#include <cstdint>
#include <tuple>
#include <torch/extension.h>
#include "kernels/mega_moe/mega_moe_tiling.hpp"
#include "kernels/mega_moe/common/mega_moe_tp_layout.hpp"

namespace deepep::megamoe {

struct PlatformInfo {
    uint32_t aic;
    uint32_t aiv;
    uint32_t ub;
    uint32_t block_dim;
    uint32_t system_workspace;
};

// 一个 BS 对应一个准备好的计划；Python owner 保证同流串行复用及销毁顺序。
class Plan {
public:
    Plan(torch::Tensor example, int64_t intermediate, int64_t topk, int64_t experts, int64_t world, int64_t max_tokens,
         bool is_a8w4, int64_t shared_experts, int64_t ranks_per_node, uint64_t ffts, uint64_t symmetric_address,
         int64_t symmetric_bytes, int64_t local_rank = -1, bool route_weight_before_quant = false,
         bool variable_token_counts = false, c10::optional<torch::Tensor> workspace_hint = c10::nullopt);

    torch::Tensor workspace_tensor() const { return workspace_; }

    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor x, c10::optional<torch::Tensor> x_scales, torch::Tensor topk_ids, torch::Tensor topk_weights,
        torch::Tensor weight1, torch::Tensor weight2, torch::Tensor weight_scales1, torch::Tensor weight_scales2,
        c10::optional<torch::Tensor> timer, c10::optional<torch::Tensor> shared_weight1,
        c10::optional<torch::Tensor> shared_weight2, c10::optional<torch::Tensor> shared_weight_scales1,
        c10::optional<torch::Tensor> shared_weight_scales2, c10::optional<double> activation_clamp,
        c10::optional<torch::Tensor> out);

    int64_t workspace_bytes() const
    {
        return workspace_.numel();
    }

    void enable_compute_replay();
    void prepare_local_compute(torch::Tensor data, torch::Tensor scales, torch::Tensor counts);
    torch::Tensor local_compute_output();

private:
    PlatformInfo platform_;
    MegaMoeTilingData tiling_;
    torch::Tensor workspace_;
    torch::Tensor context_;
    uint64_t ffts_;
    bool compute_replay_ = false;
    torch::Tensor replay_counts_;
    torch::Tensor local_buffer_;
    int64_t local_rows_ = 0;
};

int64_t buffer_size(int64_t world, int64_t experts, int64_t tokens, int64_t topk, int64_t hidden);

int64_t tp_buffer_size(int64_t tp_world, int64_t tokens, int64_t topk, int64_t hidden,
                       int64_t rs_tokens_per_chunk = MegaMoeImpl::TP_DEFAULT_RS_TOKENS_PER_CHUNK);

// TP 模式专用执行计划：路由专家沿中间维做 TP 切分，可选共享专家不做 TP（支持全 FP8 / 全 BF16）。
class TpPlan {
public:
    TpPlan(torch::Tensor example, int64_t global_intermediate, int64_t shared_intermediate, int64_t topk,
           int64_t num_experts, int64_t tp_world, int64_t max_tokens, int64_t shared_experts, uint64_t ffts,
           uint64_t symmetric_address, int64_t symmetric_bytes, bool route_weight_before_quant = false,
           int64_t m_groups_per_wave = MegaMoeImpl::TP_DEFAULT_M_GROUPS_PER_WAVE,
           int64_t ag_cores_per_peer = MegaMoeImpl::TP_DEFAULT_AG_CORES_PER_PEER,
           int64_t rs_tokens_per_chunk = MegaMoeImpl::TP_DEFAULT_RS_TOKENS_PER_CHUNK,
           c10::optional<torch::Tensor> workspace_hint = c10::nullopt, bool use_urma_allgather = true,
           int64_t shmem_instance_id = 0);

    torch::Tensor workspace_tensor() const { return workspace_; }
    int64_t workspace_bytes() const { return workspace_.numel(); }

    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor x, c10::optional<torch::Tensor> x_scales, torch::Tensor topk_ids, torch::Tensor topk_weights,
        torch::Tensor weight1, torch::Tensor weight2, torch::Tensor weight_scales1, torch::Tensor weight_scales2,
        c10::optional<torch::Tensor> timer, c10::optional<torch::Tensor> shared_weight1,
        c10::optional<torch::Tensor> shared_weight2, c10::optional<torch::Tensor> shared_weight_scales1,
        c10::optional<torch::Tensor> shared_weight_scales2, c10::optional<double> activation_clamp,
        c10::optional<torch::Tensor> out, c10::optional<bool> route_weight_before_quant = c10::nullopt);

private:
    PlatformInfo platform_;
    MegaMoeImpl::MegaMoeTpTilingData tiling_{};
    torch::Tensor workspace_;
    torch::Tensor context_;
    uint64_t ffts_{0};
};

}  // namespace deepep::megamoe
