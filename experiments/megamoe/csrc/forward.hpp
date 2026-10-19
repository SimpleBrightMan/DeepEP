// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once

#include <cstdint>
#include <tuple>
#include <torch/extension.h>
#include "kernels/mega_moe/mega_moe_tiling.hpp"

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
         int64_t symmetric_bytes, bool prefill_optimization);

    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor x, c10::optional<torch::Tensor> x_scales, torch::Tensor topk_ids, torch::Tensor topk_weights,
        torch::Tensor weight1, torch::Tensor weight2, torch::Tensor weight_scales1, torch::Tensor weight_scales2,
        c10::optional<torch::Tensor> timer, c10::optional<torch::Tensor> shared_weight1,
        c10::optional<torch::Tensor> shared_weight2, c10::optional<torch::Tensor> shared_weight_scales1,
        c10::optional<torch::Tensor> shared_weight_scales2, c10::optional<torch::Tensor> out);

    int64_t workspace_bytes() const
    {
        return workspace_.numel();
    }

    bool prefill_optimized() const
    {
        return tiling_.prefillCommOptimized != 0U;
    }
    bool prefill_requested() const
    {
        return prefill_requested_;
    }
    bool is_a8w4() const
    {
        return is_a8w4_;
    }
    int64_t dispatch_buffer_count() const
    {
        return tiling_.dispatchBufferConfig.bufferCount;
    }
    int64_t combine_buffer_count() const
    {
        return prefill_optimized() ? 8 : (tiling_.bs <= 256U ? 6 : 2);
    }
    int64_t prefill_metadata_bytes() const
    {
        return prefill_metadata_.defined() ? prefill_metadata_.numel() * sizeof(int32_t) : 0;
    }

private:
    PlatformInfo platform_;
    MegaMoeTilingData tiling_;
    torch::Tensor workspace_;
    torch::Tensor context_;
    // Immutable identity records: allocated once per prepared BS, outside forward.
    torch::Tensor prefill_metadata_;
    bool prefill_requested_;
    bool is_a8w4_;
    uint64_t ffts_;
};

int64_t buffer_size(int64_t world, int64_t experts, int64_t tokens, int64_t topk, int64_t hidden);

}  // namespace deepep::megamoe
