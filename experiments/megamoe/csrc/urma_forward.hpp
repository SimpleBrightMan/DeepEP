// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include <torch/extension.h>
#include <map>
#include <string>
#include "forward.hpp"
#include "kernels/mega_moe/urma/abi.hpp"

namespace deepep::megamoe {
int64_t urma_buffer_size(int64_t world, int64_t max_tokens, int64_t topk, int64_t hidden, int64_t max_recv_rows,
                         int64_t cv_rows);

class UrmaPlan {
public:
    UrmaPlan(torch::Tensor example, torch::Tensor counts_cpu, int64_t intermediate, int64_t topk, int64_t cv_rows,
             int64_t max_recv_rows, int64_t max_tokens, int64_t rank, uint64_t ffts, uint64_t symmetric_address,
             int64_t symmetric_bytes, int64_t gmm1_ahead_mgroups);
    void bind_reduce_dependencies(torch::Tensor reduce_index);
    torch::Tensor forward(torch::Tensor x, torch::Tensor x_scales, torch::Tensor order, torch::Tensor reduce_index,
                          torch::Tensor topk_weights, torch::Tensor weight1, torch::Tensor weight2,
                          torch::Tensor weight_scales1, torch::Tensor weight_scales2,
                          c10::optional<torch::Tensor> timer, c10::optional<torch::Tensor> out, int64_t epoch,
                          c10::optional<torch::Tensor> trace = c10::nullopt);
    int64_t workspace_bytes() const;
    // Explicit diagnostic only: synchronize before copying owned intermediates.
    std::map<std::string, torch::Tensor> debug_snapshot() const;
    int64_t gmm1_startup_mgroups() const
    {
        return gmm1_startup_mgroups_;
    }

private:
    PlatformInfo platform_{};
    MegaMoeUrma::Arguments args_{};
    torch::Tensor expanded_x_, expanded_scale_, gmm1_, activation_, activation_scale_, gmm2_;
    torch::Tensor gathered_input_scale_, input_scale_done_;
    torch::Tensor gmm2_contiguous_, gmm2_matmul_done_;
    torch::Tensor fragments_, send_chunks_, recv_chunks_, gmm1_done_, activation_done_, gmm2_done_, local_flags_;
    torch::Tensor segments_, compute_batches_, compute_tasks_, chunk_group_indices_;
    torch::Tensor peer_reorder_ranges_, peer_reorder_spans_;
    torch::Tensor pack_rounds_, pack_ready_;
    torch::Tensor expanded_reduce_dependencies_, reduce_dependencies_, bound_reduce_index_;
    torch::Tensor system_workspace_;
    int64_t bound_reduce_index_version_ = -1;
    int64_t last_epoch_ = 0;
    int64_t gmm1_startup_mgroups_ = 0;
};
}  // namespace deepep::megamoe
