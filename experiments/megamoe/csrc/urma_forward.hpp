// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include <torch/extension.h>
#include "forward.hpp"
#include "kernels/mega_moe/urma/abi.hpp"

namespace deepep::megamoe {
int64_t urma_buffer_size(int64_t world, int64_t max_tokens, int64_t topk, int64_t hidden, int64_t max_recv_rows,
                         int64_t cv_rows);

class UrmaPlan {
public:
    UrmaPlan(torch::Tensor example, torch::Tensor counts_cpu, int64_t intermediate, int64_t topk, int64_t cv_rows,
             int64_t max_recv_rows, int64_t max_tokens, int64_t rank, uint64_t ffts, uint64_t symmetric_address,
             int64_t symmetric_bytes);
    torch::Tensor forward(torch::Tensor x, torch::Tensor x_scales, torch::Tensor order, torch::Tensor reduce_index,
                          torch::Tensor topk_weights, torch::Tensor weight1, torch::Tensor weight2,
                          torch::Tensor weight_scales1, torch::Tensor weight_scales2,
                          c10::optional<torch::Tensor> timer, c10::optional<torch::Tensor> out, int64_t epoch);
    int64_t workspace_bytes() const;

private:
    PlatformInfo platform_{};
    MegaMoeUrma::Arguments args_{};
    torch::Tensor expanded_x_, expanded_scale_, gmm1_, activation_, activation_scale_, gmm2_;
    torch::Tensor fragments_, send_chunks_, recv_chunks_, gmm1_done_, activation_done_, gmm2_done_, local_flags_;
    torch::Tensor system_workspace_;
    int64_t last_epoch_ = 0;
};
}  // namespace deepep::megamoe
