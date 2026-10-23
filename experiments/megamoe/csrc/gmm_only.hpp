// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include <map>
#include <string>
#include <vector>
#include "forward.hpp"
#include "kernels/mega_moe/urma/abi.hpp"

namespace deepep::megamoe {
class GmmOnlyPlan {
public:
    GmmOnlyPlan(torch::Tensor counts_cpu, torch::Tensor weight1, torch::Tensor weight2, torch::Tensor scale1,
                torch::Tensor scale2, int64_t rank, int64_t cv_rows, int64_t ahead, int64_t bs);
    void forward(int64_t mode);
    std::map<std::string, int64_t> describe() const;

private:
    MegaMoeUrma::Arguments args_{};
    PlatformInfo platform_{};
    std::vector<torch::Tensor> storage_;
    int64_t startup_groups_ = 0;
    int device_ = 0;
};
}  // namespace deepep::megamoe
