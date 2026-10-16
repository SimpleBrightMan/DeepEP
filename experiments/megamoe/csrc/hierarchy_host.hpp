// Copyright (c) 2026, SimpleBright_Man
#pragma once
#include <functional>
#include "forward.hpp"
#include "hierarchy_types.hpp"

namespace deepep::megamoe {
struct HierarchyState {
    MegaMoeHierarchyConfig config;
    torch::Tensor token_slots, edge_slots, cache, edge_outputs, weights, expected_ids, route_owner;
    torch::Tensor copy_cross, copy_intra, copy_metadata;
    torch::Tensor local_rows, local_inputs, group_rows, group_inputs, source_rows, source_inputs;
    uint32_t cross_count = 0, intra_count = 0, metadata_count = 0;
    uint32_t local_count = 0, group_count = 0, source_count = 0;
    uint64_t weight_source = 0;
    const void* route_identity = nullptr;
    int64_t route_version = -1;
    std::map<std::string, double> metrics;
};

void ExecuteHierarchy(HierarchyState& state, torch::Tensor ids, torch::Tensor weights, torch::Tensor output,
                      const PlatformInfo& platform, void* stream, uint64_t ffts,
                      const std::function<void(MegaMoeHierarchyConfig)>& launch);
}  // namespace deepep::megamoe
