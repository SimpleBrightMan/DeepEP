// Copyright (c) 2026, SimpleBright_Man
#pragma once
#include <cstdint>

// Separate from the public 224-byte tiling ABI. Zero keeps the original fused path.
struct MegaMoeHierarchyConfig {
    uint64_t local_cache = 0;
    uint64_t token_slots = 0;
    uint64_t edge_slots = 0;
    uint64_t edge_outputs = 0;
    uint32_t mode = 0;  // 0 direct, 1 prepare only, 2 cached compute only
    uint32_t batch = 0;
};

struct HierarchyCopyTask {
    uint64_t source;
    uint64_t destination;
    uint32_t bytes;
    uint32_t reserved = 0;
};
struct HierarchyReduceRow {
    uint64_t destination;
    uint32_t begin;
    uint32_t count;
};
struct HierarchyReduceInput {
    uint64_t source;
    uint64_t weight = 0;  // zero means an unweighted FP32 partial
};

extern "C" void mega_moe_hierarchy_copy_do(uint32_t blocks, uint32_t ub, void* stream, uint8_t* tasks, uint32_t count,
                                           uint64_t ffts);
extern "C" void mega_moe_hierarchy_reduce_do(uint32_t blocks, uint32_t ub, void* stream, uint8_t* rows, uint8_t* inputs,
                                             uint32_t count, uint32_t hidden, uint32_t mode, uint8_t* output,
                                             uint64_t ffts);
