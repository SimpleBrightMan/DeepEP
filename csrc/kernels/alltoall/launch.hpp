// Copyright (c) 2026, SimpleBrightMan

#pragma once
#include <cstdint>
#include "../notify/launch.hpp"

extern "C" ASCEND_DEEPEP_EXPORT void alltoall_mte_kernel_do(void* stream, uint8_t* send, uint8_t* workspace,
                                                            uint8_t* timing, uint32_t rank, uint32_t world,
                                                            uint32_t group, uint32_t token_bytes, uint32_t tokens,
                                                            uint32_t mode, uint32_t num_cores);
extern "C" ASCEND_DEEPEP_EXPORT void alltoall_mte_drain_kernel_do(void* stream, uint8_t* workspace, uint8_t* out,
                                                                 uint64_t total_bytes, uint32_t num_cores);
