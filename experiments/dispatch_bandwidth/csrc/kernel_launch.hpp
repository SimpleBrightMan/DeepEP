// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once

#include "dispatch_types.hpp"

extern "C" void dispatch_bandwidth_kernel_do(uint32_t block_dim, uint32_t ub_bytes, void* stream, uint8_t* peer_bases,
                                             uint8_t* tasks, uint8_t* metadata_input, uint8_t* workers,
                                             uint8_t* output_tokens, uint8_t* output_scales, uint8_t* output_metadata,
                                             uint8_t* timer, deepep::dispatch_bench::KernelConfig config);
