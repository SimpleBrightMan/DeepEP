// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once

#include <cstdint>

namespace deepep::dispatch_bench {

constexpr uint32_t kTrafficKinds = 3U;
constexpr uint32_t kMaxRanks = 16U;
constexpr uint32_t kMetaWords = 8U;
constexpr uint32_t kIntraWorkers = 12U;
constexpr uint32_t kCrossWorkers = 16U;
constexpr uint32_t kWorkerRotationPeriod = 48U;

enum class Traffic : uint32_t { Local = 0U, Group = 1U, Cross = 2U, Intra = 3U };

inline uint32_t RotatedWorkerIndex(uint32_t physical_worker, uint32_t rotation)
{
    if (rotation == 0U) {
        return physical_worker;
    }
    return physical_worker < kIntraWorkers
               ? (physical_worker + rotation) % kIntraWorkers
               : kIntraWorkers + (physical_worker - kIntraWorkers + rotation) % kCrossWorkers;
}

enum class MetadataMode : uint32_t { Scalar = 0U, Dma = 1U, Preload = 2U };

struct CopyTask {
    uint32_t peer;
    uint32_t token;
    uint32_t topk_slot;
    uint32_t destination;
};

struct Worker {
    uint32_t traffic;
    uint32_t begin;
    uint32_t count;
    uint32_t reserved;
};

struct alignas(32) TimerRecord {
    uint64_t start;
    uint64_t end;
    uint64_t traffic;
    uint64_t rows;
};

struct KernelConfig {
    uint32_t rank;
    uint32_t world;
    uint32_t cores;
    uint32_t hidden;
    uint32_t scale_elements;
    uint32_t row_bytes;
    uint32_t ring_slots;
    uint32_t repeats;
    uint32_t metadata_mode;
    uint32_t worker_rotation;
};

static_assert(sizeof(CopyTask) == 16U);
static_assert(sizeof(Worker) == 16U);
static_assert(sizeof(TimerRecord) == 32U);
static_assert(sizeof(KernelConfig) == 40U);

}  // namespace deepep::dispatch_bench
