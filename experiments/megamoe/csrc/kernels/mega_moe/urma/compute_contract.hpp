// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"

namespace MegaMoeUrma {

// Pure host/device scheduling arithmetic. Compute groups contain at most M256
// rows of one expert, independently of the peer/CV communication boundaries.
constexpr uint32_t kComputeCores = kAic;
constexpr uint32_t kReduceWorkers = kLocalWorkers - kCommunicationWorkers;
constexpr uint32_t kComputeTileM = 256U;
constexpr uint32_t kComputeTileN = 256U;
// Keep matrix scheduling at M256; only the AIV1 epilogue works in M128
// slices. The W4 AIV0 prologue remains the largest UB user at exactly 192KiB.
constexpr uint32_t kActivationTileM = 128U;
// The URMA v1 input is H3584: each row has 112 E8M0 bytes. GM stays
// compact while DMA uses 32-byte-aligned rows in phase-private activation UB.
constexpr uint32_t kInputScaleBytes = 3584U / 32U;
constexpr uint32_t kInputScaleUbStride = ((kInputScaleBytes + 31U) / 32U) * 32U;
constexpr uint32_t kInputScaleUbBytes = kComputeTileM * kInputScaleUbStride;
constexpr uint32_t kRequiredUbBytes = 192U * 1024U;
struct ActivationUbContract {
    static constexpr uint32_t elements = kActivationTileM * kComputeTileN;
    static constexpr uint32_t gateOffset = 0U;
    static constexpr uint32_t upOffset = elements * sizeof(uint16_t);
    static constexpr uint32_t inputEnd = upOffset + elements * sizeof(uint16_t);
    // Gate/up are dead before quantization reuses their storage.
    static constexpr uint32_t activationOffset = 0U;
    static constexpr uint32_t quantOffset = elements * sizeof(uint16_t);
    static constexpr uint32_t scaleElements = elements / 32U;
    static constexpr uint32_t scaleOffset = quantOffset + elements;
    static constexpr uint32_t maxExpOffset = scaleOffset + scaleElements;
    static constexpr uint32_t reciprocalOffset = maxExpOffset + scaleElements * sizeof(uint16_t);
    static constexpr uint32_t scratchEnd = reciprocalOffset + scaleElements * sizeof(uint16_t);
};
static_assert(ActivationUbContract::inputEnd == 128U * 1024U);
static_assert(ActivationUbContract::scratchEnd == 101U * 1024U);
static_assert(ActivationUbContract::inputEnd <= kRequiredUbBytes &&
              ActivationUbContract::scratchEnd <= kRequiredUbBytes);
static_assert(kInputScaleUbBytes == 32768U && kInputScaleUbBytes <= ActivationUbContract::inputEnd);

struct ActivationRowSlice {
    uint32_t rowBegin;
    uint32_t rows;
};

enum class ComputeBatchStage : uint32_t { None = 0U, Gmm1 = 1U, Gmm2 = 2U };

struct ComputeBatchTask {
    ComputeBatchStage stage;
    uint32_t batchIndex;
};

struct ComputeBatchRange {
    uint32_t begin;
    uint32_t end;
};

#if defined(__NPU_DEVICE__)
#define MEGAMOE_URMA_COMPUTE_INLINE __aicore__ inline
#else
#define MEGAMOE_URMA_COMPUTE_INLINE inline
#endif

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t LocalWorkerIndex(uint32_t block, uint32_t subblock)
{
    return 2U * block + subblock;
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint64_t LocalOwnedRows(uint64_t rows, uint32_t worker)
{
    return worker < rows ? (rows - 1U - worker) / kLocalWorkers + 1U : 0U;
}

// Contiguous partitions preserve the old strided schedule's per-worker count,
// including empty workers and remainder rows. Pack can therefore DMA
// their route metadata in batches without changing timer payload accounting.
MEGAMOE_URMA_COMPUTE_INLINE constexpr uint64_t LocalRowBegin(uint64_t rows, uint32_t worker)
{
    const uint64_t remainder = rows % kLocalWorkers;
    return static_cast<uint64_t>(worker) * (rows / kLocalWorkers) + (worker < remainder ? worker : remainder);
}

// The 15 peer owners never reduce. All AIV0 and the 13 activation AIV1
// workers enter Reduce independently after releasing their compute resources.
// Return kReduceWorkers as the sentinel for a communication worker.
MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t ReduceWorkerIndex(uint32_t block, uint32_t subblock)
{
    return subblock == 0U ? block
                          : (block >= kCommunicationWorkers ? kAic + block - kCommunicationWorkers : kReduceWorkers);
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint64_t ReduceOwnedRows(uint64_t rows, uint32_t worker)
{
    return worker < kReduceWorkers && worker < rows ? (rows - 1U - worker) / kReduceWorkers + 1U : 0U;
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint64_t ReduceRowBegin(uint64_t rows, uint32_t worker)
{
    const uint64_t remainder = rows % kReduceWorkers;
    return static_cast<uint64_t>(worker) * (rows / kReduceWorkers) + (worker < remainder ? worker : remainder);
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr bool IsSelfExpandedRow(uint64_t row, const Config& config)
{
    return row >= config.selfSendRow && row - config.selfSendRow < config.selfRows;
}

// Call only for a row in the self interval. Stable expert ordering is identical
// on both sides, so the inverse transform also maps self GMM2 rows to Reduce.
MEGAMOE_URMA_COMPUTE_INLINE constexpr uint64_t SelfReceiveRow(uint64_t expandedRow, const Config& config)
{
    return config.selfRecvRow + (expandedRow - config.selfSendRow);
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint64_t SelfReturnRow(uint64_t receivedRow, const Config& config)
{
    return config.selfSendRow + (receivedRow - config.selfRecvRow);
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint64_t ComputeBatchTaskCount(uint32_t batchCount)
{
    return static_cast<uint64_t>(batchCount) * 2U;
}

// A batch contains all M256 groups of one expert, including an empty batch for
// an empty expert. Warmup G1(0), then G1(i) -> G2(i-1), then drain G2(last).
// Every AIC/AIV0 follows this order independently; there is no expert barrier.
MEGAMOE_URMA_COMPUTE_INLINE constexpr ComputeBatchTask ComputeBatchTaskAt(uint64_t step, uint32_t batchCount)
{
    const uint64_t taskCount = ComputeBatchTaskCount(batchCount);
    if (step >= taskCount) return {ComputeBatchStage::None, 0U};
    if (step == 0U) return {ComputeBatchStage::Gmm1, 0U};
    if (step == taskCount - 1U) return {ComputeBatchStage::Gmm2, batchCount - 1U};
    if ((step & 1U) != 0U) return {ComputeBatchStage::Gmm1, static_cast<uint32_t>((step + 1U) / 2U)};
    return {ComputeBatchStage::Gmm2, static_cast<uint32_t>(step / 2U - 1U)};
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr ComputeBatchRange ComputeBatchFragments(const ComputeBatch& batch)
{
    return {batch.fragmentBegin, batch.fragmentBegin + batch.fragmentCount};
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr ActivationRowSlice ActivationSlice(uint32_t tileRowBegin, uint32_t tileRows,
                                                                         uint32_t offset)
{
    const uint32_t remaining = offset < tileRows ? tileRows - offset : 0U;
    return {tileRowBegin + offset, remaining < kActivationTileM ? remaining : kActivationTileM};
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t ComputeTileCount(uint32_t rows, uint32_t columns)
{
    return ((rows + kComputeTileM - 1U) / kComputeTileM) * ((columns + kComputeTileN - 1U) / kComputeTileN);
}

// All stage task IDs are global prefix sums. Sparse experts cannot repeatedly
// give the same physical core their first tile. AIC and its AIV0 call this with
// identical inputs; the independent AIV1 activation team uses workerCount=13.
MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t FirstOwnedComputeTile(uint64_t tileBegin, uint32_t worker,
                                                                     uint32_t workerCount)
{
    const uint32_t startOwner = tileBegin % workerCount;
    return worker >= startOwner ? worker - startOwner : worker + workerCount - startOwner;
}

#undef MEGAMOE_URMA_COMPUTE_INLINE
}  // namespace MegaMoeUrma
