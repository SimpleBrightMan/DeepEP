// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"

namespace MegaMoeUrma {

// Pure host/device scheduling arithmetic. Fragments retain the peer/CV row
// order; an expert boundary always starts a new fragment, not a new Wave.
constexpr uint32_t kComputeCores = kAic;
constexpr uint32_t kComputeTileM = 256U;
constexpr uint32_t kComputeTileN = 256U;
// Keep matrix scheduling at M256; only the AIV1 epilogue works in M128
// slices. The W4 AIV0 prologue remains the largest UB user at exactly 192KiB.
constexpr uint32_t kActivationTileM = 128U;
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

struct ActivationRowSlice {
    uint32_t rowBegin;
    uint32_t rows;
};

enum class ComputeBatchStage : uint32_t { None = 0U, Gmm1 = 1U, Gmm2 = 2U };

struct ComputeBatchTask {
    ComputeBatchStage stage;
    uint32_t chunkIndex;
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

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint64_t ComputeBatchTaskCount(uint32_t chunkCount)
{
    return static_cast<uint64_t>(chunkCount) * 2U;
}

// A batch is one existing receive chunk, uniquely identified by (peer, cv).
// Warmup G1(0), then G1(i) -> G2(i-1), then drain G2(last). No new communication
// boundary, expert split, or task ownership is introduced by this ordering.
MEGAMOE_URMA_COMPUTE_INLINE constexpr ComputeBatchTask ComputeBatchTaskAt(uint64_t step, uint32_t chunkCount)
{
    const uint64_t taskCount = ComputeBatchTaskCount(chunkCount);
    if (step >= taskCount) return {ComputeBatchStage::None, 0U};
    if (step == 0U) return {ComputeBatchStage::Gmm1, 0U};
    if (step == taskCount - 1U) return {ComputeBatchStage::Gmm2, chunkCount - 1U};
    if ((step & 1U) != 0U) return {ComputeBatchStage::Gmm1, static_cast<uint32_t>((step + 1U) / 2U)};
    return {ComputeBatchStage::Gmm2, static_cast<uint32_t>(step / 2U - 1U)};
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr ComputeBatchRange ComputeBatchFragments(const Chunk& chunk)
{
    return {chunk.fragmentBegin, chunk.fragmentBegin + chunk.fragmentCount};
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
