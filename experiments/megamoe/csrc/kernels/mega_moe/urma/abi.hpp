// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include <cstdint>

namespace MegaMoeUrma {
constexpr uint32_t kWorld = 16;
constexpr uint32_t kAic = 28;
constexpr uint32_t kCommunicationWorkers = 15;
constexpr uint32_t kActivationWorkers = 13;
constexpr uint64_t kFlagBytes = 128;
constexpr uint32_t kProtocol = 1;
constexpr uint32_t kTimerCores = kAic * 3U;
constexpr uint32_t kTimerPhases = 8U;
constexpr uint32_t kTimerRecordElements = 16U;
constexpr uint64_t kTimerElements = static_cast<uint64_t>(kTimerCores) * kTimerPhases * kTimerRecordElements;

// Detailed traces have independent, bounded single-writer slices per engine.
// A slice contains a 128B header followed by capacity 128B event records.
constexpr uint32_t kTraceSchema = 1U;
constexpr uint32_t kTraceRecordElements = 16U;
constexpr uint32_t kTraceHeaderElements = 16U;
constexpr uint32_t kTraceMaxCapacity = 65536U;
constexpr uint64_t kTraceUnknown = UINT64_MAX;

enum class TraceEvent : uint32_t {
    Pack = 0,
    DispatchPutX,
    DispatchPutScale,
    DispatchQuiet,
    DispatchPublish,
    SelfDispatch,
    WaitStartPeer,
    PublishStart,
    WaitLocalDispatch,
    WaitCombineReady,
    CombinePut,
    CombineQuiet,
    SelfCombine,
    FinalPeerHandshake,
    WaitDispatch,
    Gmm1Tile,
    Gmm1TileDrain,
    Gmm1Dequant,
    WaitActivation,
    Gmm2Tile,
    Gmm2Dequant,
    Gmm2Drain,
    WaitGmm1,
    ActivationSlice,
    RankBarrier,
    Reduce,
    LocalSync,
    Gmm1Drain
};

struct TraceMeta {
    uint64_t peer = kTraceUnknown;
    uint64_t cv = kTraceUnknown;
    uint64_t fragment = kTraceUnknown;
    uint64_t expert = kTraceUnknown;
    uint64_t row = kTraceUnknown;
    uint64_t rows = kTraceUnknown;
    uint64_t tileM = kTraceUnknown;
    uint64_t tileN = kTraceUnknown;
    uint64_t columns = kTraceUnknown;
    uint64_t bytes = 0U;
};

// One intersection of a peer CV interval and an expert's stable route stream.
struct Fragment {
    uint32_t expert;
    uint32_t sourceRank;
    uint64_t rowBegin;
    uint32_t rows;
    uint32_t inputReadyIndex;
    uint64_t gmm1TileBegin;
    uint64_t gmm2TileBegin;
};

// Send chunks use local expanded-row offsets. Receive chunks describe the
// corresponding source-major input and full-width GMM2 output interval.
struct Chunk {
    uint32_t peer;
    uint32_t cv;
    uint64_t localRow;
    uint64_t remoteRow;
    uint32_t rows;
    uint32_t fragmentBegin;
    uint32_t fragmentCount;
    uint32_t reserved;
};

struct Config {
    uint32_t bs;
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t topk;
    uint32_t localExperts;
    uint32_t cvRows;
    uint32_t rank;
    uint32_t maxPeerChunks;
    uint32_t fragmentCount;
    uint32_t sendChunkCount;
    uint32_t recvChunkCount;
    uint32_t reserved;
    uint64_t receiveRows;
    uint64_t gmm1Tiles;
    uint64_t gmm2Tiles;
    uint64_t epoch;
    uint64_t receiveXOffset;
    uint64_t receiveScaleOffset;
    uint64_t returnYOffset;
    uint64_t dispatchReadyOffset;
    uint64_t barrierOffset;
    uint64_t symmetricBytes;
};

// This POD is copied as kernel launch arguments; every address is device GM.
struct Arguments {
    Config config;
    uint64_t symmetric;
    uint64_t x;
    uint64_t xScale;
    uint64_t order;
    uint64_t reduceIndex;
    uint64_t weights;
    uint64_t weight1;
    uint64_t weight2;
    uint64_t weightScale1;
    uint64_t weightScale2;
    uint64_t output;
    uint64_t expandedX;
    uint64_t expandedScale;
    uint64_t gmm1Output;
    uint64_t activation;
    uint64_t activationScale;
    uint64_t gmm2Output;
    uint64_t fragments;
    uint64_t sendChunks;
    uint64_t recvChunks;
    uint64_t gmm1Done;
    uint64_t activationDone;
    uint64_t gmm2Done;
    uint64_t localFlags;
    uint64_t timer;
    uint64_t trace;
    uint64_t traceCapacity;
    uint64_t systemWorkspace;
    uint64_t ffts;
};
static_assert(sizeof(Fragment) == 40);
static_assert(sizeof(Chunk) == 40);
}  // namespace MegaMoeUrma

extern "C" void mega_moe_urma_kernel_do(uint32_t block_dim, uint32_t ub_bytes, void* stream,
                                        const MegaMoeUrma::Arguments* arguments);
