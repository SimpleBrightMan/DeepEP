// Copyright (c) 2026, SimpleBright_Man
// Execute the real DMA helper: copies read their source only when their engine
// runs, so an unsafe UB overwrite cannot be hidden by an eager memcpy mock.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#define __aicore__
constexpr int INT32_PER_256B = 8;

static void Require(bool ok)
{
    if (!ok) throw std::runtime_error("metadata DMA regression failed");
}

namespace AscendC {
using TEventID = int;
enum class HardEvent { MTE2_MTE3, MTE3_MTE2, MTE2_S, MTE3_S };
struct Memory {
    std::vector<uint8_t> bytes;
    explicit Memory(size_t size) : bytes(size, 0xcd) {}
};
struct Job {
    std::function<bool()> ready;
    std::function<void()> execute;
};
struct EngineModel {
    std::array<std::deque<Job>, 2> queues;
    std::array<std::array<int, 8>, 4> issued{};
    std::array<std::array<int, 8>, 4> waited{};
    std::array<std::array<int, 8>, 4> signaled{};
    int scalarWaits = 0;
    int completedCopies = 0;
    int copiesIssued = 0;
    int overlappingQueues = 0;
    int policy = 0;
    int step = 0;

    bool Advance()
    {
        const int first = policy == 0 ? 0 : policy == 1 ? 1 : step++ % 2;
        for (int engine : {first, 1 - first}) {
            auto& queue = queues[engine];
            if (queue.empty() || !queue.front().ready()) continue;
            if (!queues[0].empty() && !queues[1].empty()) ++overlappingQueues;
            auto job = std::move(queue.front());
            queue.pop_front();
            job.execute();
            return true;
        }
        return false;
    }
    void Finish()
    {
        while (!queues[0].empty() || !queues[1].empty()) Require(Advance());
        Require(issued == waited);
        Require(issued == signaled);
    }
};
static EngineModel* model = nullptr;

template <typename T, bool Local>
struct Tensor {
    Memory* memory;
    size_t offset = 0;
    Tensor operator[](uint64_t index) const
    {
        return {memory, offset + index * sizeof(T)};
    }
    template <typename U>
    Tensor<U, Local> ReinterpretCast() const
    {
        return {memory, offset};
    }
    T GetValue(uint64_t index) const
    {
        T value;
        Require(offset + (index + 1) * sizeof(T) <= memory->bytes.size());
        std::memcpy(&value, memory->bytes.data() + offset + index * sizeof(T), sizeof(T));
        return value;
    }
};
template <typename T>
using LocalTensor = Tensor<T, true>;
template <typename T>
using GlobalTensor = Tensor<T, false>;

static void EnqueueCopy(int engine, Memory* dst, size_t dstOffset, Memory* src, size_t srcOffset, size_t bytes)
{
    Require(dstOffset + bytes <= dst->bytes.size());
    Require(srcOffset + bytes <= src->bytes.size());
    ++model->copiesIssued;
    model->queues[engine].push_back({[] { return true; },
                                     [=] {
                                         std::memcpy(dst->bytes.data() + dstOffset, src->bytes.data() + srcOffset,
                                                     bytes);
                                         ++model->completedCopies;
                                     }});
}
template <typename T, bool DstLocal, bool SrcLocal>
void DataCopy(Tensor<T, DstLocal> dst, Tensor<T, SrcLocal> src, uint64_t count)
{
    static_assert(DstLocal != SrcLocal);
    EnqueueCopy(DstLocal ? 0 : 1, dst.memory, dst.offset, src.memory, src.offset, count * sizeof(T));
}
struct CopyParams {
    uint16_t count;
    uint32_t bytes;
    uint32_t srcStride;
    uint32_t dstStride;
    uint32_t reserved;
};
template <typename T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, CopyParams params)
{
    Require(params.count == 1 && params.srcStride == 0 && params.dstStride == 0 && params.reserved == 0);
    EnqueueCopy(1, dst.memory, dst.offset, src.memory, src.offset, params.bytes);
}
template <HardEvent Event>
void SetFlag(TEventID id)
{
    constexpr int type = static_cast<int>(Event);
    constexpr int source = Event == HardEvent::MTE2_MTE3 || Event == HardEvent::MTE2_S ? 0 : 1;
    Require(id >= 0 && id < 8);
    const int generation = ++model->issued[type][id];
    model->queues[source].push_back({[] { return true; }, [=] { model->signaled[type][id] = generation; }});
}
template <HardEvent Event>
void WaitFlag(TEventID id)
{
    constexpr int type = static_cast<int>(Event);
    const int generation = ++model->waited[type][id];
    Require(generation <= model->issued[type][id]);
    auto ready = [=] { return model->signaled[type][id] >= generation; };
    if constexpr (Event == HardEvent::MTE2_S || Event == HardEvent::MTE3_S) {
        ++model->scalarWaits;
        while (!ready()) Require(model->Advance());
    } else {
        constexpr int target = Event == HardEvent::MTE2_MTE3 ? 1 : 0;
        model->queues[target].push_back({ready, [] {}});
    }
}
}  // namespace AscendC

#include "stage/mega_moe_dispatch_metadata_dma.hpp"

static void CheckCase(int slots, int hidden, int workers, int policy)
{
    using namespace AscendC;
    using namespace MegaMoeImpl;
    constexpr int peers = 16;
    constexpr int tokens = 17;
    constexpr int topk = 8;
    constexpr int batchRows = 5;
    const int scaleBytes = hidden / 32;
    const int alignedScaleBytes = (scaleBytes + 31) / 32 * 32;
    const int rowBytes = hidden + alignedScaleBytes;
    const std::array<int, 7> sliceRows{0, 1, slots - 1, slots, slots + 1, 19, 37};
    int totalRows = 0;
    for (int rows : sliceRows) totalRows += rows;
    EngineModel engine;
    engine.policy = policy;
    model = &engine;
    std::vector<Memory> sources;
    for (int peer = 0; peer < peers; ++peer) {
        sources.emplace_back(tokens * rowBytes);
        for (int token = 0; token < tokens; ++token) {
            for (int byte = 0; byte < rowBytes; ++byte) {
                sources.back().bytes[token * rowBytes + byte] = static_cast<uint8_t>(peer * 73 + token * 11 + byte * 3);
            }
        }
    }
    Memory metadata(peers * tokens * topk * 32);
    for (int peer = 0; peer < peers; ++peer) {
        for (int token = 0; token < tokens; ++token) {
            for (int slot = 0; slot < topk; ++slot) {
                const std::array<int32_t, 8> values{
                    peer, token, slot, 0x12345678, peer ^ token, -slot - 1, token * 8191 + slot, 0x31415926};
                const int row = (peer * tokens + token) * topk + slot;
                std::memcpy(metadata.bytes.data() + row * 32, values.data(), 32);
            }
        }
    }
    Memory tokenUb(slots * rowBytes), metaUb(slots * 32);
    Memory tokenOut(totalRows * hidden), scaleOut(totalRows * scaleBytes), metaOut(totalRows * 32);
    Memory expectedTokens(tokenOut.bytes.size()), expectedScales(scaleOut.bytes.size()),
        expectedMeta(metaOut.bytes.size());
    Memory routeGm(batchRows * 4), routeUb(batchRows * 4);
    GlobalTensor<uint8_t> dataDst{&tokenOut}, scalesDst{&scaleOut};
    GlobalTensor<int32_t> metadataSrc{&metadata}, metadataDst{&metaOut};
    int rowBase = 0;
    int slice = 0;
    int routeWaits = 0;
    int sliceWaits = 0;
    for (int rows : sliceRows) {
        int issued = 0;
        int pendingDst = -1;
        auto store = [&](int slot, int dst) {
            const int scalarBefore = engine.scalarWaits;
            StoreDispatchTokenWithDmaMetadata<uint8_t, uint8_t>(
                dataDst, scalesDst, metadataDst, LocalTensor<uint8_t>{&tokenUb, static_cast<size_t>(slot * rowBytes)},
                LocalTensor<int32_t>{&metaUb, static_cast<size_t>(slot * 32)}, hidden, hidden, scaleBytes, dst, slot);
            Require(engine.scalarWaits == scalarBefore);
        };
        for (int batch = 0; batch < rows; batch += batchRows) {
            const int count = std::min(batchRows, rows - batch);
            for (int i = 0; i < count; ++i) {
                const int row = batch + i;
                const int token = (row * 7 + slice * 3) % tokens;
                const int slot = (row + workers + slice) % topk;
                const int32_t topkIndex = token * topk + slot;
                std::memcpy(routeGm.bytes.data() + i * 4, &topkIndex, 4);
            }
            // Existing route batches synchronize Scalar through event 6. They
            // can occur with a previous peer's payload/metadata still pending.
            DataCopy(LocalTensor<int32_t>{&routeUb}, GlobalTensor<int32_t>{&routeGm}, count);
            SetFlag<HardEvent::MTE2_S>(6);
            WaitFlag<HardEvent::MTE2_S>(6);
            ++routeWaits;
            for (int i = 0; i < count; ++i) {
                const int row = batch + i;
                const int peer = (row * 5 + workers + slice) % peers;
                const int topkIndex = LocalTensor<int32_t>{&routeUb}.GetValue(i);
                const int token = topkIndex / topk;
                const int metadataRow = peer * tokens * topk + topkIndex;
                // Alternate row order models different core ownership/rotation
                // without permitting the test to assume contiguous output.
                const int dst = rowBase + (workers % 2 == 0 ? rows - 1 - row : row);
                if (slots == 1 && pendingDst >= 0) {
                    store(0, pendingDst);
                    pendingDst = -1;
                }
                const int slot = issued % slots;
                const int scalarBefore = engine.scalarWaits;
                const int copiesBefore = engine.completedCopies;
                const LocalTensor<uint8_t> copy{&tokenUb, static_cast<size_t>(slot * rowBytes)};
                const LocalTensor<int32_t> meta{&metaUb, static_cast<size_t>(slot * 32)};
                const GlobalTensor<uint8_t> remote{&sources[peer]};
                if (issued < slots) {
                    FetchDispatchTokenWithDmaMetadata<false>(copy, remote, token * rowBytes, rowBytes, meta,
                                                             metadataSrc, metadataRow, slot);
                } else {
                    FetchDispatchTokenWithDmaMetadata<true>(copy, remote, token * rowBytes, rowBytes, meta, metadataSrc,
                                                            metadataRow, slot);
                }
                Require(engine.scalarWaits == scalarBefore);
                Require(engine.completedCopies == copiesBefore);
                if (pendingDst >= 0) store((issued - 1) % slots, pendingDst);
                pendingDst = dst;
                ++issued;
                std::memcpy(expectedTokens.bytes.data() + dst * hidden, sources[peer].bytes.data() + token * rowBytes,
                            hidden);
                std::memcpy(expectedScales.bytes.data() + dst * scaleBytes,
                            sources[peer].bytes.data() + token * rowBytes + hidden, scaleBytes);
                std::memcpy(expectedMeta.bytes.data() + dst * 32, metadata.bytes.data() + metadataRow * 32, 32);
            }
        }
        if (pendingDst >= 0) store((issued - 1) % slots, pendingDst);
        const int scalarBefore = engine.scalarWaits;
        const int copiesBefore = engine.completedCopies;
        for (int slot = 0; slot < std::min(issued, slots); ++slot) DrainDispatchMetadataDmaSlot(slot);
        Require(engine.scalarWaits == scalarBefore);
        Require(engine.completedCopies == copiesBefore);
        if (rows > 0) {
            // The production ready-publication boundary still waits MTE3_S.
            // Slot drain itself only enqueues reuse dependencies on MTE2.
            SetFlag<HardEvent::MTE3_S>(5);
            WaitFlag<HardEvent::MTE3_S>(5);
            ++sliceWaits;
        }
        Require(tokenOut.bytes == expectedTokens.bytes);
        Require(scaleOut.bytes == expectedScales.bytes);
        Require(metaOut.bytes == expectedMeta.bytes);
        rowBase += rows;
        ++slice;
    }
    engine.Finish();
    Require(engine.scalarWaits == routeWaits + sliceWaits);
    Require(engine.copiesIssued == totalRows * 5 + routeWaits);
    Require(engine.completedCopies == engine.copiesIssued);
    Require(engine.overlappingQueues > 0);
}

// Prove the model detects the original class of problem: without the reuse
// event, the next MTE2 metadata read overwrites the slot before MTE3 consumes it.
static void CheckMissingReuseNegativeControl()
{
    using namespace AscendC;
    using namespace MegaMoeImpl;
    EngineModel engine;
    model = &engine;
    Memory remote(64), metadata(64), copy(32), metaUb(32), dataOut(32), scalesOut(32), metadataOut(64);
    std::fill_n(metadata.bytes.begin(), 32, uint8_t(0x11));
    std::fill_n(metadata.bytes.begin() + 32, 32, uint8_t(0x22));
    for (int row = 0; row < 2; ++row) {
        FetchDispatchTokenWithDmaMetadata<false>(LocalTensor<uint8_t>{&copy}, GlobalTensor<uint8_t>{&remote}, row * 32,
                                                 32, LocalTensor<int32_t>{&metaUb}, GlobalTensor<int32_t>{&metadata},
                                                 row, 0);
        StoreDispatchTokenWithDmaMetadata<uint8_t, uint8_t>(
            GlobalTensor<uint8_t>{&dataOut}, GlobalTensor<uint8_t>{&scalesOut}, GlobalTensor<int32_t>{&metadataOut},
            LocalTensor<uint8_t>{&copy}, LocalTensor<int32_t>{&metaUb}, 16, 16, 16, row, 0);
    }
    // Deliberately broken caller consumes the two reuse events only after both
    // reads; prioritize MTE2 so the first output receives the second metadata.
    DrainDispatchMetadataDmaSlot(0);
    DrainDispatchMetadataDmaSlot(0);
    engine.Finish();
    Require(metadataOut.bytes[0] == 0x22);
    Require(metadataOut.bytes[0] != metadata.bytes[0]);
}

int main()
{
    CheckMissingReuseNegativeControl();
    for (int slots : {1, 2, 6}) {
        for (int hidden : {3584, 3712, 8192}) {
            for (int workers : {1, 7, 28}) {
                for (int policy : {0, 1, 2}) CheckCase(slots, hidden, workers, policy);
            }
        }
    }
    std::cout << "metadata DMA: 1/2/6 slots, peers, route batches, slice drains and full rows passed\n";
}
