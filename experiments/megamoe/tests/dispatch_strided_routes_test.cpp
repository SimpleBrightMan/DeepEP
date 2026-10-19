// Copyright (c) 2026, SimpleBright_Man
// Production route functions are inserted by the Python harness. Route DMA
// stays deferred until the real MTE2_S fence; token slot mocks retain the
// captured identity after the shared index buffer is reused for the next peer.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

#define __aicore__
#define __gm__
static void Require(bool condition)
{
    if (!condition) throw std::runtime_error("Dispatch stride regression failed");
}
constexpr uint32_t MAX_DISPATCH_BUFFER_COUNT = 6U;
constexpr uint32_t ALIGN_32 = 32U;
using GM_ADDR = uint8_t*;
using TEventID = uint32_t;
using Transfer = std::tuple<int32_t, int32_t, int32_t>;
struct Slot {
    bool initialized = false;
    bool pending = false;
    int32_t peer = -1;
    int32_t topk = -1;
};
struct Model {
    std::vector<std::function<void()>> copies;
    std::vector<Transfer> output;
    std::array<Slot, 6> slots;
    uint32_t routeWaits = 0U;
    uint32_t slotDrains = 0U;
};
static Model* model = nullptr;
namespace AscendC {
enum class HardEvent { MTE2_S, MTE3_MTE2, MTE3_S };
template <HardEvent Event>
void WaitFlag(TEventID)
{
    // Scalar metadata mode is deliberately not selected by this fixture.
    static_assert(Event != HardEvent::MTE2_S);
    throw std::runtime_error("unexpected scalar metadata event");
}
}  // namespace AscendC
using AscendC::WaitFlag;
template <AscendC::HardEvent Event, uint32_t Id>
void SyncFuncStatic()
{
    static_assert(Event == AscendC::HardEvent::MTE2_S && Id == MAX_DISPATCH_BUFFER_COUNT);
    for (const auto& copy : model->copies) copy();
    model->copies.clear();
    ++model->routeWaits;
}
namespace Ops::Base {
template <typename T>
T CeilAlign(T value, T alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}
}  // namespace Ops::Base
template <typename T>
struct LocalTensor {
    T* data;
    uint32_t capacity;
    LocalTensor operator[](uint32_t offset) const
    {
        Require(offset <= capacity);
        return {data + offset, capacity - offset};
    }
    T GetValue(uint32_t offset) const
    {
        Require(model->copies.empty() && offset < capacity);
        return data[offset];
    }
};
template <typename T>
struct GlobalTensor {
    T* data = nullptr;
    void SetGlobalBuffer(T* pointer)
    {
        data = pointer;
    }
};
struct CopyParams {
    uint16_t blocks;
    uint32_t bytes;
    uint32_t srcStride;
    uint32_t dstStride;
    uint32_t reserved;
};
struct PadParams {
    bool enabled;
    uint8_t left;
    uint8_t right;
    uint8_t padding;
};
template <typename T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, CopyParams copy, PadParams pad)
{
    Require(copy.blocks == 1U && copy.bytes <= dst.capacity * sizeof(T));
    Require(copy.srcStride == 0U && copy.dstStride == 0U && !pad.enabled);
    model->copies.push_back([=] { std::memcpy(dst.data, src.data, copy.bytes); });
}
struct MegaMoeDispatchBufferConfig {
    int32_t bufferCount;
    int32_t routeItemsPerBatch;
};
struct TokenDispatchConfig {
    MegaMoeDispatchBufferConfig bufferConfig;
    uint64_t quantWinOffset = 0U;
    uint64_t routeIndexAlignSize;
};
struct MoeStageCommonConfig {
    uint32_t worldSize = 16U;
};
struct Params {
    struct Tiling {
        uint32_t topK = 8U;
    }* tilingData;
    struct {
        GM_ADDR maskRecvPtr;
    } peermemInfo;
};
struct Audit {
    void AuditTransfer(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t)
    {
        throw std::runtime_error("unexpected audit");
    }
};
template <typename Activation>
struct TokenDispatchScratch {
    LocalTensor<int32_t> validTopkIndexTensor;
    Audit* auditTimer = nullptr;
    uint32_t auditExpert = 0U;
    uint32_t auditExpertGlobalRow = 0U;
    GM_ADDR dmaMetadataInput = reinterpret_cast<GM_ADDR>(uintptr_t{1});
};
template <bool Reuse, bool Prefetch, typename Activation>
void FetchDispatchTokenAndMetaInfo(const TokenDispatchConfig&, const Params&, TokenDispatchScratch<Activation>&,
                                   int32_t index, int32_t topk, int32_t peer, GlobalTensor<Activation>&)
{
    static_assert(!Prefetch);
    auto& slot = model->slots.at(index);
    Require(!slot.pending && slot.initialized == Reuse);
    slot.initialized = true;
    slot.pending = true;
    slot.peer = peer;
    slot.topk = topk;
}
template <typename Activation, typename Scale, bool Prefetch>
void StoreDispatchTokenAndMetaInfo(const TokenDispatchConfig&, const Params&, TokenDispatchScratch<Activation>&,
                                   int32_t index, int32_t row)
{
    static_assert(!Prefetch);
    auto& slot = model->slots.at(index);
    Require(slot.pending);
    model->output.emplace_back(row, slot.peer, slot.topk);
    slot.pending = false;
}
static void DrainDispatchMetadataDmaSlot(int32_t index)
{
    Require(!model->slots.at(index).pending && model->slots.at(index).initialized);
    ++model->slotDrains;
}

// INSERT_ROUTE_DECLARATIONS

// INSERT_ROUTE_FUNCTIONS

static void Check(uint32_t stride, uint32_t capacity, uint32_t slots)
{
    Model state;
    model = &state;
    constexpr uint32_t routeLength = 4096U;
    constexpr uint32_t expert = 1U;
    constexpr uint32_t globalRow = 37U;
    std::vector<int32_t> routes(2U * 16U * routeLength);
    for (uint32_t i = 0U; i < routes.size(); ++i) routes[i] = static_cast<int32_t>((i * 13U + 7U) % 32768U);
    std::vector<int32_t> localRoutes(capacity, -1);
    TokenDispatchConfig context{
        {static_cast<int32_t>(slots), static_cast<int32_t>(capacity)}, 0U, routeLength * sizeof(int32_t)};
    Params::Tiling tiling;
    Params params{&tiling, {reinterpret_cast<GM_ADDR>(routes.data())}};
    TokenDispatchScratch<uint8_t> scratch{{localRoutes.data(), capacity}};
    std::array<uint8_t, 16> unusedTokens{};
    std::array<GM_ADDR, 16> windows{};
    for (uint32_t peer = 0U; peer < 16U; ++peer) windows[peer] = &unusedTokens[peer];
    DispatchRouteBatch batch{};
    DispatchCopyProgress progress{};
    std::vector<Transfer> expected;
    uint32_t destination = 5U;
    for (uint32_t i = 0U; i < 15U; ++i) {
        const uint32_t count = std::array<uint32_t, 5>{0U, 1U, 5U, 17U, 101U}[i % 5U];
        const uint32_t peer = (i * 7U + 2U) % 16U;
        const uint32_t source = i + 3U;
        const uint32_t actualStride = i % 3U == 0U ? 1U : stride;
        const uint32_t end = count == 0U ? source : source + (count - 1U) * actualStride + 1U;
        const auto copied =
            DispatchRankTokens<uint8_t, uint8_t, false>(context, {}, params, windows.data(), scratch, expert, globalRow,
                                                        peer, destination, source, end, batch, progress, actualStride);
        Require(copied == static_cast<int32_t>(count));
        for (uint32_t item = 0U; item < count; ++item) {
            const auto index = (expert * 16U + peer) * routeLength + source + item * actualStride;
            expected.emplace_back(globalRow + destination + item * actualStride, peer, routes[index]);
        }
        destination += count * actualStride + 11U;
    }
    FlushDispatchRouteBatch<uint8_t, uint8_t, false>(context, params, windows.data(), scratch, batch, progress);
    DrainDispatchCopyBuffers<uint8_t, uint8_t, false>(context, params, scratch, progress);
    Require(state.output == expected);
    Require(progress.issuedRows == expected.size());
    Require(state.slotDrains == slots && state.routeWaits > 1U);
    Require(batch.count == 0U && batch.usedItems == 0U && state.copies.empty());
}

int main()
{
    for (uint32_t stride = 1U; stride <= 27U; ++stride)
        for (uint32_t capacity : {8U, 16U, 256U, 2048U})
            for (uint32_t slots : {1U, 2U, 6U}) Check(stride, capacity, slots);
    std::cout << "Dispatch actual route batching: strides 1..27, bounded spans, holes, peers, partial tails passed\n";
}
