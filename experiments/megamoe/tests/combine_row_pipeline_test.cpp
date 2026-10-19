// Copyright (c) 2026, SimpleBright_Man
// Execute real row-ring functions. Copies read UB when the relevant engine
// advances, so removing a reuse dependency corrupts data rather than being
// hidden by an eager host memcpy.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define __aicore__
#define __gm__
#include "common/mega_moe_constants.hpp"

static void Require(bool condition)
{
    if (!condition) throw std::runtime_error("Combine row-ring regression failed");
}

namespace AscendC {
using bfloat16_t = uint16_t;
using fp8_e4m3fn_t = uint8_t;
using fp8_e5m2_t = uint8_t;
using TEventID = int;
constexpr int EVENT_ID0 = 0;
constexpr int AIC = 0;
constexpr int g_coreType = 1;
static uint32_t subBlock = 1U;
static uint32_t GetSubBlockIdx()
{
    return subBlock;
}
enum class TPosition { VECIN, VECCALC };
enum class HardEvent { MTE2_MTE3, MTE3_MTE2, MTE2_V, V_MTE3 };
struct Job {
    std::function<bool()> ready;
    std::function<void()> run;
};
struct Engine {
    std::vector<uint8_t> ub = std::vector<uint8_t>(256U * 1024U, 0xcd);
    std::array<std::vector<uint8_t>, 16> destinations;
    std::array<std::deque<Job>, 2> queue;
    std::array<std::array<uint32_t, 8>, 4> issued{};
    std::array<std::array<uint32_t, 8>, 4> consumed{};
    std::array<std::array<uint32_t, 8>, 4> signaled{};
    uint32_t copies = 0U;
    uint32_t writes = 0U;
    uint32_t overlap = 0U;
    uint32_t steps = 0U;
    uint32_t policy = 0U;

    bool Advance()
    {
        const uint32_t first = policy == 2U ? steps++ % 2U : policy;
        for (uint32_t pipe : {first, 1U - first}) {
            if (queue[pipe].empty() || !queue[pipe].front().ready()) continue;
            if (!queue[0].empty() && !queue[1].empty()) ++overlap;
            auto job = std::move(queue[pipe].front());
            queue[pipe].pop_front();
            job.run();
            return true;
        }
        return false;
    }
    void Finish()
    {
        while (!queue[0].empty() || !queue[1].empty()) Require(Advance());
        Require(issued == consumed && issued == signaled);
    }
};
static Engine* engine = nullptr;

template <typename T>
struct LocalTensor {
    uint8_t* bytes = nullptr;
    size_t length = 0U;
    LocalTensor() = default;
    LocalTensor(TPosition, uint32_t offset, uint32_t elements)
        : bytes(engine->ub.data() + offset), length(elements * sizeof(T))
    {
        Require(offset + length <= engine->ub.size());
    }
    LocalTensor operator[](uint64_t index) const
    {
        Require(index * sizeof(T) <= length);
        LocalTensor result;
        result.bytes = bytes + index * sizeof(T);
        result.length = length - index * sizeof(T);
        return result;
    }
    template <typename U>
    LocalTensor<U> ReinterpretCast() const
    {
        LocalTensor<U> result;
        result.bytes = bytes;
        result.length = length;
        return result;
    }
    T GetValue(uint64_t index) const
    {
        Require((index + 1U) * sizeof(T) <= length);
        T value;
        std::memcpy(&value, bytes + index * sizeof(T), sizeof(T));
        return value;
    }
    void SetValue(uint64_t index, T value)
    {
        Require((index + 1U) * sizeof(T) <= length);
        std::memcpy(bytes + index * sizeof(T), &value, sizeof(T));
    }
};
template <typename T>
struct GlobalTensor {
    uint8_t* bytes = nullptr;
    void SetGlobalBuffer(T* address)
    {
        bytes = reinterpret_cast<uint8_t*>(address);
    }
    GlobalTensor operator[](uint64_t index) const
    {
        return {bytes + index * sizeof(T)};
    }
};
struct DataCopyExtParams {
    uint16_t blockCount;
    uint32_t blockLen;
    uint32_t srcStride;
    uint32_t dstStride;
    uint32_t reserved;
};
template <typename T>
struct DataCopyPadExtParams {
    bool enabled;
    uint8_t left;
    uint8_t right;
    T padding;
};
template <typename T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams params, DataCopyPadExtParams<T> pad)
{
    Require(params.blockCount == 1U && params.blockLen <= dst.length && !pad.enabled);
    Require(params.srcStride == 0U && params.dstStride == 0U && params.reserved == 0U);
    ++engine->copies;
    engine->queue[0].push_back({[] { return true; }, [=] { std::memcpy(dst.bytes, src.bytes, params.blockLen); }});
}
template <typename T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams params)
{
    Require(params.blockCount == 1U && params.blockLen <= src.length);
    Require(params.srcStride == 0U && params.dstStride == 0U && params.reserved == 0U);
    ++engine->writes;
    engine->queue[1].push_back({[] { return true; }, [=] { std::memcpy(dst.bytes, src.bytes, params.blockLen); }});
}
template <HardEvent Event>
void SetFlag(TEventID id)
{
    constexpr size_t type = static_cast<size_t>(Event);
    constexpr size_t pipe = Event == HardEvent::MTE2_MTE3 ? 0U : 1U;
    Require(id >= 0 && id <= 7);
    const auto generation = ++engine->issued[type][id];
    engine->queue[pipe].push_back({[] { return true; }, [=] { engine->signaled[type][id] = generation; }});
}
template <HardEvent Event>
void WaitFlag(TEventID id)
{
    constexpr size_t type = static_cast<size_t>(Event);
    constexpr size_t pipe = Event == HardEvent::MTE2_MTE3 ? 1U : 0U;
    Require(id >= 0 && id <= 7);
    const auto generation = ++engine->consumed[type][id];
    Require(generation <= engine->issued[type][id]);
    engine->queue[pipe].push_back({[=] { return engine->signaled[type][id] >= generation; }, [] {}});
}
}  // namespace AscendC

namespace Ops::Base {
template <typename T>
T CeilDiv(T a, T b)
{
    return (a + b - 1U) / b;
}
template <typename T>
T CeilAlign(T a, T b)
{
    return CeilDiv(a, b) * b;
}
}  // namespace Ops::Base

namespace MegaMoeImpl {
using namespace AscendC;
using GM_ADDR = uint8_t*;
struct MoeStageCommonConfig {
    uint32_t tokenHiddenDim = 3584U;
    uint32_t tokenNum = 4096U;
};
struct Params {
    struct {
        GM_ADDR combineSendPtr;
        GM_ADDR rankSyncInWorldPtr;
    } peermemInfo;
    struct Tiling {
        uint32_t topK = 8U;
    }* tilingData;
};
struct AscendTimerDevice {
    void AuditTransfer(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t)
    {
        throw std::runtime_error("unexpected audit in slot test");
    }
};
namespace CombineImpl {
struct CombineTokenRoute {
    uint32_t dstRankId;
    uint32_t tokenIdx;
    uint32_t topkIdx;
};
static CombineTokenRoute LoadCombineTokenRoute(const LocalTensor<int32_t>& meta, uint32_t row)
{
    return {static_cast<uint32_t>(meta.GetValue(row * META_INFO_SIZE + RANK_ID)),
            static_cast<uint32_t>(meta.GetValue(row * META_INFO_SIZE + TOKEN_ID)),
            static_cast<uint32_t>(meta.GetValue(row * META_INFO_SIZE + TOPK_INDEX))};
}
template <typename T>
void SendCombineTokenRow(uint32_t elements, uint64_t remoteOffset, LocalTensor<int32_t>& meta, LocalTensor<T>& row,
                         const Params& params)
{
    const auto route = LoadCombineTokenRoute(meta, 0U);
    const uint64_t destination = static_cast<uint64_t>(route.tokenIdx) * params.tilingData->topK + route.topkIdx;
    auto& memory = engine->destinations.at(route.dstRankId);
    Require(remoteOffset + (destination + 1U) * elements * sizeof(T) <= memory.size());
    GlobalTensor<T> target;
    target.SetGlobalBuffer(reinterpret_cast<T*>(memory.data() + remoteOffset));
    DataCopyPad(target[destination * elements], row, {1U, elements * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U});
}
}  // namespace CombineImpl
namespace Mxfp8 {
template <uint8_t Mode, typename T>
void QuantMxFp8(LocalTensor<T>, LocalTensor<T>, LocalTensor<float>, uint32_t);
}

// INSERT_PRODUCTION_DECLARATIONS

// INSERT_PRODUCTION_FUNCTIONS

}  // namespace MegaMoeImpl

using namespace MegaMoeImpl;

static void CheckBuffers(uint32_t hidden, uint32_t count, uint32_t bs)
{
    Engine state;
    engine = &state;
    WaveCombineScratch scratch;
    scratch.rowBufferCountOverride = count;
    MoeStageCommonConfig common{hidden, bs};
    const auto buffers = InitTokenCombineSliceBuffers(common, scratch);
    const uint32_t actual = count != 0U ? count : bs <= W4A8_SMALL_BATCH_MAX_TOKENS ? 6U : 2U;
    Require(buffers.rowBufferCount == actual);
    Require(buffers.rowBytes == hidden * 2U);
    Require(buffers.rowStrideBytes == (hidden * 2U + 31U) / 32U * 32U);
    Require(buffers.slotStrideBytes == buffers.rowStrideBytes);
    Require(scratch.rowBufferTensor.bytes == state.ub.data() + WAVE_COMBINE_UB_BASE);
    Require(scratch.rowBufferTensor.length == actual * buffers.slotStrideBytes);
    Require(WAVE_COMBINE_UB_BASE + scratch.rowBufferTensor.length <= WAVE_COMBINE_UB_LIMIT);
    Require(scratch.metaInfoTensor.bytes == state.ub.data() + 200U * 1024U);
    Require(scratch.metaInfoTensor.length == 48U * 1024U);
    subBlock = 0U;
    Require(InitTokenCombineSliceBuffers(common, scratch).rowBufferCount == 0U);
    subBlock = 1U;
}

static void CheckPipeline(uint32_t hidden, uint32_t slots, uint32_t policy)
{
    Engine state;
    state.policy = policy;
    engine = &state;
    MoeStageCommonConfig common{hidden, 4096U};
    WaveCombineScratch scratch;
    scratch.rowBufferCountOverride = slots;
    const auto buffers = InitTokenCombineSliceBuffers(common, scratch);
    constexpr uint32_t rows = 43U;
    const uint32_t rowBytes = hidden * sizeof(bfloat16_t);
    std::vector<uint8_t> input(rows * rowBytes);
    for (uint32_t row = 0U; row < rows; ++row) {
        for (uint32_t byte = 0U; byte < rowBytes; ++byte)
            input[row * rowBytes + byte] = static_cast<uint8_t>(row * 29U + byte * 7U);
        scratch.metaInfoTensor.SetValue(row * META_INFO_SIZE + RANK_ID, static_cast<int32_t>((row * 7U + 3U) % 16U));
        scratch.metaInfoTensor.SetValue(row * META_INFO_SIZE + TOKEN_ID, static_cast<int32_t>(row / 8U));
        scratch.metaInfoTensor.SetValue(row * META_INFO_SIZE + TOPK_INDEX, static_cast<int32_t>(row % 8U));
    }
    for (auto& peer : state.destinations) peer.assign(rows * rowBytes, 0xa5);
    std::array<uint8_t, 1> peerBase{};
    Params::Tiling tiling;
    Params params{{peerBase.data(), peerBase.data()}, &tiling};
    // Multiple slices share the UB but each drains and restarts. Segment cuts
    // deliberately fall inside the first-use region and on ring wraparound.
    for (uint32_t total : {0U, 1U, slots - 1U, slots, slots + 1U, 29U, rows}) {
        for (auto& peer : state.destinations) std::fill(peer.begin(), peer.end(), 0xa5);
        uint32_t sequence = 0U, done = 0U;
        for (uint32_t chunk : {0U, 1U, 2U, 7U, 3U, rows}) {
            const uint32_t take = std::min(chunk, total - done);
            CombineWaveTokenRange<COMBINE_NO_QUANT>(common, buffers, scratch, params, input.data(), done, take, done,
                                                    sequence);
            done += take;
            Require(sequence == done);
        }
        Require(done == total);
        const auto issuedBeforeDrain = state.issued;
        const auto waitedBeforeDrain = state.consumed;
        DrainCombineRowBuffers(sequence, slots);
        Require(sequence == 0U && state.issued == issuedBeforeDrain);
        for (uint32_t id = 0U; id < 8U; ++id) {
            const uint32_t expected = id < std::min(total, slots) ? 1U : 0U;
            Require(state.consumed[1][id] == waitedBeforeDrain[1][id] + expected);
        }
        state.Finish();
        for (uint32_t row = 0U; row < total; ++row) {
            const auto& peer = state.destinations[(row * 7U + 3U) % 16U];
            Require(std::memcmp(peer.data() + row * rowBytes, input.data() + row * rowBytes, rowBytes) == 0);
        }
        // Ready-scan storage immediately follows the allowed row ring and must
        // remain untouched even when all eight slots are live.
        Require(std::all_of(state.ub.begin() + WAVE_COMBINE_UB_LIMIT, state.ub.begin() + META_INFO_TENSOR_ADDR,
                            [](uint8_t value) { return value == 0xcd; }));
    }
    Require(state.copies == state.writes && state.overlap > 0U);
    Require(state.issued[0][slots - 1U] > 0U && state.issued[1][slots - 1U] > 0U);
    for (uint32_t id = slots; id < 8U; ++id) Require(state.issued[0][id] == 0U && state.issued[1][id] == 0U);
}

int main()
{
    static_assert(WAVE_COMBINE_STEADY_ROW_BUFFER_COUNT == 1U, "steady-wave policy must remain unchanged");
    static_assert(WAVE_COMBINE_MAX_ROW_BUFFER_COUNT == 6U, "production defaults must remain at six");
    static_assert(WAVE_COMBINE_REPLAY_MAX_ROW_BUFFER_COUNT == 8U, "replay must support event slots 0..7");
    Require(CombineReplayRowRingFits(3584U, 8U));
    Require(CombineReplayRowRingFits(6144U, 8U));
    Require(CombineReplayRowRingFits(7680U, 8U));
    Require(!CombineReplayRowRingFits(7681U, 8U));
    Require(!CombineReplayRowRingFits(8192U, 8U));
    Require(CombineReplayRowRingFits(8192U, 6U));
    Require(!CombineReplayRowRingFits(UINT32_MAX, 8U));
    for (uint32_t hidden : {3584U, 6144U, 7680U})
        for (uint32_t slots : {2U, 4U, 6U, 8U}) {
            CheckBuffers(hidden, slots, 4096U);
            for (uint32_t policy = 0U; policy < 3U; ++policy) CheckPipeline(hidden, slots, policy);
        }
    for (uint32_t slots : {2U, 4U, 6U}) CheckBuffers(8192U, slots, 4096U);
    for (uint32_t bs : {1U, 256U, 257U, 2048U, 4096U, 8192U}) CheckBuffers(3584U, 0U, bs);
    std::cout << "Combine 2/4/6/8 slots: deferred DMA, events 0..7, peer segments and tail drains passed\n";
    std::cout << "UB: H3584/6144/7680 fit, H8192 eight-slot rejection, original Decode6/Prefill2 passed\n";
}
