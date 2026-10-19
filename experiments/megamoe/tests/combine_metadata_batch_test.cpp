// Copyright (c) 2026, SimpleBright_Man
// Production functions are inserted by the Python harness. Metadata copies
// remain invisible until the real MTE2_S wait, detecting premature route reads.
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __gm__
#include "common/mega_moe_comm_split.hpp"

static void Require(bool ok)
{
    if (!ok) throw std::runtime_error("Combine metadata batching regression failed");
}

namespace AscendC {
enum class HardEvent { MTE2_S };
constexpr int EVENT_ID0 = 0;
struct Model {
    std::vector<std::function<void()>> copies;
    std::vector<uint32_t> sent;
    uint32_t slots = 6U;
    uint32_t sets = 0U;
    uint32_t waits = 0U;
    uint32_t drains = 0U;
    uint32_t starts = 0U;
    uint32_t ends = 0U;
    uint32_t clears = 0U;
    uint32_t copiesIssued = 0U;
    uint32_t rowsAtSet = 0U;
    uint32_t expectedRows = 0U;
    uint64_t expertGlobal = 0U;
};
static Model* model = nullptr;

template <typename T>
struct LocalTensor {
    T* base;
    size_t capacity;
    size_t offset = 0U;
    LocalTensor operator[](uint64_t index) const
    {
        Require(offset + index <= capacity);
        return {base, capacity, offset + static_cast<size_t>(index)};
    }
    T GetValue(uint64_t index) const
    {
        Require(model->copies.empty());
        Require(offset + index < capacity);
        return base[offset + index];
    }
};
template <typename T>
struct GlobalTensor {
    T* base = nullptr;
    size_t offset = 0U;
    void SetGlobalBuffer(T* pointer)
    {
        base = pointer;
    }
    GlobalTensor operator[](uint64_t index) const
    {
        return {base, offset + static_cast<size_t>(index)};
    }
};
template <typename T>
void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t count)
{
    Require(dst.offset + count <= dst.capacity);
    ++model->copiesIssued;
    model->copies.push_back([=] { std::copy_n(src.base + src.offset, count, dst.base + dst.offset); });
}
template <HardEvent Event>
void SetFlag(int event)
{
    static_assert(Event == HardEvent::MTE2_S);
    Require(event == EVENT_ID0 && model->sets == model->waits);
    Require(!model->copies.empty());
    model->rowsAtSet = static_cast<uint32_t>(model->sent.size());
    ++model->sets;
}
template <HardEvent Event>
void WaitFlag(int event)
{
    static_assert(Event == HardEvent::MTE2_S);
    Require(event == EVENT_ID0 && model->sets == model->waits + 1U);
    for (const auto& copy : model->copies) copy();
    model->copies.clear();
    ++model->waits;
}
}  // namespace AscendC

namespace MegaMoeImpl {
using namespace AscendC;
using bfloat16_t = uint16_t;
using GM_ADDR = uint8_t*;
constexpr uint32_t META_INFO_SIZE = 8U;
constexpr uint32_t WAVE_COMBINE_META_INFO_TOKEN_CAPACITY = 1536U;
constexpr uint8_t COMBINE_NO_QUANT = 0U;
struct Params {
    struct Workspace {
        GM_ADDR metaInfoPtr;
        GM_ADDR gmm2MmadResPtr;
    } workspaceInfo;
    struct Tiling {
        uint32_t combineIntraCores;
    }* tilingData;
};
struct MoeStageCommonConfig {
    uint32_t tokenHiddenDim = 1U;
};
struct ExpertLoopState {
    uint32_t expertIdx = 7U;
    uint64_t globalTokenStartIndex = 17U;
};
struct WaveCombineScratch {
    uint32_t auditRowBase = 999U;
    LocalTensor<int32_t> metaInfoTensor;
};
struct Buffers {
    uint32_t rowBufferCount;
};
namespace AscendTimer {
enum Span { AIV_COMBINE_INTRA, AIV_COMBINE_CROSS };
}
struct AscendTimerDevice {
    uint32_t expectedRows;
    uint32_t expectedStart;
    AscendTimer::Span expectedSpan;
    void SetSlice(uint32_t expert, uint32_t rowStart, uint32_t rows)
    {
        Require(expert == 7U && rowStart == expectedStart && rows == expectedRows);
    }
    void StartSpan(AscendTimer::Span type)
    {
        Require(type == expectedSpan && model->sent.empty() && model->copies.empty());
        ++model->starts;
    }
    void EndSpan(AscendTimer::Span type)
    {
        Require(type == expectedSpan && model->drains == 1U && model->copies.empty());
        ++model->ends;
    }
    void ClearSlice()
    {
        Require(model->ends == 1U);
        ++model->clears;
    }
};
static Buffers InitTokenCombineSliceBuffers(const MoeStageCommonConfig&, WaveCombineScratch&)
{
    return {model->slots};
}
template <uint8_t Mode>
void CombineWaveTokenRange(const MoeStageCommonConfig& common, const Buffers& buffers, WaveCombineScratch& scratch,
                           const Params& params, GM_ADDR output, uint32_t tokenStart, uint32_t tokenCount,
                           uint32_t metadataOffset, uint32_t& sequence)
{
    static_assert(Mode == COMBINE_NO_QUANT);
    Require(buffers.rowBufferCount == model->slots);
    Require(output ==
            params.workspaceInfo.gmm2MmadResPtr + model->expertGlobal * common.tokenHiddenDim * sizeof(bfloat16_t));
    Require(model->drains == 0U && model->starts == 1U);
    Require(model->waits == model->sets && model->waits > 0U);
    Require(sequence == model->sent.size());
    Require(metadataOffset == model->sent.size() - model->rowsAtSet);
    for (uint32_t row = 0U; row < tokenCount; ++row) {
        const uint32_t globalRow = static_cast<uint32_t>(model->expertGlobal) + tokenStart + row;
        for (uint32_t field = 0U; field < META_INFO_SIZE; ++field) {
            const auto actual = scratch.metaInfoTensor.GetValue((metadataOffset + row) * META_INFO_SIZE + field);
            Require(actual == static_cast<int32_t>(globalRow * 13U + field));
        }
        // The actual row sender consumes metadata synchronously to calculate
        // addresses; only its independent payload slots remain in flight.
        Require(sequence % buffers.rowBufferCount == model->sent.size() % model->slots);
        model->sent.push_back(tokenStart + row);
        ++sequence;
    }
}
static void DrainCombineRowBuffers(uint32_t& sequence, uint32_t slots)
{
    Require(sequence == model->expectedRows && sequence == model->sent.size());
    Require(slots == model->slots && model->drains == 0U);
    ++model->drains;
    sequence = 0U;
}

// INSERT_PRODUCTION_FUNCTIONS

}  // namespace MegaMoeImpl

using namespace MegaMoeImpl;
struct Counts {
    std::array<uint32_t, 16> values{};
    uint32_t Get(uint32_t peer) const
    {
        return values.at(peer);
    }
};

static void Check(const CommOwnedSegments& owned, uint32_t slots, uint32_t core = 17U, uint32_t intra = 4U,
                  uint32_t rowStart = 0U)
{
    Model state;
    state.slots = slots;
    state.expectedRows = owned.rows;
    state.expertGlobal = 17U;
    model = &state;
    uint32_t largestRow = 0U;
    std::vector<uint32_t> expected;
    for (uint32_t i = 0U; i < owned.count; ++i) {
        const auto& segment = owned.segments[i];
        largestRow = std::max(largestRow, segment.rowBegin + segment.count);
        for (uint32_t j = 0U; j < segment.count; ++j) expected.push_back(segment.rowBegin + j);
    }
    Require(expected.size() == owned.rows);
    const uint32_t globalRows = static_cast<uint32_t>(state.expertGlobal) + largestRow + 1U;
    std::vector<int32_t> metadata(globalRows * META_INFO_SIZE);
    for (uint32_t row = 0; row < globalRows; ++row)
        for (uint32_t field = 0; field < META_INFO_SIZE; ++field)
            metadata[row * META_INFO_SIZE + field] = static_cast<int32_t>(row * 13U + field);
    std::vector<int32_t> ub(WAVE_COMBINE_META_INFO_TOKEN_CAPACITY * META_INFO_SIZE, -1);
    std::vector<uint8_t> output(globalRows * sizeof(bfloat16_t));
    Params::Tiling tiling{intra};
    Params params{{reinterpret_cast<GM_ADDR>(metadata.data()), output.data()}, &tiling};
    WaveCombineScratch scratch{999U, {ub.data(), ub.size()}};
    AscendTimerDevice timer{owned.rows, rowStart,
                            core < intra ? AscendTimer::AIV_COMBINE_INTRA : AscendTimer::AIV_COMBINE_CROSS};
    RunCommSplitCombineSlice({}, params, {}, rowStart, owned, core, timer, scratch);
    Require(state.sent == expected && state.copies.empty());
    Require(state.sets ==
            (owned.rows + WAVE_COMBINE_META_INFO_TOKEN_CAPACITY - 1U) / WAVE_COMBINE_META_INFO_TOKEN_CAPACITY);
    Require(state.sets == state.waits);
    const uint32_t nonempty = owned.rows != 0U;
    Require(state.drains == nonempty && state.starts == nonempty && state.ends == nonempty && state.clears == nonempty);
    Require(scratch.auditRowBase == (nonempty ? 0U : 999U));
}

static void CheckLegacyPreload()
{
    Model state;
    model = &state;
    std::vector<int32_t> gm(40U * META_INFO_SIZE);
    for (uint32_t i = 0U; i < gm.size(); ++i) gm[i] = static_cast<int32_t>(i + 1U);
    std::vector<int32_t> ub(40U * META_INFO_SIZE, -1);
    Params::Tiling tiling{4U};
    Params params{{reinterpret_cast<GM_ADDR>(gm.data()), nullptr}, &tiling};
    WaveCombineScratch scratch{0U, {ub.data(), ub.size()}};
    PreloadWaveCombineMetaInfo(params, scratch, 13U, 0U, 5U);
    Require(state.sets == 0U && state.copiesIssued == 0U);
    EnqueueWaveCombineMetaInfo(params, scratch, 13U, 0U, 5U);
    Require(state.sets == 0U && state.copiesIssued == 0U);
    EnqueueWaveCombineMetaInfo(params, scratch, 13U, 3U, 5U);
    Require(ub[5U * META_INFO_SIZE] == -1 && state.sets == 0U && state.copiesIssued == 1U);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    Require(ub[5U * META_INFO_SIZE] == gm[13U * META_INFO_SIZE]);
    PreloadWaveCombineMetaInfo(params, scratch, 7U, 9U, 18U);
    Require(state.sets == 2U && state.waits == 2U && state.copies.empty());
    Require(ub[18U * META_INFO_SIZE] == gm[7U * META_INFO_SIZE]);
    Require(ub[27U * META_INFO_SIZE - 1U] == gm[16U * META_INFO_SIZE - 1U]);
}

static CommOwnedSegments MakeSegments(const std::vector<uint32_t>& lengths)
{
    CommOwnedSegments owned{};
    uint32_t row = 5U;
    for (const auto count : lengths) {
        auto& segment = owned.segments[owned.count++];
        segment.peer = owned.count % 16U;
        segment.rowBegin = row;
        segment.sourceBegin = row / 2U;
        segment.count = count;
        owned.rows += count;
        row += count + 19U;
    }
    return owned;
}

int main()
{
    CheckLegacyPreload();
    const std::vector<std::vector<uint32_t>> lengths{{},
                                                     {0U},
                                                     {0U, 0U},
                                                     {1U},
                                                     {4U, 17U, 22U},
                                                     {0U, 3U, 0U, 0U, 7U, 0U},
                                                     {1535U},
                                                     {1536U},
                                                     {1537U},
                                                     {3073U},
                                                     {500U, 1036U, 1U},
                                                     {500U, 1037U, 17U},
                                                     {1536U, 0U, 1536U},
                                                     {2300U, 3500U, 1U},
                                                     {1U, 1536U, 0U, 4U, 3073U}};
    for (uint32_t slots : {2U, 4U, 6U})
        for (const auto& counts : lengths) Check(MakeSegments(counts), slots);
    std::cout << "metadata batching: delayed copies, mapping, capacity, zero tasks, 2/4/6 slots passed\n";

    uint64_t cases = 0U;
    uint32_t bs4096CrossRows = 0U, bs4096OldWaits = 0U, bs4096NewWaits = 0U;
    // All ranks and all legal ratios exercise the production ownership planner
    // with exact rank-first balanced routing and MegaMoE's 12x256 wave capacity.
    for (uint32_t bs : {2048U, 4096U, 8192U}) {
        for (uint32_t rank = 0U; rank < 16U; ++rank) {
            uint32_t usedGroups = 0U, wave = 0U;
            uint64_t globalRow = 0U;
            for (uint32_t expert = 0U; expert < 24U; ++expert) {
                Counts counts;
                uint32_t total = 0U;
                for (uint32_t source = 0U; source < 16U; ++source) {
                    const uint32_t length = bs * 8U;
                    const uint32_t first = (expert * 16U + rank + 384U - (source * length) % 384U) % 384U;
                    counts.values[source] = (length - first + 383U) / 384U;
                    total += counts.values[source];
                }
                for (uint32_t begin = 0U; begin < total;) {
                    if (usedGroups == 12U) {
                        usedGroups = 0U;
                        ++wave;
                    }
                    const uint32_t count = std::min(total - begin, (12U - usedGroups) * 256U);
                    CommPeerSegment peers[16];
                    BuildCommPeerSegments(counts, rank, 16U, begin, begin + count, true, peers);
                    for (bool paired : {false, true})
                        for (uint32_t intra = 1U; intra < 28U; ++intra)
                            for (uint32_t core = 0U; core < 28U; ++core) {
                                const auto owned =
                                    SelectCommOwnedSegments(peers, rank, core, 28U, intra, globalRow + wave, paired);
                                if (bs == 4096U && rank == 0U && paired && intra == 4U && core >= intra) {
                                    bs4096CrossRows += owned.rows;
                                    bs4096OldWaits += owned.count;
                                    bs4096NewWaits += (owned.rows + WAVE_COMBINE_META_INFO_TOKEN_CAPACITY - 1U) /
                                                      WAVE_COMBINE_META_INFO_TOKEN_CAPACITY;
                                }
                                Check(owned, std::array<uint32_t, 3>{2U, 4U, 6U}[cases++ % 3U], core, intra, begin);
                            }
                    begin += count;
                    globalRow += count;
                    usedGroups += (count + 255U) / 256U;
                }
            }
            Require(globalRow == static_cast<uint64_t>(bs) * 8U);
        }
    }
    Require(bs4096CrossRows == 24576U && bs4096OldWaits == 1944U && bs4096NewWaits == 576U);
    std::cout << "BS2048/4096/8192 EP16: real planner, all 27 ratios, both group orders passed (" << cases
              << " core slices)\n";
}
