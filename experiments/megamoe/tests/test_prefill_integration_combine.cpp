// Copyright (c) 2026, SimpleBright_Man
#include <cstdint>
#undef NDEBUG
#include <cassert>
#include <iostream>
#include <vector>

#define __aicore__
using GM_ADDR = uint8_t*;
using bfloat16_t = uint16_t;
constexpr uint8_t COMBINE_NO_QUANT = 0U;
constexpr uint32_t WAVE_COMBINE_PREFILL_ROW_BUFFER_COUNT = 8U;
constexpr uint32_t WAVE_COMBINE_META_INFO_TOKEN_CAPACITY = 1536U;
namespace AscendTimer {
constexpr uint32_t AIV_COMBINE = 12U;
}
struct MoeStageCommonConfig {
    uint32_t tokenHiddenDim = 4U;
};
struct Params {
    struct {
        GM_ADDR gmm2MmadResPtr = nullptr;
    } workspaceInfo;
};
struct ExpertLoopState {
    uint32_t expertIdx = 2U;
    uint64_t globalTokenStartIndex = 1000U;
};
struct Segment {
    uint32_t peer = 0U, sourceBegin = 0U, rowBegin = 0U, count = 0U;
};
struct CommOwnedSegments {
    Segment segments[16]{};
    uint32_t count = 0U, rows = 0U;
};
struct WaveCombineScratch {};
struct BufferConfig {
    uint32_t rowBufferCount = 0U;
};
enum class Call { Init, TimerStart, Preload, Send, Drain, TimerEnd, Clear };
static std::vector<Call> calls;
static std::vector<uint32_t> rowStarts, rowCounts, incomingSequence;
static uint32_t preloadedStart = 0U, preloadedCount = 0U, drainSequence = 0U;
static uint8_t memory[128000];

struct AscendTimerDevice {
    bool enabled = false;
    void SetSlice(uint32_t expert, uint32_t row, uint32_t count)
    {
        assert(expert == 2U && row == 0U && count == 2003U);
    }
    void StartSpan(uint32_t type)
    {
        assert(type == AscendTimer::AIV_COMBINE);
        if (enabled) calls.push_back(Call::TimerStart);
    }
    void EndSpan(uint32_t type)
    {
        assert(type == AscendTimer::AIV_COMBINE);
        if (enabled) calls.push_back(Call::TimerEnd);
    }
    void ClearSlice()
    {
        calls.push_back(Call::Clear);
    }
};

template <uint8_t Mode, bool IncludeAiv0, uint32_t Slots>
static BufferConfig InitWaveCombineBuffers(const MoeStageCommonConfig&, WaveCombineScratch&)
{
    static_assert(Mode == COMBINE_NO_QUANT && !IncludeAiv0 && Slots == 8U);
    calls.push_back(Call::Init);
    return {Slots};
}
static void PreloadWaveCombineMetaInfo(const Params&, WaveCombineScratch&, uint64_t row, uint32_t count,
                                       uint32_t offset)
{
    assert(offset == 0U && count <= 1536U);
    preloadedStart = static_cast<uint32_t>(row);
    preloadedCount = count;
    calls.push_back(Call::Preload);
}
template <uint8_t Mode>
static void CombineWaveTokenRange(const MoeStageCommonConfig&, const BufferConfig& buffers, WaveCombineScratch&,
                                  const Params&, GM_ADDR output, uint32_t row, uint32_t count, uint32_t metadataOffset,
                                  uint32_t& sequence)
{
    static_assert(Mode == COMBINE_NO_QUANT);
    assert(buffers.rowBufferCount == 8U && metadataOffset == 0U && output == memory + 8000U);
    assert(preloadedStart == 1000U + row && preloadedCount == count);
    rowStarts.push_back(row);
    rowCounts.push_back(count);
    incomingSequence.push_back(sequence);
    sequence += count;
    calls.push_back(Call::Send);
}
static void DrainCombineRowBuffers(uint32_t& sequence, uint32_t slots)
{
    assert(slots == 8U);
    drainSequence = sequence;
    sequence = 0U;
    calls.push_back(Call::Drain);
}

// INSERT_ACTUAL_COMBINE

int main()
{
    MoeStageCommonConfig common;
    Params params;
    params.workspaceInfo.gmm2MmadResPtr = memory;
    ExpertLoopState state;
    CommOwnedSegments owned;
    owned.segments[0] = {8U, 31U, 12U, 2000U};
    owned.segments[1] = {4U, 42U, 6000U, 3U};
    owned.count = 2U;
    owned.rows = 2003U;
    WaveCombineScratch scratch;
    for (bool enabled : {false, true}) {
        AscendTimerDevice timer{enabled};
        calls.clear();
        rowStarts.clear();
        rowCounts.clear();
        incomingSequence.clear();
        drainSequence = 0U;
        RunCommSplitCombineSlice(common, params, state, 0U, owned, timer, scratch);
        assert((rowStarts == std::vector<uint32_t>{12U, 1548U, 6000U}));
        assert((rowCounts == std::vector<uint32_t>{1536U, 464U, 3U}));
        assert((incomingSequence == std::vector<uint32_t>{0U, 1536U, 2000U}));
        assert(drainSequence == 2003U);
        std::vector<Call> expected{Call::Init};
        if (enabled) expected.push_back(Call::TimerStart);
        for (uint32_t chunk = 0U; chunk < 3U; ++chunk) {
            expected.push_back(Call::Preload);
            expected.push_back(Call::Send);
        }
        expected.push_back(Call::Drain);
        if (enabled) expected.push_back(Call::TimerEnd);
        expected.push_back(Call::Clear);
        assert(calls == expected);
        calls.clear();
        owned.rows = 0U;
        RunCommSplitCombineSlice(common, params, state, 0U, owned, timer, scratch);
        assert(calls.empty());
        owned.rows = 2003U;
    }
    std::cout << "eight slots persist across metadata/peer chunks; one final drain also with timer off passed\n";
}
