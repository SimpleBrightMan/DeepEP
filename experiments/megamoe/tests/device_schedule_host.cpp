// CPU regression for the actual device-independent policy and Wave arithmetic.
#include "common/mega_moe_shared_policy.hpp"
#include "common/mega_moe_wave_schedule.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

using namespace MegaMoeImpl;

// Frozen pre-refactor arithmetic (8ffafeb), independent of the production helper.
static uint32_t LegacyAdvance(uint32_t count, uint32_t target, uint32_t& used, ExpertTokenPosition& p)
{
    uint32_t remaining = count - p.tokenIndexInExpert;
    uint32_t capacity = (target - used) * 256U;
    uint32_t rows = remaining < capacity ? remaining : capacity;
    used += (rows + 255U) / 256U;
    p.tokenIndexInExpert += rows;
    p.globalTokenIndex += rows;
    if (p.tokenIndexInExpert >= count) {
        ++p.expertIdx;
        p.tokenIndexInExpert = 0U;
    }
    return rows;
}

static void CheckPolicy()
{
    for (bool w4 : {false, true}) {
        for (uint32_t shared = 0; shared <= 8; ++shared) {
            for (uint32_t bs : {1U, 96U, 255U, 256U, 257U, 512U, 4095U, 4096U, 65536U}) {
                const bool oldInputReady = w4 && bs <= 256U;
                const bool oldSharedDecode = w4 && shared > 0U && bs <= 256U;
                const bool oldRouteAiv1 = w4 && shared > 0U && !oldSharedDecode;
                const auto mode = SelectInputPreparationMode(w4, shared, bs);
                assert((mode == InputPreparationMode::CountEpoch) == oldInputReady);
                assert((mode == InputPreparationMode::Aiv1RouteOverlap) == oldRouteAiv1);
                assert((mode == InputPreparationMode::AllAivRankBarrier) == (!oldInputReady && !oldRouteAiv1));
                assert(UsesCountEpochInputReady(w4, bs) == oldInputReady);
                const auto gemm = SelectSharedExpertPolicy(w4, shared, bs);
                assert(gemm.useSmallBatchTiles == oldSharedDecode);
                assert(gemm.gmm1TileN == (oldSharedDecode ? 192U : 256U));
                assert(gemm.gmm2TileN == (oldSharedDecode ? 224U : 256U));
            }
        }
    }
}

// Consume the real helper across expert and Wave boundaries. A large GM offset catches narrowing.
static uint32_t CheckWaves(const std::vector<uint32_t>& counts, uint32_t target)
{
    ExpertTokenPosition current{}, legacy{};
    current.globalTokenIndex = legacy.globalTokenIndex = uint64_t{1} << 33;
    uint32_t nonemptyWaves = 0;
    uint64_t consumed = 0;
    while (current.expertIdx < counts.size()) {
        uint32_t used = 0, oldUsed = 0;
        ExpertWaveRange wave{current, current, static_cast<uint32_t>(counts.size())};
        std::vector<ExpertSlice> slices;
        while (HasWaveCapacity(current, counts.size(), used, target)) {
            const auto before = current;
            const uint32_t count = counts[current.expertIdx];
            const auto slice = TakeNextExpertSliceInWave<256>(count, target, used, current);
            const uint32_t rows = slice.rowCount;
            const uint32_t oldRows = LegacyAdvance(count, target, oldUsed, legacy);
            assert(rows == oldRows && used == oldUsed && used <= target);
            assert(slice.expertIdx == before.expertIdx);
            assert(slice.rowStartInExpert == before.tokenIndexInExpert);
            assert(slice.globalRowStart == before.globalTokenIndex);
            assert(current.expertIdx == legacy.expertIdx);
            assert(current.tokenIndexInExpert == legacy.tokenIndexInExpert);
            assert(current.globalTokenIndex == legacy.globalTokenIndex);
            assert(current.expertIdx > before.expertIdx || current.tokenIndexInExpert > before.tokenIndexInExpert);
            if (rows != 0) {
                assert(rows <= count - before.tokenIndexInExpert);
                wave.lastActiveExpert = before.expertIdx;
                consumed += rows;
                slices.push_back(slice);
            }
            wave.end = current;
        }
        if (used != 0) {
            ++nonemptyWaves;
            assert(wave.lastActiveExpert < counts.size());
            assert(wave.end.globalTokenIndex > wave.begin.globalTokenIndex);
            // GMM2 必须重放 GMM1 的同一批 slice，包括跨专家、空专家、尾块和 >32-bit GM 行号。
            uint64_t expertStart = uint64_t{1} << 33;
            for (uint32_t e = 0; e < wave.begin.expertIdx; ++e) expertStart += counts[e];
            const uint32_t endExpert = wave.end.expertIdx + (wave.end.tokenIndexInExpert != 0U);
            uint32_t replayed = 0;
            for (uint32_t e = wave.begin.expertIdx; e < endExpert; ++e) {
                const auto replay = GetExpertSliceInWave(wave, e, counts[e], expertStart);
                if (replay.rowCount != 0U) {
                    assert(replayed < slices.size());
                    const auto& original = slices[replayed++];
                    assert(replay.expertIdx == original.expertIdx);
                    assert(replay.rowStartInExpert == original.rowStartInExpert);
                    assert(replay.rowCount == original.rowCount);
                    assert(replay.globalRowStart == original.globalRowStart);
                }
                expertStart += counts[e];
            }
            assert(replayed == slices.size());
        }
    }
    uint64_t expected = 0;
    for (auto count : counts) expected += count;
    assert(consumed == expected);
    assert(!HasWaveCapacity(current, counts.size(), 0, target));
    return nonemptyWaves;
}

static void CheckEp64PrefillSlices()
{
    // EP64/BS4096、384 专家、TopK6 完全均衡：本卡六专家各4096行，W2 跨 E0/E1。
    ExpertTokenPosition position{0U, 3072U, 3072U};
    uint32_t used = 0;
    const auto tail = TakeNextExpertSliceInWave<256>(4096U, 6U, used, position);
    assert(tail.expertIdx == 0U && tail.rowStartInExpert == 3072U);
    assert(tail.rowCount == 1024U && tail.globalRowStart == 3072U);
    assert(position.expertIdx == 1U && position.tokenIndexInExpert == 0U);
    assert(position.globalTokenIndex == 4096U && used == 4U);

    const auto head = TakeNextExpertSliceInWave<256>(4096U, 6U, used, position);
    assert(head.expertIdx == 1U && head.rowStartInExpert == 0U);
    assert(head.rowCount == 512U && head.globalRowStart == 4096U);
    assert(position.expertIdx == 1U && position.tokenIndexInExpert == 512U);
    assert(position.globalTokenIndex == 4608U && used == 6U);
}

int main()
{
    CheckPolicy();
    CheckEp64PrefillSlices();
    assert(CheckWaves(std::vector<uint32_t>(6, 96), 3) == 2);
    assert(CheckWaves(std::vector<uint32_t>(6, 4096), 6) == 16);
    assert(CheckWaves({0, 0, 0}, 3) == 0);
    assert(CheckWaves({255, 256, 257, 0}, 3) == 2);
    assert(CheckWaves({0, 1, 0, 256, 0}, 1) == 2);
    uint32_t seed = 20261008U;
    for (uint32_t i = 0; i < 2000; ++i) {
        std::vector<uint32_t> counts;
        for (uint32_t e = 0; e < 1 + i % 128; ++e) {
            seed = seed * 1664525U + 1013904223U;
            counts.push_back(e % 7 == 0 ? 0U : seed % 65537U);
        }
        CheckWaves(counts, 1U + i % 32);
    }
}
