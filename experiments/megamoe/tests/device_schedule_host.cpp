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
        while (HasWaveCapacity(current, counts.size(), used, target)) {
            const auto before = current;
            const uint32_t count = counts[current.expertIdx];
            const uint32_t rows = AdvanceExpertTokenPositionInWave<256>(count, target, used, current);
            const uint32_t oldRows = LegacyAdvance(count, target, oldUsed, legacy);
            assert(rows == oldRows && used == oldUsed && used <= target);
            assert(current.expertIdx == legacy.expertIdx);
            assert(current.tokenIndexInExpert == legacy.tokenIndexInExpert);
            assert(current.globalTokenIndex == legacy.globalTokenIndex);
            assert(current.expertIdx > before.expertIdx || current.tokenIndexInExpert > before.tokenIndexInExpert);
            if (rows != 0) {
                assert(rows <= count - before.tokenIndexInExpert);
                wave.lastActiveExpert = before.expertIdx;
                consumed += rows;
            }
            wave.end = current;
        }
        if (used != 0) {
            ++nonemptyWaves;
            assert(wave.lastActiveExpert < counts.size());
            assert(wave.end.globalTokenIndex > wave.begin.globalTokenIndex);
        }
    }
    uint64_t expected = 0;
    for (auto count : counts) expected += count;
    assert(consumed == expected);
    assert(!HasWaveCapacity(current, counts.size(), 0, target));
    return nonemptyWaves;
}

int main()
{
    CheckPolicy();
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
