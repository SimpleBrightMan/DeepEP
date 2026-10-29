"""Host execution of the device tile assignment and dependency specification.

These checks cover route coverage/progress, not NPU arithmetic or URMA ordering.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import numpy as np
import pytest

from deep_ep_experimental.megamoe.urma_routing import plan_routes


@pytest.fixture(scope="module")
def scheduler_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for the compute scheduler test")
    temporary = tmp_path_factory.mktemp("urma_compute")
    root = Path(__file__).resolve().parents[3]
    source = temporary / "compute.cpp"
    source.write_text(
        r"""
#include "compute_contract.hpp"
#include "expert_plan.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace MegaMoeUrma;
struct Task { uint32_t kind, fragment; uint64_t tile; };
void CheckOwnedGroups(uint64_t prefix, uint32_t groups, uint32_t width, uint32_t workers) {
    for (uint32_t worker = 0; worker < workers; ++worker) {
        std::vector<uint64_t> reference, actual;
        std::vector<uint32_t> expectedDone(groups), actualDone(groups);
        for (uint32_t group = 0; group < groups; ++group)
            for (uint32_t tile = 0; tile < width; ++tile) {
                const uint64_t id = uint64_t(group) * width + tile;
                if ((prefix + id) % workers == worker) {
                    reference.push_back(id);
                    ++expectedDone[group];
                }
            }
        const uint64_t limit = uint64_t(groups) * width;
        uint32_t lastGroup = groups;
        for (uint64_t tile = FirstOwnedComputeTile(prefix, worker, workers); tile < limit;) {
            const auto owned = OwnedComputeGroupAt(tile, width, workers);
            assert(owned.group < groups && owned.firstTile < width && owned.tileCount > 0);
            assert(actualDone[owned.group] == 0);  // One visit/notification per owned group.
            actualDone[owned.group] = owned.tileCount;
            for (uint32_t local = owned.firstTile; local < width; local += workers)
                actual.push_back(uint64_t(owned.group) * width + local);
            const uint64_t next = tile + uint64_t(owned.tileCount) * workers;
            if (next >= limit) lastGroup = owned.group;
            tile = next;
        }
        assert(actual == reference && actualDone == expectedDone);
        assert(lastGroup == LastOwnedComputeGroup(prefix, groups, width, worker, workers));
    }
}
void CheckUniformWindow(uint32_t groupsPerExpert, uint32_t target, uint32_t initialExperts) {
    std::vector<ComputeBatch> batches;
    for (uint32_t e = 0; e < 24; ++e) batches.push_back({e * groupsPerExpert, groupsPerExpert});
    const auto tasks = BuildComputeTasks(batches, target);
    assert(tasks.size() == 48);
    uint32_t cursor = 0;
    auto expect = [&](ComputeBatchStage stage, uint32_t expert) {
        assert(tasks[cursor].stage == stage && tasks[cursor].batchIndex == expert);
        ++cursor;
    };
    for (uint32_t e = 0; e < initialExperts; ++e) expect(ComputeBatchStage::Gmm1, e);
    for (uint32_t e = 0; e < 24; ++e) {
        expect(ComputeBatchStage::Gmm2, e);
        if (e + initialExperts < 24) expect(ComputeBatchStage::Gmm1, e + initialExperts);
    }
    assert(cursor == tasks.size());
}
int main() {
    // Include empty batches, multiple owned tiles within a group, partial
    // ownership, and tile prefixes above 32 bits. No routing assumption.
    for (uint64_t prefix : {0ULL, 27ULL, (1ULL << 32) + 31ULL})
        for (uint32_t groups : {0U, 1U, 3U, 11U})
            for (uint32_t width : {1U, 5U, 14U, 28U, 29U, 57U})
                for (uint32_t workers : {13U, 28U}) CheckOwnedGroups(prefix, groups, width, workers);
    // Balanced BS2048/4096/8192: preserve the enlarged window throughout
    // computation, including the final drain, rather than just front-loading.
    CheckUniformWindow(3, 12, 4);
    CheckUniformWindow(6, 12, 2);
    CheckUniformWindow(11, 12, 2);
    CheckUniformWindow(3, 0, 2);
    CheckUniformWindow(3, 18, 6);
    CheckUniformWindow(3, 65536, 24);
    CheckUniformWindow(0, 12, 24);
    assert(BuildComputeTasks({}, 12).empty());
    static_assert(kComputeTileM == 256 && kActivationTileM == 128);
    static_assert(kRequiredUbBytes == 196608);
    static_assert(ActivationUbContract::inputEnd == 131072);
    static_assert(ActivationUbContract::scratchEnd == 103424);
    static_assert(ActivationUbContract::upOffset == 65536);
    static_assert(ActivationUbContract::quantOffset == 65536);
    static_assert(ActivationUbContract::scaleOffset == 98304);
    static_assert(ActivationUbContract::maxExpOffset == 99328);
    static_assert(ActivationUbContract::reciprocalOffset == 101376);
    static_assert(kInputScaleBytes == 112 && kInputScaleUbStride == 128 && kInputScaleUbBytes == 32768);
    assert(ActivationSlice(512, 0, 0).rows == 0);
    assert(ActivationSlice(512, 128, 128).rows == 0);
    Config config{};
    assert(std::cin >> config.bs >> config.rank >> config.cvRows >> config.gmm1AheadMGroups);
    config.hidden = 3584;
    config.intermediate = 1280;
    config.localExperts = 24;
    config.topk = 8;
    config.maxPeerChunks = (config.bs * config.topk + config.cvRows - 1) / config.cvRows;
    std::vector<int64_t> counts(16 * 384);
    for (auto& count : counts) assert(std::cin >> count);
    const auto plan = BuildExpertPlan(counts.data(), config, uint64_t(16) * config.bs * config.topk);
    const auto& fragments = plan.fragments;
    const uint32_t groupM = ComputeGroupRows(config);
    const auto& chunks = plan.recvChunks;
    const auto& batches = plan.computeBatches;
    const auto prefix1 = plan.config.gmm1Tiles, prefix2 = plan.config.gmm2Tiles;
    const auto chunkCount = static_cast<uint32_t>(chunks.size());
    const auto batchCount = static_cast<uint32_t>(batches.size());
    assert(batchCount == config.localExperts);
    std::vector<uint32_t> readyToChunk(16 * config.maxPeerChunks, UINT32_MAX);
    std::vector<std::vector<uint32_t>> fragmentChunks(fragments.size()), peerChunks(16);
    for (uint32_t index = 0; index < chunkCount; ++index) {
        const auto& chunk = chunks[index];
        readyToChunk[chunk.peer * config.maxPeerChunks + chunk.cv] = index;
        peerChunks[chunk.peer].push_back(index);
    }
    for (uint32_t f = 0; f < fragments.size(); ++f) {
        const auto& group = fragments[f];
        uint32_t rows = 0;
        for (uint32_t index = 0; index < group.segmentCount; ++index) {
            const auto& segment = plan.segments[group.segmentBegin + index];
            assert(segment.tileRow == rows && segment.rows > 0);
            rows += segment.rows;
            const auto chunk = readyToChunk[segment.inputReadyIndex];
            assert(chunk != UINT32_MAX);
            fragmentChunks[f].push_back(chunk);
        }
        assert(rows == group.rows && rows > 0 && rows <= groupM);
        // The actual scale DMA uses padded UB rows and compact GM rows.
        // Exercise ragged source/CV spans, including 43/85/171-row boundaries.
        std::vector<uint8_t> scratch(kInputScaleUbBytes + 32, 0xcd);
        std::vector<uint8_t> compact(group.rows * kInputScaleBytes);
        auto value = [](uint64_t row, uint32_t byte) {
            return static_cast<uint8_t>((row * 73U + byte * 19U + (row >> 8)) & 0xffU);
        };
        for (uint32_t index = 0; index < group.segmentCount; ++index) {
            const auto& segment = plan.segments[group.segmentBegin + index];
            for (uint32_t row = 0; row < segment.rows; ++row)
                for (uint32_t byte = 0; byte < kInputScaleBytes; ++byte)
                    scratch[(segment.tileRow + row) * kInputScaleUbStride + byte] =
                        value(segment.receiveRow + row, byte);
        }
        for (uint32_t row = 0; row < group.rows; ++row) {
            std::copy_n(scratch.begin() + row * kInputScaleUbStride, kInputScaleBytes,
                        compact.begin() + row * kInputScaleBytes);
        }
        for (uint32_t index = 0; index < group.segmentCount; ++index) {
            const auto& segment = plan.segments[group.segmentBegin + index];
            for (uint32_t row = 0; row < segment.rows; ++row)
                for (uint32_t byte = 0; byte < kInputScaleBytes; ++byte)
                    assert(compact[(segment.tileRow + row) * kInputScaleBytes + byte] ==
                           value(segment.receiveRow + row, byte));
        }
        assert(std::all_of(scratch.begin() + group.rows * kInputScaleUbStride, scratch.end(),
                           [](auto byte) { return byte == 0xcd; }));
    }
    std::vector<std::vector<Task>> tasks(28 + 13);
    std::vector<uint32_t> g1Owners(prefix1), g2Owners(prefix2), actOwners(prefix1);
    for (uint32_t f = 0; f < fragments.size(); ++f) {
        auto &d = fragments[f];
        auto n1 = Gmm1ComputeTileCount(d.rows, config), n2 = Gmm2ComputeTileCount(d.rows, config);
        const uint32_t logicalN = Gmm1LogicalTileN(config);
        const uint32_t g1Columns = 1280 / logicalN;
        assert(n1 == ((d.rows + groupM - 1) / groupM) * g1Columns);
        assert(n2 == ((d.rows + groupM - 1) / groupM) * 14);
        std::vector<uint32_t> outputVisits(d.rows * g1Columns), scaleVisits(d.rows * 40);
        uint32_t originalTileContributions = 0;
        for (uint32_t m = 0; m < d.rows; m += groupM) {
            const auto tileRows = std::min(groupM, d.rows - m);
            for (uint32_t n = 0; n < 1280; n += logicalN) {
                uint32_t slices = 0, visitedRows = 0;
                for (uint32_t offset = 0; offset < tileRows; offset += kActivationTileM) {
                    auto slice = ActivationSlice(m, tileRows, offset);
                    assert(slice.rows > 0 && slice.rows <= 128);
                    assert(slice.rowBegin == m + offset && slice.rowBegin + slice.rows <= d.rows);
                    assert(ActivationUbContract::upOffset + slice.rows * 256 * 2 <=
                           ActivationUbContract::inputEnd);
                    assert(ActivationUbContract::reciprocalOffset + slice.rows * 8 * 2 <=
                           ActivationUbContract::scratchEnd);
                    for (uint32_t row = slice.rowBegin; row < slice.rowBegin + slice.rows; ++row) {
                        ++outputVisits[row * g1Columns + n / logicalN];
                        for (uint32_t scale = n / 32; scale < (n + logicalN) / 32; ++scale)
                            ++scaleVisits[row * 40 + scale];
                    }
                    visitedRows += slice.rows;
                    ++slices;
                }
                assert(visitedRows == tileRows && slices == (tileRows + 127) / 128);
                // Completion belongs to the original ready tile, never each
                // sub-slice: GMM2 must not observe a half-written M256 tile.
                ++originalTileContributions;
            }
        }
        assert(originalTileContributions == n1);
        for (auto* visits : {&outputVisits, &scaleVisits})
            assert(std::all_of(visits->begin(), visits->end(), [](auto n) { return n == 1; }));
        for (uint32_t core = 0; core < 28; ++core) {
            for (auto j = FirstOwnedComputeTile(d.gmm1TileBegin, core, 28); j < n1; j += 28) {
                auto id = d.gmm1TileBegin + j;
                assert(id % 28 == core); ++g1Owners[id];
            }
            for (auto j = FirstOwnedComputeTile(d.gmm2TileBegin, core, 28); j < n2; j += 28) {
                auto id = d.gmm2TileBegin + j;
                assert(id % 28 == core); ++g2Owners[id];
            }
        }
        for (uint32_t worker = 0; worker < 13; ++worker)
            for (auto j = FirstOwnedComputeTile(d.gmm1TileBegin, worker, 13); j < n1; j += 13) {
                auto id = d.gmm1TileBegin + j;
                assert(id % 13 == worker); ++actOwners[id]; tasks[28 + worker].push_back({1, f, id});
            }
    }
    for (auto* owners : {&g1Owners, &g2Owners, &actOwners})
        assert(std::all_of(owners->begin(), owners->end(), [](auto n) { return n == 1; }));

    // Exercise the real expert-batch schedule. Empty experts remain explicit
    // batches; communication CV count/identity never controls compute order.
    using Stage = ComputeBatchStage;
    std::vector<ComputeBatchTask> legacy;
    if (!batches.empty()) {
        legacy.push_back({Stage::Gmm1, 0});
        for (uint32_t i = 1; i < batches.size(); ++i) {
            legacy.push_back({Stage::Gmm1, i});
            legacy.push_back({Stage::Gmm2, i - 1});
        }
        legacy.push_back({Stage::Gmm2, batchCount - 1});
    }
    const auto oldTasks = BuildComputeTasks(batches, 0);
    assert(oldTasks.size() == legacy.size());
    for (uint32_t step = 0; step < legacy.size(); ++step) {
        const auto old = ComputeBatchTaskAt(step, batchCount);
        assert(old.stage == legacy[step].stage && old.batchIndex == legacy[step].batchIndex);
        assert(oldTasks[step].stage == old.stage && oldTasks[step].batchIndex == old.batchIndex);
    }
    assert(ComputeBatchTaskCount(batchCount) == legacy.size());
    assert(ComputeBatchTaskAt(0, 0).stage == Stage::None);
    assert(ComputeBatchTaskAt(legacy.size(), batchCount).stage == Stage::None);
    for (uint32_t n : {0U, 1U, 2U, 3U, 7U, 24U}) {
        assert(ComputeBatchTaskCount(n) == 2ULL * n);
        assert(ComputeBatchTaskAt(2ULL * n, n).stage == Stage::None);
        if (!n) continue;
        auto first = ComputeBatchTaskAt(0, n), last = ComputeBatchTaskAt(2ULL * n - 1, n);
        assert(first.stage == Stage::Gmm1 && first.batchIndex == 0);
        assert(last.stage == Stage::Gmm2 && last.batchIndex == n - 1);
    }
    assert(plan.computeTasks.size() == legacy.size());
    assert(plan.config.computeTaskCount == plan.computeTasks.size());
    uint32_t nextG1 = 0, nextG2 = 0;
    std::vector<uint32_t> g1Visits(prefix1), g2Visits(prefix2), g1BatchVisits(batchCount), g2BatchVisits(batchCount);
    for (const auto task : plan.computeTasks) {
        assert(task.batchIndex < batches.size());
        if (task.stage == Stage::Gmm1) {
            assert(task.batchIndex == nextG1++);
        } else {
            assert(task.stage == Stage::Gmm2 && task.batchIndex == nextG2);
            assert(nextG1 >= std::min(batchCount, nextG2 + 2U));
            uint64_t queued = 0;
            for (uint32_t b = nextG2; b < nextG1; ++b) queued += batches[b].fragmentCount;
            assert(nextG1 == batchCount || queued >= config.gmm1AheadMGroups);
            ++nextG2;
        }
        auto& batchVisits = task.stage == Stage::Gmm1 ? g1BatchVisits : g2BatchVisits;
        assert(++batchVisits[task.batchIndex] == 1);
        if (task.stage == Stage::Gmm2) assert(g1BatchVisits[task.batchIndex] == 1);
        const auto& batch = batches[task.batchIndex];
        const auto range = ComputeBatchFragments(batch);
        assert(range.begin == batch.fragmentBegin && range.end == batch.fragmentBegin + batch.fragmentCount);
        std::vector<std::vector<Task>> reference(28);
        for (uint32_t f = range.begin; f < range.end; ++f) {
            const auto& fragment = fragments[f];
            assert(fragment.expert == task.batchIndex);
            const bool first = task.stage == Stage::Gmm1;
            const auto prefix = first ? fragment.gmm1TileBegin : fragment.gmm2TileBegin;
            const auto count = (first ? Gmm1ComputeTileCount(fragment.rows, config) : Gmm2ComputeTileCount(fragment.rows, config));
            for (uint32_t core = 0; core < 28; ++core)
                for (uint32_t tile = FirstOwnedComputeTile(prefix, core, 28); tile < count; tile += 28) {
                    const auto id = prefix + tile;
                    assert(id % 28 == core);  // Global prefix ownership remains independent of source/CV.
                    auto& visits = first ? g1Visits : g2Visits;
                    assert(++visits[id] == 1);
                    reference[core].push_back({first ? 0U : 2U, f, id});
                }
        }
        if (range.begin == range.end) continue;
        const bool firstStage = task.stage == Stage::Gmm1;
        const auto& firstFragment = fragments[range.begin];
        const uint64_t prefix = firstStage ? firstFragment.gmm1TileBegin : firstFragment.gmm2TileBegin;
        const auto width = (firstStage ? Gmm1ComputeTileCount(firstFragment.rows, config) : Gmm2ComputeTileCount(firstFragment.rows, config));
        const uint64_t limit = uint64_t(batch.fragmentCount) * width;
        for (uint32_t core = 0; core < 28; ++core) {
            size_t cursor = 0;
            for (uint64_t tile = FirstOwnedComputeTile(prefix, core, 28); tile < limit;) {
                const auto owned = OwnedComputeGroupAt(tile, width, 28);
                const auto f = range.begin + owned.group;
                const auto& fragment = fragments[f];
                assert((firstStage ? Gmm1ComputeTileCount(fragment.rows, config) : Gmm2ComputeTileCount(fragment.rows, config)) == width);
                const uint64_t fragmentPrefix = firstStage ? fragment.gmm1TileBegin : fragment.gmm2TileBegin;
                assert(fragmentPrefix == prefix + uint64_t(owned.group) * width);
                uint32_t completed = 0;
                for (uint32_t local = owned.firstTile; local < width; local += 28) {
                    const auto expected = reference[core].at(cursor++);
                    assert(expected.fragment == f && expected.tile == fragmentPrefix + local);
                    tasks[core].push_back(expected);  // Dependency simulation uses the new traversal.
                    ++completed;
                }
                assert(completed == owned.tileCount);  // Same gmm2Done AtomicAdd contribution.
                const uint64_t next = tile + uint64_t(owned.tileCount) * 28;
                assert((next >= limit) ==
                       (owned.group == LastOwnedComputeGroup(prefix, batch.fragmentCount, width, core, 28)));
                tile = next;
            }
            assert(cursor == reference[core].size());
        }
    }
    assert(nextG1 == batchCount && nextG2 == batchCount);
    for (auto* visits : {&g1Visits, &g2Visits, &g1BatchVisits, &g2BatchVisits})
        assert(std::all_of(visits->begin(), visits->end(), [](auto n) { return n == 1; }));
    // Prefix arithmetic must remain correct beyond a 32-bit global tile ID.
    for (uint64_t prefix : {0ULL, 27ULL, 28ULL, (1ULL << 32) + 31ULL})
        for (uint32_t workers : {13U, 28U})
            for (uint32_t worker = 0; worker < workers; ++worker) {
                auto start = FirstOwnedComputeTile(prefix, worker, workers);
                assert(start < workers && (prefix + start) % workers == worker);
            }
    for (uint32_t seed = 0; seed < 10; ++seed) {
        std::mt19937 random(seed);
        std::vector<size_t> cursor(tasks.size());
        std::vector<uint32_t> g1Done(prefix1), actDone(fragments.size()), g2Done(fragments.size()),
            scaleDone(fragments.size());
        std::vector<uint32_t> nextGather(13);
        std::iota(nextGather.begin(), nextGather.end(), 0U);
        std::vector<uint32_t> actors(tasks.size());
        std::iota(actors.begin(), actors.end(), 0);
        // Dispatch producers are independent of GMM2/Combine. Different CVs
        // feeding the same M256 may arrive far apart; the gather owner joins
        // them before publishing the single input-ready flag consumed by AIC.
        std::vector<uint32_t> readyAt(chunkCount);
        for (uint32_t index = 0; index < chunkCount; ++index)
            readyAt[index] = chunks[index].peer == config.rank ? 0 : 1 + random() % 80;
        uint32_t activationStart[13]{};
        activationStart[0] = 95;  // Stress an independently delayed activation worker.
        const uint32_t dispatchCompleteAt = 140;
        std::vector<uint32_t> outputWrites(plan.config.receiveRows * 14), nextPeerChunk(16), sent(chunkCount);
        uint32_t remainingChunks = 0;
        for (uint32_t index = 0; index < chunkCount; ++index) {
            if (chunks[index].peer == config.rank) sent[index] = 1;
            else ++remainingChunks;
        }
        uint64_t left = 0;
        for (auto& actor : tasks) left += actor.size();
        left += fragments.size();  // One independent scale gather per group.
        const auto totalTasks = left;
        uint64_t tick = 0;
        while (left || remainingChunks) {
            std::shuffle(actors.begin(), actors.end(), random);
            uint32_t progress = 0;
            for (auto actor : actors) {
                // Scale ownership and activation ownership are independent.
                // Even a worker with no activation task must prepare its
                // scales; waiting for G1 here would create a circular wait.
                if (actor >= 28) {
                    auto& group = nextGather[actor - 28];
                    if (group < fragments.size() &&
                        std::all_of(fragmentChunks[group].begin(), fragmentChunks[group].end(),
                                    [&](auto chunk) { return tick >= readyAt[chunk]; })) {
                        assert(++scaleDone[group] == 1);
                        group += 13;
                        --left; ++progress;
                    }
                }
                if (cursor[actor] == tasks[actor].size()) continue;
                auto t = tasks[actor][cursor[actor]];
                if (t.kind == 0 && !scaleDone[t.fragment]) continue;
                if (t.kind == 1 && tick < activationStart[actor - 28]) continue;
                if (t.kind == 1 && !g1Done[t.tile]) continue;
                if (t.kind == 2 && actDone[t.fragment] != Gmm1ComputeTileCount(fragments[t.fragment].rows, config))
                    continue;
                if (t.kind == 0) {
                    // AIC gates only on scaleDone. Under arbitrary arrivals,
                    // that aggregate flag must imply every input is ready.
                    assert(std::all_of(fragmentChunks[t.fragment].begin(), fragmentChunks[t.fragment].end(),
                                       [&](auto chunk) { return tick >= readyAt[chunk]; }));
                    ++g1Done[t.tile];
                }
                else if (t.kind == 1) ++actDone[t.fragment];
                else {
                    const auto& group = fragments[t.fragment];
                    const uint32_t n = t.tile - group.gmm2TileBegin;
                    assert(n < 14);
                    // This models the actual completion boundary: scatter
                    // every span of this N tile before incrementing gmm2Done.
                    for (uint32_t j = 0; j < group.segmentCount; ++j) {
                        const auto& segment = plan.segments[group.segmentBegin + j];
                        for (uint32_t row = 0; row < segment.rows; ++row)
                            assert(++outputWrites[(segment.receiveRow + row) * 14 + n] == 1);
                    }
                    ++g2Done[t.fragment];
                }
                ++cursor[actor]; --left; ++progress;
            }
            // One owner per remote peer, preserving CV order. Producers may
            // complete before local Dispatch; no return PUT is then permitted.
            for (uint32_t peer = 0; peer < 16; ++peer) {
                if (peer == config.rank || nextPeerChunk[peer] == peerChunks[peer].size()) continue;
                const auto index = peerChunks[peer][nextPeerChunk[peer]];
                const auto& chunk = chunks[index];
                bool ready = true;
                for (uint32_t j = 0; j < chunk.fragmentCount; ++j) {
                    const auto group = plan.chunkGroupIndices[chunk.fragmentBegin + j];
                    if (g2Done[group] != Gmm2ComputeTileCount(fragments[group].rows, config)) ready = false;
                }
                if (!ready || tick < dispatchCompleteAt) continue;
                for (uint32_t row = 0; row < chunk.rows; ++row)
                    for (uint32_t n = 0; n < 14; ++n)
                        assert(outputWrites[(chunk.localRow + row) * 14 + n] == 1);
                assert(++sent[index] == 1);
                ++nextPeerChunk[peer]; --remainingChunks; ++progress;
            }
            ++tick;
            // Before all independent producers arrive, an idle tick is legal.
            // Thereafter expert lookahead must not introduce a circular wait.
            assert((progress || tick <= dispatchCompleteAt) && "expert lookahead must not deadlock");
            assert(tick <= totalTasks + chunkCount + dispatchCompleteAt + 1);
        }
        for (uint32_t f = 0; f < fragments.size(); ++f) {
            assert(scaleDone[f] == 1);
            assert(actDone[f] == Gmm1ComputeTileCount(fragments[f].rows, config));
            assert(g2Done[f] == Gmm2ComputeTileCount(fragments[f].rows, config));
        }
        assert(std::all_of(sent.begin(), sent.end(), [](auto n) { return n == 1; }));
        assert(std::all_of(outputWrites.begin(), outputWrites.end(), [](auto n) { return n == 1; }));
    }
}
"""
    )
    binary = temporary / ("compute.exe" if os.name == "nt" else "compute")
    result = subprocess.run(
        command
        + [
            "-std=c++17",
            "-O1",
            "-UNDEBUG",
            "-I" + str(root / "experiments/megamoe/csrc/kernels/mega_moe/urma"),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


def run_scheduler(scheduler_binary, counts, bs, rank=7, cv_rows=256, ahead_mgroups=12):
    """Build the production host plan, then execute its real device schedule."""
    counts = np.asarray(counts, dtype=np.int64)
    assert counts.shape == (16, 384)
    assert np.all(counts.sum(axis=1) == bs * 8)
    lines = [f"{bs} {rank} {cv_rows} {ahead_mgroups}", " ".join(map(str, counts.reshape(-1)))]
    result = subprocess.run(
        [str(scheduler_binary)],
        input="\n".join(lines),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def empty_destination_counts(bs):
    counts = np.zeros((16, 384), dtype=np.int64)
    counts[:, 0] = bs * 8
    return counts


@pytest.mark.parametrize("bs", [2048, 4096, 8192])
@pytest.mark.parametrize("cv_rows", [128, 256, 1024, 4096])
def test_expert_group_assignment_and_independent_activation_combine_progress(scheduler_binary, bs, cv_rows):
    counts = []
    for rank in range(16):
        linear = np.arange(rank * bs * 8, (rank + 1) * bs * 8)
        ids = ((linear % 16) * 24 + (linear // 16) % 24).reshape(bs, 8)
        counts.append(plan_routes(ids, 384, 16).expert_counts)
    run_scheduler(scheduler_binary, counts, bs, cv_rows=cv_rows)


def test_cross_source_cv_groups_and_late_input_do_not_release_combine_early(scheduler_binary):
    counts = empty_destination_counts(1024)
    for source, expert, rows in [(0, 0, 129), (1, 0, 256), (2, 0, 511), (0, 1, 171), (15, 1, 42), (7, 23, 1)]:
        counts[source, 7 * 24 + expert] += rows
        counts[source, 0] -= rows
    run_scheduler(scheduler_binary, counts, 1024, cv_rows=128)


@pytest.mark.parametrize("rows", [1, 43, 85, 127, 128, 129, 171, 255, 256, 257, 511, 512, 513, 4096])
def test_activation_subtile_bounds_and_one_completion_per_original_tile(scheduler_binary, rows):
    counts = empty_destination_counts(512)
    counts[0, 7 * 24] = rows
    counts[0, 0] -= rows
    run_scheduler(scheduler_binary, counts, 512)


@pytest.mark.parametrize("active_experts", [0, 1, 2, 3, 7, 16, 24])
@pytest.mark.parametrize("ahead_mgroups", [0, 12, 18, 65536])
def test_expert_warmup_alternating_lookahead_empty_batches_and_tail(scheduler_binary, active_experts, ahead_mgroups):
    counts = empty_destination_counts(128)
    for expert in range(active_experts):
        source = expert % 16
        counts[source, 7 * 24 + expert] = expert + 1
        counts[source, 0] -= expert + 1
    run_scheduler(scheduler_binary, counts, 128, ahead_mgroups=ahead_mgroups)


@pytest.mark.parametrize("cv_rows", [128, 256, 1024, 4096])
def test_sparse_routes_empty_peers_and_ragged_multi_mtile_cv(scheduler_binary, cv_rows):
    counts = []
    bs = 1031
    for rank in range(16):
        # Only some sources contribute to rank 7. The remainder target rank 0,
        # leaving rank 15 completely empty and source-major CV ids sparse.
        destination = 7 if rank % 3 == 1 else 0
        ids = np.full((bs, 8), destination * 24, dtype=np.int64)
        ids[-13:, 4:] += 1
        counts.append(plan_routes(ids, 384, 16).expert_counts)
    for destination in (7, 15):
        run_scheduler(scheduler_binary, counts, bs, rank=destination, cv_rows=cv_rows)


def test_same_cv_ids_on_different_sources_keep_independent_ready_dependencies(scheduler_binary):
    counts = empty_destination_counts(64)
    for source in range(16):
        rows = 15 + source
        counts[source, 7 * 24] = rows
        counts[source, 0] -= rows
    run_scheduler(scheduler_binary, counts, 64)
