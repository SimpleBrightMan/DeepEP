"""Execute the production AIC-owned reorder cursor with asynchronous flags.

This verifies ownership, task order and one-shot completion; no NPU is used.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_reorder_matches_each_aic_and_drains_only_published_prefix(tmp_path):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required")
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "reorder_progress.cpp"
    source.write_text(
        r"""
#include "reorder_progress.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>
using namespace MegaMoeUrma;
using Tile = std::pair<uint32_t, uint32_t>;

struct Work {
    uint32_t columns;
    std::vector<Fragment> fragments;
    std::vector<ComputeBatch> batches;
    std::vector<ComputeBatchTask> tasks;
};

struct Backend {
    const Work& work;
    const std::vector<Tile>& expected;
    std::vector<std::vector<uint32_t>>& seen;
    uint32_t block;
    uint32_t published = 0, consumed = 0, completions = 0, probes = 0;
    uint32_t TaskCount() const { return work.tasks.size(); }
    ComputeBatchTask TaskAt(uint32_t i) const { return work.tasks.at(i); }
    ComputeBatchRange BatchAt(uint32_t i) const { return ComputeBatchFragments(work.batches.at(i)); }
    Fragment FragmentAt(uint32_t i) const { return work.fragments.at(i); }
    uint32_t TileCount(const Fragment& f) const { return ComputeTileCount(f.rows, work.columns); }
    bool Ready(const Fragment& f, uint32_t tile) {
        ++probes;
        assert(consumed < expected.size());
        const auto [index, wanted] = expected.at(consumed);
        assert(tile == wanted && f.gmm2TileBegin == work.fragments.at(index).gmm2TileBegin);
        assert((f.gmm2TileBegin + tile) % kAic == block);
        return consumed < published;
    }
    void Run(const Fragment& f, uint32_t index, uint32_t tile) {
        assert(consumed < published && std::make_pair(index, tile) == expected.at(consumed));
        assert((f.gmm2TileBegin + tile) % kAic == block);
        assert(++seen.at(index).at(tile) == 1U);
        ++consumed;
    }
    void Complete() {
        assert(consumed == expected.size());
        assert(++completions == 1U);
    }
};

// Reference is the actual RunCubeAndPrologue nested scheduling shape, not
// the helper's former global-tile modulo ActivationWorkers partition.
std::vector<Tile> AicTiles(const Work& w, uint32_t block) {
    std::vector<Tile> result;
    for (const auto& task : w.tasks) {
        if (task.stage != ComputeBatchStage::Gmm2) continue;
        const auto range = ComputeBatchFragments(w.batches.at(task.batchIndex));
        for (uint32_t group = range.begin; group < range.end; ++group) {
            const auto& f = w.fragments.at(group);
            const uint32_t count = ComputeTileCount(f.rows, w.columns);
            for (uint32_t tile = FirstOwnedComputeTile(f.gmm2TileBegin, block, kAic);
                 tile < count; tile += kAic)
                result.emplace_back(group, tile);
        }
    }
    return result;
}

void Check(const Work& w, std::mt19937& random) {
    std::vector<std::vector<uint32_t>> seen;
    for (const auto& f : w.fragments) seen.emplace_back(ComputeTileCount(f.rows, w.columns), 0U);
    for (uint32_t block = 0; block < kAic; ++block) {
        const auto expected = AicTiles(w, block);
        Backend backend{w, expected, seen, block};
        OwnedReorderProgress<Backend> progress(backend, block);
        assert(progress.Done() == expected.empty());
        assert(backend.completions == (expected.empty() ? 1U : 0U));
        while (!progress.Done()) {
            // A not-ready owner must yield immediately and must not reorder
            // anything even if unrelated AICs are already far ahead.
            const auto before = backend.probes;
            assert(!progress.DrainReady());
            assert(backend.probes == before + 1U);
            assert(backend.consumed == backend.published);
            backend.published = std::min<uint32_t>(expected.size(), backend.published + 1U + random() % 9U);
            const auto readyProbe = backend.probes;
            assert(progress.Ready());
            assert(progress.Ready());
            assert(backend.probes == readyProbe + 1U); // cache a positive head check
            assert(progress.FragmentIndex() == expected.at(backend.consumed).first);
            assert(progress.RunOneReady());
            assert(backend.probes == readyProbe + 1U); // no duplicate GM read
            progress.DrainReady();
            assert(backend.consumed == backend.published);
        }
        assert(backend.completions == 1U && backend.consumed == expected.size());
        assert(!progress.Ready() && !progress.RunOneReady() && !progress.DrainReady());
        assert(backend.completions == 1U);
    }
    for (const auto& group : seen) for (auto count : group) assert(count == 1U);
}

int main() {
    std::mt19937 random(20261026U);
    Check(Work{3584U, {}, {}, {}}, random);
    for (uint32_t world : {4U, 8U, 16U}) {
        for (uint32_t bs : {2048U, 4096U, 8192U}) {
            for (bool sparse : {false, true}) {
                for (uint32_t columns : {3584U, 8960U}) {
                    Work w{columns, {}, {}, {}};
                    uint64_t tileBegin = (uint64_t{1} << 35U) + random() % 28U;
                    const uint32_t experts = 384U / world;
                    for (uint32_t expert = 0; expert < experts; ++expert) {
                        const uint32_t rowCount = sparse ? random() % 1301U : bs * 8U / experts;
                        const uint32_t begin = w.fragments.size();
                        for (uint32_t m = 0; m < rowCount; m += 256U) {
                            const uint32_t rows = std::min(256U, rowCount - m);
                            w.fragments.push_back({expert, 0U, m, rows, 0U, 0U, tileBegin});
                            tileBegin += ComputeTileCount(rows, columns);
                        }
                        w.batches.push_back({begin, static_cast<uint32_t>(w.fragments.size()) - begin});
                    }
                    // Include empty experts, GMM1 lookahead and arbitrary GMM2
                    // batch order: the cursor follows computeTasks itself.
                    std::vector<uint32_t> order(experts);
                    for (uint32_t i = 0; i < experts; ++i) order[i] = i;
                    std::shuffle(order.begin(), order.end(), random);
                    for (uint32_t batch : order) {
                        w.tasks.push_back({ComputeBatchStage::Gmm1, batch});
                        if (random() % 3U == 0U) w.tasks.push_back({ComputeBatchStage::None, 0U});
                        w.tasks.push_back({ComputeBatchStage::Gmm2, batch});
                    }
                    Check(w, random);
                }
            }
        }
    }
    // Most physical cores own no work, yet all 28 must publish completion.
    Work tiny{3584U, {{0U, 0U, 0U, 1U, 0U, 0U, 19U}}, {{0U, 0U}, {0U, 1U}},
              {{ComputeBatchStage::Gmm1, 0U}, {ComputeBatchStage::Gmm2, 0U},
               {ComputeBatchStage::Gmm1, 1U}, {ComputeBatchStage::Gmm2, 1U}}};
    Check(tiny, random);
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / ("reorder_progress.exe" if os.name == "nt" else "reorder_progress")
    result = subprocess.run(
        [
            *command,
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
    executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert executed.returncode == 0, executed.stdout + executed.stderr
