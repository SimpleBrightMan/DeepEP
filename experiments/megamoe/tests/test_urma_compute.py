"""Host execution of the device tile assignment and dependency specification.

These checks cover route coverage/progress, not NPU arithmetic or URMA ordering.
"""

import os
from itertools import groupby
from pathlib import Path
import shlex
import shutil
import subprocess

import numpy as np
import pytest

from deep_ep_experimental.megamoe.urma_routing import cv_fragments, plan_routes


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
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace MegaMoeUrma;
struct Task { uint32_t kind, fragment; uint64_t tile; };
int main() {
    static_assert(kComputeTileM == 256 && kActivationTileM == 128);
    static_assert(kRequiredUbBytes == 196608);
    static_assert(ActivationUbContract::inputEnd == 131072);
    static_assert(ActivationUbContract::scratchEnd == 103424);
    static_assert(ActivationUbContract::upOffset == 65536);
    static_assert(ActivationUbContract::quantOffset == 65536);
    static_assert(ActivationUbContract::scaleOffset == 98304);
    static_assert(ActivationUbContract::maxExpOffset == 99328);
    static_assert(ActivationUbContract::reciprocalOffset == 101376);
    assert(ActivationSlice(512, 0, 0).rows == 0);
    assert(ActivationSlice(512, 128, 128).rows == 0);
    uint32_t maxPeerChunks = 0, chunkCount = 0;
    assert(std::cin >> maxPeerChunks >> chunkCount);
    assert(maxPeerChunks > 0);
    std::vector<Chunk> chunks;
    std::vector<Fragment> fragments;
    uint64_t prefix1 = 0, prefix2 = 0, rowBegin = 0;
    std::vector<uint32_t> fragmentChunk;
    for (uint32_t index = 0; index < chunkCount; ++index) {
        uint32_t peer = 0, cv = 0, count = 0;
        assert(std::cin >> peer >> cv >> count);
        assert(peer < 16 && cv < maxPeerChunks);
        const auto begin = static_cast<uint32_t>(fragments.size());
        const auto chunkRow = rowBegin;
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t expert = 0, row = 0;
            assert(std::cin >> expert >> row);
            assert(expert < 24 && row > 0);
            fragments.push_back({expert, peer, rowBegin, row, peer * maxPeerChunks + cv, prefix1, prefix2});
            fragmentChunk.push_back(index);
            prefix1 += ComputeTileCount(row, 1280);
            prefix2 += ComputeTileCount(row, 3584);
            rowBegin += row;
        }
        chunks.push_back({peer, cv, chunkRow, 0, static_cast<uint32_t>(rowBegin - chunkRow), begin, count, 0});
    }
    std::vector<std::vector<Task>> tasks(28 + 13);
    std::vector<uint32_t> g1Owners(prefix1), g2Owners(prefix2), actOwners(prefix1);
    for (uint32_t f = 0; f < fragments.size(); ++f) {
        auto &d = fragments[f];
        auto n1 = ComputeTileCount(d.rows, 1280), n2 = ComputeTileCount(d.rows, 3584);
        assert(n1 == ((d.rows + 255) / 256) * 5);
        assert(n2 == ((d.rows + 255) / 256) * 14);
        std::vector<uint32_t> outputVisits(d.rows * 5), scaleVisits(d.rows * 40);
        uint32_t originalTileContributions = 0;
        for (uint32_t m = 0; m < d.rows; m += kComputeTileM) {
            const auto tileRows = std::min(kComputeTileM, d.rows - m);
            for (uint32_t n = 0; n < 1280; n += 256) {
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
                        ++outputVisits[row * 5 + n / 256];
                        for (uint32_t scale = n / 32; scale < (n + 256) / 32; ++scale)
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

    // Use the same production task and chunk-range helpers as the device
    // routine. CVs are the supplied communication descriptors, never inferred
    // from rows or conflated merely because two sources have the same CV id.
    using Stage = ComputeBatchStage;
    std::vector<ComputeBatchTask> expected;
    if (!chunks.empty()) {
        expected.push_back({Stage::Gmm1, 0});
        for (uint32_t i = 1; i < chunks.size(); ++i) {
            expected.push_back({Stage::Gmm1, i});
            expected.push_back({Stage::Gmm2, i - 1});
        }
        expected.push_back({Stage::Gmm2, static_cast<uint32_t>(chunks.size() - 1)});
    }
    assert(ComputeBatchTaskCount(chunkCount) == expected.size());
    assert(ComputeBatchTaskAt(0, 0).stage == Stage::None);
    assert(ComputeBatchTaskAt(expected.size(), chunkCount).stage == Stage::None);
    std::vector<uint32_t> g1Visits(prefix1), g2Visits(prefix2), g1ChunkVisits(chunkCount), g2ChunkVisits(chunkCount);
    for (uint32_t step = 0; step < expected.size(); ++step) {
        const auto task = ComputeBatchTaskAt(step, chunkCount);
        assert(task.stage == expected[step].stage && task.chunkIndex == expected[step].chunkIndex);
        assert(task.chunkIndex < chunks.size());
        auto& chunkVisits = task.stage == Stage::Gmm1 ? g1ChunkVisits : g2ChunkVisits;
        assert(++chunkVisits[task.chunkIndex] == 1);
        if (task.stage == Stage::Gmm2) assert(g1ChunkVisits[task.chunkIndex] == 1);
        const auto& chunk = chunks[task.chunkIndex];
        const auto range = ComputeBatchFragments(chunk);
        assert(range.begin == chunk.fragmentBegin && range.end == chunk.fragmentBegin + chunk.fragmentCount);
        for (uint32_t f = range.begin; f < range.end; ++f) {
            const auto& fragment = fragments[f];
            assert(fragmentChunk[f] == task.chunkIndex);
            assert(fragment.sourceRank == chunk.peer);
            assert(fragment.inputReadyIndex == chunk.peer * maxPeerChunks + chunk.cv);
            const bool first = task.stage == Stage::Gmm1;
            const auto prefix = first ? fragment.gmm1TileBegin : fragment.gmm2TileBegin;
            const auto count = ComputeTileCount(fragment.rows, first ? 1280 : 3584);
            for (uint32_t core = 0; core < 28; ++core)
                for (uint32_t tile = FirstOwnedComputeTile(prefix, core, 28); tile < count; tile += 28) {
                    const auto id = prefix + tile;
                    assert(id % 28 == core);  // Identical ownership to the original fragment schedule.
                    auto& visits = first ? g1Visits : g2Visits;
                    assert(++visits[id] == 1);
                    tasks[core].push_back({first ? 0U : 2U, f, id});
                }
        }
    }
    for (auto* visits : {&g1Visits, &g2Visits, &g1ChunkVisits, &g2ChunkVisits})
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
        std::vector<uint32_t> g1Done(prefix1), actDone(fragments.size()), g2Done(fragments.size());
        std::vector<uint32_t> actors(tasks.size());
        std::iota(actors.begin(), actors.end(), 0);
        // Dispatch has an independent producer. The earliest compute CV may
        // arrive last; ready events do not depend on downstream GMM2 progress.
        std::vector<uint32_t> readyAt(chunkCount);
        for (auto& ready : readyAt) ready = random() % 53;
        if (chunkCount) readyAt[0] = 80;
        uint32_t activationStart[13]{};
        activationStart[0] = 95;  // Stress an independently delayed activation worker.
        uint64_t left = 0;
        for (auto& actor : tasks) left += actor.size();
        const auto totalTasks = left;
        uint64_t tick = 0;
        while (left) {
            std::shuffle(actors.begin(), actors.end(), random);
            uint32_t progress = 0;
            for (auto actor : actors) {
                if (cursor[actor] == tasks[actor].size()) continue;
                auto t = tasks[actor][cursor[actor]];
                if (t.kind == 0 && tick < readyAt[fragmentChunk[t.fragment]]) continue;
                if (t.kind == 1 && tick < activationStart[actor - 28]) continue;
                if (t.kind == 1 && !g1Done[t.tile]) continue;
                if (t.kind == 2 && actDone[t.fragment] != ComputeTileCount(fragments[t.fragment].rows, 1280))
                    continue;
                if (t.kind == 0) ++g1Done[t.tile];
                else if (t.kind == 1) ++actDone[t.fragment];
                else ++g2Done[t.fragment];
                ++cursor[actor]; --left; ++progress;
            }
            ++tick;
            // Before tick 95 a whole iteration may legitimately wait for an
            // independent arrival. Thereafter a nonempty DAG must progress.
            assert((progress || tick <= 95) && "CV lookahead must not create a circular dependency");
            assert(tick <= totalTasks + 96 && "bounded arrivals must lead to bounded completion");
        }
        for (uint32_t f = 0; f < fragments.size(); ++f) {
            assert(actDone[f] == ComputeTileCount(fragments[f].rows, 1280));
            assert(g2Done[f] == ComputeTileCount(fragments[f].rows, 3584));
        }
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


def run_scheduler(scheduler_binary, chunks):
    """Send explicit peer/CV descriptors and their expert fragments to C++."""
    max_peer_chunks = max((cv + 1 for _, cv, _ in chunks), default=1)
    lines = [f"{max_peer_chunks} {len(chunks)}"]
    for peer, cv, fragments in chunks:
        lines.append(f"{peer} {cv} {len(fragments)}")
        lines.extend(f"{expert} {rows}" for expert, rows in fragments)
    result = subprocess.run(
        [str(scheduler_binary)],
        input="\n".join(lines),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def communication_chunks(fragments):
    """Preserve the oracle's actual source/CV intersections and expert ids."""
    return [
        (source, cv, [(f.local_expert, f.rows) for f in group])
        for (source, cv), group in groupby(fragments, lambda f: (f.source_rank, f.cv_id))
    ]


@pytest.mark.parametrize("bs", [2048, 4096, 8192])
@pytest.mark.parametrize("cv_rows", [128, 256, 1024, 4096])
def test_actual_fragment_assignment_and_independent_activation_progress(scheduler_binary, bs, cv_rows):
    routes = []
    for rank in range(16):
        linear = np.arange(rank * bs * 8, (rank + 1) * bs * 8)
        ids = ((linear % 16) * 24 + (linear // 16) % 24).reshape(bs, 8)
        routes.append(plan_routes(ids, 384, 16))
    fragments = cv_fragments(routes, destination_rank=7, cv_rows=cv_rows)
    run_scheduler(scheduler_binary, communication_chunks(fragments))


def test_partial_and_multi_mtile_fragments(scheduler_binary):
    # Both within-CV and peer-boundary ragged tails, with multiple M tiles in
    # the same fragment. CV 0 on source 3 must remain separate from source 0.
    run_scheduler(
        scheduler_binary,
        [
            (0, 0, [(0, 1), (1, 255), (2, 256)]),
            (0, 1, [(2, 257), (3, 511), (4, 512)]),
            (3, 0, [(5, 513), (6, 1024)]),
            (15, 7, [(7, 170), (8, 43), (9, 85)]),
        ],
    )


@pytest.mark.parametrize("rows", [1, 43, 85, 127, 128, 129, 171, 255, 256, 257, 511, 512, 513, 4096])
def test_activation_subtile_bounds_and_one_completion_per_original_tile(scheduler_binary, rows):
    run_scheduler(scheduler_binary, [(0, 0, [(0, rows)])])


@pytest.mark.parametrize("chunk_count", [0, 1, 2, 3, 7, 16, 33])
def test_cv_warmup_alternating_lookahead_and_tail(scheduler_binary, chunk_count):
    # The real TaskAt helper is checked against the full requested stage
    # sequence inside the executable, including zero and out-of-range steps.
    chunks = [(i // 8, i % 8, [(i % 24, 1 + i % 517)]) for i in range(chunk_count)]
    run_scheduler(scheduler_binary, chunks)


@pytest.mark.parametrize("cv_rows", [128, 256, 1024, 4096])
def test_sparse_routes_empty_peers_and_ragged_multi_mtile_cv(scheduler_binary, cv_rows):
    routes = []
    for rank in range(16):
        # Only some sources contribute to rank 7. The remainder target rank 0,
        # leaving rank 15 completely empty and source-major CV ids sparse.
        destination = 7 if rank % 3 == 1 else 0
        rows = 769 + 17 * rank
        ids = np.full((rows, 8), destination * 24, dtype=np.int64)
        ids[-13:, 4:] += 1
        routes.append(plan_routes(ids, 384, 16))
    for destination in (7, 15):
        fragments = cv_fragments(routes, destination_rank=destination, cv_rows=cv_rows)
        run_scheduler(scheduler_binary, communication_chunks(fragments))


def test_ragged_descriptor_ranges_do_not_merge_sources_with_same_cv_id(scheduler_binary):
    run_scheduler(
        scheduler_binary,
        [
            (0, 0, [(0, 42), (1, 43), (2, 43), (3, 43), (4, 43), (5, 42)]),
            (1, 0, [(0, 256)]),
            (1, 1, [(1, 3), (2, 126)]),
            (15, 0, [(23, 1)]),
        ],
    )
