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
    std::vector<uint32_t> rows;
    uint32_t row;
    while (std::cin >> row) { assert(row > 0); rows.push_back(row); }
    assert(!rows.empty());
    std::vector<Fragment> fragments;
    uint64_t prefix1 = 0, prefix2 = 0, rowBegin = 0;
    for (uint32_t i = 0; i < rows.size(); ++i) {
        fragments.push_back({i % 24, i % 16, rowBegin, rows[i], i, prefix1, prefix2});
        prefix1 += ComputeTileCount(rows[i], 1280);
        prefix2 += ComputeTileCount(rows[i], 3584);
        rowBegin += rows[i];
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
                assert(id % 28 == core); ++g1Owners[id]; tasks[core].push_back({0, f, id});
            }
            for (auto j = FirstOwnedComputeTile(d.gmm2TileBegin, core, 28); j < n2; j += 28) {
                auto id = d.gmm2TileBegin + j;
                assert(id % 28 == core); ++g2Owners[id]; tasks[core].push_back({2, f, id});
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
    // Prefix arithmetic must remain correct beyond a 32-bit global tile ID.
    for (uint64_t prefix : {0ULL, 27ULL, 28ULL, (1ULL << 32) + 31ULL})
        for (uint32_t workers : {13U, 28U})
            for (uint32_t worker = 0; worker < workers; ++worker) {
                auto start = FirstOwnedComputeTile(prefix, worker, workers);
                assert(start < workers && (prefix + start) % workers == worker);
            }
    for (uint32_t seed = 0; seed < 5; ++seed) {
        std::mt19937 random(seed);
        std::vector<size_t> cursor(tasks.size());
        std::vector<uint32_t> g1Done(prefix1), actDone(fragments.size()), g2Done(fragments.size());
        std::vector<uint32_t> actors(tasks.size());
        std::iota(actors.begin(), actors.end(), 0);
        uint64_t left = 0;
        for (auto& actor : tasks) left += actor.size();
        while (left) {
            std::shuffle(actors.begin(), actors.end(), random);
            uint32_t progress = 0;
            for (auto actor : actors) {
                if (cursor[actor] == tasks[actor].size()) continue;
                auto t = tasks[actor][cursor[actor]];
                if (t.kind == 1 && !g1Done[t.tile]) continue;
                if (t.kind == 2 && actDone[t.fragment] != ComputeTileCount(fragments[t.fragment].rows, 1280))
                    continue;
                if (t.kind == 0) ++g1Done[t.tile];
                else if (t.kind == 1) ++actDone[t.fragment];
                else ++g2Done[t.fragment];
                ++cursor[actor]; --left; ++progress;
            }
            assert(progress && "independent 28/13 task sequences must not deadlock");
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


@pytest.mark.parametrize("bs", [2048, 4096, 8192])
@pytest.mark.parametrize("cv_rows", [128, 256, 1024, 4096])
def test_actual_fragment_assignment_and_independent_activation_progress(scheduler_binary, bs, cv_rows):
    routes = []
    for rank in range(16):
        linear = np.arange(rank * bs * 8, (rank + 1) * bs * 8)
        ids = ((linear % 16) * 24 + (linear // 16) % 24).reshape(bs, 8)
        routes.append(plan_routes(ids, 384, 16))
    fragments = cv_fragments(routes, destination_rank=7, cv_rows=cv_rows)
    result = subprocess.run(
        [str(scheduler_binary)],
        input=" ".join(str(fragment.rows) for fragment in fragments),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_partial_and_multi_mtile_fragments(scheduler_binary):
    result = subprocess.run(
        [str(scheduler_binary)],
        input="1 255 256 257 511 512 513 1024 170 43 85",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("rows", [1, 43, 85, 127, 128, 129, 171, 255, 256, 257, 511, 512, 513, 4096])
def test_activation_subtile_bounds_and_one_completion_per_original_tile(scheduler_binary, rows):
    result = subprocess.run([str(scheduler_binary)], input=str(rows), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
