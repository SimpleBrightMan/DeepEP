"""CPU checks of production deferred completion and arbitrary-route reorder.

These tests cannot validate Ascend Unit Flag hardware ordering or arithmetic.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_deferred_output_and_segment_reorder(tmp_path):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required")
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "reorder.cpp"
    source.write_text(
        r"""
#include "deferred_output.hpp"
#include "reorder_plan.hpp"
#include "compute_contract.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <vector>
using namespace MegaMoeUrma;

struct Backend {
    uint32_t submissions = 0, captured = 0, completions = 0;
    bool armed = false;
    std::vector<bool> stages;
    std::vector<uint64_t> done;
    void Submit() { ++submissions; }
    void Arm() { assert(!armed); captured = submissions; armed = true; }
    void Complete(bool gmm1, uint64_t tile, const TraceMeta& meta) {
        assert(armed && tile + 1 == captured && meta.fragment == tile);
        // The same production fence covers two GMM1 projections as one
        // logical task and only the one captured tile, not its successor.
        assert(submissions == captured || submissions == captured + 1);
        stages.push_back(gmm1); done.push_back(tile); armed = false; ++completions;
    }
};

int main() {
    static_assert(kReorderUbBytes == 128U * 1024U);
    static_assert(kReorderUbBytes <= kRequiredUbBytes);
    for (uint32_t pattern = 0; pattern < 256; ++pattern) {
        Backend b;
        DeferredOutput<Backend> completion(b);
        for (uint32_t tile = 0; tile < 8; ++tile) {
            const bool ready = ((pattern >> tile) & 1) != 0;
            completion.BeforeDependency(ready);
            if (!ready) assert(b.completions == tile); // dependency can require prior tile
            b.Submit();
            TraceMeta meta; meta.fragment = tile;
            completion.AfterSubmit(tile % 3 != 2, tile, meta); // G1/G1/G2, cross-expert/stage
            assert(b.completions == tile && b.captured == tile + 1);
        }
        completion.Flush(); completion.Flush();
        assert(b.completions == 8 && !b.armed);
        for (uint32_t tile = 0; tile < 8; ++tile) {
            assert(b.done[tile] == tile && b.stages[tile] == (tile % 3 != 2));
        }
    }
    // No-work cores never arm or wait on a nonexistent event.
    Backend empty; DeferredOutput<Backend> none(empty); none.Flush(); assert(!empty.armed);

    std::mt19937 random(0x81703);
    for (uint32_t world : {4U, 8U, 16U}) {
        for (uint32_t rows : {1U, 15U, 16U, 43U, 85U, 127U, 128U, 129U, 255U, 256U}) {
            for (uint32_t rank = 0; rank < world; ++rank) {
                std::vector<RowSegment> segments;
                std::vector<uint64_t> expected(rows);
                std::vector<bool> expectedSelf(rows);
                for (uint32_t m = 0; m < rows;) {
                    const uint32_t count = std::min(rows - m, 1U + random() % 39U);
                    const uint32_t peer = random() % world;
                    RowSegment segment{peer, random() % 512U,
                                       uint64_t(1) << 34U | m * 13U,
                                       uint64_t(1) << 35U | m * 17U,
                                       m, count, 0U, 0U};
                    segments.push_back(segment);
                    for (uint32_t row = m; row < m + count; ++row) {
                        expectedSelf[row] = peer == rank;
                        expected[row] = (peer == rank ? segment.returnRow : segment.receiveRow) + row - m;
                    }
                    m += count;
                }
                std::vector<uint32_t> seen(rows * 14U);
                uint32_t groupDone = 0;
                std::vector<uint32_t> workerDone(kAic);
                for (uint32_t tile = 0; tile < 14U; ++tile) {
                    // Arrival into contiguous scratch alone does not increment
                    // the Combine/Reduce-visible counter.
                    const uint32_t before = groupDone;
                    bool matmulDone = true;
                    assert(matmulDone && groupDone == before);
                    for (uint32_t m = 0; m < rows; m += kReorderRows) {
                        const uint32_t sliceRows = std::min(rows - m, kReorderRows);
                        for (const auto& segment : segments) {
                            auto span = PlanReorderSpan(segment, rank, m, sliceRows);
                            for (uint32_t i = 0; i < span.rows; ++i) {
                                const uint32_t row = m + span.scratchRow + i;
                                assert(row < rows && span.self == expectedSelf[row]);
                                assert(span.outputRow + i == expected[row]);
                                ++seen[row * 14U + tile];
                            }
                        }
                    }
                    ++groupDone;
                    ++workerDone[(uint64_t(1) << 33U | tile) % kAic];
                }
                assert(groupDone == ComputeTileCount(rows, 3584U));
                assert(std::all_of(seen.begin(), seen.end(), [](auto n) { return n == 1U; }));
            }
        }
    }
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / ("reorder.exe" if os.name == "nt" else "reorder")
    compiled = subprocess.run(
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
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
