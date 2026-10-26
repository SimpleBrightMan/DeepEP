"""Execute the real owned-work cursor and deferred GMM2 completion on the CPU.

This checks scheduling/notification lifetimes, not Ascend instruction execution.
The source checks connect those tested helpers to the device implementation.
"""

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
KERNEL = ROOT / "experiments/megamoe/csrc/kernels/mega_moe"


CPP = r"""
#include "compute_work.hpp"
#include "gmm2_completion.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>
using namespace MegaMoeUrma;

struct Plan {
    std::vector<ComputeBatchTask> tasks;
    std::vector<ComputeBatch> batches;
    std::vector<Fragment> groups;
    std::vector<uint32_t> width1, width2;
    uint32_t TaskCount() const { return tasks.size(); }
    ComputeBatchTask Task(uint32_t i) const { return tasks.at(i); }
    ComputeBatch Batch(uint32_t i) const { return batches.at(i); }
    Fragment Group(uint32_t i) const { return groups.at(i); }
    uint32_t TileCount(ComputeBatchStage s, const Fragment& f) const {
        return (s == ComputeBatchStage::Gmm1 ? width1 : width2).at(f.expert);
    }
};

Plan MakePlan(uint32_t seed, uint64_t prefix) {
    std::mt19937 rng(seed);
    Plan p;
    uint64_t g1 = prefix, g2 = prefix + 17;
    const uint32_t widths[] = {1, 3, 5, 14, 28, 33, 56};
    for (uint32_t e = 0; e < 13; ++e) {
        uint32_t count = e == 0 || e == 6 ? 0U : rng() % 8U;
        p.width1.push_back(widths[rng() % 7]);
        p.width2.push_back(widths[rng() % 7]);
        p.batches.push_back({uint32_t(p.groups.size()), count});
        for (uint32_t i = 0; i < count; ++i) {
            Fragment f{};
            f.expert = e; f.rows = i + 1 == count ? 1U + rng() % 256U : 256U;
            f.rowBegin = p.groups.size() * uint64_t{256};
            f.gmm1TileBegin = g1; f.gmm2TileBegin = g2;
            p.groups.push_back(f);
            g1 += p.width1.back(); g2 += p.width2.back();
        }
        // Alternate stage boundaries and runs of consecutive GMM2 experts.
        p.tasks.push_back({ComputeBatchStage::Gmm1, e});
        if (e % 3 == 2) {
            for (uint32_t j = e - 2; j <= e; ++j)
                p.tasks.push_back({ComputeBatchStage::Gmm2, j});
        }
    }
    p.tasks.push_back({ComputeBatchStage::Gmm2, 12});
    return p;
}

std::vector<ComputeWork> Reference(const Plan& p, uint32_t worker, uint32_t workers) {
    std::vector<ComputeWork> result;
    for (const auto& task : p.tasks) {
        const auto batch = p.Batch(task.batchIndex);
        size_t batchStart = result.size();
        for (uint32_t i = 0; i < batch.fragmentCount; ++i) {
            const auto f = p.Group(batch.fragmentBegin + i);
            uint32_t width = p.TileCount(task.stage, f), first = width;
            uint64_t prefix = task.stage == ComputeBatchStage::Gmm1 ? f.gmm1TileBegin : f.gmm2TileBegin;
            for (uint32_t tile = 0; tile < width; ++tile)
                if ((prefix + tile) % workers == worker) { first = tile; break; }
            if (first == width) continue;
            ComputeWork w{};
            w.stage = task.stage; w.fragmentIndex = batch.fragmentBegin + i;
            w.batchBegin = batch.fragmentBegin; w.firstTile = first;
            w.tilesPerGroup = width; w.fragment = f;
            result.push_back(w);
        }
        if (result.size() != batchStart) result.back().endBatch = true;
    }
    return result;
}

std::vector<ComputeWork> ReadCursor(const Plan& p, uint32_t worker, uint32_t workers) {
    ComputeWorkCursor<Plan> cursor(p, worker, workers);
    std::vector<ComputeWork> result;
    ComputeWork w;
    while (cursor.Next(w)) result.push_back(w);
    assert(w.stage == ComputeBatchStage::None);
    assert(!cursor.Next(w));
    return result;
}

void CheckCursor(const Plan& p, uint32_t worker, uint32_t workers) {
    const auto want = Reference(p, worker, workers);
    const auto got = ReadCursor(p, worker, workers);
    const auto paired = ReadCursor(p, worker, workers);
    assert(want.size() == got.size() && got.size() == paired.size());
    for (size_t i = 0; i < want.size(); ++i) {
        const auto& a = want[i]; const auto& b = got[i];
        assert(a.stage == b.stage && a.fragmentIndex == b.fragmentIndex);
        assert(a.batchBegin == b.batchBegin && a.firstTile == b.firstTile);
        assert(a.tilesPerGroup == b.tilesPerGroup && a.endBatch == b.endBatch);
        assert(a.fragment.gmm1TileBegin == b.fragment.gmm1TileBegin);
        assert(a.fragment.gmm2TileBegin == b.fragment.gmm2TileBegin);
        assert(a.fragment.rows == b.fragment.rows);
        assert(b.firstTile == paired[i].firstTile && b.fragmentIndex == paired[i].fragmentIndex);
        if (i + 1 < got.size())
            assert(ContinueComputeContext(b, got[i + 1]) == (b.stage == got[i + 1].stage));
    }
    ComputeWork end{};
    assert(!ContinueComputeContext(end, end));
    if (!got.empty()) assert(!ContinueComputeContext(got.back(), end));
}

struct Backend {
    using Key = std::pair<uint32_t, uint32_t>; // group, physical tile
    uint32_t worker, workers;
    std::vector<Key> submitted;
    std::vector<std::string> events;
    std::map<uint32_t, uint32_t> published;
    size_t cut = 0;
    bool armed = false;
    void Submit(uint32_t group, uint32_t tile) {
        submitted.emplace_back(group, tile); events.push_back("submit");
    }
    void Fence() {
        assert(!armed); armed = true; cut = submitted.size(); events.push_back("fence");
    }
    void Retire(const Gmm2CompletionRecord& r) {
        assert(armed); events.push_back("retire");
        for (uint32_t g = 0; g < r.groupCount; ++g) {
            uint32_t owned = 0;
            for (uint32_t t = 0; t < r.tilesPerGroup; ++t) {
                if ((r.globalTileBegin + uint64_t(g) * r.tilesPerGroup + t) % workers != worker) continue;
                const Key key{r.batchBegin + g, t};
                assert(std::count(submitted.begin(), submitted.begin() + cut, key) == 1);
                ++owned;
            }
            if (owned) {
                assert(published.count(r.batchBegin + g) == 0);
                published[r.batchBegin + g] = owned;
            }
        }
        armed = false;
    }
};

void CheckLifetime(const Plan& p, uint32_t worker, uint32_t workers, bool fallback) {
    const auto work = ReadCursor(p, worker, workers);
    Backend backend{worker, workers};
    DeferredGmm2Completion<Backend> completion(backend);
    std::map<uint32_t, uint32_t> expected;
    for (size_t i = 0; i < work.size(); ++i) {
        const auto& w = work[i];
        if (w.stage == ComputeBatchStage::Gmm1) { assert(!completion.Pending()); continue; }
        if (fallback && completion.Pending()) {
            completion.Flush();
            backend.events.push_back("wait-input");
            assert(!backend.armed);
        }
        for (uint32_t t = w.firstTile; t < w.tilesPerGroup; t += workers) {
            backend.Submit(w.fragmentIndex, t);
            ++expected[w.fragmentIndex];
            completion.Flush();
        }
        if (w.endBatch) {
            const auto first = p.Group(w.batchBegin);
            completion.Arm({w.batchBegin, w.fragmentIndex - w.batchBegin + 1U,
                            w.tilesPerGroup, first.gmm2TileBegin, w.fragmentIndex, w.fragment});
        }
        if (i + 1 == work.size() || !ContinueComputeContext(w, work[i + 1])) {
            completion.Flush();
            backend.events.push_back("close-context");
            assert(!backend.armed);
        }
    }
    completion.Flush(); completion.Flush();
    assert(!completion.Pending() && !backend.armed);
    assert(backend.published == expected);
}

void CheckFenceOrder() {
    Backend b{0, 1};
    DeferredGmm2Completion<Backend> c(b);
    Gmm2CompletionRecord a{0, 1, 1, 0}, d{1, 1, 1, 1};
    b.Submit(0, 0); c.Arm(a);
    const auto boundaryA = b.cut;
    b.Submit(1, 0); c.Flush();
    assert(boundaryA == 1 && b.submitted.size() == 2);
    assert((b.events == std::vector<std::string>{"submit", "fence", "submit", "retire"}));
    c.Arm(d); c.Flush(); c.Flush();
    assert(b.published.size() == 2);
    // Defensive Arm must retire an existing event before setting it again.
    Backend b2{0, 1}; DeferredGmm2Completion<Backend> c2(b2);
    b2.Submit(0, 0); c2.Arm(a); b2.Submit(1, 0); c2.Arm(d); c2.Flush();
    assert((b2.events == std::vector<std::string>{"submit", "fence", "submit", "retire", "fence", "retire"}));
}

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode(argv[1]);
    if (mode == "fence") { CheckFenceOrder(); return 0; }
    for (uint32_t seed = 0; seed < 45; ++seed) {
        const uint64_t prefix = seed % 2 ? (uint64_t{1} << 40) + seed * 31U : seed * 7U;
        const auto plan = MakePlan(seed, prefix);
        for (uint32_t workers : {4U, 8U, 28U}) {
            for (uint32_t worker = 0; worker < workers; ++worker) {
                if (mode == "cursor") CheckCursor(plan, worker, workers);
                else CheckLifetime(plan, worker, workers, mode == "fallback");
            }
        }
    }
    // All-empty schedules are valid and must not open/close any context.
    Plan empty;
    empty.batches.push_back({0, 0});
    empty.tasks.push_back({ComputeBatchStage::Gmm2, 0});
    CheckCursor(empty, 27, 28); CheckLifetime(empty, 27, 28, true);
    std::cout << mode << " passed\n";
}
"""


@pytest.fixture(scope="module")
def pipeline_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for GMM2 pipeline tests")
    temporary = tmp_path_factory.mktemp("urma_gmm2_pipeline")
    source = temporary / "pipeline.cpp"
    source.write_text(CPP, encoding="utf-8")
    binary = temporary / ("pipeline.exe" if os.name == "nt" else "pipeline")
    subprocess.run(
        [*command, "-std=c++17", "-O1", "-I", str(KERNEL / "urma"), str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.mark.parametrize("mode", ["cursor", "lifetime", "fallback", "fence"])
def test_real_pipeline_helpers(pipeline_binary, mode):
    subprocess.run([str(pipeline_binary), mode], check=True, capture_output=True, text=True)


def test_double_l0c_and_device_publication_wiring():
    compute = (KERNEL / "urma/compute.hpp").read_text(encoding="utf-8")
    pipeline = (KERNEL / "catlass_gmm/mega_moe_grouped_matmul.hpp").read_text(encoding="utf-8")
    block = (KERNEL / "catlass_gmm/block_mmad_a8w4.hpp").read_text(encoding="utf-8")
    assert "UrmaOutputRows, kGmm2TileN, 2U>" in compute
    assert "MmadA8W4Mx<Arch, !RowIO::kScatterOutput, false, 16, L0CStages," in pipeline
    assert "L0C_TILE_SIZE * i" in block
    assert "SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i])" in block
    assert "(l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0" in block
    assert "SetFlag<HardEvent::FIX_S>(EVENT_ID1)" in compute
    assert "WaitFlag<HardEvent::FIX_S>(EVENT_ID1)" in compute
    run = compute.split("inline void RunGmm2Fragment", 1)[1].split("inline ComputeBatchRange", 1)[0]
    first_flush = run.index("completion.Flush()")
    assert first_flush < run.index("WaitComputeFlag(args.activationDone")
    submit = run.index("gmm2.Compute(")
    retire = run.index("completion.Flush()", submit)
    arm = run.index("completion.Arm(")
    assert submit < retire < arm
    teardown = run[run.index("if (endContext)") :]
    assert teardown.index("completion.Flush()") < teardown.index("gmm2.End(g_coreType == AIC)")
    assert "CompleteComputeContext();" in teardown
    driver = compute.split("inline void RunCubeAndPrologue", 1)[1].split("inline bool AreGroupInputsReady", 1)[0]
    assert "!hasNext || !ContinueComputeContext(current, next)" in driver
    assert re.search(r"current\.endBatch,\s*endContext, completion", driver)
