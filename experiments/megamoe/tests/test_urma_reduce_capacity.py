"""Check Reduce's UB/cache bounds against the actual worst-case host planner."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from test_urma_reduce_pipeline import FAKE_ASCEND


DRIVER = r"""
#include "expert_plan.hpp"
#include "reduce_pipeline.hpp"
#include <cassert>
#include <iostream>
#include <string>
using namespace MegaMoeUrma;
using Ub = ReduceUbContract;

void CheckUb() {
    static_assert(Ub::maxTokens >= (8192U + kReduceWorkers - 1U) / kReduceWorkers);
    static_assert(Ub::maxRoutes == Ub::maxTokens * Ub::topk);
    static_assert(Ub::remoteCacheBytes * 8U >= Ub::maxRemoteDependencies);
    static_assert(Ub::selfCacheBytes * 8U >= Ub::maxSelfDependencies);
    static_assert(Ub::issuedBytes * 8U >= Ub::maxTokens);
    static_assert(Ub::routeRingBytes >= Ub::inputs * sizeof(uint32_t));
    const uint32_t offsets[] = {
        Ub::inputBf16Offset, Ub::inputFp32Offset, Ub::accumulatorOffset,
        Ub::outputOffset, Ub::inverseOffset, Ub::weightsOffset,
        Ub::dependenciesOffset, Ub::remoteCacheOffset, Ub::selfCacheOffset,
        Ub::issuedOffset, Ub::routeRingOffset, Ub::end};
    const uint32_t sizes[] = {
        Ub::inputs * Ub::bf16RowBytes, Ub::inputs * Ub::fp32RowBytes,
        Ub::fp32RowBytes, Ub::outputs * Ub::bf16RowBytes,
        Ub::maxRoutes * sizeof(int64_t), Ub::maxRoutes * sizeof(float),
        Ub::maxRoutes * sizeof(uint64_t), Ub::remoteCacheBytes,
        Ub::selfCacheBytes, Ub::issuedBytes, Ub::routeRingBytes};
    for (uint32_t i = 0; i < 11U; ++i) {
        assert(offsets[i] % 32U == 0U);
        assert(offsets[i] + sizes[i] <= offsets[i + 1U]);
    }
    assert(Ub::end % 32U == 0U && Ub::end <= kRequiredUbBytes);
}

void CheckPlan(const ExpertPlan& plan) {
    assert(plan.config.fragmentCount <= Ub::maxSelfDependencies);
    assert(uint64_t(kWorld) * plan.config.maxPeerChunks <= Ub::maxRemoteDependencies);
    for (const uint64_t dependency : plan.reduceDependencies) {
        const uint64_t index = dependency & ~kReduceSelfDependency;
        const bool self = (dependency & kReduceSelfDependency) != 0U;
        const auto capacity = self ? Ub::maxSelfDependencies : Ub::maxRemoteDependencies;
        const auto offset = self ? Ub::selfCacheOffset : Ub::remoteCacheOffset;
        const auto bytes = self ? Ub::selfCacheBytes : Ub::remoteCacheBytes;
        assert(index < capacity && index / 8U < bytes);
        assert(offset + index / 8U < Ub::end);
        if (self) assert(index < plan.config.fragmentCount);
        else assert(index < uint64_t(kWorld) * plan.config.maxPeerChunks);
    }
}

int main(int argc, char** argv) {
    assert(argc == 4);
    const uint32_t bs = uint32_t(std::stoul(argv[1]));
    const uint32_t cv = uint32_t(std::stoul(argv[2]));
    const bool fragmented = std::string(argv[3]) == "fragmented";
    const uint64_t routes = uint64_t(bs) * 8U;
    const uint64_t globalRoutes = uint64_t(kWorld) * routes;
    Config config{};
    config.bs = bs; config.hidden = 3584; config.intermediate = 1280;
    config.topk = 8; config.localExperts = 24; config.cvRows = cv;
    config.maxPeerChunks = uint32_t((routes + cv - 1U) / cv);
    CheckUb();

    // Every source sends its entire expanded stream to rank 15. The planner
    // accepts arbitrary legal expert counts, including this extreme skew.
    std::array<int64_t, 16U * 384U> counts{};
    for (uint32_t source = 0; source < kWorld; ++source)
        counts[source * 384U + 15U * 24U + 23U] = int64_t(routes);
    if (fragmented) {
        // Twenty-three single-row experts maximize partial M256 groups.
        // The last expert holds all other rows. With globalRoutes divisible
        // by 256, this achieves globalRoutes / 256 + 23 groups exactly.
        for (uint32_t expert = 0; expert < 23U; ++expert)
            counts[15U * 24U + expert] = 1;
        counts[15U * 24U + 23U] -= 23;
    }

    config.rank = 15;
    const auto receiver = BuildExpertPlan(counts.data(), config, globalRoutes);
    CheckPlan(receiver);
    const uint32_t expectedGroups = uint32_t(globalRoutes / 256U) + (fragmented ? 23U : 0U);
    assert(receiver.config.receiveRows == globalRoutes);
    assert(receiver.config.fragmentCount == expectedGroups);
    const uint64_t lastSelf = receiver.reduceDependencies.back();
    assert((lastSelf & kReduceSelfDependency) == 0U);
    assert(lastSelf == uint64_t(kWorld) * config.maxPeerChunks - 1U);
    if (bs == 8192U && fragmented) assert(expectedGroups == 4119U);

    // Rank 0 has no incoming compute rows, but its final returned row depends
    // on peer 15's final CV. This reaches remote key 8191 at BS8192/CV128.
    config.rank = 0;
    const auto sender = BuildExpertPlan(counts.data(), config, globalRoutes);
    CheckPlan(sender);
    assert(sender.config.receiveRows == 0U && sender.config.fragmentCount == 0U);
    const uint64_t lastRemote = sender.reduceDependencies.back();
    assert(lastRemote == uint64_t(kWorld) * config.maxPeerChunks - 1U);
    if (bs == 8192U && cv == 128U) assert(lastRemote == 8191U);

    std::vector<uint32_t> coverage(bs);
    uint64_t maximumOwned = 0;
    for (uint32_t worker = 0; worker < kReduceWorkers; ++worker) {
        const uint64_t first = ReduceRowBegin(bs, worker);
        const uint64_t owned = ReduceOwnedRows(bs, worker);
        maximumOwned = std::max(maximumOwned, owned);
        assert(owned <= Ub::maxTokens);
        assert(owned * Ub::topk <= Ub::maxRoutes);
        assert(first + owned <= bs);
        for (uint64_t token = first; token < first + owned; ++token) ++coverage[token];
    }
    for (uint32_t visits : coverage) assert(visits == 1U);
    if (bs == 8192U) assert(maximumOwned == 200U);
    std::cout << "groups=" << expectedGroups << " last_remote=" << lastRemote
              << " max_owned=" << maximumOwned << " ub_bytes=" << Ub::end << "\n";
}
"""


@pytest.fixture(scope="module")
def reduce_capacity_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for Reduce capacity tests")
    temporary = tmp_path_factory.mktemp("urma_reduce_capacity")
    (temporary / "kernel_operator.h").write_text(FAKE_ASCEND, encoding="utf-8")
    source = temporary / "capacity.cpp"
    source.write_text(DRIVER, encoding="utf-8")
    binary = temporary / ("capacity.exe" if os.name == "nt" else "capacity")
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        command
        + [
            "-std=c++17",
            "-O1",
            "-UNDEBUG",
            f"-I{temporary}",
            f"-I{root / 'experiments/megamoe/csrc/kernels/mega_moe/urma'}",
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
@pytest.mark.parametrize("cv", [128, 256, 4096])
@pytest.mark.parametrize("routing", ["single-expert", "fragmented"])
def test_reduce_capacity_extreme_planner_layouts(reduce_capacity_binary, bs, cv, routing):
    result = subprocess.run(
        [str(reduce_capacity_binary), str(bs), str(cv), routing],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ub_bytes=" in result.stdout
