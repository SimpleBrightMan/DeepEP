"""Run the production ready monitor and CSR planner on asynchronous CPU queues."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from test_urma_reduce_pipeline import FAKE_ASCEND


# Observe every asynchronous store and every engine step without modifying the
# shared simulator or the production monitor. These hooks never advance queues.
MONITOR_ASCEND = (
    FAKE_ASCEND.replace(
        "inline unsigned priority;",
        "inline unsigned priority;\n"
        "inline std::function<void()> afterStep;\n"
        "inline std::function<void(uintptr_t, uint32_t)> beforeStore;",
    )
    .replace("    return progress;\n", "    if (afterStep) afterStep();\n    return progress;\n")
    .replace(
        "        std::copy_n(source.Data(), count, destination.data);",
        "        if (beforeStore) beforeStore(reinterpret_cast<uintptr_t>(destination.data), count);\n"
        "        std::copy_n(source.Data(), count, destination.data);",
    )
)

DRIVER = r"""
#include "ready_monitor.hpp"
#include "reduce_ready_plan.hpp"
#include <iostream>
#include <map>
#include <set>
#include <string>
using namespace MegaMoeUrma;
using namespace AscendC;
static_assert(ReadyMonitorUb::end == 92224U);
int main(int argc, char** argv) {
    assert(argc == 4);
    const std::string mode = argv[1];
    const uint32_t owner = std::stoul(argv[2]);
    priority = std::stoul(argv[3]);
    Arguments args{};
    auto& config = args.config;
    config.bs = mode == "maximum" ? 8192U : mode == "empty" ? 1U : 61U;
    config.topk = 8; config.rank = 0; config.maxPeerChunks = 512;
    config.fragmentCount = 4119; config.hidden = 3584;
    config.combineReadyOffset = 128;
    std::vector<uint64_t> tokenDependencies(config.bs * 8U);
    uint32_t firstToken = 0U;
    for (uint32_t worker = 0U; worker < 15U; ++worker) {
        const uint32_t tokens = config.bs / 15U + (worker < config.bs % 15U ? 1U : 0U);
        for (uint32_t local = 0U; local < tokens; ++local) {
            const uint32_t token = firstToken + local;
            for (uint32_t slot = 0U; slot < 8U; ++slot) {
                uint64_t dependency;
                if (mode == "maximum") dependency = 512U + local * 8U + slot;
                else if (mode == "multiplicity") dependency = 512U + token;
                else if (slot < 3U) dependency = 512U + token;
                else if (slot < 7U) dependency = 1024U + token / 2U;
                else dependency = kReduceSelfDependency | token;
                tokenDependencies[token * 8U + slot] = dependency;
            }
        }
        firstToken += tokens;
    }
    const auto plan = BuildReduceReadyPlan(tokenDependencies.data(), config);
    const auto& owned = plan.owners.at(owner);
    if (mode == "maximum" && owner < 2U) {
        assert(owned.tokenCount == 547U && owned.dependencyCount == 4376U && owned.edgeCount == 4376U);
    }
    args.reduceReadyOwners = reinterpret_cast<uint64_t>(plan.owners.data());
    args.reduceReadyDependencies = reinterpret_cast<uint64_t>(plan.dependencies.data());
    args.reduceReadyEdges = reinterpret_cast<uint64_t>(plan.edges.data());
    std::vector<uint64_t> remote(16U * config.maxPeerChunks * 16U + 16U);
    std::vector<int32_t> self(config.fragmentCount * 32U);
    constexpr uint64_t sentinel = 0xdadabeef12345678ULL;
    std::vector<uint64_t> output(config.bs * 16U, sentinel);
    args.symmetric = reinterpret_cast<uint64_t>(remote.data());
    args.gmm2Done = reinterpret_cast<uint64_t>(self.data());
    args.tokenReady = reinterpret_cast<uint64_t>(output.data());
    std::set<uint64_t> relevant;
    for (uint32_t i = 0U; i < owned.dependencyCount; ++i)
        relevant.insert(plan.dependencies[owned.dependencyBegin + i].dependency);
    auto isReady = [&](uint64_t dependency) {
        const uint64_t index = dependency & ~kReduceSelfDependency;
        return (dependency & kReduceSelfDependency) != 0U ? self[index * 32U] >= 14
            : remote[16U + index * 16U] == config.epoch;
    };
    auto arrive = [&](uint64_t dependency) {
        const uint64_t index = dependency & ~kReduceSelfDependency;
        if ((dependency & kReduceSelfDependency) != 0U) self[index * 32U] = 14;
        else remote[16U + index * 16U] = config.epoch;
    };
    auto tokenReady = [&](uint32_t token) {
        for (uint32_t slot = 0; slot < 8; ++slot)
            if (!isReady(tokenDependencies[token * 8U + slot])) return false;
        return true;
    };
    ReadyMonitor monitor;
    for (uint64_t epoch : {7ULL, 19ULL}) {
        config.epoch = epoch;
        std::fill(remote.begin(), remote.end(), epoch - 1U);
        std::fill(self.begin(), self.end(), 13);
        std::fill(output.begin(), output.end(), sentinel);
        std::vector<uint32_t> stores(config.bs);
        std::map<uint64_t, uint32_t> reads;
        uint64_t totalReads = 0U;
        bool completeDuringFinish = false;
        auto checkOutputs = [&] {
            for (uint32_t token = 0; token < config.bs; ++token) {
                const bool mine = token >= owned.tokenBegin && token < owned.tokenBegin + owned.tokenCount;
                const uint64_t* flag = output.data() + token * 16U;
                if (flag[0] == config.epoch) {
                    assert(mine && stores[token] == 1U && tokenReady(token));
                    assert(flag[1] == 0U && flag[2] == 0U && flag[3] == 0U);
                } else {
                    assert(flag[0] == sentinel && stores[token] == 0U);
                    assert(flag[1] == sentinel && flag[2] == sentinel && flag[3] == sentinel);
                }
                for (uint32_t word = 4; word < 16U; ++word) assert(flag[word] == sentinel);
            }
        };
        afterStep = checkOutputs;
        beforeStore = [&](uintptr_t address, uint32_t count) {
            assert(count == 4U && address >= args.tokenReady);
            const uint64_t offset = address - args.tokenReady;
            assert(offset % 128U == 0U && offset < output.size() * sizeof(uint64_t));
            const uint32_t token = offset / 128U;
            assert(token >= owned.tokenBegin && token < owned.tokenBegin + owned.tokenCount);
            assert(tokenReady(token) && stores[token]++ == 0U);
        };
        readFlag = [&](uintptr_t address) {
            uint64_t dependency, value;
            if (address >= args.gmm2Done && address < args.gmm2Done + self.size() * sizeof(int32_t)) {
                assert((address - args.gmm2Done) % 128U == 0U);
                const auto index = (address - args.gmm2Done) / 128U;
                dependency = kReduceSelfDependency | index;
                value = self[index * 32U];
            } else {
                assert(address >= args.symmetric + config.combineReadyOffset);
                const auto offset = address - args.symmetric - config.combineReadyOffset;
                assert(offset % 128U == 0U && offset < uint64_t(16U * 512U) * 128U);
                const auto index = offset / 128U;
                dependency = index; value = remote[16U + index * 16U];
            }
            assert(relevant.count(dependency) == 1U);
            ++reads[dependency];
            ++totalReads;
            assert(totalReads <= 20U * std::max<size_t>(1U, relevant.size()) *
                                     std::max<uint32_t>(1U, owned.tokenCount));
            if (completeDuringFinish && totalReads > 2U * relevant.size())
                for (auto item : relevant) arrive(item);
            return value;
        };
        const auto oldCopies = outputCopies;
        const auto oldCompleted = completedOutputs;
        const auto oldMetadata = metadataCopies;
        monitor.Init(args, owner);
        assert(monitor.TokenBegin() == owned.tokenBegin && monitor.TokenCount() == owned.tokenCount);
        assert(monitor.Done() == (owned.tokenCount == 0U));
        assert(metadataCopies - oldMetadata == (owned.dependencyCount ? 2U : 0U));
        // Old remote epochs and self=13 must never release payloads. A scan
        // returns after exactly one read of each still-unresolved dependency.
        for (uint32_t pass = 1; pass <= 2; ++pass) {
            assert(monitor.RefreshOnce(args) == 0U);
            assert(outputCopies == oldCopies);
            assert(reads.size() == relevant.size());
            for (auto dependency : relevant) assert(reads[dependency] == pass);
        }
        if (mode == "tail") {
            completeDuringFinish = true;
        } else if (mode == "maximum") {
            for (auto dependency : relevant) arrive(dependency);
            assert(monitor.RefreshOnce(args) == owned.tokenCount);
            const auto priorReads = reads;
            assert(monitor.RefreshOnce(args) == 0U && reads == priorReads);
        } else {
            std::set<uint64_t> consumed;
            uint32_t admitted = 0;
            for (uint32_t local = 0; local < owned.tokenCount; ++local) {
                const uint32_t token = owned.tokenBegin + local;
                for (uint32_t slot = 0; slot < 8; ++slot) arrive(tokenDependencies[token * 8U + slot]);
                const auto priorReads = reads;
                const uint32_t newlyReady = monitor.RefreshOnce(args);
                uint32_t expectedReady = 0;
                for (uint32_t candidate = owned.tokenBegin; candidate < owned.tokenBegin + owned.tokenCount;
                     ++candidate) expectedReady += tokenReady(candidate);
                assert(newlyReady == expectedReady - admitted);
                admitted += newlyReady;
                for (auto dependency : relevant) {
                    assert(reads[dependency] - priorReads.at(dependency) == (consumed.count(dependency) ? 0U : 1U));
                    if (isReady(dependency)) consumed.insert(dependency);
                }
                // Repeated calls, without Drain, exercise reuse of the
                // S_MTE3 gate while older 32-byte stores remain queued.
                const auto beforeRepeated = reads;
                assert(monitor.RefreshOnce(args) == 0U);
                for (auto dependency : consumed) assert(reads[dependency] == beforeRepeated.at(dependency));
                checkOutputs();
            }
        }
        if (owned.tokenCount && mode != "tail") {
            assert(monitor.Done());
            assert(outputCopies - oldCopies == owned.tokenCount);
            assert(completedOutputs - oldCompleted < owned.tokenCount);
        }
        monitor.Finish(args);
        assert(monitor.Done());
        PipeBarrier<PIPE_ALL>();
        checkOutputs();
        assert(outputCopies - oldCopies == owned.tokenCount);
        assert(completedOutputs - oldCompleted == owned.tokenCount);
        for (uint32_t token = 0; token < config.bs; ++token)
            assert(stores[token] == (token >= owned.tokenBegin && token < owned.tokenBegin + owned.tokenCount));
        for (const auto& queue : queues) assert(queue.empty());
        for (const auto& kind : events) for (bool flag : kind) assert(!flag);
        afterStep = {};
        beforeStore = {};
        readFlag = {};
    }
}
"""


@pytest.fixture(scope="module")
def monitor_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for ready-monitor tests")
    temporary = tmp_path_factory.mktemp("urma_ready_monitor")
    (temporary / "kernel_operator.h").write_text(MONITOR_ASCEND)
    source = temporary / "monitor.cpp"
    source.write_text(DRIVER)
    root = Path(__file__).resolve().parents[3]
    binary = temporary / ("monitor.exe" if os.name == "nt" else "monitor")
    result = subprocess.run(
        command
        + [
            "-std=c++17",
            "-O1",
            "-UNDEBUG",
            "-I" + str(temporary),
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


@pytest.mark.parametrize("priority", [0, 1, 2])
@pytest.mark.parametrize(
    "mode,owner",
    [
        ("staged", 0),
        ("staged", 7),
        ("staged", 14),
        ("multiplicity", 0),
        ("multiplicity", 14),
        ("empty", 14),
        ("maximum", 0),
        ("maximum", 1),
        ("maximum", 14),
        ("tail", 0),
        ("tail", 14),
    ],
)
def test_actual_monitor_pipeline(monitor_binary, mode, owner, priority):
    result = subprocess.run(
        [str(monitor_binary), mode, str(owner), str(priority)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_kernel_refresh_stays_between_put_and_quiet():
    root = Path(__file__).resolve().parents[3]
    source = (root / "experiments/megamoe/csrc/kernels/mega_moe/urma/kernel.hpp").read_text()
    combine = source[source.index("ReadyMonitor readyMonitor;") : source.index("__aicore__ inline void Process")]
    assert (
        combine.index("WaitChunkComputed(args, chunk, timer);")
        < combine.index("Put(symmetric,")
        < combine.index("readyMonitor.RefreshOnce(args);")
        < combine.index("Quiet(peer);")
        < combine.index("PublishDone(symmetric,")
        < combine.index("readyMonitor.Finish(args);")
    )
    assert combine.count("readyMonitor.RefreshOnce(args);") == 1
