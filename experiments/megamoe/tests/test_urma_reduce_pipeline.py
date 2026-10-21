"""Run the production Reduce header with asynchronous CPU pipeline queues.

The fake Ascend operators model FIFO engines, cross-engine events, UB reuse,
and BF16/FP32 arithmetic. The single Reduce trace must cover dependency waits
and end only after output drain. This does not model NPU timing or replace
device precision validation.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


FAKE_ASCEND = r"""
#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <vector>
#define __aicore__
#define __gm__
namespace AscendC {
enum class TPosition { VECCALC };
enum class CacheMode { CACHE_MODE_DISABLE };
enum class RoundMode { CAST_NONE, CAST_RINT };
using TEventID = uint32_t;
constexpr TEventID EVENT_ID0 = 0;
enum class HardEvent { V_MTE2, MTE2_V, MTE3_V, V_MTE3, S_MTE2, MTE2_S };
enum { PIPE_V, PIPE_ALL };
enum Engine : unsigned { Scalar, Mte2, Vector, Mte3 };
struct EventEngines { Engine source, destination; };
constexpr EventEngines Engines(HardEvent event) {
    switch (event) {
    case HardEvent::V_MTE2: return {Vector, Mte2};
    case HardEvent::MTE2_V: return {Mte2, Vector};
    case HardEvent::MTE3_V: return {Mte3, Vector};
    case HardEvent::V_MTE3: return {Vector, Mte3};
    case HardEvent::S_MTE2: return {Scalar, Mte2};
    case HardEvent::MTE2_S: return {Mte2, Scalar};
    }
    assert(false);
    return {};
}
struct bfloat16_t {
    uint16_t bits{};
    bfloat16_t() = default;
    explicit bfloat16_t(float value) {
        uint32_t raw;
        std::memcpy(&raw, &value, sizeof(raw));
        bits = uint16_t((raw + 0x7fffU + ((raw >> 16U) & 1U)) >> 16U);
    }
    operator float() const {
        uint32_t raw = uint32_t(bits) << 16U;
        float value;
        std::memcpy(&value, &raw, sizeof(value));
        return value;
    }
};
static_assert(sizeof(bfloat16_t) == 2);
using Command = std::function<bool()>;
inline std::array<std::deque<Command>, 4> queues;
inline bool events[6][8]{};
alignas(64) inline std::array<unsigned char, 192 * 1024> ub;
inline unsigned priority;
inline uint64_t inputCopies, outputCopies, metadataCopies, completedOutputs;
inline uint64_t readyGateSubmissions, readyGateExecutions;
inline std::vector<uintptr_t> metadataSources;
inline bool bypass;
inline std::function<void(uintptr_t)> beforePayload;
inline std::function<void(uintptr_t)> executePayload;
inline std::function<uint64_t(uintptr_t)> readFlag;
inline std::function<void()> onCycle;
inline std::function<void(uintptr_t)> submitOutput;
inline std::function<void(uintptr_t)> executeOutput;
inline constexpr uintptr_t returnedBase = uintptr_t(1) << 40U;
inline constexpr uintptr_t returnedEnd = returnedBase + (uintptr_t(1) << 36U);
inline bfloat16_t Returned(uint64_t row, uint32_t column) {
    return bfloat16_t(float(int((row * 37U + column * 19U) % 257U) - 128) / 16.0F);
}
inline void Enqueue(Engine engine, Command command) {
    if (engine == Scalar) { assert(command()); }
    else queues[engine].push_back(std::move(command));
}
inline bool Step() {
    bool progress = false;
    for (unsigned n = 0; n < 3; ++n) {
        auto& queue = queues[1U + (n + priority) % 3U];
        if (!queue.empty() && queue.front()()) { queue.pop_front(); progress = true; }
    }
    return progress;
}
inline void Drain() {
    while (!queues[Mte2].empty() || !queues[Vector].empty() || !queues[Mte3].empty()) {
        assert(Step() && "pipeline event deadlock");
    }
}
inline uint64_t GetSystemCycle() {
    static uint64_t cycle = 0;
    if (onCycle) onCycle();
    return cycle += 200U;
}
template <typename T> T ReadGmByPassDCache(T* address) {
    assert(readFlag);
    return static_cast<T>(readFlag(reinterpret_cast<uintptr_t>(address)));
}
template <HardEvent Event> void SetFlag(TEventID id) {
    assert(id < 8);
    if constexpr (Event == HardEvent::S_MTE2) if (id == 6U) ++readyGateSubmissions;
    Enqueue(Engines(Event).source, [=] {
        auto& flag = events[unsigned(Event)][id];
        assert(!flag && "event reset without consumption");
        flag = true;
        return true;
    });
}
template <HardEvent Event> void WaitFlag(TEventID id) {
    assert(id < 8);
    auto wait = [=] {
        auto& flag = events[unsigned(Event)][id];
        if (!flag) return false;
        flag = false;
        if constexpr (Event == HardEvent::S_MTE2) if (id == 6U) ++readyGateExecutions;
        return true;
    };
    if (Engines(Event).destination == Scalar) {
        while (!wait()) { assert(Step() && "Scalar event deadlock"); }
    } else Enqueue(Engines(Event).destination, wait);
}
template <unsigned Pipe> void PipeBarrier() {
    if constexpr (Pipe == PIPE_ALL) Drain();
    else Enqueue(Vector, [] { return true; });
}
template <typename T> struct LocalTensor {
    uint32_t address{}, count{};
    LocalTensor() = default;
    LocalTensor(TPosition, uint32_t offset, uint32_t n) : address(offset), count(n) {
        assert(address % 32U == 0U);
        assert(uint64_t(address) + uint64_t(count) * sizeof(T) <= ub.size());
    }
    LocalTensor(uint32_t offset, uint32_t n) : address(offset), count(n) {}
    LocalTensor operator[](uint64_t offset) const {
        assert(offset <= count);
        return LocalTensor(address + uint32_t(offset * sizeof(T)), count - uint32_t(offset));
    }
    T* Data() const { return reinterpret_cast<T*>(ub.data() + address); }
    T GetValue(uint64_t offset) const { assert(offset < count); return Data()[offset]; }
    void SetValue(uint64_t offset, T value) const { assert(offset < count); Data()[offset] = value; }
};
template <typename T> struct GlobalTensor {
    T* data{};
    void SetGlobalBuffer(T* pointer) { data = pointer; }
    void SetL2CacheHint(CacheMode) { bypass = true; }
    GlobalTensor operator[](uint64_t offset) const {
        GlobalTensor result;
        result.data = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(data) + offset * sizeof(T));
        return result;
    }
};
template <typename T> void DataCopy(LocalTensor<T> destination, GlobalTensor<T> source, uint32_t count) {
    assert(count <= destination.count);
    assert(count * sizeof(T) % 32U == 0U);
    const uintptr_t address = reinterpret_cast<uintptr_t>(source.data);
    if (address >= returnedBase && address < returnedEnd) {
        assert(beforePayload);
        beforePayload(address);
        ++inputCopies;
        Enqueue(Mte2, [=] {
            if (executePayload) executePayload(address);
            const uint64_t index = (address - returnedBase) / sizeof(bfloat16_t);
            for (uint32_t i = 0; i < count; ++i) {
                if constexpr (sizeof(T) == 2) {
                    destination.Data()[i] = Returned((index + i) / 3584U, (index + i) % 3584U);
                } else assert(false);
            }
            return true;
        });
    } else {
        ++metadataCopies;
        metadataSources.push_back(address);
        Enqueue(Mte2, [=] {
            std::copy_n(source.data, count, destination.Data());
            return true;
        });
    }
}
template <typename T> void DataCopy(GlobalTensor<T> destination, LocalTensor<T> source, uint32_t count) {
    assert(count <= source.count);
    if (submitOutput) submitOutput(reinterpret_cast<uintptr_t>(destination.data));
    ++outputCopies;
    Enqueue(Mte3, [=] {
        std::copy_n(source.Data(), count, destination.data);
        if (executeOutput) executeOutput(reinterpret_cast<uintptr_t>(destination.data));
        ++completedOutputs;
        return true;
    });
}
template <typename D, typename S> void Cast(LocalTensor<D> destination, LocalTensor<S> source,
                                           RoundMode mode, uint32_t count) {
    assert(count <= source.count && count <= destination.count);
    Enqueue(Vector, [=] {
        for (uint32_t i = 0; i < count; ++i) destination.Data()[i] = D(float(source.Data()[i]));
        return true;
    });
}
inline void Muls(LocalTensor<float> destination, LocalTensor<float> source, float weight, uint32_t count) {
    Enqueue(Vector, [=] {
        for (uint32_t i = 0; i < count; ++i) destination.Data()[i] = source.Data()[i] * weight;
        return true;
    });
}
inline void Add(LocalTensor<float> destination, LocalTensor<float> lhs, LocalTensor<float> rhs, uint32_t count) {
    Enqueue(Vector, [=] {
        for (uint32_t i = 0; i < count; ++i) destination.Data()[i] = lhs.Data()[i] + rhs.Data()[i];
        return true;
    });
}
}  // namespace AscendC
"""

DRIVER = r"""
#include "reduce_pipeline.hpp"
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
using namespace MegaMoeUrma;
using namespace AscendC;
static_assert(ReduceUbContract::end == 191136U);
struct TraceSink {
    uint64_t begins{}, ends{}, expectedRow{}, expectedRows{};
    bool active{};
    uint64_t TraceBegin() {
        assert(!active && begins == 0U && ends == 0U);
        assert(metadataCopies == 0U && inputCopies == 0U && completedOutputs == 0U);
        active = true;
        ++begins;
        return 73U;
    }
    void TraceEnd(TraceEvent event, uint64_t start, const TraceMeta& meta = TraceMeta{}) {
        assert(event == TraceEvent::Reduce && start == 73U);
        assert(active && begins == 1U && ends == 0U);
        assert(meta.row == expectedRow && meta.rows == expectedRows);
        assert(meta.bytes == expectedRows * 3584U * sizeof(bfloat16_t));
        assert(completedOutputs == expectedRows);
        for (const auto& queue : queues) assert(queue.empty());
        for (const auto& kind : events) for (auto event : kind) assert(!event);
        active = false;
        ++ends;
    }
};
int main(int argc, char** argv) {
    if (argc == 2) {
        for (uint32_t rows : {1U, 40U, 41U, 42U, 2048U, 4096U, 8192U}) {
            std::vector<uint32_t> coverage(rows), workers(kReduceWorkers);
            for (uint32_t block = 0; block < 28; ++block) {
                const auto first = ReduceWorkerIndex(block, 0);
                assert(first == block && ++workers[first] == 1);
                if (block >= 15) {
                    const auto second = ReduceWorkerIndex(block, 1);
                    assert(second == 28U + block - 15U && ++workers[second] == 1);
                }
            }
            for (uint32_t worker = 0; worker < kReduceWorkers; ++worker) {
                const auto start = ReduceRowBegin(rows, worker), count = ReduceOwnedRows(rows, worker);
                for (uint64_t row = start; row < start + count; ++row) assert(++coverage[row] == 1);
            }
            for (auto n : coverage) assert(n == 1U);
            for (auto n : workers) assert(n == 1U);
        }
        return 0;
    }
    assert(argc == 5 || argc == 6);
    const std::string mode = argc == 6 ? argv[5] : "delayed";
    const bool switchMode = mode.rfind("switch:", 0U) == 0U;
    const uint32_t switchAfter = switchMode ? std::stoul(mode.substr(7U)) : 0U;
    const uint32_t bs = std::stoul(argv[1]), worker = std::stoul(argv[2]);
    priority = std::stoul(argv[3]);
    std::mt19937 random(std::stoul(argv[4]));
    const uint32_t rows = bs * 8U;
    const auto count = ReduceOwnedRows(bs, worker), begin = ReduceRowBegin(bs, worker);
    std::vector<int64_t> inverse(rows);
    std::vector<float> weights(rows);
    std::vector<uint64_t> dependencies(rows), routeForRow(rows);
    for (uint32_t i = 0; i < rows; ++i) {
        inverse[i] = i;
        weights[i] = float(int(random() % 2001U) - 1000) / 997.0F;
    }
    std::shuffle(inverse.begin(), inverse.end(), random);
    for (uint32_t route = 0; route < rows; ++route) routeForRow[inverse[route]] = route;
    for (uint32_t local = 0; local < count; ++local) {
        const std::array<uint64_t, 8> tokenDependencies = {
            512U + local, 8191U, kReduceSelfDependency | local, kReduceSelfDependency | 2753U,
            512U + local, 8191U, kReduceSelfDependency | local, kReduceSelfDependency | 2753U};
        for (uint32_t slot = 0; slot < 8; ++slot)
            dependencies[(begin + local) * 8U + slot] = tokenDependencies[slot];
    }
    std::vector<bfloat16_t> output(uint64_t(bs) * 3584U);
    std::vector<uint32_t> desired;
    for (uint32_t local : {1U, 7U, 2U, 0U}) if (local < count) desired.push_back(local);
    for (uint32_t local = 0; local < count; ++local)
        if (std::find(desired.begin(), desired.end(), local) == desired.end()) desired.push_back(local);
    ub.fill(0xa5);
    for (uint32_t invocation = 0U; invocation < (mode == "epochs" ? 2U : 1U); ++invocation) {
        // Keep the previous invocation's UB bitmaps deliberately intact.
        inputCopies = outputCopies = metadataCopies = completedOutputs = 0U;
        readyGateSubmissions = readyGateExecutions = 0U;
        metadataSources.clear(); bypass = false;
        for (auto& value : output) value.bits = 0x7fc1;
        Arguments args{};
        args.config.hidden = 3584; args.config.topk = 8; args.config.bs = bs;
        args.config.maxPeerChunks = 512; args.config.fragmentCount = 2754;
        args.config.epoch = 7U + invocation;
        args.symmetric = returnedBase;
        args.config.combineReadyOffset = uintptr_t(1) << 35U;
        args.gmm2Done = uintptr_t(1) << 42U;
        args.localFlags = uintptr_t(1) << 43U;
        args.output = reinterpret_cast<uint64_t>(output.data());
        args.reduceIndex = reinterpret_cast<uint64_t>(inverse.data());
        args.weights = reinterpret_cast<uint64_t>(weights.data());
        args.reduceDependencies = reinterpret_cast<uint64_t>(dependencies.data());
        uint64_t pollClock = 0U, aggregatePolls = 0U, gatesBeforeLatch = 0U, crossTokenPrefetches = 0U;
        uint64_t cycleCalls = 0U, backoffRounds = 0U;
        uint32_t readyCandidate = count;
        bool allInputsObserved = false;
        std::vector<uint32_t> admissionOrder, outputOrder;
        std::vector<bool> admitted(count), written(count);
        std::vector<uint64_t> payloadVisits(rows), payloadGates(rows);
        std::map<uint64_t, uint32_t> positiveReads, failedReads;
        std::set<uint32_t> failedTokens;
        TraceSink timer;
        timer.expectedRow = begin; timer.expectedRows = count;
        auto tokenCondition = [&](uint32_t local) {
            if (mode == "head") return local != 0U || written[count - 1U];
            if (mode == "last") return local == count - 1U || written[count - 1U];
            if (mode == "sequence")
                return admissionOrder.size() < desired.size() && desired[admissionOrder.size()] == local;
            if (mode == "stalled") return backoffRounds >= 2U;
            if (mode == "wait-switch") return false;
            if (mode == "delayed" || mode == "epochs" || mode == "future" || mode == "self-delayed")
                return pollClock >= 20U + ((local * 37U) % std::max<uint64_t>(count, 1U)) * 7U;
            return true;
        };
        auto ready = [&](uint64_t dependency) {
            if (dependency == 8191U || dependency == (kReduceSelfDependency | 2753U)) return true;
            const auto index = dependency & ~kReduceSelfDependency;
            const uint32_t local = dependency & kReduceSelfDependency ? index : index - 512U;
            assert(local < count);
            if (mode == "self-delayed" && !(dependency & kReduceSelfDependency)) return true;
            return tokenCondition(local);
        };
        readFlag = [&](uintptr_t address) {
            assert(timer.active && !allInputsObserved && "ready GM read after aggregate latch");
            assert(readyCandidate == count && "a ready token was not submitted immediately");
            ++pollClock;
            assert(pollClock < 200000U && "ready-first scanner made no progress");
            Step();
            if (address == args.localFlags + uint64_t(kLocalFlagIncomingDone) * kFlagBytes) {
                ++aggregatePolls;
                const bool complete = mode == "fast" || (switchMode && admissionOrder.size() >= switchAfter) ||
                                      (mode == "wait-switch" && aggregatePolls >= 3U);
                if (complete) { allInputsObserved = true; gatesBeforeLatch = readyGateSubmissions; }
                return uint64_t(complete ? 16U : mode == "overcount" ? 17U : 15U);
            }
            uint64_t dependency;
            if (address >= args.gmm2Done && address < args.gmm2Done + uint64_t(2754U) * 128U) {
                assert((address - args.gmm2Done) % 128U == 0U);
                dependency = kReduceSelfDependency | ((address - args.gmm2Done) / 128U);
            } else {
                const uintptr_t remote = args.symmetric + args.config.combineReadyOffset;
                assert(address >= remote && address < remote + uint64_t(8192U) * 128U);
                assert((address - remote) % 128U == 0U);
                dependency = (address - remote) / 128U;
            }
            assert(positiveReads[dependency] == 0U && "positive readiness must be cached for the invocation");
            const bool complete = ready(dependency);
            if (complete) ++positiveReads[dependency];
            else {
                ++failedReads[dependency];
                const uint64_t index = dependency & ~kReduceSelfDependency;
                failedTokens.insert(dependency & kReduceSelfDependency ? index : index - 512U);
            }
            if (complete) for (uint32_t local = 0U; local < count; ++local) {
                if (admitted[local]) continue;
                bool tokenReady = true;
                for (uint32_t slot = 0U; slot < 8U; ++slot)
                    tokenReady &= positiveReads[dependencies[(begin + local) * 8U + slot]] == 1U;
                if (tokenReady) {
                    assert(readyCandidate == count);
                    readyCandidate = local;
                }
            }
            if (dependency & kReduceSelfDependency) return uint64_t(complete ? 28U : 27U);
            return complete ? args.config.epoch : mode == "future" ? args.config.epoch + 1U : args.config.epoch - 1U;
        };
        beforePayload = [&](uintptr_t address) {
            assert(timer.active);
            const auto row = (address - returnedBase) / sizeof(bfloat16_t) / 3584U;
            const auto route = routeForRow[row], token = route / 8U, local = token - begin;
            assert(token >= begin && local < count && readyGateSubmissions > 0U);
            if (!allInputsObserved) for (uint32_t slot = 0; slot < 8; ++slot)
                assert(positiveReads[dependencies[token * 8U + slot]] == 1U && "payload precedes ready acquire");
            if (route % 8U == 0U) {
                assert(!admitted[local]);
                if (!allInputsObserved) assert(readyCandidate == local);
                readyCandidate = count;
                admitted[local] = true;
                admissionOrder.push_back(local);
                failedTokens.clear();
            }
            assert(admitted[local] && admissionOrder.back() == local);
            assert(inputCopies % 8U == route % 8U && "TopK route order changed");
            if (admissionOrder.size() > outputCopies + 1U) ++crossTokenPrefetches;
            payloadGates[route] = readyGateSubmissions;
            assert(++payloadVisits[route] == 1U);
        };
        executePayload = [&](uintptr_t address) {
            const auto row = (address - returnedBase) / sizeof(bfloat16_t) / 3584U;
            const auto route = routeForRow[row];
            assert(payloadGates[route] != 0U && readyGateExecutions >= payloadGates[route]);
        };
        submitOutput = [&](uintptr_t address) {
            const uint32_t token = (address - args.output) / (3584U * sizeof(bfloat16_t));
            assert(address >= args.output && token >= begin && token - begin < count);
            assert(outputOrder.size() < admissionOrder.size());
            assert(token - begin == admissionOrder[outputOrder.size()]);
            outputOrder.push_back(token - begin);
        };
        executeOutput = [&](uintptr_t address) {
            const uint32_t token = (address - args.output) / (3584U * sizeof(bfloat16_t));
            assert(!written[token - begin]); written[token - begin] = true;
        };
        onCycle = [&] {
            // Only a whole unsuccessful scan can enter the 200-cycle delay.
            if ((cycleCalls++ & 1U) == 0U) {
                for (uint32_t local = 0; local < count; ++local)
                    if (!admitted[local]) assert(failedTokens.count(local) == 1U);
                ++backoffRounds;
            }
            Step();
        };
        Reduce(args, worker, timer);
        assert(inputCopies == count * 8U && outputCopies == count && completedOutputs == count);
        assert(metadataCopies == (count == 0U ? 0U : mode == "fast" ? 2U : 3U));
        assert(admissionOrder.size() == count && outputOrder == admissionOrder);
        assert(timer.begins == (count != 0U) && timer.ends == (count != 0U) && !timer.active);
        if (allInputsObserved) assert(readyGateSubmissions == gatesBeforeLatch + 1U);
        else assert(readyGateSubmissions == count);
        assert(readyGateExecutions == readyGateSubmissions);
        assert(bypass == (count != 0U));
        if (count != 0U) {
            assert(aggregatePolls > 0U);
            if (mode == "fast") {
                assert(aggregatePolls == 1U && positiveReads.empty() && failedReads.empty());
                assert(cycleCalls == 0U);
            }
            if (mode == "head") assert(admissionOrder.front() == 1U && admissionOrder.back() == 0U);
            if (mode == "last") assert(admissionOrder.front() == count - 1U);
            if (mode == "sequence") assert(admissionOrder == desired && cycleCalls == 0U);
            if (mode == "stalled") assert(backoffRounds == 2U && failedReads[512U] >= 2U);
            if (mode == "overcount") assert(!allInputsObserved);
            if (mode == "wait-switch") {
                assert(allInputsObserved);
                for (const auto& entry : positiveReads) assert(entry.second == 0U);
            }
            if (switchMode) assert(allInputsObserved);
            if (mode == "ready" || mode == "overcount" || mode == "fast") {
                if (count > 1U) assert(crossTokenPrefetches > 0U);
                assert(cycleCalls == 0U);
            }
            if (mode != "fast" && mode != "wait-switch") {
                assert(positiveReads[8191U] == 1U && positiveReads[kReduceSelfDependency | 2753U] == 1U);
                if (!allInputsObserved) {
                    for (uint32_t local = 0; local < count; ++local)
                        assert(positiveReads[512U + local] == 1U && positiveReads[kReduceSelfDependency | local] == 1U);
                }
            }
        } else assert(aggregatePolls == 0U);
        for (const auto& queue : queues) assert(queue.empty());
        for (const auto& kind : events) for (auto event : kind) assert(!event);
        for (uint32_t token = 0; token < bs; ++token) {
            const bool mine = token >= begin && token - begin < count;
            for (uint32_t column = 0; column < 3584; ++column) {
                if (!mine) { assert(output[uint64_t(token) * 3584U + column].bits == 0x7fc1); continue; }
                float accumulator = 0.0F;
                for (uint32_t slot = 0; slot < 8; ++slot) {
                    const auto route = uint64_t(token) * 8U + slot;
                    volatile float term = float(Returned(inverse[route], column)) * weights[route];
                    accumulator = slot == 0 ? term : accumulator + term;
                }
                assert(output[uint64_t(token) * 3584U + column].bits == bfloat16_t(accumulator).bits);
            }
        }
    }
}
"""


@pytest.fixture(scope="module")
def reduce_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for the Reduce pipeline test")
    temporary = tmp_path_factory.mktemp("urma_reduce_pipeline")
    (temporary / "kernel_operator.h").write_text(FAKE_ASCEND, encoding="utf-8")
    source = temporary / "reduce.cpp"
    source.write_text(DRIVER, encoding="utf-8")
    binary = temporary / ("reduce.exe" if os.name == "nt" else "reduce")
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            *command,
            "-std=c++17",
            "-O1",
            "-UNDEBUG",
            "-ffp-contract=off",
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


@pytest.mark.parametrize("bs,worker", [(1, 0), (1, 40), (40, 39), (41, 40), (42, 0), (42, 40)])
@pytest.mark.parametrize("priority", [0, 1, 2])
def test_reduce_short_and_empty_workers(reduce_binary, bs, worker, priority):
    result = subprocess.run(
        [str(reduce_binary), str(bs), str(worker), str(priority), "17"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bs", [2048, 4096, 8192])
@pytest.mark.parametrize("worker", [0, 40])
@pytest.mark.parametrize("priority", [0, 1, 2])
def test_reduce_full_cases_preserve_arithmetic_and_events(reduce_binary, bs, worker, priority):
    result = subprocess.run(
        [str(reduce_binary), str(bs), str(worker), str(priority), str(bs + worker)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_reduce_41_worker_partition_covers_every_token(reduce_binary):
    result = subprocess.run([str(reduce_binary), "coverage"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bs", [42, 2048, 4096, 8192])
@pytest.mark.parametrize("priority", [0, 1, 2])
def test_ready_tokens_do_not_overwrite_unconsumed_gate_events(reduce_binary, bs, priority):
    result = subprocess.run(
        [str(reduce_binary), str(bs), "0", str(priority), "29", "ready"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bs,worker", [(1, 40), (42, 0), (2048, 40), (4096, 0), (8192, 0)])
@pytest.mark.parametrize("priority", [0, 1, 2])
def test_aggregate_ready_restores_cross_token_pipeline_without_further_polls(reduce_binary, bs, worker, priority):
    result = subprocess.run(
        [str(reduce_binary), str(bs), str(worker), str(priority), "31", "fast"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "mode",
    [
        "switch:8",
        "switch:9",
        "switch:120",
        "switch:127",
        "switch:128",
        "wait-switch",
        "overcount",
    ],
)
@pytest.mark.parametrize("priority", [0, 1, 2])
def test_aggregate_ready_latches_during_full_token_scans(reduce_binary, mode, priority):
    result = subprocess.run(
        [str(reduce_binary), "8192", "40", str(priority), "37", mode],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["head", "last", "sequence", "stalled", "epochs", "future", "self-delayed"])
@pytest.mark.parametrize("priority", [0, 1, 2])
def test_ready_first_skips_blocked_tokens_without_batching_or_reusing_positive_cache(reduce_binary, mode, priority):
    result = subprocess.run(
        [str(reduce_binary), "8192", "0", str(priority), "43", mode],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
