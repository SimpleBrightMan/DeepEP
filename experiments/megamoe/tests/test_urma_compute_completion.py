"""Exercise the real deferred GMM completion queue with host AscendC stubs."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


URMA = Path(__file__).resolve().parents[1] / "csrc/kernels/mega_moe/urma"


@pytest.mark.parametrize("engine", [0, 1])
@pytest.mark.parametrize("enabled", [False, True])
def test_deferred_completion_protocol(tmp_path, engine, enabled):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("A host C++ compiler is required to execute the completion queue")
    (tmp_path / "kernel_operator.h").write_text(
        r"""
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <vector>
#define __aicore__
#define __gm__
namespace AscendC {
constexpr int AIC = 0, AIV = 1;
enum class HardEvent { FIX_S };
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
enum Kind { SUBMIT, SET, WAIT, WRITE, ATOMIC, READ };
struct Operation { Kind kind; uint64_t value; };
inline std::vector<Operation> operations;
inline uint64_t issued = 0, completed = 0, cycles = 100, flushes = 0;
inline uint64_t marker[2]{};
inline bool armed[2]{};
inline std::map<int32_t*, uint64_t> required;
inline int32_t* delayedInput = nullptr;
inline int32_t* inputDependency = nullptr;
inline uint32_t delayedReads = 0;
inline uint64_t GetSystemCycle() { return cycles++; }
inline uint32_t GetBlockIdx() { return 0; }
inline uint32_t GetTaskRation() { return 2; }
inline uint32_t GetSubBlockIdx() { return 0; }
inline void Submit() { operations.push_back({SUBMIT, ++issued}); }
template<HardEvent Event> void SetFlag(uint16_t id) {
    static_assert(Event == HardEvent::FIX_S);
    assert(id < 2 && !armed[id]);
    armed[id] = true; marker[id] = issued;
    operations.push_back({SET, id});
}
template<HardEvent Event> void WaitFlag(uint16_t id) {
    static_assert(Event == HardEvent::FIX_S);
    assert(id < 2 && armed[id]);
    completed = std::max(completed, marker[id]);
    armed[id] = false;
    operations.push_back({WAIT, id});
}
inline void WriteGmByPassDCache(int32_t* flag, int32_t value) {
    assert(required.count(flag) && completed >= required.at(flag));
    operations.push_back({WRITE, required.at(flag)});
    *flag = value;
}
inline void AtomicAdd(int32_t* flag, int32_t value) {
    assert(required.count(flag) && completed >= required.at(flag));
    operations.push_back({ATOMIC, required.at(flag)});
    *flag += value;
}
inline int32_t ReadGmByPassDCache(int32_t* flag) {
    operations.push_back({READ, issued});
    if (flag == delayedInput) {
        // Fail promptly instead of hanging if the pending GMM1 notification
        // is held while waiting for the activation consumer to produce input.
        assert(++delayedReads < 8);
        if (*inputDependency == 1 && delayedReads >= 3) *flag = 5;
    }
    return *flag;
}
template<class T> struct GlobalTensor { void SetGlobalBuffer(T*) {} };
template<class T, CacheLine, DcciDst>
void DataCacheCleanAndInvalid(GlobalTensor<T>&) { ++flushes; }
}
constexpr int g_coreType = ENGINE == 0 ? AscendC::AIC : AscendC::AIV;
""",
        encoding="utf-8",
    )
    source = tmp_path / "compute_completion.cpp"
    source.write_text(
        r"""
#include "compute_completion.hpp"
#include <cassert>
#include <vector>
using namespace MegaMoeUrma;
struct Record { TraceEvent event; uint64_t start; uint32_t fragment; };
int main() {
    constexpr uint64_t guard = 0x123456789abcdef0ULL;
    constexpr uint64_t capacity = 12;
    constexpr uint64_t stride = kTraceHeaderElements + capacity * kTraceRecordElements;
    std::vector<uint64_t> memory(kTimerCores * stride + 2, guard);
    auto* buffer = memory.data() + 1;
    Timer timer;
    timer.Init(0, reinterpret_cast<uint64_t>(buffer), capacity);
    DeferredGmmCompletion completion;
    int32_t first = 0, second = 0, output = 0, ready = 5;
    std::vector<Record> expected;
    uint32_t published = 0;
    auto arm = [&](int32_t* flag, TraceEvent event, uint64_t start, uint32_t fragment) {
        TraceMeta meta;
        meta.fragment = fragment; meta.expert = 2; meta.rows = 17; meta.tileN = 256;
        AscendC::required[flag] = AscendC::issued;
        completion.Arm(flag, event, start, meta);
        expected.push_back({event, start, fragment});
    };
    auto checkRecord = [&]() {
#ifdef DEEPEP_MEGAMOE_TIMER
        auto* record = buffer + kTraceHeaderElements + published * kTraceRecordElements;
        const auto& want = expected.at(published);
        assert(record[0] == want.start && record[1] > want.start);
        assert(record[2] == static_cast<uint64_t>(want.event));
        assert(record[5] == want.fragment && record[6] == 2);
        assert(record[8] == 17 && record[10] == 256);
        assert(record[13] == kTraceFlagDeferredCompletion);
        assert(record[14] == 0 && record[15] == 0);
#else
        for (auto value : memory) assert(value == guard);
#endif
        ++published;
    };
    if constexpr (g_coreType == AscendC::AIV) {
        // The shared AIC/AIV0 call path must not claim FIX events, inspect
        // readiness, modify completion flags or generate deferred AIC traces.
        completion.Arm(&first, TraceEvent::Gmm1Tile, 10, TraceMeta{});
        completion.WaitInput(&first, 1, timer);
        completion.Publish(timer);
        completion.Arm(&output, TraceEvent::Gmm2Tile, 20, TraceMeta{});
        completion.Publish(timer);
        assert(first == 0 && output == 0 && AscendC::operations.empty());
        assert(AscendC::cycles == 100 && AscendC::flushes == 0);
        for (auto value : memory) assert(value == guard);
        return 0;
    }
    completion.Publish(timer);
    assert(AscendC::operations.empty());
    auto start1 = timer.TraceBegin();
    AscendC::Submit();
    arm(&first, TraceEvent::Gmm1Tile, start1, 11);
    assert(first == 0 && AscendC::completed == 0 && AscendC::armed[1]);
    assert(AscendC::flushes == 0);  // Arm does not end the tile trace.

    auto start2 = timer.TraceBegin();
    AscendC::Submit();
    completion.Publish(timer);
    assert(first == 1 && AscendC::completed == 1 && AscendC::issued == 2);
    assert(AscendC::operations[2].kind == AscendC::SUBMIT);
    assert(AscendC::operations[3].kind == AscendC::WAIT);
    assert(AscendC::operations[3].value == 1);
    assert(AscendC::operations[4].kind == AscendC::WRITE);
    checkRecord();
    arm(&output, TraceEvent::Gmm2Tile, start2, 12);
    assert(output == 0);

    // Context End uses FIX_S0. It may drain the tile's stores but must not
    // consume FIX_S1 or publish the pending completion ahead of the next task.
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
    assert(AscendC::armed[1] && !AscendC::armed[0] && output == 0);
    auto start3 = timer.TraceBegin();
    AscendC::Submit();
    completion.Publish(timer);
    assert(output == 1 && AscendC::completed == 2 && AscendC::issued == 3);
    checkRecord();
    auto operations = AscendC::operations.size();
    completion.Publish(timer);
    assert(AscendC::operations.size() == operations && output == 1);
    arm(&output, TraceEvent::Gmm2Tile, start3, 12);

    // A ready dependency keeps the prior notification pending so the next
    // Compute can be issued before the previous FIX completion is consumed.
    operations = AscendC::operations.size();
    completion.WaitInput(&ready, 5, timer);
    assert(AscendC::operations.size() == operations + 1);
    assert(AscendC::operations.back().kind == AscendC::READ);
    assert(AscendC::armed[1] && output == 1);
    auto start4 = timer.TraceBegin();
    AscendC::Submit();
    completion.Publish(timer);
    assert(output == 2 && AscendC::completed == 3);
    checkRecord();
    arm(&second, TraceEvent::Gmm1Tile, start4, 13);

    // An unready input must release pending output first. The mocked consumer
    // cannot make input ready until that GMM1 flag has actually been written.
    int32_t delayed = 0;
    AscendC::delayedInput = &delayed;
    AscendC::inputDependency = &second;
    operations = AscendC::operations.size();
    completion.WaitInput(&delayed, 5, timer);
    assert(second == 1 && delayed == 5 && AscendC::delayedReads == 3);
    assert(AscendC::operations[operations].kind == AscendC::READ);
    assert(AscendC::operations[operations + 1].kind == AscendC::WAIT);
    assert(AscendC::operations[operations + 2].kind == AscendC::WRITE);
    assert(AscendC::operations[operations + 3].kind == AscendC::READ);
    assert(!AscendC::armed[1]);
    checkRecord();

    // No subsequent task exists at the end of the kernel. Explicit final
    // Publish still counts the last GMM2 tile once and closes its trace.
    auto start5 = timer.TraceBegin();
    AscendC::Submit();
    arm(&output, TraceEvent::Gmm2Tile, start5, 14);
    assert(output == 2);
    completion.Publish(timer);
    assert(output == 3 && AscendC::completed == 5 && !AscendC::armed[1]);
    checkRecord();
    operations = AscendC::operations.size();
    completion.Publish(timer);
    assert(AscendC::operations.size() == operations && output == 3);
    timer.FinishTrace();
    assert(memory.front() == guard && memory.back() == guard);
#ifdef DEEPEP_MEGAMOE_TIMER
    assert(buffer[0] == kTraceSchema && buffer[2] == published && buffer[3] == 0);
    assert(AscendC::flushes == published + 1);
#else
    for (auto value : memory) assert(value == guard);
    assert(AscendC::cycles == 100 && AscendC::flushes == 0);
#endif
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "compute_completion"
    command = [*compiler, "-std=c++17", f"-DENGINE={engine}"]
    if enabled:
        command.append("-DDEEPEP_MEGAMOE_TIMER")
    subprocess.run(
        [*command, "-I", str(tmp_path), "-I", str(URMA), str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_compute_hooks_deferred_completion_after_next_submit():
    source = (URMA / "compute.hpp").read_text(encoding="utf-8")
    gmm1 = source.split("void RunGmm1Fragment(", 1)[1].split("void RunGmm2Fragment(", 1)[0]
    gmm2 = source.split("void RunGmm2Fragment(", 1)[1].split("ComputeBatchRange LoadComputeBatch(", 1)[0]
    for stage, fragment in (("gmm1", gmm1), ("gmm2", gmm2)):
        submit = fragment.index(f"{stage}.Compute(")
        publish = fragment.index("completion.Publish(timer)")
        arm = fragment.index("completion.Arm(")
        assert fragment.index("completion.WaitInput(") < submit < publish < arm
        assert f"{stage}.End()" in fragment
        assert "PIPE_ALL" not in fragment
        assert "WriteGmByPassDCache" not in fragment and "AtomicAdd" not in fragment
    runner = source.split("void RunCubeAndPrologue(", 1)[1].split("bool AreGroupInputsReady(", 1)[0]
    assert runner.count("DeferredGmmCompletion completion;") == 1
    assert runner.count("completion.Publish(timer)") == 1
    assert runner.index("completion.Publish(timer)") > runner.index("tile = nextTile;")
    assert "PIPE_ALL" not in (URMA / "compute_completion.hpp").read_text(encoding="utf-8")
