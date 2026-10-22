"""Execute the real bounded timer writer against host-only AscendC stubs."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("engine", [0, 1, 2])
@pytest.mark.parametrize("block", [0, 5, 27])
@pytest.mark.parametrize("enabled", [False, True])
def test_trace_writer_bounds_and_disabled_path(tmp_path, engine, block, enabled):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("A host C++ compiler is required to execute the timer writer")
    root = Path(__file__).resolve().parents[3]
    (tmp_path / "kernel_operator.h").write_text(
        r"""
#pragma once
#include <cstdint>
#define __aicore__
#define __gm__
namespace AscendC {
constexpr int AIC = 0, AIV = 1;
inline uint64_t cycles = 100, flushes = 0;
inline uint64_t GetSystemCycle() { return cycles++; }
inline uint32_t GetBlockIdx() { return ENGINE == 0 ? BLOCK : BLOCK * 2; }
inline uint32_t GetTaskRation() { return 2; }
inline uint32_t GetSubBlockIdx() { return ENGINE == 2 ? 1 : 0; }
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
template<class T> struct GlobalTensor { void SetGlobalBuffer(T*) {} };
template<class T, CacheLine, DcciDst> void DataCacheCleanAndInvalid(GlobalTensor<T>&) { ++flushes; }
}
constexpr int g_coreType = ENGINE == 0 ? AscendC::AIC : AscendC::AIV;
""",
        encoding="utf-8",
    )
    source = tmp_path / "trace_writer.cpp"
    source.write_text(
        r"""
#include "timer.hpp"
#include <cassert>
#include <vector>
using namespace MegaMoeUrma;
int main() {
    constexpr uint64_t guard = 0x123456789abcdef0ULL;
    constexpr uint64_t stride = kTraceHeaderElements + kTraceRecordElements;
    constexpr uint32_t core = BLOCK * 3 + ENGINE;
    std::vector<uint64_t> memory(kTimerCores * stride + 2, guard);
    auto* buffer = memory.data() + 1;
    Timer timer;
    timer.Init(0, reinterpret_cast<uint64_t>(buffer), 1);
    TraceMeta meta;
    meta.peer = 7; meta.cv = 3; meta.bytes = 7168;
    auto start = timer.TraceBegin();
    timer.TraceEnd(TraceEvent::DispatchPutX, start, meta);
    start = timer.TraceBegin();
    timer.TraceEnd(TraceEvent::CombinePut, start, meta);
    timer.FinishTrace();
    assert(memory.front() == guard && memory.back() == guard);
#ifdef DEEPEP_MEGAMOE_TIMER
    auto* header = buffer + core * stride;
    auto* record = header + kTraceHeaderElements;
    assert(header[0] == kTraceSchema && header[1] == 1);
    assert(header[2] == 1 && header[3] == 1 && header[4] == core);
    assert(record[0] == 100 && record[1] == 101 && record[2] == 1);
    assert(record[3] == 7 && record[4] == 3 && record[5] == kTraceUnknown);
    assert(record[12] == 7168 && record[13] == 0 && record[15] == 0);
    for (uint32_t other = 0; other < kTimerCores; ++other)
        if (other != core)
            for (uint32_t i = 0; i < stride; ++i) assert(buffer[other * stride + i] == guard);
    assert(AscendC::flushes == 2);
#else
    for (auto value : memory) assert(value == guard);
    assert(AscendC::cycles == 100 && AscendC::flushes == 0);
#endif
    const auto cycles = AscendC::cycles, flushes = AscendC::flushes;
    Timer disabled;
    disabled.Init(0);
    assert(disabled.TraceBegin() == 0);
    disabled.TraceEnd(TraceEvent::Pack, 0);
    disabled.FinishTrace();
    assert(AscendC::cycles == cycles && AscendC::flushes == flushes);
    // Aggregated transport markers disambiguate helper block 15 from old
    // self-copy traces. Payload accumulates only on this worker's phase slot.
    std::vector<uint64_t> totals(kTimerCores * kTimerPhases * kTimerRecordElements + 2, guard);
    auto* aggregate = totals.data() + 1;
    Timer aggregateTimer;
    aggregateTimer.Init(reinterpret_cast<uint64_t>(aggregate));
    aggregateTimer.Begin(Phase::Pack);
    aggregateTimer.End(Phase::Pack, 32);
    aggregateTimer.Begin(Phase::Combine);
    aggregateTimer.End(Phase::Combine, 7168, 7);
    aggregateTimer.Begin(Phase::Combine);
    aggregateTimer.End(Phase::Combine, 14336, 7);
    assert(totals.front() == guard && totals.back() == guard);
#ifdef DEEPEP_MEGAMOE_TIMER
    auto* packed = aggregate + (core * kTimerPhases + Phase::Pack) * kTimerRecordElements;
    auto* combined = aggregate + (core * kTimerPhases + Phase::Combine) * kTimerRecordElements;
    assert(packed[3] == 32 && packed[4] == 1 && packed[5] == 0 && packed[6] == kProtocol);
    assert(combined[3] == 21504 && combined[4] == 2 && combined[5] == 8 && combined[6] == kProtocol);
    assert(combined[0] < combined[1]);
#else
    for (auto value : totals) assert(value == guard);
    assert(AscendC::cycles == cycles && AscendC::flushes == flushes);
#endif
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "trace_writer"
    command = [*compiler, "-std=c++17", f"-DENGINE={engine}", f"-DBLOCK={block}"]
    if enabled:
        command.append("-DDEEPEP_MEGAMOE_TIMER")
    subprocess.run(
        [
            *command,
            "-I",
            str(tmp_path),
            "-I",
            str(root / "experiments/megamoe/csrc/kernels/mega_moe/urma"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
