"""Run the production Reduce header with asynchronous CPU pipeline queues.

The fake Ascend operators model FIFO engines, cross-engine events, UB reuse,
and BF16/FP32 arithmetic. This checks layout/lifetime and exact accumulation
order; it does not model NPU timing or replace device precision validation.
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
inline uint64_t inputCopies, outputCopies, metadataCopies;
inline bool bypass;
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
template <HardEvent Event> void SetFlag(TEventID id) {
    assert(id < 8);
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
        ++inputCopies;
        Enqueue(Mte2, [=] {
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
        Enqueue(Mte2, [=] {
            std::copy_n(source.data, count, destination.Data());
            return true;
        });
    }
}
template <typename T> void DataCopy(GlobalTensor<T> destination, LocalTensor<T> source, uint32_t count) {
    assert(count <= source.count);
    ++outputCopies;
    Enqueue(Mte3, [=] {
        std::copy_n(source.Data(), count, destination.data);
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
#include <random>
using namespace MegaMoeUrma;
using namespace AscendC;
int main(int argc, char** argv) {
    assert(argc == 5);
    const uint32_t bs = std::stoul(argv[1]), worker = std::stoul(argv[2]);
    priority = std::stoul(argv[3]);
    std::mt19937 random(std::stoul(argv[4]));
    const uint32_t rows = bs * 8U;
    std::vector<int64_t> inverse(rows);
    std::vector<float> weights(rows);
    for (uint32_t i = 0; i < rows; ++i) {
        inverse[i] = i;
        weights[i] = float(int(random() % 2001U) - 1000) / 997.0F;
    }
    std::shuffle(inverse.begin(), inverse.end(), random);
    std::vector<bfloat16_t> output(uint64_t(bs) * 3584U);
    for (auto& value : output) value.bits = 0x7fc1;
    ub.fill(0xa5);
    Arguments args{};
    args.config.hidden = 3584;
    args.config.topk = 8;
    args.config.bs = bs;
    args.symmetric = returnedBase;
    args.output = reinterpret_cast<uint64_t>(output.data());
    args.reduceIndex = reinterpret_cast<uint64_t>(inverse.data());
    args.weights = reinterpret_cast<uint64_t>(weights.data());
    Reduce(args, worker);
    const auto count = LocalOwnedRows(bs, worker), begin = LocalRowBegin(bs, worker);
    assert(inputCopies == count * 8U && outputCopies == count);
    assert(metadataCopies == 2U * ((count + 127U) / 128U));
    assert(bypass == (count != 0U));
    for (const auto& queue : queues) assert(queue.empty());
    for (const auto& kind : events) for (auto event : kind) assert(!event);
    for (uint32_t token = 0; token < bs; ++token) {
        const bool owned = token >= begin && token - begin < count;
        for (uint32_t column = 0; column < 3584; ++column) {
            if (!owned) {
                assert(output[uint64_t(token) * 3584 + column].bits == 0x7fc1);
                continue;
            }
            float accumulator = 0.0F;
            for (uint32_t slot = 0; slot < 8; ++slot) {
                const auto route = uint64_t(token) * 8U + slot;
                volatile float term = float(Returned(inverse[route], column)) * weights[route];
                accumulator = slot == 0 ? term : accumulator + term;
            }
            assert(output[uint64_t(token) * 3584 + column].bits == bfloat16_t(accumulator).bits);
        }
    }
    std::cout << "rows=" << count << " input_dma=" << inputCopies << " metadata_dma=" << metadataCopies << "\n";
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


@pytest.mark.parametrize("bs,worker", [(1, 0), (1, 55), (55, 54), (56, 55), (57, 0), (57, 55)])
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
@pytest.mark.parametrize("worker", [0, 55])
@pytest.mark.parametrize("priority", [0, 1, 2])
def test_reduce_full_cases_preserve_arithmetic_and_events(reduce_binary, bs, worker, priority):
    result = subprocess.run(
        [str(reduce_binary), str(bs), str(worker), str(priority), str(bs + worker)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
