"""Execute the real Pack header with queued, dependency-driven AscendC DMA.

The model validates values, ownership, UB lifetimes, event consumption and
the existence of MTE2/MTE3 overlap. It does not model NPU execution latency.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def pack_pipeline_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for Pack pipeline tests")
    temporary = tmp_path_factory.mktemp("urma_pack_pipeline")
    root = Path(__file__).resolve().parents[3]
    (temporary / "kernel_operator.h").write_text(
        r"""
#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <vector>
#define __aicore__
#define __gm__
namespace AscendC {
enum class HardEvent { MTE3_MTE2, MTE2_MTE3, S_MTE2, MTE2_S, MTE3_S };
enum class TPosition { VECCALC };
using TEventID = int;
constexpr TEventID EVENT_ID0 = 0;
constexpr int PIPE_ALL = 0;
enum Pipe { Scalar, Mte2, Mte3 };
struct Node {
    std::vector<std::shared_ptr<Node>> deps;
    std::function<void()> work;
    uint64_t start = 0, finish = 0, cost = 0;
    bool done = false;
    int payload = 0; // 1=input, 2=output
};
using Ptr = std::shared_ptr<Node>;
inline std::array<uint8_t, 192 * 1024> ub;
inline std::array<Ptr, 3> tail;
inline std::vector<Ptr> nodes;
inline std::map<std::pair<HardEvent, int>, Ptr> flags;
inline uint32_t scalarWaits = 0, indexCopies = 0, epoch = 0, hidden = 0, topk = 8;
inline uint64_t totalRows = 0, selfBegin = 0, selfRows = 0, selfReceive = 0, scaleOffset = 0;
inline std::vector<int64_t> order;
inline std::vector<uint32_t> tokenVisits, scaleVisits;
constexpr uintptr_t xBase = 0x100000000ULL, scaleBase = 0x200000000ULL, orderBase = 0x300000000ULL;
constexpr uintptr_t expandedBase = 0x400000000ULL, expandedScaleBase = 0x500000000ULL;
constexpr uintptr_t symmetricBase = 0x600000000ULL;
inline Ptr Append(Pipe pipe, uint64_t cost, std::function<void()> work = {}, Ptr extra = {}, int payload = 0) {
    auto node = std::make_shared<Node>();
    if (tail[pipe]) node->deps.push_back(tail[pipe]);
    if (extra) node->deps.push_back(extra);
    node->cost = cost; node->work = std::move(work); node->payload = payload;
    nodes.push_back(node); tail[pipe] = node;
    return node;
}
inline void Execute(Ptr root) {
    if (!root || root->done) return;
    std::vector<std::pair<Ptr, bool>> stack{{root, false}};
    while (!stack.empty()) {
        auto [node, visited] = stack.back(); stack.pop_back();
        if (node->done) continue;
        if (!visited) {
            stack.push_back({node, true});
            for (auto& dep : node->deps) if (!dep->done) stack.push_back({dep, false});
        } else {
            for (auto& dep : node->deps) { assert(dep->done); node->start = std::max(node->start, dep->finish); }
            if (node->work) node->work();
            node->finish = node->start + node->cost;
            node->done = true;
        }
    }
}
inline std::pair<Pipe, Pipe> Direction(HardEvent event) {
    switch (event) {
    case HardEvent::MTE3_MTE2: return {Mte3, Mte2};
    case HardEvent::MTE2_MTE3: return {Mte2, Mte3};
    case HardEvent::S_MTE2: return {Scalar, Mte2};
    case HardEvent::MTE2_S: return {Mte2, Scalar};
    case HardEvent::MTE3_S: return {Mte3, Scalar};
    }
    std::abort();
}
template<HardEvent event> void SetFlag(TEventID id) {
    auto key = std::make_pair(event, id);
    assert(id >= 0 && id < 8 && flags.count(key) == 0);
    // Include the set itself, so an initial credit has a real producer node.
    flags[key] = Append(Direction(event).first, 0);
}
template<HardEvent event> void WaitFlag(TEventID id) {
    auto key = std::make_pair(event, id);
    assert(flags.count(key) == 1);
    auto producer = flags[key]; flags.erase(key);
    auto consumer = Direction(event).second;
    auto wait = Append(consumer, 0, {}, producer);
    if (consumer == Scalar) { ++scalarWaits; Execute(wait); }
}
template<int> void PipeBarrier() {
    // Stress reuse: execute the MTE2 queue first. Its ring waits must pull in
    // the required MTE3 consumers before any UB slot can be overwritten.
    Execute(tail[Mte2]); Execute(tail[Mte3]); Execute(tail[Scalar]);
    for (auto& node : nodes) assert(node->done);
    assert(flags.empty());
}
template<typename T> struct GlobalTensor {
    uintptr_t address = 0;
    void SetGlobalBuffer(T* ptr) { address = reinterpret_cast<uintptr_t>(ptr); }
    GlobalTensor operator[](uint64_t i) const { return {address + i * sizeof(T)}; }
};
template<typename T> struct LocalTensor {
    uint32_t offset, count;
    LocalTensor(TPosition, uint32_t start, uint32_t size) : offset(start), count(size) {
        assert(start % 32 == 0 && uint64_t(start) + uint64_t(size) * sizeof(T) <= ub.size());
    }
    T GetValue(uint32_t i) const {
        assert(i < count);
        T value; std::memcpy(&value, ub.data() + offset + i * sizeof(T), sizeof(T)); return value;
    }
};
struct DataCopyExtParams { uint16_t blockCount; uint32_t blockLen, srcStride, dstStride, reserved; };
struct DataCopyPadExtParams { bool pad; uint8_t left, right, value; };
inline uint8_t Value(uint64_t token, uint32_t column, bool scale) {
    return static_cast<uint8_t>((token * 131 + column * 17 + epoch * 23 + (scale ? 31 : 0)) % 251);
}
inline void Read(uintptr_t address, uint32_t offset, uint32_t bytes) {
    if (address >= orderBase && address < orderBase + order.size() * sizeof(int64_t)) {
        assert(address + bytes <= orderBase + order.size() * sizeof(int64_t));
        std::memcpy(ub.data() + offset, reinterpret_cast<uint8_t*>(order.data()) + address - orderBase, bytes);
        return;
    }
    const bool scales = address >= scaleBase && address < scaleBase + (totalRows / topk) * (hidden / 32);
    const auto base = scales ? scaleBase : xBase;
    const auto stride = scales ? hidden / 32 : hidden;
    assert(address >= base && address + bytes <= base + (totalRows / topk) * stride);
    assert(bytes == stride && (address - base) % stride == 0);
    auto token = (address - base) / stride;
    for (uint32_t j = 0; j < bytes; ++j) ub[offset + j] = Value(token, j, scales);
}
inline void Write(uintptr_t address, uint32_t offset, uint32_t bytes) {
    bool scales = bytes == hidden / 32;
    uint64_t row;
    if (address >= symmetricBase) {
        const auto base = symmetricBase + (scales ? scaleOffset : 0);
        const auto stride = scales ? hidden / 32 : hidden;
        assert(address >= base && (address - base) % stride == 0);
        const auto received = (address - base) / stride;
        assert(received >= selfReceive && received < selfReceive + selfRows);
        row = received - selfReceive + selfBegin;
    } else {
        const auto base = scales ? expandedScaleBase : expandedBase;
        const auto stride = scales ? hidden / 32 : hidden;
        assert(address >= base && (address - base) % stride == 0);
        row = (address - base) / stride;
        assert(row < totalRows && (row < selfBegin || row >= selfBegin + selfRows));
    }
    assert(bytes == (scales ? hidden / 32 : hidden));
    auto token = uint64_t(order[row]) / topk;
    for (uint32_t j = 0; j < bytes; ++j) assert(ub[offset + j] == Value(token, j, scales));
    ++(scales ? scaleVisits : tokenVisits)[row];
}
template<typename T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams p, DataCopyPadExtParams pad) {
    assert(p.blockCount == 1 && p.srcStride == 0 && p.dstStride == 0 && !pad.pad);
    const auto rounded = (p.blockLen + 31) / 32 * 32;
    assert(rounded <= dst.count * sizeof(T));
    const bool indices = src.address >= orderBase && src.address < orderBase + order.size() * sizeof(int64_t);
    if (indices) ++indexCopies;
    Append(Mte2, 7 + p.blockLen / 32, [=] {
        Read(src.address, dst.offset, p.blockLen);
        std::fill(ub.begin() + dst.offset + p.blockLen, ub.begin() + dst.offset + rounded, 0);
    }, {}, indices ? 0 : 1);
}
template<typename T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams p) {
    assert(p.blockCount == 1 && p.srcStride == 0 && p.dstStride == 0 && p.blockLen <= src.count * sizeof(T));
    Append(Mte3, 11 + p.blockLen / 24, [=] { Write(dst.address, src.offset, p.blockLen); }, {}, 2);
}
inline bool Overlap() {
    std::vector<Ptr> loads, stores;
    for (auto& node : nodes) {
        if (node->payload == 1) loads.push_back(node);
        if (node->payload == 2) stores.push_back(node);
    }
    size_t j = 0;
    for (auto& load : loads) {
        while (j < stores.size() && stores[j]->finish <= load->start) ++j;
        if (j < stores.size() && std::max(load->start, stores[j]->start) <
                                 std::min(load->finish, stores[j]->finish)) return true;
    }
    return false;
}
inline void ResetWorker() {
    tail = {}; nodes.clear(); flags.clear(); ub.fill(0xff); scalarWaits = indexCopies = 0;
}
} // namespace AscendC
"""
    )
    source = temporary / "pack_pipeline.cpp"
    source.write_text(
        r"""
#include "pack_pipeline.hpp"
#include <iostream>
#include <string>
using namespace MegaMoeUrma;
int main(int argc, char** argv) {
    using namespace AscendC;
    assert(argc == 4);
    uint32_t bs = std::stoul(argv[1]);
    hidden = std::stoul(argv[2]);
    std::string pattern = argv[3];
    totalRows = uint64_t(bs) * topk;
    selfBegin = pattern == "all" ? 0 : totalRows / 3;
    selfRows = pattern == "all" ? totalRows : pattern == "empty" ? 0 : totalRows / 5;
    selfReceive = 11;
    scaleOffset = ((totalRows + 64) * hidden + 511) / 512 * 512;
    order.resize(totalRows);
    uint64_t stride = 137;
    while (std::gcd(stride, totalRows) != 1 && totalRows != 0) stride += 2;
    for (uint64_t row = 0; row < totalRows; ++row) order[row] = (row * stride + 3) % totalRows;
    Arguments args{};
    args.config.bs = bs; args.config.topk = topk; args.config.hidden = hidden;
    args.config.selfSendRow = selfBegin; args.config.selfRecvRow = selfReceive;
    args.config.selfRows = selfRows; args.config.receiveScaleOffset = scaleOffset;
    args.x = xBase; args.xScale = scaleBase; args.order = orderBase;
    args.expandedX = expandedBase; args.expandedScale = expandedScaleBase; args.symmetric = symmetricBase;
    static_assert(kPackPipelineSlots == 6 && kPackUbBytes <= 192 * 1024);
    for (epoch = 1; epoch <= 2; ++epoch) {
        tokenVisits.assign(totalRows, 0); scaleVisits.assign(totalRows, 0);
        for (uint32_t index = 0; index < kLocalWorkers; ++index) {
            const uint32_t worker = index * 37 % kLocalWorkers;
            const auto count = LocalOwnedRows(totalRows, worker);
            ResetWorker();
            Expand(args, worker); // Run the production function, not a duplicate algorithm.
            const auto batches = (count + kPackOrderRows - 1) / kPackOrderRows;
            // Each metadata batch becomes scalar-readable once; the final
            // MTE3_S additionally certifies both GM stores, even for no rows.
            assert(scalarWaits == batches + 1 && indexCopies == batches && flags.empty());
            if (count >= kPackPipelineSlots + 1) assert(Overlap());
            const auto begin = LocalRowBegin(totalRows, worker);
            for (uint64_t row = begin; row < begin + count; ++row) {
                assert(tokenVisits[row] == 1 && scaleVisits[row] == 1);
            }
        }
        for (uint64_t row = 0; row < totalRows; ++row) {
            assert(tokenVisits[row] == 1 && scaleVisits[row] == 1);
        }
    }
}
"""
    )
    binary = temporary / ("pack_pipeline.exe" if os.name == "nt" else "pack_pipeline")
    result = subprocess.run(
        command
        + [
            "-std=c++17",
            "-O2",
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


@pytest.mark.parametrize("bs", [0, 1, 6, 7, 8, 41, 42, 43, 2048, 4096, 8192, 16384])
@pytest.mark.parametrize("pattern", ["empty", "mixed", "all"])
def test_pack_dma_ring_payloads_tails_self_destinations_and_reuse(pack_pipeline_binary, bs, pattern):
    # H3584 covers the real 112-byte, non-32B-aligned scale plane. The larger
    # unsupported BS is only a CPU stress case for a second metadata batch.
    result = subprocess.run(
        [str(pack_pipeline_binary), str(bs), "3584" if bs <= 8192 else "128", pattern],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pack_maximum_reserved_slot_size(pack_pipeline_binary):
    result = subprocess.run(
        [str(pack_pipeline_binary), "113", "8192", "mixed"], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
