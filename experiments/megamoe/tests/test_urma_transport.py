"""Exercise the actual PUT helper against deferred-transfer SDK doubles."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_put_completion_and_signal_ordering(tmp_path):
    root = Path(__file__).resolve().parents[3]
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for the transport simulation")
    (tmp_path / "kernel_operator.h").write_text(
        r"""
#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#define __aicore__
#define __gm__
enum pipe_t { PIPE_S };
inline std::vector<int> events;
namespace AscendC {
using TEventID = uint32_t;
enum class HardEvent { S_MTE3, MTE3_S };
template <typename T> struct LocalTensor {
    T* data;
    void SetValue(uint32_t i, T value) { data[i] = value; }
};
template <typename T> struct GlobalTensor {
    T* data;
    void SetGlobalBuffer(T* address) { data = address; }
};
template <HardEvent E> void SetFlag(TEventID id) {
    assert(id == 3); events.push_back(E == HardEvent::S_MTE3 ? 1 : 4);
}
template <HardEvent E> void WaitFlag(TEventID id) {
    assert(id == 3); events.push_back(E == HardEvent::S_MTE3 ? 2 : 5);
}
template <typename T> void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t n) {
    assert(events.size() >= 2 && events.back() == 2);
    assert(n * sizeof(T) == 32);
    events.push_back(3); std::memcpy(dst.data, src.data, n * sizeof(T));
}
template <typename T> T ReadGmByPassDCache(T* ptr) { return *ptr; }
}
"""
    )
    (tmp_path / "shmem.h").write_text(
        r"""
#pragma once
#include <stdexcept>
#include <vector>
inline uint8_t windows[16][4096]{};
inline uint32_t currentRank = 5;
struct Transfer { uint8_t* dst; uint8_t* src; uint32_t bytes; int peer; };
inline std::vector<Transfer> pending;
inline uint32_t submitted = 0, completed = 0;
inline void* aclshmem_ptr(uint8_t* local, int peer) {
    auto offset = local - windows[currentRank]; assert(offset >= 0 && offset < 4096);
    return windows[peer] + offset;
}
template <typename T, pipe_t P>
void aclshmemx_udma_put_nbi(T* dst, T* src, T* scratch, uint32_t n, int peer, uint32_t event) {
    static_assert(P == PIPE_S); assert(scratch == nullptr && event == 0 && peer != int(currentRank));
    pending.push_back({static_cast<uint8_t*>(aclshmem_ptr(dst, peer)), src, n, peer}); ++submitted;
}
inline void aclshmemx_udma_quiet(int peer) {
    for (auto& t : pending) { assert(t.peer == peer); std::memcpy(t.dst, t.src, t.bytes); ++completed; }
    pending.clear();
}
template <typename... T> void aclshmemi_kernel_abort(const char*, T...) { throw std::runtime_error("abort"); }
"""
    )
    source = tmp_path / "transport.cpp"
    source.write_text(
        r"""
#include "transport.hpp"
#include <set>
int main() {
    using namespace MegaMoeUrma;
    for (uint32_t rank = 0; rank < 16; ++rank) {
        std::set<uint32_t> peers;
        for (uint32_t owner = 0; owner < 15; ++owner) {
            auto peer = PeerForOwner(rank, owner); assert(peer < 16 && peer != rank); peers.insert(peer);
        }
        assert(peers.size() == 15);
    }
    uint8_t x[64], scale[32]; std::memset(x, 7, 64); std::memset(scale, 9, 32);
    uint64_t signal[4]{};
    auto base = windows[currentRank];
    Put(base, 0, x, 64, 9); Put(base, 64, scale, 32, 9);
    assert(submitted == 2 && completed == 0 && windows[9][0] == 0);
    Quiet(9);
    assert(completed == 2 && pending.empty() && windows[9][0] == 7 && windows[9][64] == 9);
    std::memset(windows[9] + 128, 0xab, 128);
    PublishDone(base, 128, 9, 123, {signal}, 3);
    assert((events == std::vector<int>{1,2,3,4,5}));
    WaitLocalDone(windows[9] + 128, 123);
    for (uint32_t i = 32; i < 128; ++i) assert(windows[9][128 + i] == 0xab);
    Put(base, 0, x, 0, 9); assert(submitted == 2);
    bool rejected = false;
    try { Put(base, 0, x, kMaxPutBytes + 1, 9); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected && submitted == 2);
    // Other ranks have reached the barrier already. A later generation may
    // replace one flag; the monotone barrier must not wait for exact equality.
    for (uint32_t p = 0; p < 16; ++p) *reinterpret_cast<uint64_t*>(base + 1024 + p*128) = 202;
    RankBarrier(base, 1024, currentRank, 201, {signal}, 3);
    for (uint32_t p = 0; p < 16; ++p)
        assert(*reinterpret_cast<uint64_t*>(windows[p] + 1024 + currentRank*128) == 201);
}
"""
    )
    binary = tmp_path / ("transport.exe" if os.name == "nt" else "transport")
    compiled = subprocess.run(
        command
        + [
            "-std=c++17",
            "-O1",
            "-I" + str(tmp_path),
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
    executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert executed.returncode == 0, executed.stdout + executed.stderr
