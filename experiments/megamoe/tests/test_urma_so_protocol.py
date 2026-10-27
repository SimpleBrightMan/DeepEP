"""Exercise SO ordering and the receiver-clear/start handshake without an NPU.

The compiled test uses the real transport header. The scheduler model explores
cross-invocation skew, empty edges, changing CV sets and delayed queue progress;
it does not claim to validate the device's hardware ordering implementation.
"""

import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess

import pytest


def test_real_transport_ro_data_so_flags_and_final_quiet(tmp_path):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for the SO transport simulation")
    root = Path(__file__).resolve().parents[3]
    (tmp_path / "kernel_operator.h").write_text(
        r"""
#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#define __aicore__
#define __gm__
enum pipe_t { PIPE_S };
namespace AscendC {
using TEventID = uint32_t;
enum class HardEvent { S_MTE3, MTE3_S };
template<class T> struct LocalTensor {
    T* data;
    void SetValue(uint32_t index, T value) { data[index] = value; }
};
template<class T> struct GlobalTensor {
    T* data;
    void SetGlobalBuffer(T* pointer) { data = pointer; }
};
template<HardEvent> void SetFlag(TEventID) {}
template<HardEvent> void WaitFlag(TEventID) {}
template<class T> void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t n) {
    std::memcpy(dst.data, src.data, n * sizeof(T));
}
inline uint32_t polls = 0;
template<class T> T ReadGmByPassDCache(T* ptr) {
    if (++polls > 20) throw std::runtime_error("poll remains blocked");
    return *ptr;
}
}
"""
    )
    (tmp_path / "shmem.h").write_text(
        r"""
#pragma once
#include <type_traits>
struct aclshmemx_udma_op_config_t { uint32_t cqe, se, fence, odr; };
inline constexpr aclshmemx_udma_op_config_t ACLSHMEMX_UDMA_OP_CONFIG_RO{1,0,0,5};
inline constexpr aclshmemx_udma_op_config_t ACLSHMEMX_UDMA_OP_CONFIG_SO{1,0,0,6};
alignas(128) inline uint8_t windows[4][4096]{};
inline uint32_t rank = 0, quiets = 0, executed = 0;
struct Op { uint8_t* dst; uint8_t* src; uint32_t bytes, peer; uint64_t add; };
inline std::vector<Op> pending;
inline void* aclshmem_ptr(void* address, uint32_t peer) {
    auto* ptr = reinterpret_cast<uint8_t*>(address);
    assert(ptr >= windows[rank] && ptr < windows[rank] + sizeof(windows[rank]));
    return windows[peer] + (ptr - windows[rank]);
}
template<class T, pipe_t Pipe, const aclshmemx_udma_op_config_t& Config>
void aclshmemx_udma_qp_put_nbi(T* dst, T* src, T* scratch, uint32_t n,
                              uint32_t peer, uint32_t qp, uint32_t event) {
    static_assert(Pipe == PIPE_S && Config.odr == 5 && Config.cqe == 1);
    assert(qp == 0 && peer != rank && scratch == nullptr && event == 0);
    pending.push_back({reinterpret_cast<uint8_t*>(aclshmem_ptr(dst, peer)),
                       reinterpret_cast<uint8_t*>(src), static_cast<uint32_t>(n * sizeof(T)), peer, 0});
}
template<class T, const aclshmemx_udma_op_config_t& Config>
void aclshmemx_udma_atomic_add(T* dst, T value, int32_t peer) {
    static_assert(std::is_same_v<T,uint64_t> && Config.odr == 6 && Config.cqe == 1);
    assert(peer != rank && reinterpret_cast<uintptr_t>(dst) % sizeof(T) == 0);
    pending.push_back({reinterpret_cast<uint8_t*>(aclshmem_ptr(dst, peer)),
                       nullptr, 0, static_cast<uint32_t>(peer), value});
}
inline void ProgressOne(uint32_t peer) {
    for (auto it = pending.begin(); it != pending.end(); ++it) {
        if (it->peer != peer) continue;
        if (it->src) std::memcpy(it->dst, it->src, it->bytes);
        else *reinterpret_cast<uint64_t*>(it->dst) += it->add;
        pending.erase(it); ++executed; return;
    }
}
inline void aclshmemx_udma_qp_quiet(uint32_t peer, uint32_t qp) {
    assert(qp == 0); ++quiets;
    while (true) {
        const auto before = pending.size(); ProgressOne(peer);
        if (pending.size() == before) return;
    }
}
template<class... Args> void aclshmemi_kernel_abort(const char*, Args...) {
    throw std::runtime_error("abort");
}
"""
    )
    source = tmp_path / "so_transport.cpp"
    source.write_text(
        r"""
#include "transport.hpp"
int main() {
    using namespace MegaMoeUrma;
    static_assert(kCommunicationQp == 0);
    uint8_t x[64], scale[32]; std::memset(x,7,sizeof(x)); std::memset(scale,9,sizeof(scale));
    auto* base = windows[rank];
    auto flag = [](uint32_t peer, uint32_t offset) -> uint64_t& {
        return *reinterpret_cast<uint64_t*>(windows[peer] + offset);
    };
    const uint64_t epoch = (uint64_t{1} << 40) + 123;
    Put(base,0,x,sizeof(x),1); Put(base,64,scale,sizeof(scale),1);
    PublishOrdered(base,128,1,epoch);
    Put(base,0,x,sizeof(x),2); PublishOrdered(base,128,2,epoch);
    assert(quiets == 0 && executed == 0 && pending.size() == 5);
    assert(flag(1,128) == 0);
    ProgressOne(1); assert(flag(1,128) == 0);
    ProgressOne(1); assert(flag(1,128) == 0 && windows[1][0] == 7 && windows[1][64] == 9);
    ProgressOne(1); assert(flag(1,128) == epoch && pending.size() == 2);
    AscendC::polls = 0; WaitLocalDone(windows[1]+128,epoch);
    // An empty edge still queues a finished flag. Completion covers it and
    // all prior same-peer work, while another peer remains in flight.
    PublishOrdered(base,256,1,epoch);
    assert(flag(1,256) == 0 && quiets == 0);
    Quiet(1); assert(quiets == 1 && flag(1,256) == epoch && pending.size() == 2);
    assert(flag(2,128) == 0); Quiet(2); assert(flag(2,128) == epoch && pending.empty());
    // Receiver clear before its next start permits a skipped CV to reappear,
    // with an arbitrary increasing 64-bit epoch rather than a +1 counter.
    std::memset(windows[1]+128,0,256);
    PublishOrdered(base,128,1,epoch+17); Quiet(1); assert(flag(1,128) == epoch+17);
    AscendC::polls = 0; bool blocked = false;
    try { WaitLocalDone(windows[1]+128,epoch); } catch (const std::runtime_error&) { blocked = true; }
    assert(blocked); AscendC::polls = 0; WaitStartAtLeast(windows[1]+128,epoch);
    const auto before = pending.size(); Put(base,0,x,0,1); assert(pending.size() == before);
    blocked = false;
    try { Put(base,0,x,kMaxPutBytes+1,1); } catch (const std::runtime_error&) { blocked = true; }
    assert(blocked && pending.size() == before);
}
"""
    )
    binary = tmp_path / ("so_transport.exe" if os.name == "nt" else "so_transport")
    compiled = subprocess.run(
        command + ["-std=c++17", "-O1", "-I" + str(tmp_path),
                   "-I" + str(root / "experiments/megamoe/csrc/kernels/mega_moe/urma"),
                   str(source), "-o", str(binary)],
        capture_output=True, text=True, timeout=120,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert executed.returncode == 0, executed.stdout + executed.stderr


class _Fabric:
    """RO operations may complete out of order; SO waits for its predecessors."""

    def __init__(self, routes):
        self.routes = routes
        self.world = len(routes[0])
        self.start = [dict() for _ in range(self.world)]
        self.flags = [dict() for _ in range(self.world)]
        self.data = [dict() for _ in range(self.world)]
        self.queues = {(s, d): [] for s in range(self.world) for d in range(self.world) if s != d}
        self.ranks = [_Rank(self, rank) for rank in range(self.world)]
        self.reads = 0

    def progress(self, source, target, choice):
        queue = self.queues[source, target]
        if not queue:
            return False
        # SO is an execution boundary. Choose any preceding RO, but execute
        # the boundary itself only once all its preceding WQEs have retired.
        boundary = next((i for i, op in enumerate(queue) if op[0] == "flag"), len(queue))
        index = choice % boundary if boundary else 0
        kind, epoch, key, value = queue.pop(index)
        if kind == "data":
            assert self.ranks[target].epoch == epoch, "payload crossed receiver reset/start"
            self.data[target][key] = value
        else:
            assert self.ranks[target].epoch == epoch, "late atomic added to another epoch"
            self.flags[target][key] = self.flags[target].get(key, 0) + epoch
        return True


class _Rank:
    def __init__(self, fabric, rank):
        self.fabric, self.rank = fabric, rank
        self.epoch = 0
        self.complete = False
        self.stage = "reset"

    def step(self):
        f, rank = self.fabric, self.rank
        peers = [p for p in range(f.world) if p != rank]
        if self.complete:
            return False
        if self.stage == "reset":
            # The prior local kernel joined reduction, inbound observations
            # and all per-owner final quiets before this stream-ordered clear.
            assert all(not f.queues[rank, peer] for peer in peers)
            self.epoch += 1
            f.flags[rank].clear()
            self.started, self.submitted, self.quieted, self.consumed = set(), set(), set(), set()
            self.incoming = set()
            self.stage = "run"
            return True
        for peer in peers:
            if peer not in self.started:
                f.start[peer][rank] = self.epoch
                self.started.add(peer)
                return True
        for peer in peers:
            if peer in self.submitted or f.start[rank].get(peer, 0) < self.epoch:
                continue
            # Empty edges obey the same start guard, and publish finished.
            count = f.routes[self.epoch - 1][rank][peer]
            queue = f.queues[rank, peer]
            for cv in range(count):
                for phase in ("dispatch", "combine"):
                    key = rank, phase, cv
                    parts = 2 if phase == "dispatch" else 1
                    for part in range(parts):
                        payload = self.epoch, rank, peer, phase, cv, part
                        queue.append(("data", self.epoch, (*key, part), payload))
                    queue.append(("flag", self.epoch, key, None))
            queue.append(("flag", self.epoch, (rank, "finished"), None))
            self.submitted.add(peer)
            return True
        for peer in peers:
            if peer in self.submitted and peer not in self.quieted and not f.queues[rank, peer]:
                self.quieted.add(peer)
                return True
        for peer in peers:
            count = f.routes[self.epoch - 1][peer][rank]
            for cv in range(count):
                for phase in ("dispatch", "combine"):
                    key = peer, phase, cv
                    if key in self.consumed or f.flags[rank].get(key, 0) != self.epoch:
                        continue
                    parts = 2 if phase == "dispatch" else 1
                    for part in range(parts):
                        assert f.data[rank][(*key, part)] == (self.epoch, peer, rank, phase, cv, part)
                        f.reads += 1
                    self.consumed.add(key)
                    return True
            if peer not in self.incoming and f.flags[rank].get((peer, "finished"), 0) >= self.epoch:
                self.incoming.add(peer)
                return True
        expected = 2 * sum(f.routes[self.epoch - 1][peer][rank] for peer in peers)
        if len(self.quieted) == len(peers) and len(self.incoming) == len(peers) and len(self.consumed) == expected:
            self.complete = self.epoch == len(f.routes)
            self.stage = "reset"
            return True
        return False


@pytest.mark.parametrize("seed", range(32))
@pytest.mark.parametrize("world", [2, 4])
def test_clear_start_so_protocol_with_empty_edges_and_changing_routes(seed, world):
    rng = random.Random(seed)
    routes = [
        [[0 if source == target else rng.randrange(4) for target in range(world)] for source in range(world)]
        for _ in range(4)
    ]
    # Force both a disappearing/reappearing CV and an empty round.
    routes[0][0][1], routes[1][0][1], routes[2][0][1] = 3, 0, 2
    routes[3] = [[0] * world for _ in range(world)]
    fabric = _Fabric(routes)
    for _ in range(30000):
        actions = [("rank", rank) for rank in range(world)] + [("queue", pair) for pair in fabric.queues]
        rng.shuffle(actions)
        progressed = False
        for kind, key in actions:
            if kind == "rank":
                progressed = fabric.ranks[key].step() or progressed
            else:
                progressed = fabric.progress(*key, rng.randrange(1000)) or progressed
        if all(rank.complete for rank in fabric.ranks):
            break
        assert progressed, "protocol deadlocked"
    else:
        pytest.fail("protocol failed to finish")
    assert all(not queue for queue in fabric.queues.values())
    assert fabric.reads == 3 * sum(sum(row) for epoch in routes for row in epoch)


def test_model_rejects_start_published_before_receiver_reset():
    fabric = _Fabric([[[0, 1], [0, 0]]])
    sender = fabric.ranks[0]
    assert sender.step()  # sender clears and enters epoch 1
    fabric.start[0][1] = 1  # inject an invalid early start from the receiver
    assert sender.step()  # sender publishes its own start
    assert sender.step()  # sender submits data and SO flags
    with pytest.raises(AssertionError, match="payload crossed receiver reset/start"):
        fabric.progress(0, 1, 0)
