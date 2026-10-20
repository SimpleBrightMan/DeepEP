"""Adversarial CPU oracle for per-CV returns and independently advancing ranks.

Payload delivery, quiet and publication are distinct actions. Every payload
read/write checks its epoch; no final peer handshake or rank barrier exists.
The oracle checks protocol causality, not hardware cache/URMA guarantees.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess

import pytest


@dataclass
class Transfer:
    sender: int
    peer: int
    cv: int
    epoch: int
    combine: bool
    delivered: bool = False
    quieted: bool = False


class Fabric:
    def __init__(self, routes, epochs=3, start_at_least=True):
        self.routes = routes
        self.epochs = epochs
        self.start_at_least = start_at_least
        self.transfers = []
        self.ranks = [Rank(self, rank) for rank in range(len(routes))]
        self.reads = 0

    def put(self, sender, peer, cv, combine):
        rank = self.ranks[sender]
        assert rank.start.get(peer, 0) >= rank.epoch
        transfer = Transfer(sender, peer, cv, rank.epoch, combine)
        self.transfers.append(transfer)
        return transfer

    def deliver(self, transfer):
        if transfer.delivered:
            return False
        source = self.ranks[transfer.sender]
        target = self.ranks[transfer.peer]
        # Source buffers are looked up NOW, not snapshotted at PUT submission:
        # reuse before quiet would turn this into a next-epoch payload.
        value = (source.computed if transfer.combine else source.expanded)[transfer.peer, transfer.cv]
        assert value[0] == transfer.epoch
        assert target.epoch == transfer.epoch
        destination = target.returned if transfer.combine else target.received
        destination[transfer.sender, transfer.cv] = value
        transfer.delivered = True
        return True

    @staticmethod
    def quiet(transfer):
        if not transfer.delivered:
            return False
        transfer.quieted = True
        return True

    def publish(self, transfer):
        assert transfer.quieted and transfer.delivered
        target = self.ranks[transfer.peer]
        flags = target.return_ready if transfer.combine else target.input_ready
        flags[transfer.sender, transfer.cv] = transfer.epoch


class Owner:
    def __init__(self, rank, peer):
        self.rank = rank
        self.peer = peer
        self.state = "publish-start"
        self.cv = 0
        self.transfer = None
        self.dispatch_done = False
        self.done = False

    def step(self):
        rank, peer, fabric = self.rank, self.peer, self.rank.fabric
        if self.done:
            return False
        if self.state == "publish-start":
            fabric.ranks[peer].start[rank.index] = rank.epoch
            self.state = "wait-start"
        elif self.state == "wait-start":
            observed = rank.start.get(peer, 0)
            ready = observed >= rank.epoch if fabric.start_at_least else observed == rank.epoch
            if not ready:
                return False
            self.state = "dispatch"
        elif self.state == "dispatch":
            if self.cv < fabric.routes[rank.index][peer]:
                self.transfer = fabric.put(rank.index, peer, self.cv, False)
                self.state = "dispatch-quiet"
            else:
                self.dispatch_done = True
                self.cv = 0
                self.state = "wait-local-dispatch"
        elif self.state == "wait-local-dispatch":
            if not all(owner.dispatch_done for owner in rank.owners):
                return False
            self.state = "combine"
        elif self.state == "combine":
            if self.cv == fabric.routes[peer][rank.index]:
                self.done = True
            elif (peer, self.cv) not in rank.computed:
                return False
            else:
                self.transfer = fabric.put(rank.index, peer, self.cv, True)
                self.state = "combine-quiet"
        elif self.state.endswith("-quiet"):
            if not fabric.quiet(self.transfer):
                return False
            self.state = self.state.replace("-quiet", "-publish")
        elif self.state.endswith("-publish"):
            fabric.publish(self.transfer)
            self.cv += 1
            self.state = self.state.replace("-publish", "")
        else:
            raise AssertionError(self.state)
        return True


class Rank:
    def __init__(self, fabric, index):
        self.fabric, self.index = fabric, index
        self.epoch = 1
        self.packed = self.finished = False
        self.start, self.input_ready, self.return_ready = {}, {}, {}
        self.received, self.returned, self.expanded, self.computed = {}, {}, {}, {}
        self.consumed = set()
        self.owners = []

    def pack(self):
        if self.packed or self.finished:
            return False
        self.expanded = {
            (peer, cv): (self.epoch, self.index, peer, cv)
            for peer, count in enumerate(self.fabric.routes[self.index])
            for cv in range(count)
        }
        self.computed = {}
        self.consumed = set()
        for cv in range(self.fabric.routes[self.index][self.index]):
            key = self.index, cv
            self.received[key] = self.expanded[key]
            self.input_ready[key] = self.epoch
        self.owners = [Owner(self, peer) for peer in range(len(self.fabric.ranks)) if peer != self.index]
        self.packed = True
        return True

    def compute(self):
        for source, counts in enumerate(self.fabric.routes):
            for cv in range(counts[self.index]):
                key = source, cv
                if key not in self.computed and self.input_ready.get(key) == self.epoch:
                    value = self.received[key]
                    assert value == (self.epoch, source, self.index, cv)
                    self.computed[key] = (*value, "ffn")
                    return True
        return False

    def reduce(self):
        for expert, count in enumerate(self.fabric.routes[self.index]):
            for cv in range(count):
                key = expert, cv
                if key in self.consumed:
                    continue
                if expert == self.index:
                    if key not in self.computed:
                        continue
                    value = self.computed[key]
                else:
                    if self.return_ready.get(key) != self.epoch:
                        continue
                    value = self.returned[key]
                assert value == (self.epoch, self.index, expert, cv, "ffn")
                self.consumed.add(key)
                self.fabric.reads += 1
                return True
        return False

    def finish(self):
        expected_compute = sum(counts[self.index] for counts in self.fabric.routes)
        expected_reduce = sum(self.fabric.routes[self.index])
        # The final LOCAL join covers all computing, peer-owner and reducing
        # work. No condition depends on another rank's final phase.
        if (
            not all(owner.done for owner in self.owners)
            or len(self.computed) != expected_compute
            or len(self.consumed) != expected_reduce
        ):
            return False
        if self.epoch == self.fabric.epochs:
            self.finished = True
        else:
            self.epoch += 1
            self.packed = False
        return True


def run_adversarial(fabric, seed):
    rng = random.Random(seed)
    for _ in range(100_000):
        if all(rank.finished for rank in fabric.ranks):
            assert fabric.reads == fabric.epochs * sum(map(sum, fabric.routes))
            assert all(transfer.quieted for transfer in fabric.transfers)
            return
        actions = []
        for rank in fabric.ranks:
            if rank.finished:
                continue
            if not rank.packed:
                actions.append(rank.pack)
            else:
                actions.extend(owner.step for owner in rank.owners)
                actions.extend((rank.compute, rank.reduce, rank.finish))
        actions.extend(lambda t=t: fabric.deliver(t) for t in fabric.transfers if not t.delivered)
        rng.shuffle(actions)
        assert any(action() for action in actions), "Protocol deadlock"
    raise AssertionError("Protocol failed to terminate")


@pytest.mark.parametrize("monotonic", [False, True])
def test_empty_peer_start_overwrite_requires_monotonic_wait(monotonic):
    fabric = Fabric([[0, 0], [0, 0]], start_at_least=monotonic)
    fast, slow = fabric.ranks
    fast.pack()
    slow.pack()
    fast_owner, slow_owner = fast.owners[0], slow.owners[0]
    assert fast_owner.step() and slow_owner.step()  # Both publish Start(1).
    assert slow_owner.state == "wait-start"
    # Pause the slow owner BEFORE it observes the fast peer's Start(1).
    while not fast_owner.done:
        assert fast_owner.step()
    assert fast.finish() and fast.epoch == 2
    assert fast.pack()
    assert fast.owners[0].step()  # Start(2) overwrites slow's old Start(1).
    assert slow.start[0] == 2 and slow.epoch == 1
    if monotonic:
        assert slow_owner.step()
        run_adversarial(fabric, 123)
    else:
        assert not slow_owner.step()
        with pytest.raises(AssertionError, match="Protocol deadlock"):
            run_adversarial(fabric, 123)


@pytest.mark.parametrize("seed", [0, 5, 17])
@pytest.mark.parametrize("pattern", ["empty", "self", "one-way", "alltoall", "skewed"])
def test_asynchronous_three_epoch_protocol(seed, pattern):
    world = 4
    if pattern == "empty":
        routes = [[0] * world for _ in range(world)]
    elif pattern == "self":
        routes = [[2 * (source == peer) for peer in range(world)] for source in range(world)]
    elif pattern == "one-way":
        routes = [[2 * (peer == (source + 1) % world) for peer in range(world)] for source in range(world)]
    elif pattern == "alltoall":
        routes = [[2] * world for _ in range(world)]
    else:
        routes = [
            [(source + peer) % 3 if peer == 1 or source == peer else 0 for peer in range(world)]
            for source in range(world)
        ]
    run_adversarial(Fabric(routes), seed)


def test_ep16_sparse_protocol_with_idle_peer_owners():
    routes = [[2 if peer in {source, (source + 3) % 16} else 0 for peer in range(16)] for source in range(16)]
    run_adversarial(Fabric(routes), 41)


def test_quiet_publication_and_exact_epoch_guard_payload_reads():
    fabric = Fabric([[0, 1], [0, 0]])
    source, expert = fabric.ranks
    source.pack()
    expert.pack()
    source.start[1] = expert.start[0] = 1
    dispatch = fabric.put(0, 1, 0, False)
    assert not fabric.quiet(dispatch)
    with pytest.raises(AssertionError):
        fabric.publish(dispatch)
    assert not expert.compute()
    fabric.deliver(dispatch)
    assert not expert.compute()  # Payload arrival alone is not readiness.
    assert fabric.quiet(dispatch)
    fabric.publish(dispatch)
    assert expert.compute()
    combine = fabric.put(1, 0, 0, True)
    fabric.deliver(combine)
    assert not source.reduce()
    assert fabric.quiet(combine)
    fabric.publish(combine)
    source.return_ready[1, 0] = 0
    assert not source.reduce()
    source.return_ready[1, 0] = 2
    assert not source.reduce()  # CV readiness must NOT use the Start >= rule.
    source.return_ready[1, 0] = 1
    assert source.reduce()


def test_next_epoch_put_waits_for_target_to_finish_old_reduce():
    fabric = Fabric([[0, 1], [0, 0]])
    source, expert = fabric.ranks
    source.pack()
    expert.pack()
    # Expert has finished epoch 1 locally, while source is still reducing it.
    expert.epoch = 2
    expert.packed = False
    expert.pack()
    expert.start[0] = 1
    owner = expert.owners[0]
    assert owner.step()  # Publish Start(2).
    assert not owner.step()  # No target Start(2), hence no new payload write.
    with pytest.raises(AssertionError):
        fabric.put(1, 0, 0, True)
    source.epoch = 2
    source.packed = False
    source.pack()
    expert.start[0] = 2
    assert owner.step()


@pytest.fixture(scope="module")
def reduce_workers_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for reducer ownership tests")
    temporary = tmp_path_factory.mktemp("urma_tail_workers")
    root = Path(__file__).resolve().parents[3]
    (temporary / "kernel_operator.h").write_text(
        r"""
#pragma once
#include <cstdint>
#include <stdexcept>
#define __aicore__
#define __gm__
constexpr int PIPE_S = 0;
namespace AscendC {
enum class HardEvent { S_MTE3, MTE3_S };
using TEventID = int;
inline unsigned polls = 0;
inline uint64_t ReadGmByPassDCache(uint64_t* address) {
    if (++polls > 100) throw std::runtime_error("still waiting");
    return *address;
}
template<typename T> struct LocalTensor { void SetValue(unsigned, T) {} };
template<typename T> struct GlobalTensor { void SetGlobalBuffer(T*) {} };
template<HardEvent> void SetFlag(TEventID) {}
template<HardEvent> void WaitFlag(TEventID) {}
template<typename T> void DataCopy(GlobalTensor<T>, LocalTensor<T>, unsigned) {}
}
"""
    )
    (temporary / "shmem.h").write_text(
        r"""
#pragma once
inline void aclshmemi_kernel_abort(const char*, ...) {}
template<typename T, int> void aclshmemx_udma_put_nbi(T*, T*, void*, uint32_t, uint32_t, uint32_t) {}
inline void aclshmemx_udma_quiet(int) {}
inline void* aclshmem_ptr(void* address, unsigned) { return address; }
"""
    )
    source = temporary / "workers.cpp"
    source.write_text(
        r"""
#include "compute_contract.hpp"
#include "transport.hpp"
#include <algorithm>
#include <cassert>
#include <string>
#include <vector>
using namespace MegaMoeUrma;
int main(int argc, char** argv) {
    assert(argc == 2);
    if (std::string(argv[1]) == "flags") {
        uint64_t observed = 2;
        auto* address = reinterpret_cast<uint8_t*>(&observed);
        AscendC::polls = 0;
        WaitStartAtLeast(address, 1);
        assert(AscendC::polls == 1);
        bool blocked = false;
        AscendC::polls = 0;
        try { WaitLocalDone(address, 1); } catch (const std::runtime_error&) { blocked = true; }
        assert(blocked); // A future CV flag must never authorize an old payload.
        observed = 1; AscendC::polls = 0;
        WaitLocalDone(address, 1);
        assert(AscendC::polls == 1);
        AscendC::polls = 0; blocked = false;
        try { WaitStartAtLeast(address, 2); } catch (const std::runtime_error&) { blocked = true; }
        assert(blocked);
        return 0;
    }
    uint64_t rows = std::stoull(argv[1]), total = 0;
    std::vector<unsigned> visits(rows), workers(kReduceWorkers);
    unsigned communications = 0;
    static_assert(kReduceWorkers == 41);
    for (unsigned block = 0; block < kAic; ++block) {
        for (unsigned sub = 0; sub < 2; ++sub) {
            auto worker = ReduceWorkerIndex(block, sub);
            if (worker == kReduceWorkers) {
                assert(sub == 1 && block < kCommunicationWorkers);
                assert(ReduceOwnedRows(rows, worker) == 0);
                ++communications;
                continue;
            }
            assert(worker < kReduceWorkers && ++workers[worker] == 1);
            auto begin = ReduceRowBegin(rows, worker);
            auto count = ReduceOwnedRows(rows, worker);
            assert(begin + count <= rows);
            total += count;
            for (uint64_t i = begin; i < begin + count; ++i) assert(++visits[i] == 1);
        }
    }
    assert(communications == 15 && total == rows);
    assert(std::all_of(visits.begin(), visits.end(), [](auto n) { return n == 1; }));
    assert(std::all_of(workers.begin(), workers.end(), [](auto n) { return n == 1; }));
    const uint64_t large = (1ULL << 32) + 29;
    uint64_t next = 0;
    for (unsigned worker = 0; worker < kReduceWorkers; ++worker) {
        assert(ReduceRowBegin(large, worker) == next);
        next += ReduceOwnedRows(large, worker);
    }
    assert(next == large);
}
"""
    )
    binary = temporary / ("workers.exe" if os.name == "nt" else "workers")
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


@pytest.mark.parametrize("rows", [0, 1, 40, 41, 42, 127, 128, 129, 2048, 4096, 8192])
def test_actual_41_reducer_partition(reduce_workers_binary, rows):
    result = subprocess.run([str(reduce_workers_binary), str(rows)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_start_and_payload_wait_predicates(reduce_workers_binary):
    result = subprocess.run([str(reduce_workers_binary), "flags"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
