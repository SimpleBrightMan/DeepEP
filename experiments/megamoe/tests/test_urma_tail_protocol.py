"""Adversarial CPU oracle for per-CV returns and independently advancing ranks.

Payload delivery, quiet and publication are distinct actions. Every payload
read/write checks its epoch. The original per-CV model remains a baseline;
the finished model adds per-peer notifications and a local all-ready latch.
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


class FinishedFabric(Fabric):
    def __init__(self, routes, epochs=3):
        super().__init__(routes, epochs)
        self.ranks = [FinishedRank(self, rank) for rank in range(len(routes))]
        self.finished_publications = set()

    def publish_finished(self, sender, peer):
        source, target = self.ranks[sender], self.ranks[peer]
        assert source.start.get(peer, 0) >= source.epoch
        assert target.packed and target.epoch == source.epoch
        key = sender, peer, source.epoch
        assert key not in self.finished_publications
        transfers = [
            transfer
            for transfer in self.transfers
            if transfer.sender == sender
            and transfer.peer == peer
            and transfer.epoch == source.epoch
            and transfer.combine
        ]
        assert len(transfers) == self.routes[peer][sender]
        for transfer in transfers:
            assert transfer.delivered and transfer.quieted
            assert target.return_ready.get((sender, transfer.cv)) == source.epoch
        self.finished_publications.add(key)
        target.peer_finished[sender] = source.epoch


class FinishedOwner(Owner):
    def __init__(self, rank, peer):
        super().__init__(rank, peer)
        self.incoming_counted = False
        self.self_counted = False
        self.observed_before_quiet = False

    def observe(self):
        rank = self.rank
        if not self.incoming_counted and rank.peer_finished.get(self.peer, 0) == rank.epoch:
            self.incoming_counted = True
            rank.add_incoming(self.peer)
        if self is rank.owners[0] and not self.self_counted and rank.aic_done == 28:
            self.self_counted = True
            rank.add_incoming(rank.index)
        return self.incoming_counted and (self is not rank.owners[0] or self.self_counted)

    def step(self):
        rank, fabric = self.rank, self.rank.fabric
        if self.done:
            return False
        if self.state == "combine" and self.cv == fabric.routes[self.peer][rank.index]:
            self.state = "publish-finished"
        elif self.state == "publish-finished":
            fabric.publish_finished(rank.index, self.peer)
            self.state = "wait-finished"
        elif self.state == "wait-finished":
            before = rank.incoming_count
            self.done = self.observe()
            return self.done or rank.incoming_count != before
        else:
            if self.state == "combine-quiet" and not self.observed_before_quiet:
                self.observe()
                self.observed_before_quiet = True
            progressed = super().step()
            if self.state == "combine":
                self.observed_before_quiet = False
            return progressed
        return True


class FinishedRank(Rank):
    def __init__(self, fabric, index):
        super().__init__(fabric, index)
        self.peer_finished = {}
        self.incoming_sources = set()
        self.incoming_count = self.aic_done = 0
        self.fast_latched = False
        self.fast_reads = 0
        self.completed_epochs = []

    def pack(self):
        if not super().pack():
            return False
        # This reset models host memset before any PublishStart of this epoch.
        self.incoming_sources = set()
        self.incoming_count = self.aic_done = 0
        self.fast_latched = False
        self.owners = [FinishedOwner(self, peer) for peer in range(len(self.fabric.ranks)) if peer != self.index]
        return True

    def add_incoming(self, source):
        assert source not in self.incoming_sources
        self.incoming_sources.add(source)
        self.incoming_count += 1
        assert self.incoming_count == len(self.incoming_sources) <= len(self.fabric.ranks)

    def complete_next_aic(self):
        # A conservative completion model: all group FIX stores precede
        # the 28 separate AIC completion contributions.
        if self.aic_done == 28 or len(self.computed) != sum(row[self.index] for row in self.fabric.routes):
            return False
        self.aic_done += 1
        return True

    def reduce(self):
        if self.incoming_count == len(self.fabric.ranks):
            self.fast_latched = True
        if not self.fast_latched:
            return super().reduce()
        for expert, count in enumerate(self.fabric.routes[self.index]):
            for cv in range(count):
                key = expert, cv
                if key in self.consumed:
                    continue
                # The fast path reads payloads without any per-CV flag lookup.
                value = (self.computed if expert == self.index else self.returned)[key]
                assert value == (self.epoch, self.index, expert, cv, "ffn")
                self.consumed.add(key)
                self.fast_reads += 1
                self.fabric.reads += 1
                return True
        return False

    def finish(self):
        epoch = self.epoch
        if not super().finish():
            return False
        assert self.aic_done == 28
        assert self.incoming_sources == set(range(len(self.fabric.ranks)))
        self.completed_epochs.append(epoch)
        return True


def run_finished_adversarial(fabric, seed, *, delay_reduce_until_finished=False):
    rng = random.Random(seed)
    for _ in range(150_000):
        if all(rank.finished for rank in fabric.ranks):
            world = len(fabric.ranks)
            assert fabric.reads == fabric.epochs * sum(map(sum, fabric.routes))
            assert len(fabric.finished_publications) == fabric.epochs * world * (world - 1)
            assert all(rank.completed_epochs == list(range(1, fabric.epochs + 1)) for rank in fabric.ranks)
            return
        actions = []
        for rank in fabric.ranks:
            if rank.finished:
                continue
            if not rank.packed:
                actions.append(rank.pack)
            else:
                actions.extend(owner.step for owner in rank.owners)
                actions.extend((rank.compute, rank.complete_next_aic, rank.finish))
                if not delay_reduce_until_finished or rank.incoming_count == len(fabric.ranks):
                    actions.append(rank.reduce)
        actions.extend(
            lambda transfer=transfer: fabric.deliver(transfer)
            for transfer in fabric.transfers
            if not transfer.delivered
        )
        rng.shuffle(actions)
        assert any(action() for action in actions), "Finished protocol deadlock"
    raise AssertionError("Finished protocol failed to terminate")


@pytest.mark.parametrize("seed", [0, 9, 31])
@pytest.mark.parametrize("pattern", ["empty", "self", "one-way", "alltoall", "skewed"])
def test_finished_protocol_three_epochs(seed, pattern):
    world = 4
    routes = [
        [
            0
            if pattern == "empty"
            else 2 * (source == peer)
            if pattern == "self"
            else 2 * (peer == (source + 1) % world)
            if pattern == "one-way"
            else 2
            if pattern == "alltoall"
            else (source + peer) % 3
            if peer == 1 or source == peer
            else 0
            for peer in range(world)
        ]
        for source in range(world)
    ]
    run_finished_adversarial(FinishedFabric(routes), seed)


def test_finished_ep16_and_fast_reduce_without_cv_flag_checks():
    routes = [[2 if peer in {source, (source + 3) % 16} else 0 for peer in range(16)] for source in range(16)]
    fabric = FinishedFabric(routes)
    run_finished_adversarial(fabric, 42, delay_reduce_until_finished=True)
    assert sum(rank.fast_reads for rank in fabric.ranks) == fabric.reads


def test_empty_finished_sends_before_wait_and_self_requires_all_28_aic():
    fabric = FinishedFabric([[0, 0], [0, 0]], epochs=1)
    left, right = fabric.ranks
    for rank in fabric.ranks:
        rank.pack()
        rank.start[1 - rank.index] = rank.epoch
        rank.owners[0].state = "combine"
    first, second = left.owners[0], right.owners[0]
    assert first.step() and first.state == "publish-finished"
    assert first.step() and first.state == "wait-finished"
    assert left.incoming_count == 0  # Outbound finished is not inbound finished.
    assert not first.step()
    assert second.step() and second.step()  # Both peers publish before waiting.
    assert first.step() and second.step()
    assert left.incoming_count == right.incoming_count == 1
    for rank in fabric.ranks:
        for _ in range(27):
            assert rank.complete_next_aic()
        assert not rank.owners[0].step()
        assert rank.incoming_count == 1
        assert rank.complete_next_aic() and rank.owners[0].step()
        assert rank.incoming_count == 2 and not rank.owners[0].step()
        assert rank.finish()


def test_finished_cannot_precede_last_quiet_or_cv_publish():
    fabric = FinishedFabric([[0, 1], [0, 0]])
    source, expert = fabric.ranks
    source.pack()
    expert.pack()
    source.start[1] = expert.start[0] = 1
    dispatch = fabric.put(0, 1, 0, False)
    fabric.deliver(dispatch)
    fabric.quiet(dispatch)
    fabric.publish(dispatch)
    assert expert.compute()
    combine = fabric.put(1, 0, 0, True)
    fabric.deliver(combine)
    with pytest.raises(AssertionError):
        fabric.publish_finished(1, 0)
    fabric.quiet(combine)
    with pytest.raises(AssertionError):
        fabric.publish_finished(1, 0)
    fabric.publish(combine)
    fabric.publish_finished(1, 0)
    assert source.peer_finished[1] == 1


def test_incoming_observation_before_quiet_is_nonblocking_and_not_outbound_completion():
    fabric = FinishedFabric([[0, 1], [0, 0]])
    source, expert = fabric.ranks
    source.pack()
    expert.pack()
    source.start[1] = expert.start[0] = 1
    dispatch = fabric.put(0, 1, 0, False)
    fabric.deliver(dispatch)
    fabric.quiet(dispatch)
    fabric.publish(dispatch)
    assert expert.compute()
    owner = expert.owners[0]
    owner.state = "combine"
    assert owner.step() and owner.state == "combine-quiet"
    # An empty reverse edge can announce finished while this PUT is in flight.
    fabric.publish_finished(0, 1)
    assert not owner.step()  # Nonblocking observation ran; quiet awaits delivery.
    assert expert.incoming_count == 1 and not owner.done
    assert not owner.step() and expert.incoming_count == 1
    assert source.peer_finished.get(1) is None
    fabric.deliver(owner.transfer)
    assert owner.step() and owner.step()  # Quiet, then the per-CV signal.
    assert owner.step() and owner.step()  # End loop, then finished publication.
    assert source.peer_finished[1] == 1


def test_finished_epoch_reset_precedes_start_and_old_or_future_epochs_do_not_count():
    fabric = FinishedFabric([[0, 0], [0, 0]])
    rank, peer = fabric.ranks
    peer.pack()
    rank.epoch = peer.epoch = 2
    rank.incoming_count = rank.aic_done = 28
    rank.incoming_sources = {0, 1}
    rank.pack()
    assert rank.incoming_count == rank.aic_done == 0 and not rank.incoming_sources
    owner = rank.owners[0]
    assert owner.step() and peer.start[0] == 2
    owner.state = "wait-finished"
    for epoch in (1, 3):
        rank.peer_finished[1] = epoch
        assert not owner.step() and rank.incoming_count == 0
    rank.peer_finished[1] = 2
    assert owner.step() and rank.incoming_count == 1
    assert not owner.step() and rank.incoming_count == 1


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
#include <array>
#include <cstdint>
#include <functional>
#include <stdexcept>
#define __aicore__
#define __gm__
constexpr int PIPE_S = 0;
namespace AscendC {
enum class HardEvent { S_MTE3, MTE3_S };
using TEventID = int;
constexpr TEventID EVENT_ID0 = 0;
inline unsigned polls = 0;
inline unsigned selfPolls = 0, publications = 0, backoffs = 0;
inline std::function<uint64_t(uint64_t*)> readPeer;
inline std::function<int32_t(int32_t*)> readSelf;
inline uint64_t ReadGmByPassDCache(uint64_t* address) {
    if (++polls > 100) throw std::runtime_error("still waiting");
    return readPeer ? readPeer(address) : *address;
}
inline int32_t ReadGmByPassDCache(int32_t* address) {
    if (++selfPolls > 100) throw std::runtime_error("still waiting for self");
    return readSelf ? readSelf(address) : *address;
}
inline void AtomicAdd(int32_t* address, int32_t value) { *address += value; }
inline uint64_t GetSystemCycle() { return uint64_t(++backoffs) * 200U; }
template<typename T> struct LocalTensor {
    std::array<T, 4> values{};
    void SetValue(unsigned index, T value) { values.at(index) = value; }
};
template<typename T> struct GlobalTensor {
    T* pointer{};
    void SetGlobalBuffer(T* address) { pointer = address; }
};
template<HardEvent> void SetFlag(TEventID) {}
template<HardEvent> void WaitFlag(TEventID) {}
template<typename T> void DataCopy(GlobalTensor<T> target, LocalTensor<T> source, unsigned count) {
    if (count != 4) throw std::runtime_error("unexpected signal size");
    for (unsigned i = 0; i < count; ++i) target.pointer[i] = source.values[i];
    ++publications;
}
}
"""
    )
    (temporary / "shmem.h").write_text(
        r"""
#pragma once
inline void aclshmemi_kernel_abort(const char*, ...) {}
template<typename T, int> void aclshmemx_udma_qp_put_nbi(T*, T*, void*, uint32_t, uint32_t, uint32_t, uint32_t) {}
inline void aclshmemx_udma_qp_quiet(uint32_t, uint32_t) {}
inline void* aclshmem_ptr(void* address, unsigned) { return address; }
"""
    )
    source = temporary / "workers.cpp"
    source.write_text(
        r"""
#include "compute_contract.hpp"
#include "tail_completion.hpp"
#include "transport.hpp"
#include <algorithm>
#include <cassert>
#include <string>
#include <vector>
using namespace MegaMoeUrma;
int main(int argc, char** argv) {
    assert(argc == 3);
    const uint32_t world = std::stoul(argv[2]);
    assert(world == 4 || world == 8 || world == 16);
    if (std::string(argv[1]) == "completion") {
        using namespace AscendC;
        Arguments args{};
        args.config.rank = world - 1;
        args.config.world = world;
        args.config.epoch = 7;
        args.config.barrierOffset = 128;
        std::vector<uint64_t> symmetric(16 + 4 * 16 * 16);
        std::vector<int32_t> local(16 * 32);
        args.symmetric = reinterpret_cast<uint64_t>(symmetric.data());
        args.localFlags = reinterpret_cast<uint64_t>(local.data());
        auto* incoming = LocalCompletionFlag(args, kLocalFlagIncomingDone);
        auto* aic = LocalCompletionFlag(args, kLocalFlagAicDone);
        auto flag = [&](unsigned owner) {
            return reinterpret_cast<uint64_t*>(args.symmetric +
                PeerPhaseSlot(args, 1, PeerForOwner(args.config.rank, owner)));
        };
        bool peerCounted = false, selfCounted = false;
        *flag(0) = 6;
        assert(!ObserveIncomingCompletion(args, 0, peerCounted, selfCounted));
        *flag(0) = 8;
        assert(!ObserveIncomingCompletion(args, 0, peerCounted, selfCounted));
        assert(peerCounted && !selfCounted && *incoming == 1 && polls == 2 && selfPolls == 2);
        for (unsigned i = 0; i < 27; ++i) MarkCubeComplete(args);
        *flag(0) = 7;
        assert(!ObserveIncomingCompletion(args, 0, peerCounted, selfCounted));
        assert(peerCounted && !selfCounted && *incoming == 1);
        const auto peerReads = polls;
        assert(!ObserveIncomingCompletion(args, 0, peerCounted, selfCounted));
        assert(polls == peerReads && *incoming == 1);
        MarkCubeComplete(args);
        assert(*aic == 28 && ObserveIncomingCompletion(args, 0, peerCounted, selfCounted));
        assert(*incoming == 2);
        const auto selfReads = selfPolls;
        assert(ObserveIncomingCompletion(args, 0, peerCounted, selfCounted));
        assert(polls == peerReads && selfPolls == selfReads && *incoming == 2);

        // Reset happens before a new invocation's Start. Old finished epochs
        // remain in symmetric slots and must not contribute after this reset.
        args.config.epoch = 9;
        std::fill(local.begin(), local.end(), 0);
        std::array<bool, kWorld - 1> peerSeen{}, selfSeen{};
        assert(!ObserveIncomingCompletion(args, 0, peerSeen[0], selfSeen[0]));
        assert(*incoming == 0);
        for (unsigned i = 0; i < 28; ++i) MarkCubeComplete(args);
        for (unsigned owner = 0; owner < world - 1; ++owner) {
            *flag(owner) = 9;
            assert(ObserveIncomingCompletion(args, owner, peerSeen[owner], selfSeen[owner]));
        }
        assert(*incoming == world);
        const auto noMorePeerReads = polls, noMoreSelfReads = selfPolls;
        for (unsigned owner = 0; owner < world - 1; ++owner)
            assert(ObserveIncomingCompletion(args, owner, peerSeen[owner], selfSeen[owner]));
        assert(*incoming == world && polls == noMorePeerReads && selfPolls == noMoreSelfReads);

        // Empty edges still publish first, even while the incoming peer and
        // self completion are delayed. Polling cannot begin before publication.
        args.config.epoch = 11;
        std::fill(local.begin(), local.end(), 0);
        peerCounted = selfCounted = false;
        publications = polls = selfPolls = backoffs = 0;
        readPeer = [&](uint64_t* address) {
            assert(publications == 1 && address == flag(0));
            return polls == 1 ? 10ULL : 11ULL;
        };
        readSelf = [&](int32_t* address) {
            assert(publications == 1 && address == aic);
            return selfPolls < 3 ? 27 : 28;
        };
        LocalTensor<uint64_t> signal;
        FinishCombinePeer(args, 0, signal, peerCounted, selfCounted);
        assert(publications == 1 && polls == 2 && selfPolls == 3 && backoffs > 0);
        assert(*incoming == 2 && peerCounted && selfCounted);
        auto* published = reinterpret_cast<uint64_t*>(args.symmetric +
            PeerPhaseSlot(args, 1, args.config.rank));
        assert(published[0] == 11 && published[1] == 0 && published[2] == 0 && published[3] == 0);

        // Early observation before quiet does not suppress the eventual
        // outbound finished publication, and already-counted flags are skipped.
        publications = polls = selfPolls = 0;
        readPeer = [](uint64_t*) -> uint64_t { assert(false); return 0; };
        readSelf = [](int32_t*) -> int32_t { assert(false); return 0; };
        FinishCombinePeer(args, 0, signal, peerCounted, selfCounted);
        assert(publications == 1 && polls == 0 && selfPolls == 0 && *incoming == 2);
        return 0;
    }
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
    const uint32_t reducers = ReduceWorkers(world);
    const uint32_t comm = CommunicationWorkers(world);
    const uint32_t helpers = ActivationWorkers(world);
    assert(comm == world - 1 && helpers == 29 - world && reducers == 57 - world);
    std::vector<unsigned> visits(rows), workers(reducers), steadyPack(kAic), startupPack(kLocalWorkers);
    unsigned communications = 0;
    for (unsigned block = 0; block < kAic; ++block) {
        if (block < comm) ++steadyPack[block];
        else ++steadyPack[comm + (block - comm)];
        for (unsigned sub = 0; sub < 2; ++sub) {
            assert(++startupPack[LocalWorkerIndex(block, sub)] == 1);
            auto worker = ReduceWorkerIndex(block, sub, world);
            if (worker == reducers) {
                assert(sub == 1 && block < comm);
                assert(ReduceOwnedRows(rows, worker, world) == 0);
                ++communications;
                continue;
            }
            assert(worker < reducers && ++workers[worker] == 1);
            auto begin = ReduceRowBegin(rows, worker, world);
            auto count = ReduceOwnedRows(rows, worker, world);
            assert(count <= 200 && begin + count <= rows);
            total += count;
            for (uint64_t i = begin; i < begin + count; ++i) assert(++visits[i] == 1);
        }
    }
    assert(communications == comm && total == rows);
    assert(std::all_of(visits.begin(), visits.end(), [](auto n) { return n == 1; }));
    assert(std::all_of(workers.begin(), workers.end(), [](auto n) { return n == 1; }));
    assert(std::all_of(steadyPack.begin(), steadyPack.end(), [](auto n) { return n == 1; }));
    assert(std::all_of(startupPack.begin(), startupPack.end(), [](auto n) { return n == 1; }));
    // Sparse groups and shifted prefixes: every GMM1 tile and input-scale
    // group has one helper owner, including all newly available EP4/8 helpers.
    constexpr uint32_t groups = 513;
    std::vector<unsigned> scales(groups), activations(groups * 5);
    std::vector<unsigned> helperVisits(helpers);
    for (unsigned helper = 0; helper < helpers; ++helper) {
        for (unsigned group = helper; group < groups; group += helpers) {
            assert(++scales[group] == 1);
            ++helperVisits[helper];
        }
        for (unsigned group = 0; group < groups; ++group) {
            const uint64_t prefix = 37 + group * 5;
            for (unsigned tile = FirstOwnedComputeTile(prefix, helper, helpers); tile < 5; tile += helpers)
                assert(++activations[group * 5 + tile] == 1);
        }
    }
    assert(std::all_of(scales.begin(), scales.end(), [](auto n) { return n == 1; }));
    assert(std::all_of(activations.begin(), activations.end(), [](auto n) { return n == 1; }));
    assert(std::all_of(helperVisits.begin(), helperVisits.end(), [](auto n) { return n != 0; }));
    const uint64_t large = (1ULL << 32) + 29;
    uint64_t next = 0;
    for (unsigned worker = 0; worker < reducers; ++worker) {
        assert(ReduceRowBegin(large, worker, world) == next);
        next += ReduceOwnedRows(large, worker, world);
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


@pytest.mark.parametrize("world", [4, 8, 16])
@pytest.mark.parametrize("rows", [0, 1, 40, 41, 42, 49, 53, 127, 128, 129, 2048, 4096, 8192])
def test_actual_reducer_and_helper_partition(reduce_workers_binary, rows, world):
    result = subprocess.run(
        [str(reduce_workers_binary), str(rows), str(world)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_start_and_payload_wait_predicates(reduce_workers_binary):
    result = subprocess.run([str(reduce_workers_binary), "flags", "16"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("world", [4, 8, 16])
def test_actual_finished_helpers_count_once_publish_first_and_require_28_aic(reduce_workers_binary, world):
    result = subprocess.run(
        [str(reduce_workers_binary), "completion", str(world)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_finished_observation_and_aic_publication_integration():
    root = Path(__file__).resolve().parents[3]
    source = (root / "experiments/megamoe/csrc/kernels/mega_moe/urma/kernel.hpp").read_text()
    retire = source[
        source.index("__aicore__ inline void RetireCombineBatch") : source.index(
            "__aicore__ inline void CommunicatePeer"
        )
    ]
    assert (
        retire.index("ObserveIncomingCompletion(args, owner, peerCounted, selfCounted);")
        < retire.index("Quiet(peer, kCombineQp);")
        < retire.index("PublishDone(symmetric,")
    )
    combine = source[source.index("while (remaining != 0U)") : source.index("__aicore__ inline void Process")]
    assert (
        combine.index("Put(symmetric,")
        < combine.index("if (pendingCount != 0U)")
        < combine.index("timer.End(Phase::Combine, combineBytes)")
        < combine.index("FinishCombinePeer(args, owner, signal, peerCounted, selfCounted);")
    )
    process = source[source.index("__aicore__ inline void Process") :]
    assert process.index("RunCubeAndPrologue(args, block, timer);") < process.index("MarkCubeComplete(args);")
    assert process.count("MarkCubeComplete(args);") == 1
