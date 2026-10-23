"""CPU execution of the production host planner against independent routes."""

from collections import defaultdict
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import numpy as np
import pytest

from deep_ep_experimental.megamoe.urma_routing import plan_routes


@pytest.fixture(scope="module")
def expert_plan_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for expert-plan tests")
    temporary = tmp_path_factory.mktemp("urma_expert_plan")
    root = Path(__file__).resolve().parents[3]
    source = temporary / "expert_plan.cpp"
    source.write_text(
        r"""
#include "expert_plan.hpp"
#include <cassert>
#include <iostream>
using namespace MegaMoeUrma;
Config Dimensions(uint32_t bs, uint32_t cv, uint32_t world = 16) {
    Config c{};
    c.bs = bs; c.hidden = 3584; c.intermediate = 1280; c.topk = 8;
    c.world = world; c.localExperts = 384 / world; c.cvRows = cv; c.maxPeerChunks = (bs * 8 + cv - 1) / cv;
    return c;
}
int main(int argc, char**) {
    std::array<int64_t, 16 * 384> counts{};
    if (argc > 1) {
        auto c = Dimensions(1, 128);
        for (uint32_t s = 0; s < 16; ++s) counts[s * 384] = 8;
        auto reject = [&](const Config& config, uint64_t capacity) {
            bool threw = false;
            try { BuildExpertPlan(counts.data(), config, capacity); }
            catch (const std::invalid_argument&) { threw = true; }
            assert(threw);
        };
        assert(BuildExpertPlan(counts.data(), c, 128).config.receiveRows == 128);
        reject(c, 127);
        reject(c, 0);
        auto bad = c; bad.rank = 16; reject(bad, 128);
        bad = c; bad.cvRows = 0; reject(bad, 128);
        bad = c; bad.cvRows = 129; reject(bad, 128);
        bad = c; bad.maxPeerChunks = 0; reject(bad, 128);
        bad = c; bad.maxPeerChunks = 513; reject(bad, 128);
        bad = c; bad.world = 3; reject(bad, 128);
        bad = c; bad.world = 0; reject(bad, 128);
        bad = c; bad.localExperts = 23; reject(bad, 128);
        bad = c; bad.gmm1AheadMGroups = 65537; reject(bad, 128);
        counts[0] = -1; reject(c, 128);
        counts[0] = 9; reject(c, 128);
        counts[0] = 7; reject(c, 128);
        c = Dimensions(257, 128); c.maxPeerChunks = 1;
        for (uint32_t s = 0; s < 16; ++s) counts[s * 384] = 257 * 8;
        reject(c, 16 * 257 * 8);
        return 0;
    }
    uint32_t bs = 0, cv = 0, world = 0;
    assert(std::cin >> bs >> cv >> world);
    for (uint32_t index = 0; index < world * 384; ++index) assert(std::cin >> counts[index]);
    for (uint32_t rank = 0; rank < world; ++rank) {
        auto config = Dimensions(bs, cv, world);
        config.rank = rank;
        auto p = BuildExpertPlan(counts.data(), config, uint64_t(world) * bs * 8);
        const auto& c = p.config;
        std::cout << "P " << rank << ' ' << c.receiveRows << ' ' << c.gmm1Tiles << ' ' << c.gmm2Tiles
                  << ' ' << c.selfSendRow << ' ' << c.selfRecvRow << ' ' << c.selfRows << ' '
                  << c.maxPeerChunks << ' ' << c.fragmentCount << ' ' << c.sendChunkCount << ' '
                  << c.recvChunkCount << '\n';
        for (const auto& b : p.computeBatches)
            std::cout << "B " << b.fragmentBegin << ' ' << b.fragmentCount << '\n';
        assert(c.computeTaskCount == p.computeTasks.size());
        for (const auto& task : p.computeTasks)
            std::cout << "A " << static_cast<uint32_t>(task.stage) << ' ' << task.batchIndex << '\n';
        for (const auto& f : p.fragments)
            std::cout << "F " << f.expert << ' ' << f.segmentBegin << ' ' << f.rowBegin << ' '
                      << f.rows << ' ' << f.segmentCount << ' ' << f.gmm1TileBegin << ' '
                      << f.gmm2TileBegin << '\n';
        for (const auto& s : p.segments)
            std::cout << "S " << s.sourceRank << ' ' << s.cv << ' ' << s.receiveRow << ' '
                      << s.returnRow << ' ' << s.tileRow << ' ' << s.rows << ' '
                      << s.inputReadyIndex << ' ' << s.reserved << '\n';
        auto chunks = [](const auto& list, char label) {
            for (const auto& c : list)
                std::cout << label << ' ' << c.peer << ' ' << c.cv << ' ' << c.localRow << ' '
                          << c.remoteRow << ' ' << c.rows << ' ' << c.fragmentBegin << ' '
                          << c.fragmentCount << ' ' << c.reserved << '\n';
        };
        chunks(p.sendChunks, 'T'); chunks(p.recvChunks, 'R');
        std::cout << 'D';
        for (auto group : p.chunkGroupIndices) std::cout << ' ' << group;
        std::cout << '\n';
        assert(p.reduceDependencies.size() == uint64_t(bs) * 8);
        for (uint64_t begin = 0; begin < p.reduceDependencies.size();) {
            uint64_t end = begin + 1;
            while (end < p.reduceDependencies.size() &&
                   p.reduceDependencies[end] == p.reduceDependencies[begin]) ++end;
            std::cout << "Q " << begin << ' ' << end - begin << ' ' << p.reduceDependencies[begin] << '\n';
            begin = end;
        }
    }
}
"""
    )
    binary = temporary / ("expert_plan.exe" if os.name == "nt" else "expert_plan")
    result = subprocess.run(
        command
        + [
            "-std=c++17",
            "-O1",
            "-UNDEBUG",
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


def read_plans(binary, ids, cv_rows):
    world = len(ids)
    routes = [plan_routes(value, 384, world) for value in ids]
    counts = np.array([route.expert_counts for route in routes])
    result = subprocess.run(
        [str(binary)],
        input=f"{len(ids[0])} {cv_rows} {world}\n" + " ".join(map(str, counts.reshape(-1))),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    plans = []
    for line in result.stdout.splitlines():
        label, *values = line.split()
        values = list(map(int, values))
        if label == "P":
            plans.append(defaultdict(list))
        plans[-1][label].append(values)
    assert len(plans) == world
    return routes, plans


def check_plans(ids, cv_rows, routes, plans):
    world, local_experts = len(ids), 384 // len(ids)
    counts = np.array([r.peer_counts for r in routes])
    receive_base = np.concatenate((np.zeros((1, world), dtype=np.int64), counts.cumsum(axis=0)), axis=0)
    for rank, plan in enumerate(plans):
        (
            reported_rank,
            rows,
            g1,
            g2,
            self_send,
            self_recv,
            self_rows,
            max_chunks,
            groups,
            sends,
            receives,
        ) = plan["P"][0]
        assert reported_rank == rank and rows == counts[:, rank].sum()
        assert self_send == routes[rank].peer_offsets[rank]
        assert self_recv == receive_base[rank, rank] and self_rows == counts[rank, rank]
        assert groups == len(plan["F"]) and sends == len(plan["T"]) and receives == len(plan["R"])
        assert g1 == groups * 5 and g2 == groups * 14
        assert len(plan["B"]) == local_experts
        assert len(plan["A"]) == 2 * local_experts
        assert [expert for stage, expert in plan["A"] if stage == 1] == list(range(local_experts))
        assert [expert for stage, expert in plan["A"] if stage == 2] == list(range(local_experts))
        submitted = consumed = 0
        for stage, expert in plan["A"]:
            if stage == 1:
                submitted += 1
            else:
                assert expert == consumed and submitted >= min(local_experts, consumed + 2)
                queued = sum(count for _, count in plan["B"][consumed:submitted])
                assert submitted == local_experts or queued >= 12
                consumed += 1
        covered = np.zeros(rows, dtype=np.int8)
        compact = np.zeros(rows, dtype=np.int8)
        used_segments = np.zeros(len(plan["S"]), dtype=np.int8)
        expected_dependencies = defaultdict(set)
        expected_reduce_dependencies = np.full(ids[rank].size, (1 << 64) - 1, dtype=np.uint64)
        expected_compact_rows = []
        for expert in range(local_experts):
            global_expert = rank * local_experts + expert
            expert_rows = []
            for source, route in enumerate(routes):
                lo, hi = map(int, route.expert_offsets[global_expert : global_expert + 2])
                for return_row in range(lo, hi):
                    peer_row = return_row - int(route.peer_offsets[rank])
                    expert_rows.append((source, int(receive_base[source, rank]) + peer_row, return_row))
            first, count = plan["B"][expert]
            assert count == (len(expert_rows) + 255) // 256
            assert first == sum(b[1] for b in plan["B"][:expert])
            cursor = 0
            for group_index in range(first, first + count):
                e, begin, row_begin, group_rows, segment_count, g1_begin, g2_begin = plan["F"][group_index]
                assert e == expert and 0 < group_rows <= 256
                assert group_rows == min(256, len(expert_rows) - cursor)
                assert row_begin == len(expected_compact_rows) + cursor
                assert g1_begin == group_index * 5 and g2_begin == group_index * 14
                compact[row_begin : row_begin + group_rows] += 1
                mapping = []
                for segment_index in range(begin, begin + segment_count):
                    used_segments[segment_index] += 1
                    source, cv, receive_row, return_row, tile_row, length, ready, reserved = plan["S"][segment_index]
                    assert reserved == 0 and length > 0 and tile_row == len(mapping)
                    assert ready == source * max_chunks + cv
                    peer_row = receive_row - int(receive_base[source, rank])
                    assert cv == peer_row // cv_rows == (peer_row + length - 1) // cv_rows
                    assert return_row == int(routes[source].peer_offsets[rank]) + peer_row
                    covered[receive_row : receive_row + length] += 1
                    original = routes[source].order[return_row : return_row + length]
                    assert np.all(ids[source].reshape(-1)[original] == global_expert)
                    mapping.extend((source, receive_row + offset, return_row + offset) for offset in range(length))
                    expected_dependencies[source, cv].add(group_index)
                    if source == rank:
                        assert return_row - self_send == receive_row - self_recv
                assert mapping == expert_rows[cursor : cursor + group_rows]
                for source, _, return_row in expert_rows[cursor : cursor + group_rows]:
                    if source == rank:
                        expected_reduce_dependencies[return_row] = (1 << 63) | group_index
                cursor += group_rows
            assert cursor == len(expert_rows)
            expected_compact_rows.extend(expert_rows)
        assert len(expected_compact_rows) == rows
        assert np.all(covered == 1) and np.all(compact == 1) and np.all(used_segments == 1)

        expected_send = []
        expected_receive = []
        for peer in range(world):
            for offset in range(0, int(counts[rank, peer]), cv_rows):
                if peer != rank:
                    lo = int(routes[rank].peer_offsets[peer]) + offset
                    hi = lo + min(cv_rows, int(counts[rank, peer]) - offset)
                    expected_reduce_dependencies[lo:hi] = peer * max_chunks + offset // cv_rows
                expected_send.append(
                    [
                        peer,
                        offset // cv_rows,
                        int(routes[rank].peer_offsets[peer]) + offset,
                        int(receive_base[rank, peer]) + offset,
                        min(cv_rows, int(counts[rank, peer]) - offset),
                        0,
                        0,
                        0,
                    ]
                )
            for offset in range(0, int(counts[peer, rank]), cv_rows):
                expected_receive.append(
                    [
                        peer,
                        offset // cv_rows,
                        int(receive_base[peer, rank]) + offset,
                        int(routes[peer].peer_offsets[rank]) + offset,
                        min(cv_rows, int(counts[peer, rank]) - offset),
                    ]
                )
        assert plan["T"] == expected_send
        assert [chunk[:5] for chunk in plan["R"]] == expected_receive
        dependency_cursor = 0
        for source, cv, _, _, chunk_rows, begin, count, reserved in plan["R"]:
            assert chunk_rows > 0 and reserved == 0 and begin == dependency_cursor
            actual = plan["D"][0][begin : begin + count]
            assert actual == sorted(expected_dependencies[source, cv])
            assert len(actual) == len(set(actual)) > 0
            dependency_cursor += count
        assert dependency_cursor == len(plan["D"][0])
        actual_reduce_dependencies = np.empty(ids[rank].size, dtype=np.uint64)
        dependency_cursor = 0
        for begin, length, encoded in plan["Q"]:
            assert begin == dependency_cursor and length > 0
            assert begin + length <= actual_reduce_dependencies.size
            actual_reduce_dependencies[begin : begin + length] = encoded
            dependency_cursor += length
        assert dependency_cursor == actual_reduce_dependencies.size
        assert not np.any(expected_reduce_dependencies == np.uint64((1 << 64) - 1))
        np.testing.assert_array_equal(actual_reduce_dependencies, expected_reduce_dependencies)
        # This is the index_select contract used by native bind on the NPU:
        # each token/top-k entry receives the dependency for its original row.
        token_dependencies = actual_reduce_dependencies[routes[rank].reduce_index]
        destinations = ids[rank] // local_experts
        self_entries = (token_dependencies & np.uint64(1 << 63)) != 0
        np.testing.assert_array_equal(self_entries, destinations == rank)
        np.testing.assert_array_equal(token_dependencies[~self_entries] // max_chunks, destinations[~self_entries])
        assert np.all((token_dependencies[self_entries] & np.uint64((1 << 63) - 1)) < groups)
        # The transport workload is unchanged, regardless of compute grouping.
        assert sum(c[4] for c in plan["T"] if c[0] != rank) == counts[rank].sum() - self_rows
        assert sum(c[4] for c in plan["R"] if c[0] != rank) == counts[:, rank].sum() - self_rows


@pytest.mark.parametrize("bs,expected_groups", [(2048, 72), (4096, 144), (8192, 264)])
@pytest.mark.parametrize("cv_rows", [128, 256, 1024, 4096])
def test_balanced_expert_groups_preserve_original_transport(expert_plan_binary, bs, expected_groups, cv_rows):
    ids = []
    for rank in range(16):
        linear = np.arange(rank * bs * 8, (rank + 1) * bs * 8)
        ids.append(((linear % 16) * 24 + (linear // 16) % 24).reshape(bs, 8))
    routes, plans = read_plans(expert_plan_binary, ids, cv_rows)
    check_plans(ids, cv_rows, routes, plans)
    assert all(len(plan["F"]) == expected_groups for plan in plans)


@pytest.mark.parametrize("world", [4, 8, 16])
@pytest.mark.parametrize("pattern", ["all_self", "empty_self", "one_expert", "random", "boundary"])
@pytest.mark.parametrize("cv_rows", [128, 256, 768])
def test_sparse_skewed_ragged_and_empty_experts(expert_plan_binary, pattern, cv_rows, world):
    local_experts = 384 // world
    bs = 257
    rng = np.random.default_rng(207)
    ids = []
    for rank in range(world):
        if pattern == "random":
            value = rng.integers(0, 384, (bs, 8), dtype=np.int64)
        else:
            destination = {"all_self": rank, "empty_self": (rank + 1) % world, "one_expert": world - 1, "boundary": 0}[pattern]
            value = np.full((bs, 8), destination * local_experts, dtype=np.int64)
            if pattern != "one_expert":
                # Unequal source/expert lengths intersect both M256 and CV
                # boundaries; the remainder of the 24 experts remains empty.
                value.reshape(-1)[1 + rank * 7 : 258 + rank * 11] += 3
                value[-1, -1] += local_experts - 1
        ids.append(value)
    routes, plans = read_plans(expert_plan_binary, ids, cv_rows)
    check_plans(ids, cv_rows, routes, plans)
    if pattern == "empty_self":
        assert all(plan["P"][0][6] == 0 for plan in plans)
    if pattern == "one_expert":
        assert all(not plan["F"] for rank, plan in enumerate(plans) if rank != world - 1)


def test_invalid_counts_and_capacities_are_rejected(expert_plan_binary):
    result = subprocess.run([str(expert_plan_binary), "invalid"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
