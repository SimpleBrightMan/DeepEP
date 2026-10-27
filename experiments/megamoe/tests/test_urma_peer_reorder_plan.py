"""Execute the production full-row peer/CV planner against independent routes."""

from collections import defaultdict
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import numpy as np
import pytest


@pytest.fixture(scope="module")
def peer_reorder_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for peer-reorder tests")
    temporary = tmp_path_factory.mktemp("urma_peer_reorder")
    root = Path(__file__).resolve().parents[3]
    source = temporary / "peer_reorder.cpp"
    source.write_text(
        r"""
#include "expert_plan.hpp"
#include "peer_reorder_plan.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
using namespace MegaMoeUrma;
Config Dimensions(uint32_t bs, uint32_t cv, uint32_t world) {
    Config c{};
    c.bs = bs; c.hidden = 3584; c.intermediate = 1280; c.topk = 8;
    c.world = world; c.localExperts = 384 / world; c.cvRows = cv;
    c.maxPeerChunks = (bs * 8 + cv - 1) / cv;
    return c;
}
int main(int argc, char** argv) {
    static_assert(sizeof(PeerReorderSpan) == 24);
    static_assert(sizeof(PeerReorderRange) == 8);
    if (argc > 1 && std::string(argv[1]) == "invalid") {
        std::vector<int64_t> counts(4 * 384);
        for (uint32_t source = 0; source < 4; ++source) counts[source * 384] = 264;
        const auto original = BuildExpertPlan(counts.data(), Dimensions(33, 128, 4), 1056);
        auto build = [](const auto& p) {
            return BuildPeerReorderPlan(p.config, p.fragments, p.segments, p.recvChunks);
        };
        assert(build(original).spans.size() == original.segments.size());
        auto reject = [&](const auto& p) {
            bool threw = false;
            try { build(p); } catch (const std::invalid_argument&) { threw = true; }
            assert(threw);
        };
        auto bad = original; bad.config.world = 3; reject(bad);
        bad = original; bad.config.rank = 4; reject(bad);
        bad = original; bad.config.maxPeerChunks = 0; reject(bad);
        bad = original; bad.recvChunks.push_back(bad.recvChunks[0]); reject(bad);
        bad = original; bad.recvChunks.pop_back(); reject(bad);
        bad = original; ++bad.recvChunks[0].localRow; reject(bad);
        bad = original; bad.fragments[0].rows = 257; reject(bad);
        bad = original; ++bad.fragments[0].rowBegin; reject(bad);
        bad = original; ++bad.fragments[0].segmentBegin; reject(bad);
        bad = original; bad.segments[0].rows = 0; reject(bad);
        bad = original; ++bad.segments[0].tileRow; reject(bad);
        bad = original; ++bad.segments[0].returnRow; reject(bad);
        bad = original; ++bad.segments[0].receiveRow; reject(bad);
        bad = original; bad.segments[0].sourceRank = 4; reject(bad);
        bad = original; bad.segments.push_back(bad.segments[0]); reject(bad);
        bad = original; bad.segments.pop_back(); reject(bad);
        return 0;
    }
    uint32_t bs, cv, world;
    assert(std::cin >> bs >> cv >> world);
    std::vector<int64_t> counts(world * 384);
    for (auto& count : counts) assert(std::cin >> count);
    for (uint32_t rank = 0; rank < world; ++rank) {
        auto config = Dimensions(bs, cv, world); config.rank = rank;
        auto p = BuildExpertPlan(counts.data(), config, uint64_t(world) * bs * 8);
        if (argc > 1 && std::string(argv[1]) == "reverse")
            std::reverse(p.recvChunks.begin(), p.recvChunks.end());
        auto reorder = BuildPeerReorderPlan(p.config, p.fragments, p.segments, p.recvChunks);
        assert(reorder.spans.size() == p.segments.size());
        assert(reorder.ranges.size() == p.recvChunks.size());
        std::cout << "P " << rank << ' ' << p.config.receiveRows << '\n';
        for (uint32_t index = 0; index < p.recvChunks.size(); ++index) {
            const auto& c = p.recvChunks[index]; const auto& r = reorder.ranges[index];
            std::cout << "C " << c.peer << ' ' << c.cv << ' ' << c.localRow << ' '
                      << c.remoteRow << ' ' << c.rows << ' ' << r.spanBegin << ' ' << r.spanCount << '\n';
        }
        for (const auto& s : reorder.spans)
            std::cout << "S " << s.sourceRow << ' ' << s.destinationRow << ' ' << s.fragmentIndex
                      << ' ' << s.rows << '\n';
    }
}
""",
        encoding="utf-8",
    )
    binary = temporary / ("peer_reorder.exe" if os.name == "nt" else "peer_reorder")
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


def read_peer_plans(binary, counts, bs, cv_rows, reverse=False):
    result = subprocess.run(
        [str(binary)] + (["reverse"] if reverse else []),
        input=f"{bs} {cv_rows} {len(counts)}\n" + " ".join(map(str, counts.reshape(-1))),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    plans = []
    for line in result.stdout.splitlines():
        label, *values = line.split()
        if label == "P":
            plans.append(defaultdict(list))
        plans[-1][label].append(list(map(int, values)))
    assert len(plans) == len(counts)
    return plans


def expected_rows(counts, rank, cv_rows):
    """Flatten source expert lists independently of the segment descriptors."""
    world = len(counts)
    local_experts = 384 // world
    peer_counts = counts.reshape(world, world, local_experts).sum(axis=2)
    send_offsets = np.pad(peer_counts.cumsum(axis=1), ((0, 0), (1, 0)))
    recv_offsets = np.pad(peer_counts.cumsum(axis=0), ((1, 0), (0, 0)))
    expert_offsets = np.pad(counts.cumsum(axis=1), ((0, 0), (1, 0)))
    peers, cvs, destinations, groups = [], [], [], []
    group_base = 0
    for expert in range(rank * local_experts, (rank + 1) * local_experts):
        expert_row = 0
        for source in range(world):
            rows = int(counts[source, expert])
            expanded = np.arange(expert_offsets[source, expert], expert_offsets[source, expert] + rows)
            relative = expanded - send_offsets[source, rank]
            peers.extend([source] * rows)
            cvs.extend(relative // cv_rows)
            destinations.extend(expanded if source == rank else recv_offsets[source, rank] + relative)
            groups.extend(group_base + np.arange(expert_row, expert_row + rows) // 256)
            expert_row += rows
        group_base += (expert_row + 255) // 256
    return tuple(np.asarray(values, dtype=np.int64) for values in (peers, cvs, destinations, groups))


def check_peer_plans(counts, cv_rows, plans):
    world = len(counts)
    for rank, plan in enumerate(plans):
        peers, cvs, destinations, groups = expected_rows(counts, rank, cv_rows)
        assert plan["P"] == [[rank, len(peers)]]
        covered = np.zeros(len(peers), dtype=np.uint8)
        span_cursor = 0
        self_workers = [0] * (world - 1)
        for peer, cv, local, remote, rows, begin, length in plan["C"]:
            assert begin == span_cursor and length > 0
            owner = cv % (world - 1) if peer == rank else peer - (peer > rank)
            assert 0 <= owner < world - 1
            if peer == rank:
                self_workers[owner] += 1
            else:
                assert (owner if owner < rank else owner + 1) == peer
            copied = 0
            last_group = -1
            for source, destination, fragment, count in plan["S"][begin : begin + length]:
                assert 0 < count <= 256 and 0 <= source < source + count <= len(peers)
                span = slice(source, source + count)
                assert np.all(peers[span] == peer) and np.all(cvs[span] == cv)
                assert np.all(groups[span] == fragment) and fragment >= last_group
                np.testing.assert_array_equal(destinations[span], np.arange(destination, destination + count))
                assert destination == (remote if peer == rank else local) + copied
                covered[span] += 1
                copied += count
                last_group = fragment
            assert copied == rows
            span_cursor += length
        assert max(self_workers) - min(self_workers) <= 1
        assert span_cursor == len(plan["S"])
        assert np.all(covered == 1)


@pytest.mark.parametrize("world", [4, 8, 16])
@pytest.mark.parametrize("cv_rows", [128, 256, 768, 4096])
@pytest.mark.parametrize("pattern", ["all_self", "empty_self", "one_expert", "random", "boundary"])
def test_peer_cv_ranges_cover_arbitrary_mgroups(peer_reorder_binary, world, cv_rows, pattern):
    bs = 257
    local_experts = 384 // world
    random = np.random.default_rng(207)
    counts = np.zeros((world, 384), dtype=np.int64)
    for source in range(world):
        if pattern == "random":
            counts[source] = np.bincount(random.integers(0, 384, bs * 8), minlength=384)
        else:
            destination = {
                "all_self": source,
                "empty_self": (source + 1) % world,
                "one_expert": world - 1,
                "boundary": 0,
            }[pattern]
            first = destination * local_experts
            counts[source, first] = bs * 8
            if pattern != "one_expert":
                # Offset both the M256 and CV boundaries; include an M1 tail.
                split = 257 + source * 11
                counts[source, first] -= split + 1
                counts[source, first + 3] = split
                counts[source, first + local_experts - 1] = 1
    plans = read_peer_plans(peer_reorder_binary, counts, bs, cv_rows, reverse=pattern == "boundary")
    check_peer_plans(counts, cv_rows, plans)


def test_peer_ranges_copy_every_hidden_column_and_direct_self_return(peer_reorder_binary):
    bs, world, hidden, cv_rows = 5, 4, 3584, 128
    random = np.random.default_rng(812)
    counts = np.array([np.bincount(random.integers(0, 384, bs * 8), minlength=384) for _ in range(world)])
    plans = read_peer_plans(peer_reorder_binary, counts, bs, cv_rows)
    check_peer_plans(counts, cv_rows, plans)
    for rank, plan in enumerate(plans):
        peers, _, destinations, _ = expected_rows(counts, rank, cv_rows)
        source = np.arange(len(peers) * hidden, dtype=np.int64).reshape(-1, hidden)
        remote = np.full_like(source, -1)
        self_return = np.full((bs * 8, hidden), -1, dtype=np.int64)
        for peer, _, _, _, _, begin, length in plan["C"]:
            output = self_return if peer == rank else remote
            for first, destination, _, rows in plan["S"][begin : begin + length]:
                output[destination : destination + rows, :] = source[first : first + rows, :]
        for row, (peer, destination) in enumerate(zip(peers, destinations)):
            np.testing.assert_array_equal((self_return if peer == rank else remote)[destination], source[row])


def test_peer_reorder_rejects_gaps_overlaps_and_invalid_ranges(peer_reorder_binary):
    result = subprocess.run([str(peer_reorder_binary), "invalid"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
