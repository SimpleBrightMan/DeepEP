"""Execute the production token-ready CSR planner with a host C++ compiler."""

from collections import Counter
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess

import pytest


SELF = 1 << 63


@pytest.fixture(scope="module")
def ready_plan_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for token-ready planner tests")
    temporary = tmp_path_factory.mktemp("urma_reduce_ready_plan")
    source = temporary / "ready_plan.cpp"
    source.write_text(
        r"""
#include "reduce_ready_plan.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <limits>
using namespace MegaMoeUrma;
int main(int argc, char**) {
    Config config{};
    if (argc > 1) {
        config.bs = 15; config.topk = 8; config.rank = 0;
        config.maxPeerChunks = 2; config.fragmentCount = 4;
        std::array<uint64_t, 15 * 8> dependencies;
        dependencies.fill(2);
        auto reject = [&](const uint64_t* values, const Config& c) {
            bool threw = false;
            try { BuildReduceReadyPlan(values, c); }
            catch (const std::invalid_argument&) { threw = true; }
            assert(threw);
        };
        assert(BuildReduceReadyPlan(dependencies.data(), config).owners.size() == 15);
        reject(nullptr, config);
        auto bad = config; bad.bs = 0; reject(dependencies.data(), bad);
        bad = config; bad.bs = 8193; reject(dependencies.data(), bad);
        bad = config; bad.topk = 7; reject(dependencies.data(), bad);
        bad = config; bad.rank = 16; reject(dependencies.data(), bad);
        bad = config; bad.maxPeerChunks = 0; reject(dependencies.data(), bad);
        bad = config; bad.maxPeerChunks = 513; reject(dependencies.data(), bad);
        for (uint64_t invalid : {uint64_t{0}, uint64_t{1}, uint64_t{32},
                                 kReduceSelfDependency | 4, std::numeric_limits<uint64_t>::max()}) {
            dependencies[0] = invalid; reject(dependencies.data(), config);
        }
        dependencies[0] = kReduceSelfDependency | 3;
        assert(BuildReduceReadyPlan(dependencies.data(), config).owners.size() == 15);
        bad = config; bad.fragmentCount = 0; reject(dependencies.data(), bad);
        dependencies.fill(2);
        assert(BuildReduceReadyPlan(dependencies.data(), bad).owners.size() == 15);
        return 0;
    }
    assert(std::cin >> config.bs >> config.rank >> config.maxPeerChunks >> config.fragmentCount);
    config.topk = 8;
    std::vector<uint64_t> input(uint64_t(config.bs) * config.topk);
    for (auto& value : input) assert(std::cin >> value);
    auto plan = BuildReduceReadyPlan(input.data(), config);
    for (const auto& o : plan.owners)
        std::cout << "O " << o.tokenBegin << ' ' << o.tokenCount << ' ' << o.dependencyBegin << ' '
                  << o.dependencyCount << ' ' << o.edgeBegin << ' ' << o.edgeCount << ' '
                  << o.reserved0 << ' ' << o.reserved1 << '\n';
    for (const auto& d : plan.dependencies)
        std::cout << "D " << d.dependency << ' ' << d.edgeBegin << ' ' << d.edgeCount << '\n';
    for (const auto& e : plan.edges)
        std::cout << "E " << e.token << ' ' << e.multiplicity << '\n';
}
"""
    )
    root = Path(__file__).resolve().parents[3]
    binary = temporary / ("ready_plan.exe" if os.name == "nt" else "ready_plan")
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


def read_plan(binary, tokens, *, rank=0, chunks=512, groups=4119):
    result = subprocess.run(
        [str(binary)],
        input=f"{len(tokens)} {rank} {chunks} {groups}\n" + " ".join(str(value) for token in tokens for value in token),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    plan = {label: [] for label in ("O", "D", "E")}
    for line in result.stdout.splitlines():
        label, *values = line.split()
        plan[label].append(tuple(map(int, values)))
    return plan


def check_plan(tokens, plan):
    assert len(plan["O"]) == 15
    cursor = dependency_cursor = edge_cursor = 0
    quotient, remainder = divmod(len(tokens), 15)
    for worker, owner in enumerate(plan["O"]):
        first, count, dep_first, dep_count, edge_first, edge_count, reserved0, reserved1 = owner
        assert first == cursor and count == quotient + (worker < remainder)
        assert dep_first == dependency_cursor and edge_first == edge_cursor
        assert dep_first % 2 == 0 and edge_first % 8 == 0
        assert reserved0 == reserved1 == 0
        assert count <= 547 and edge_count <= count * 8 <= 4376
        assert dep_count <= edge_count
        actual = [Counter() for _ in range(count)]
        dependency_values = set()
        consumed_edges = 0
        for dependency, begin, number in plan["D"][dep_first : dep_first + dep_count]:
            assert dependency not in dependency_values
            dependency_values.add(dependency)
            assert begin == consumed_edges and number > 0
            local_tokens = set()
            for token, multiplicity in plan["E"][edge_first + begin : edge_first + begin + number]:
                assert 0 <= token < count and 1 <= multiplicity <= 8
                assert token not in local_tokens
                local_tokens.add(token)
                actual[token][dependency] += multiplicity
            consumed_edges += number
        assert consumed_edges == edge_count
        assert actual == [Counter(token) for token in tokens[first : first + count]]
        assert all(sum(counter.values()) == 8 for counter in actual)
        dependency_cursor = (dep_first + dep_count + 1) // 2 * 2
        edge_cursor = (edge_first + edge_count + 7) // 8 * 8
        assert all(record == (0, 0, 0) for record in plan["D"][dep_first + dep_count : dependency_cursor])
        assert all(record == (0, 0) for record in plan["E"][edge_first + edge_count : edge_cursor])
        cursor += count
    assert cursor == len(tokens)
    assert dependency_cursor == len(plan["D"])
    assert edge_cursor == len(plan["E"])


@pytest.mark.parametrize("bs", [1, 14, 15, 16, 2048, 4096, 8192])
def test_contiguous_owners_and_repeated_cv_multiplicity(ready_plan_binary, bs):
    tokens = [[512 + token % 7] * 3 + [1024 + token % 5] * 4 + [SELF | token % 24] for token in range(bs)]
    plan = read_plan(ready_plan_binary, tokens)
    check_plan(tokens, plan)
    if bs == 8192:
        assert [owner[1] for owner in plan["O"]] == [547, 547] + [546] * 13


@pytest.mark.parametrize("rank,dependency", [(0, 512), (15, 0), (7, SELF | 4118)])
def test_all_eight_routes_share_one_dependency(ready_plan_binary, rank, dependency):
    tokens = [[dependency] * 8 for _ in range(8192)]
    plan = read_plan(ready_plan_binary, tokens, rank=rank)
    check_plan(tokens, plan)
    assert all(owner[3] == 1 and owner[5] == owner[1] for owner in plan["O"])
    assert all(multiplicity == 8 for _, multiplicity in plan["E"] if multiplicity)


def test_maximum_owner_metadata_and_dma_padding(ready_plan_binary):
    tokens = []
    for worker in range(15):
        count = 547 if worker < 2 else 546
        tokens.extend([[512 + local * 8 + slot for slot in range(8)] for local in range(count)])
    plan = read_plan(ready_plan_binary, tokens)
    check_plan(tokens, plan)
    assert plan["O"][0][3] == plan["O"][0][5] == 4376
    assert plan["O"][2][3] == plan["O"][2][5] == 4368


def test_csr_arrival_order_preserves_all_eight_contributions(ready_plan_binary):
    rng = random.Random(8304)
    choices = [512 + index for index in range(9)] + [SELF | index for index in range(7)]
    tokens = [[rng.choice(choices) for _ in range(8)] for _ in range(83)]
    plan = read_plan(ready_plan_binary, tokens)
    check_plan(tokens, plan)
    for first, count, dep_first, dep_count, edge_first, _, _, _ in plan["O"]:
        records = plan["D"][dep_first : dep_first + dep_count]
        # Every invocation starts new counters: old-epoch readiness is not
        # encoded into the reusable structural plan.
        for epoch in (1, 7, 105):
            assert epoch > 0
            rng.shuffle(records)
            arrived = set()
            counters = [0] * count
            for dependency, begin, number in records:
                arrived.add(dependency)
                for token, multiplicity in plan["E"][edge_first + begin : edge_first + begin + number]:
                    counters[token] += multiplicity
                assert all(value <= 8 for value in counters)
                assert [value == 8 for value in counters] == [
                    all(dependency in arrived for dependency in token) for token in tokens[first : first + count]
                ]
            assert counters == [8] * count


def test_invalid_dependencies_and_dimensions(ready_plan_binary):
    result = subprocess.run([str(ready_plan_binary), "invalid"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
