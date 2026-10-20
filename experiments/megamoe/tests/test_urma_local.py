"""Exercise production local-worker/address helpers in a CPU routing oracle.

The scalar payloads tag complete token/scale rows. These tests check ownership,
destinations and row lifetime across calls, not NPU visibility or FFN numerics.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import numpy as np
import pytest

from deep_ep_experimental.megamoe.urma_routing import plan_routes


@pytest.fixture(scope="module")
def local_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for the local routing test")
    temporary = tmp_path_factory.mktemp("urma_local")
    root = Path(__file__).resolve().parents[3]
    source = temporary / "local.cpp"
    source.write_text(
        r"""
#include "compute_contract.hpp"
#include "expert_plan.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
using namespace MegaMoeUrma;
using Value = int64_t;
constexpr Value poison = -7;
constexpr uint32_t h = 3584, topk = 8;
struct Payload {
    Value token = poison, scale = poison;
    bool operator==(const Payload& other) const { return token == other.token && scale == other.scale; }
};
struct Rank {
    Config config{};
    ExpertPlan plan;
    uint32_t bs{};
    std::array<uint64_t, 16> counts{}, send{}, recv{};
    std::vector<uint64_t> order, inverse;
    std::vector<uint32_t> ids;
    std::vector<Payload> expanded, received;
    std::vector<Value> gmm2, returned, output;
};
Payload Input(uint32_t epoch, uint32_t rank, uint64_t token) {
    return {Value(epoch) * 100000000 + rank * 1000000 + Value(token),
            Value(epoch) * 100000 + rank * 1000 + Value(token % 251)};
}
Value Expert(const Payload& input, uint32_t expert) {
    return input.token * (expert + 1) + input.scale * 3;
}
Value Weight(uint32_t rank, uint32_t slot) { return Value(rank + 1) * (slot + 1); }
void Ownership(uint64_t rows) {
    std::vector<uint32_t> visits(rows), workers(kLocalWorkers);
    uint64_t owned = 0, packBytes = 0, outputBytes = 0;
    for (uint32_t block = 0; block < kAic; ++block) {
        for (uint32_t sub = 0; sub < 2; ++sub) {
            const auto worker = LocalWorkerIndex(block, sub);
            assert(worker < kLocalWorkers && ++workers[worker] == 1);
            uint64_t count = 0;
            for (uint64_t row = worker; row < rows; row += kLocalWorkers) {
                assert(++visits[row] == 1);
                ++count;
            }
            assert(LocalOwnedRows(rows, worker) == count);
            owned += count;
            packBytes += count * (h + h / 32);
            outputBytes += count * h * 2;
        }
    }
    assert(owned == rows && packBytes == rows * (h + h / 32) && outputBytes == rows * h * 2);
    assert(std::all_of(visits.begin(), visits.end(), [](auto n) { return n == 1; }));
    assert(std::all_of(workers.begin(), workers.end(), [](auto n) { return n == 1; }));
}
int main(int argc, char** argv) {
    static_assert(kLocalWorkers == 56 && kAic == 28);
    if (argc > 1) {
        Ownership(std::stoull(argv[1]));
        // Address arithmetic must not truncate 64-bit row prefixes.
        for (uint64_t offset : {0ULL, (1ULL << 32) + 11ULL}) {
            Config c{};
            c.selfSendRow = offset + 13;
            c.selfRecvRow = offset + 901;
            c.selfRows = 57;
            assert(!IsSelfExpandedRow(c.selfSendRow - 1, c));
            assert(!IsSelfExpandedRow(c.selfSendRow + c.selfRows, c));
            for (uint64_t row = 0; row < c.selfRows; ++row) {
                const auto expanded = c.selfSendRow + row;
                const auto received = c.selfRecvRow + row;
                assert(IsSelfExpandedRow(expanded, c));
                assert(SelfReceiveRow(expanded, c) == received);
                assert(SelfReturnRow(received, c) == expanded);
            }
            c.selfRows = 0;
            assert(!IsSelfExpandedRow(c.selfSendRow, c));
            assert(!IsSelfExpandedRow(c.selfSendRow + 1, c));
        }
        return 0;
    }
    std::array<Rank, 16> ranks;
    for (uint32_t rank = 0; rank < 16; ++rank) {
        auto& r = ranks[rank];
        assert(std::cin >> r.bs >> r.config.selfSendRow >> r.config.selfRecvRow >> r.config.selfRows);
        r.config.rank = rank;
        r.config.hidden = h;
        r.config.topk = topk;
        r.config.bs = r.bs;
        for (auto& n : r.counts) assert(std::cin >> n);
        const uint64_t rows = uint64_t(r.bs) * topk;
        r.order.resize(rows);
        r.ids.resize(rows);
        r.inverse.resize(rows);
        std::vector<uint32_t> coverage(rows);
        for (uint64_t row = 0; row < rows; ++row) {
            assert(std::cin >> r.order[row]);
            assert(r.order[row] < rows && ++coverage[r.order[row]] == 1);
            r.inverse[r.order[row]] = row;
        }
        for (auto& expert : r.ids) { assert(std::cin >> expert); assert(expert < 384); }
        std::partial_sum(r.counts.begin(), r.counts.end() - 1, r.send.begin() + 1);
        assert(r.config.selfSendRow == r.send[rank] && r.config.selfRows == r.counts[rank]);
        assert(std::accumulate(r.counts.begin(), r.counts.end(), uint64_t(0)) == rows);
        r.expanded.resize(rows);
        r.returned.assign(rows, poison);
        r.output.assign(r.bs, poison);
    }
    for (uint32_t destination = 0; destination < 16; ++destination) {
        auto& d = ranks[destination];
        uint64_t rows = 0;
        for (uint32_t source = 0; source < 16; ++source) {
            d.recv[source] = rows;
            rows += ranks[source].counts[destination];
        }
        assert(d.config.selfRecvRow == d.recv[destination]);
        d.received.resize(rows);
        d.gmm2.assign(rows, poison);
    }
    std::vector<int64_t> allCounts(16 * 384);
    for (uint32_t source = 0; source < 16; ++source) {
        assert(ranks[source].bs == ranks[0].bs);
        for (auto expert : ranks[source].ids) ++allCounts[source * 384 + expert];
    }
    for (auto& r : ranks) {
        r.config.localExperts = 24;
        r.config.intermediate = 1280;
        r.config.cvRows = 256;
        r.config.maxPeerChunks = (r.bs * topk + r.config.cvRows - 1) / r.config.cvRows;
        r.plan = BuildExpertPlan(allCounts.data(), r.config, 16ULL * r.bs * topk);
        assert(r.plan.config.selfSendRow == r.config.selfSendRow);
        assert(r.plan.config.selfRecvRow == r.config.selfRecvRow);
        assert(r.plan.config.selfRows == r.config.selfRows);
    }
    // Buffers persist between epochs. A direct self write must never touch a
    // remote interval still holding another call's data. Remote completions
    // are deliberately applied in reverse order after all self work.
    for (uint32_t epoch = 1; epoch <= 3; ++epoch) {
        std::array<std::vector<uint32_t>, 16> receiveVisits, returnVisits;
        uint64_t packBytes = 0, dispatchBytes = 0, combineBytes = 0, selfRows = 0, allRows = 0;
        for (uint32_t rank = 0; rank < 16; ++rank) {
            auto& r = ranks[rank];
            receiveVisits[rank].assign(r.received.size(), 0);
            returnVisits[rank].assign(r.returned.size(), 0);
            const auto before = r.received;
            std::vector<uint32_t> visits(r.order.size());
            for (uint32_t block = 0; block < kAic; ++block)
                for (uint32_t sub = 0; sub < 2; ++sub) {
                    auto worker = LocalWorkerIndex(block, sub);
                    uint64_t owned = 0;
                    for (uint64_t row = worker; row < r.order.size(); row += kLocalWorkers) {
                        assert(++visits[row] == 1);
                        const auto value = Input(epoch, rank, r.order[row] / topk);
                        if (IsSelfExpandedRow(row, r.config)) {
                            const auto target = SelfReceiveRow(row, r.config);
                            assert(target < r.received.size() && ++receiveVisits[rank][target] == 1);
                            r.received[target] = value;
                            // No expanded self row is needed or read.
                            assert(r.expanded[row].token == poison);
                        } else {
                            r.expanded[row] = value;
                        }
                        ++owned;
                    }
                    assert(owned == LocalOwnedRows(r.order.size(), worker));
                    packBytes += owned * (h + h / 32);
                }
            for (uint64_t row = 0; row < r.received.size(); ++row)
                if (row < r.config.selfRecvRow || row >= r.config.selfRecvRow + r.config.selfRows)
                    assert(r.received[row] == before[row]);
            allRows += r.order.size();
            selfRows += r.config.selfRows;
        }
        for (uint32_t source = 0; source < 16; ++source) {
            auto& s = ranks[source];
            for (uint32_t destination = 0; destination < 16; ++destination) {
                if (source == destination) continue;
                auto& d = ranks[destination];
                for (uint64_t row = 0; row < s.counts[destination]; ++row) {
                    const auto target = d.recv[source] + row;
                    assert(++receiveVisits[destination][target] == 1);
                    d.received[target] = s.expanded[s.send[destination] + row];
                }
                dispatchBytes += s.counts[destination] * (h + h / 32);
            }
        }
        for (uint32_t destination = 0; destination < 16; ++destination) {
            auto& d = ranks[destination];
            const auto before = d.returned;
            assert(std::all_of(receiveVisits[destination].begin(), receiveVisits[destination].end(),
                               [](auto n) { return n == 1; }));
            // Exercise the same expert-major gather/scatter descriptors as the
            // kernel. The scalar expert transform is an independent oracle;
            // communication/Reduce must preserve original token+top-k identity.
            for (const auto& group : d.plan.fragments) {
                std::vector<Payload> gathered(group.rows);
                std::vector<uint32_t> gatheredVisits(group.rows);
                for (uint32_t j = 0; j < group.segmentCount; ++j) {
                    const auto& segment = d.plan.segments[group.segmentBegin + j];
                    const auto& s = ranks[segment.sourceRank];
                    for (uint32_t row = 0; row < segment.rows; ++row) {
                        const auto original = s.order[segment.returnRow + row];
                        assert(s.ids[original] == destination * 24 + group.expert);
                        const auto input = d.received[segment.receiveRow + row];
                        assert(input == Input(epoch, segment.sourceRank, original / topk));
                        assert(++gatheredVisits[segment.tileRow + row] == 1);
                        gathered[segment.tileRow + row] = input;
                    }
                }
                assert(std::all_of(gatheredVisits.begin(), gatheredVisits.end(), [](auto n) { return n == 1; }));
                for (uint32_t j = 0; j < group.segmentCount; ++j) {
                    const auto& segment = d.plan.segments[group.segmentBegin + j];
                    for (uint32_t row = 0; row < segment.rows; ++row) {
                        const auto receiveRow = segment.receiveRow + row;
                        const auto returnRow = segment.returnRow + row;
                        const auto value = Expert(gathered[segment.tileRow + row], destination * 24 + group.expert);
                        if (segment.sourceRank == destination) {
                            assert(returnRow == SelfReturnRow(receiveRow, d.config));
                            assert(++returnVisits[destination][returnRow] == 1);
                            d.returned[returnRow] = value;
                            assert(d.gmm2[receiveRow] == poison);
                        } else {
                            d.gmm2[receiveRow] = value;
                        }
                    }
                }
            }
            for (uint64_t row = 0; row < d.returned.size(); ++row)
                if (!IsSelfExpandedRow(row, d.config)) assert(d.returned[row] == before[row]);
        }
        for (int destination = 15; destination >= 0; --destination) {
            const auto& d = ranks[destination];
            for (uint32_t source = 0; source < 16; ++source) {
                if (source == uint32_t(destination)) continue;
                auto& s = ranks[source];
                for (uint64_t row = 0; row < s.counts[destination]; ++row) {
                    const auto target = s.send[destination] + row;
                    assert(!IsSelfExpandedRow(target, s.config) && ++returnVisits[source][target] == 1);
                    s.returned[target] = d.gmm2[d.recv[source] + row];
                }
                combineBytes += s.counts[destination] * h * 2;
            }
        }
        for (uint32_t rank = 0; rank < 16; ++rank) {
            auto& r = ranks[rank];
            assert(std::all_of(returnVisits[rank].begin(), returnVisits[rank].end(), [](auto n) { return n == 1; }));
            std::vector<uint32_t> visits(r.bs);
            uint64_t outputBytes = 0;
            for (uint32_t block = 0; block < kAic; ++block)
                for (uint32_t sub = 0; sub < 2; ++sub) {
                    const auto worker = LocalWorkerIndex(block, sub);
                    uint64_t owned = 0;
                    for (uint32_t token = worker; token < r.bs; token += kLocalWorkers) {
                        assert(++visits[token] == 1);
                        Value actual = 0, expected = 0;
                        for (uint32_t slot = 0; slot < topk; ++slot) {
                            const auto route = uint64_t(token) * topk + slot;
                            actual += r.returned[r.inverse[route]] * Weight(rank, slot);
                            expected += Expert(Input(epoch, rank, token), r.ids[route]) * Weight(rank, slot);
                        }
                        assert(actual == expected);
                        r.output[token] = actual;
                        ++owned;
                    }
                    assert(owned == LocalOwnedRows(r.bs, worker));
                    outputBytes += owned * h * 2;
                }
            assert(outputBytes == uint64_t(r.bs) * h * 2);
            assert(std::all_of(visits.begin(), visits.end(), [](auto n) { return n == 1; }));
        }
        assert(packBytes == allRows * (h + h / 32));
        assert(dispatchBytes == (allRows - selfRows) * (h + h / 32));
        assert(combineBytes == (allRows - selfRows) * h * 2);
    }
}
"""
    )
    binary = temporary / ("local.exe" if os.name == "nt" else "local")
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


@pytest.mark.parametrize("rows", [0, 1, 27, 28, 55, 56, 57, 127, 128, 129, 2048, 4096, 8192, 65536])
def test_all_aiv_ownership_ragged_tails_and_64bit_addresses(local_binary, rows):
    result = subprocess.run([str(local_binary), str(rows)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def run_roundtrip(local_binary, ids):
    routes = [plan_routes(value, 384, 16) for value in ids]
    parts = []
    for rank, (value, route) in enumerate(zip(ids, routes)):
        receive_offset = sum(int(r.peer_counts[rank]) for r in routes[:rank])
        parts.append(f"{len(value)} {route.peer_offsets[rank]} {receive_offset} {route.peer_counts[rank]}")
        parts.append(" ".join(map(str, route.peer_counts)))
        parts.append(" ".join(map(str, route.order)))
        parts.append(" ".join(map(str, value.reshape(-1))))
    result = subprocess.run([str(local_binary)], input="\n".join(parts), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bs", [2048, 4096, 8192])
def test_balanced_all_ranks_direct_self_roundtrip(local_binary, bs):
    ids = []
    for rank in range(16):
        linear = np.arange(rank * bs * 8, (rank + 1) * bs * 8)
        ids.append(((linear % 16) * 24 + (linear // 16) % 24).reshape(bs, 8))
    run_roundtrip(local_binary, ids)


@pytest.mark.parametrize("pattern", ["all_self", "empty_self", "skewed", "random"])
def test_sparse_duplicate_and_ragged_direct_self_roundtrip(local_binary, pattern):
    rng = np.random.default_rng(1019)
    ids = []
    for rank in range(16):
        bs = 307
        if pattern == "random":
            value = rng.integers(0, 384, (bs, 8), dtype=np.int64)
        else:
            destination = {"all_self": rank, "empty_self": (rank + 1) % 16, "skewed": 7}[pattern]
            # Repeated expert instances must retain independent top-k weights.
            value = np.full((bs, 8), destination * 24, dtype=np.int64)
            value[::3, 3:] += 5
        ids.append(value)
    run_roundtrip(local_binary, ids)
