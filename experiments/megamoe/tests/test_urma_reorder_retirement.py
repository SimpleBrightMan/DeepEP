"""Execute production tail retirement beside a real, untouched reorder cursor.

Only transport, DMA and external producers are doubles. RetireCombineBatch,
OwnedReorder, ReorderBackend and OwnedReorderProgress are compiled unchanged.
This checks publication/cursor contracts, not hardware timing or DMA ordering.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from test_urma_reorder_comm import FAKE_BACKEND


ADAPTER_SUPPORT = r"""
inline ComputeBatchTask LoadComputeTask(const Arguments& args, uint32_t i) {
    return reinterpret_cast<const ComputeBatchTask*>(args.computeTasks)[i];
}
inline ComputeBatchRange LoadComputeBatch(const Arguments& args, uint32_t i) {
    return ComputeBatchFragments(reinterpret_cast<const ComputeBatch*>(args.computeBatches)[i]);
}
inline Fragment LoadFragment(const Arguments& args, uint32_t i) {
    return reinterpret_cast<const Fragment*>(args.fragments)[i];
}
inline uint32_t FragmentGmm2Tiles(const Arguments& args, const Fragment& f) {
    return ComputeTileCount(f.rows, args.config.hidden);
}
inline TraceMeta FragmentTraceMeta(const Fragment& f, uint32_t i) {
    TraceMeta meta; meta.fragment = i; meta.expert = f.expert; return meta;
}
inline void MarkReorderComplete(const Arguments&);
inline void ReorderGmm2Tile(const Arguments& args, const Fragment& f, uint32_t i,
                           uint32_t tile, Timer& timer) {
    assert((f.gmm2TileBegin + tile) % kAic == Test::owner);
    assert(*ComputeFlag(args.gmm2MatmulDone, f.gmm2TileBegin + tile) == 1);
    const auto start = timer.TraceBegin();
    Test::traceClock += 20U;
    Test::ownedRuns.push_back(i);
    ++*ComputeFlag(args.gmm2Done, i);
    timer.TraceEnd(TraceEvent::Gmm2Reorder, start, FragmentTraceMeta(f, i));
}
"""


QUIET_DOUBLE = r"""
inline void Quiet(uint32_t peer, uint32_t qp) {
    assert(peer == Test::peer && qp == kCombineQp);
    assert(!Test::combinePending.empty());
    assert(Test::ownedRuns.empty() && Test::cycleCalls == 0U);
    ++Test::combineQuiets;
    Test::combineBatches.push_back(Test::combinePending);
    for (auto index : Test::combinePending) {
        assert(!Test::quieted.at(index));
        Test::quieted.at(index) = true;
    }
    Test::combinePending.clear();
}
"""


DRIVER = r"""
using namespace MegaMoeUrma;
int main(int argc, char** argv) {
    assert(argc == 5);
    const std::string scenario = argv[1];
    const uint32_t world = std::stoul(argv[2]), owner = std::stoul(argv[3]);
    const bool unrelatedReady = std::stoul(argv[4]) != 0U;
    Arguments args{};
    args.config.world = world; args.config.rank = world - 1U;
    args.config.hidden = 3584U; args.config.maxPeerChunks = kMaxPeerChunks;
    args.config.epoch = 7U; args.config.combineReadyOffset = 8192U;
    Test::args = &args; Test::owner = owner; Test::peer = PeerForOwner(args.config.rank, owner);
    std::vector<uint64_t> symmetric(4096U);
    std::array<int32_t, 5U * 32U> local{};
    std::vector<Fragment> fragments(8U);
    std::vector<ComputeBatch> batches;
    std::vector<ComputeBatchTask> tasks;
    // The production cursor visits owned fragments 0,2,4,6. These tiles are
    // unrelated to already submitted immutable payloads, and must stay put.
    for (uint32_t i = 0; i < fragments.size(); ++i) {
        fragments[i].rows = 256U;
        fragments[i].gmm2TileBegin = owner + i * 14U;
        batches.push_back({i, 1U});
        tasks.push_back({ComputeBatchStage::Gmm1, i});
        tasks.push_back({ComputeBatchStage::Gmm2, i});
    }
    std::vector<int32_t> matmul(160U * 32U), computed(8U * 32U);
    args.symmetric = Test::Ptr(symmetric.data()); args.localFlags = Test::Ptr(local.data());
    args.fragments = Test::Ptr(fragments.data()); args.config.fragmentCount = fragments.size();
    args.computeBatches = Test::Ptr(batches.data()); args.computeTasks = Test::Ptr(tasks.data());
    args.config.computeTaskCount = tasks.size();
    args.gmm2MatmulDone = Test::Ptr(matmul.data()); args.gmm2Done = Test::Ptr(computed.data());
    if (unrelatedReady) *ComputeFlag(args.gmm2MatmulDone, fragments[0].gmm2TileBegin) = 1;
    constexpr uint32_t begin = 2U;
    const uint32_t count = scenario == "capacity" ? kMaxPeerChunks : scenario == "single-only" ? 1U : 4U;
    std::vector<Chunk> chunks(begin + count);
    for (uint32_t i = 0; i < count; ++i)
        chunks[begin + i] = {Test::peer, i, i * 256U, i * 256U, 32U + 16U * (i % 7U), 0U, 0U, 0U};
    args.recvChunks = Test::Ptr(chunks.data()); args.config.recvChunkCount = chunks.size();
    LocalTensor<uint32_t> pending(TPosition::VECCALC, kCommunicationPendingOffset, kMaxPeerChunks);
    LocalTensor<uint64_t> signal(TPosition::VECCALC, kCommunicationSignalOffset, 4U);
    std::vector<uint32_t> expected;
    if (scenario == "prefix" || scenario == "capacity") {
        for (uint32_t i = 0U; i + 1U < count; ++i) expected.push_back(i);
    } else if (scenario == "holes") {
        expected = {0U, 2U, 3U}; // the final actual CV will be CV1, not the highest index
    } else if (scenario == "last-single") {
        expected = {1U};
    } else if (scenario == "single-only") {
        expected = {0U};
    } else {
        assert(scenario == "empty");
    }
    uint32_t pendingCount = expected.size();
    uint64_t pendingRows = 0U;
    for (uint32_t i = 0; i < pendingCount; ++i) {
        pending.SetValue(i, expected[i]);
        pendingRows += chunks[begin + expected[i]].rows;
    }
    Test::combinePending = expected;
    Test::combinePublications.resize(count); Test::quieted.resize(count);
    if (scenario == "last-single") {
        for (uint32_t i : {0U, 2U, 3U}) {
            Test::combinePublications[i] = 1U;
            Test::quieted[i] = true;
        }
    }
    const auto oldPublications = Test::combinePublications;
    Timer timer;
    OwnedReorder reorder(args, owner, timer);
    assert(!reorder.Done() && reorder.FragmentIndex() == 0U);
    bool peerCounted = false, selfCounted = false;
    RetireCombineBatch(args, owner, begin, pending, pendingCount, pendingRows,
                       signal, timer, peerCounted, selfCounted);
    assert(pendingCount == 0U && pendingRows == 0U && Test::combinePending.empty());
    assert(Test::combineQuiets == (expected.empty() ? 0U : 1U));
    if (!expected.empty()) assert(Test::combineBatches.front() == expected);
    assert(Test::cycleCalls == 0U && Test::ownedRuns.empty());
    assert(!reorder.Done() && reorder.FragmentIndex() == 0U && reorder.Ready() == unrelatedReady);
    for (uint32_t i = 0; i < count; ++i) {
        const bool retired = std::find(expected.begin(), expected.end(), i) != expected.end();
        assert(Test::combinePublications[i] == oldPublications[i] + (retired ? 1U : 0U));
    }
    assert(Test::traces.size() == (expected.empty() ? 0U : expected.size() + 1U));
    if (!expected.empty()) {
        assert(Test::traces.front().event == TraceEvent::CombineQuiet);
        for (uint32_t i = 1U; i < Test::traces.size(); ++i) {
            assert(Test::traces[i].event == TraceEvent::CombinePublish);
            assert(Test::traces[i].start >= Test::traces.front().end);
        }
    }
    // A repeated empty retire must not quiet or republish the old entries.
    const auto publications = Test::combinePublications;
    const auto quietCount = Test::combineQuiets;
    RetireCombineBatch(args, owner, begin, pending, pendingCount, pendingRows,
                       signal, timer, peerCounted, selfCounted);
    assert(Test::combinePublications == publications && Test::combineQuiets == quietCount);
    assert(Test::cycleCalls == 0U && Test::ownedRuns.empty());
    // Prove the real cursor remains usable rather than accidentally completed.
    if (unrelatedReady) {
        assert(reorder.RunOneReady());
        assert((Test::ownedRuns == std::vector<uint32_t>{0U}));
        assert(reorder.FragmentIndex() == 2U);
    }
}
"""


@pytest.fixture(scope="module")
def retirement_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for production retirement checks")
    headers = Path(__file__).resolve().parents[3] / "experiments/megamoe/csrc/kernels/mega_moe/urma"
    compute = (headers / "compute.hpp").read_text(encoding="utf-8")
    actual_owned = compute[compute.index("struct ReorderBackend {") : compute.index("// Helpers retain their static")]
    fake = FAKE_BACKEND.replace(
        '#include "compute_contract.hpp"',
        '#include "compute_contract.hpp"\n#include "reorder_progress.hpp"',
    )
    fake = fake.replace(
        "inline int32_t* unblockFlag;", "inline int32_t* unblockFlag;\ninline std::vector<uint32_t> ownedRuns;"
    )
    first, last = fake.index("class OwnedReorder {"), fake.index("inline void Put(")
    fake = fake[:first] + ADAPTER_SUPPORT + actual_owned + fake[last:]
    first, last = fake.index("inline void Quiet("), fake.index("inline void PublishDone(")
    fake = fake[:first] + QUIET_DOUBLE + fake[last:]
    kernel = (headers / "kernel.hpp").read_text(encoding="utf-8")
    kernel = kernel[kernel.index("namespace MegaMoeUrma {") : kernel.index("__aicore__ inline void CommunicatePeer")]
    tail = (headers / "tail_completion.hpp").read_text(encoding="utf-8")
    tail = tail[tail.index("namespace MegaMoeUrma {") :]
    temporary = tmp_path_factory.mktemp("urma_reorder_retirement")
    source = temporary / "retirement.cpp"
    source.write_text(fake + "\n" + tail + "\n" + kernel + "\n}\n" + DRIVER, encoding="utf-8")
    binary = temporary / ("retirement.exe" if os.name == "nt" else "retirement")
    result = subprocess.run(
        [*command, "-std=c++17", "-O1", "-UNDEBUG", "-I" + str(headers), str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("world,owner", [(4, 0), (8, 5), (16, 14)])
@pytest.mark.parametrize("unrelated_ready", [False, True])
@pytest.mark.parametrize("scenario", ["prefix", "holes", "last-single", "single-only", "empty", "capacity"])
def test_tail_retirement_only_waits_for_submitted_puts(retirement_binary, world, owner, unrelated_ready, scenario):
    result = subprocess.run(
        [str(retirement_binary), scenario, str(world), str(owner), str(int(unrelated_ready))],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
