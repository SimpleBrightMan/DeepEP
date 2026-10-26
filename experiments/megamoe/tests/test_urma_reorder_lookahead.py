"""Execute real next-CV lookahead, owned cursor and quiet retirement on CPU.

Only external producer progress, DMA and transport are doubles. In particular,
the production OwnedReorder/ReorderBackend/OwnedReorderProgress and kernel
PrepareNextCombine/RetireCombineBatch bodies are compiled unchanged. These
checks establish scheduling and publication order, not hardware performance.
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


DRIVER = r"""
using namespace MegaMoeUrma;
int main(int argc, char** argv) {
    assert(argc == 4);
    const std::string scenario = argv[1];
    const uint32_t world = std::stoul(argv[2]), owner = std::stoul(argv[3]);
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
    // This owner's real cursor visits fragments 0,2,4,6. All other owners'
    // matmul flags stay zero, so a whole-group wait cannot accidentally pass.
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
    // Nonzero peer range + nonzero dependency offsets guard both index spaces.
    std::vector<uint32_t> dependencies{7U, 0U, 2U, 4U, 6U, 1U};
    constexpr uint32_t begin = 2U, count = 4U;
    std::vector<Chunk> chunks(begin + count);
    for (uint32_t i = 0; i < count; ++i)
        chunks[begin + i] = {Test::peer, i, i * 256U, i * 256U, 256U, i + 1U, 1U, 0U};
    // CV1 normally spans owned fragments 2 and 4; CV2 and CV3 remain unsent.
    chunks[begin + 1U].fragmentCount = 2U;
    args.recvChunks = Test::Ptr(chunks.data()); args.config.recvChunkCount = chunks.size();
    args.chunkGroupIndices = Test::Ptr(dependencies.data());
    LocalTensor<uint32_t> submitted(TPosition::VECCALC, kCommunicationSubmittedOffset, kMaxPeerChunks);
    LocalTensor<uint32_t> pending(TPosition::VECCALC, kCommunicationPendingOffset, kMaxPeerChunks);
    LocalTensor<uint64_t> signal(TPosition::VECCALC, kCommunicationSignalOffset, 4U);
    for (uint32_t i = 0; i < count; ++i) submitted.SetValue(i, i == 0U ? 1U : 0U);
    pending.SetValue(0U, 0U);
    uint32_t pendingCount = 1U;
    uint64_t pendingRows = 256U;
    Test::combinePending = {0U};
    Test::combinePublications.resize(count); Test::quieted.resize(count);
    std::vector<uint32_t> expected{0U, 2U, 4U};
    if (scenario == "holes") {
        submitted.SetValue(2U, 1U);
        chunks[begin + 1U].fragmentCount = 1U; // first hole needs fragment 2 only
        pending.SetValue(1U, 2U); pendingCount = 2U; pendingRows = 512U;
        Test::combinePending = {0U, 2U};
        expected = {0U, 2U};
    } else if (scenario == "no-owned") {
        chunks[begin + 1U].fragmentBegin = 5U;
        chunks[begin + 1U].fragmentCount = 1U; // only another owner's fragment
        expected.clear();
    } else if (scenario == "all-submitted" || scenario == "last-single") {
        for (uint32_t i = 0; i < count; ++i) submitted.SetValue(i, 1U);
        expected.clear();
        if (scenario == "last-single") {
            pending.SetValue(0U, count - 1U);
            Test::combinePending = {count - 1U};
        }
    } else if (scenario == "empty-pending") {
        pendingCount = 0U; pendingRows = 0U; Test::combinePending.clear(); expected.clear();
    }
    // The AIC publishes one owned tile after every forty backoff cycles.
    // Future and other-owner flags never become ready during this call.
    Test::tick = [&] {
        for (uint32_t j = 0; j < expected.size(); ++j)
            if (Test::cycleCalls >= 40U * (j + 1U))
                *ComputeFlag(args.gmm2MatmulDone, fragments[expected[j]].gmm2TileBegin) = 1;
    };
    uint32_t quietChecks = 0U;
    Test::beforeQuiet = [&] {
        assert(Test::ownedRuns == expected);
        // A whole CV is deliberately still not ready: only one tile out of
        // each fragment's fourteen has been reordered by this owner.
        if (!expected.empty()) assert(!IsChunkComputed(args, chunks[begin + 1U]));
        ++quietChecks;
    };
    Timer timer;
    OwnedReorder reorder(args, owner, timer);
    if (scenario == "bounded-ready") {
        // The bounded primitive itself must stop at its target, even when
        // later owned work is already ready. Opportunistic DrainReady is
        // deliberately a separate caller decision after this primitive.
        for (uint32_t index : {0U, 2U, 4U})
            *ComputeFlag(args.gmm2MatmulDone, fragments[index].gmm2TileBegin) = 1;
        reorder.DrainThroughFragment(2U);
        assert((Test::ownedRuns == std::vector<uint32_t>{0U, 2U}));
        assert(reorder.FragmentIndex() == 4U && Test::cycleCalls == 0U);
        reorder.DrainThroughFragment(1U); // target already passed
        assert((Test::ownedRuns == std::vector<uint32_t>{0U, 2U}));
        assert(Test::traces.size() == 2U);
        for (const auto& event : Test::traces) assert(event.event == TraceEvent::Gmm2Reorder);
        assert(reorder.Ready());
        return 0;
    }
    bool peerCounted = false, selfCounted = false;
    RetireCombineBatch(args, owner, begin, count, submitted, pending, pendingCount,
                       pendingRows, signal, timer, peerCounted, selfCounted, reorder);
    assert(Test::ownedRuns == expected && pendingCount == 0U && pendingRows == 0U);
    assert(quietChecks == (scenario == "empty-pending" ? 0U : 1U));
    assert(Test::cycleCalls >= expected.size() * 40U);
    if (expected.empty()) assert(Test::cycleCalls == 0U);
    assert(!reorder.Done()); // later, unrelated owned fragment 6 is still blocked
    assert(reorder.FragmentIndex() == (expected.empty() ? 0U : expected.back() + 2U));
    uint32_t waits = 0U, copies = 0U;
    for (const auto& event : Test::traces) {
        if (event.event == TraceEvent::WaitGmm2Reorder) {
            ++waits;
            for (const auto& other : Test::traces)
                if (other.event == TraceEvent::Gmm2Reorder || other.event == TraceEvent::CombineQuiet)
                    assert(event.end <= other.start || other.end <= event.start);
        }
        if (event.event == TraceEvent::Gmm2Reorder) ++copies;
    }
    // Forty-plus empty probes per tile produce one wait interval, not forty.
    assert(waits == expected.size() && copies == expected.size());
    if (scenario == "holes") assert((Test::combineBatches.front() == std::vector<uint32_t>{0U, 2U}));
    if (scenario == "last-single") assert((Test::combineBatches.front() == std::vector<uint32_t>{3U}));
    for (uint32_t i = 0; i < count; ++i) {
        const bool published = scenario == "empty-pending" ? false :
            scenario == "last-single" ? i == 3U : i == 0U || (scenario == "holes" && i == 2U);
        assert(Test::combinePublications[i] == (published ? 1U : 0U));
    }
}
"""


@pytest.fixture(scope="module")
def reorder_lookahead_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for production lookahead checks")
    headers = Path(__file__).resolve().parents[3] / "experiments/megamoe/csrc/kernels/mega_moe/urma"
    compute = (headers / "compute.hpp").read_text(encoding="utf-8")
    actual_owned = compute[compute.index("struct ReorderBackend {") : compute.index("// Helpers retain their static")]
    fake = FAKE_BACKEND.replace(
        '#include "compute_contract.hpp"',
        '#include "compute_contract.hpp"\n#include "reorder_progress.hpp"\n#include <functional>',
    )
    fake = fake.replace(
        "inline int32_t* unblockFlag;",
        "inline int32_t* unblockFlag;\ninline std::function<void()> tick, beforeQuiet;\ninline std::vector<uint32_t> ownedRuns;",
    )
    fake = fake.replace(
        "return Test::traceClock += 200U;", "if (Test::tick) Test::tick();\n    return Test::traceClock += 200U;"
    )
    first, last = fake.index("class OwnedReorder {"), fake.index("inline void Put(")
    fake = fake[:first] + ADAPTER_SUPPORT + actual_owned + fake[last:]
    fake = fake.replace(
        'assert(Test::lastDrainSerial > Test::lastQuietSerial && "every quiet must first check paired reorder");',
        "if (Test::beforeQuiet) Test::beforeQuiet();",
    )
    kernel = (headers / "kernel.hpp").read_text(encoding="utf-8")
    kernel = kernel[kernel.index("namespace MegaMoeUrma {") : kernel.index("__aicore__ inline void Process")]
    tail = (headers / "tail_completion.hpp").read_text(encoding="utf-8")
    tail = tail[tail.index("namespace MegaMoeUrma {") :]
    temporary = tmp_path_factory.mktemp("urma_reorder_lookahead")
    source = temporary / "lookahead.cpp"
    source.write_text(fake + "\n" + tail + "\n" + kernel + "\n}\n" + DRIVER, encoding="utf-8")
    binary = temporary / ("lookahead.exe" if os.name == "nt" else "lookahead")
    result = subprocess.run(
        [*command, "-std=c++17", "-O1", "-UNDEBUG", "-I" + str(headers), str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("world,owner", [(4, 0), (8, 5), (16, 14)])
@pytest.mark.parametrize(
    "scenario", ["prefix", "holes", "no-owned", "all-submitted", "last-single", "empty-pending", "bounded-ready"]
)
def test_next_cv_owned_reorder_finishes_before_quiet_without_waiting_for_other_owners(
    reorder_lookahead_binary, world, owner, scenario
):
    result = subprocess.run(
        [str(reorder_lookahead_binary), scenario, str(world), str(owner)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
