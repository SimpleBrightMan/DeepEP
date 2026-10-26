"""Execute the production peer loop with adversarial Pack/Reorder DMA doubles.

The doubles supply external AIC progress and transport completion; scheduling,
CV batching, UB offsets and finished handling come from kernel.hpp and
tail_completion.hpp. This validates host-visible protocol/UB contracts, not
hardware timing, atomics or DMA event ordering.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


FAKE_BACKEND = r"""
#include "abi.hpp"
#include "compute_contract.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define __aicore__
#define __gm__
namespace Test {
using namespace MegaMoeUrma;
inline std::array<unsigned char, kRequiredUbBytes> ub{};
inline const Arguments* args;
inline uint32_t owner, peer, round, lastPackRound = UINT32_MAX;
inline uint32_t serial, lastPackSerial, lastDrainSerial, lastQuietSerial;
inline uint32_t drainCalls, reorderRemaining, reorderCompleted, reorderDoneMarks;
inline uint32_t finishedPublications, dispatchQuiets, combineQuiets, cycleCalls;
inline bool deferReorderUntilFinished, enableInitialReorder, markOnFirstDrain;
inline uint32_t readyChecks, readyThreshold;
inline uint64_t traceClock;
inline int32_t* unblockFlag;
struct Trace { TraceEvent event; uint64_t start, end; };
inline std::vector<Trace> traces;
inline std::vector<uint32_t> combinePending, combineSends, combinePublications;
inline std::vector<uint32_t> dispatchPublications, packRounds;
inline std::vector<std::vector<uint32_t>> combineBatches;
inline std::vector<bool> quieted;
inline uint64_t Ptr(const void* ptr) { return reinterpret_cast<uint64_t>(ptr); }
inline int32_t* Flag(uint64_t address, uint32_t index) {
    return reinterpret_cast<int32_t*>(address + index * kFlagBytes);
}
inline void CompleteReorder() {
    if (reorderRemaining == 0U && reorderDoneMarks == 0U) {
        ++*Flag(args->localFlags, kLocalFlagReorderDone);
        ++reorderDoneMarks;
    }
}
}
namespace AscendC {
using bfloat16_t = uint16_t;
enum class TPosition { VECCALC };
constexpr uint32_t EVENT_ID0 = 0U;
template <typename T> struct LocalTensor {
    uint32_t address, count;
    LocalTensor(TPosition, uint32_t offset, uint32_t length) : address(offset), count(length) {
        assert(uint64_t(address) + uint64_t(count) * sizeof(T) <= Test::ub.size());
    }
    void SetValue(uint32_t i, T value) {
        assert(i < count); std::memcpy(Test::ub.data() + address + i * sizeof(T), &value, sizeof(T));
    }
    T GetValue(uint32_t i) const {
        assert(i < count); T value;
        std::memcpy(&value, Test::ub.data() + address + i * sizeof(T), sizeof(T)); return value;
    }
};
template <typename T> T ReadGmByPassDCache(T* ptr) { return *ptr; }
inline void AtomicAdd(int32_t* ptr, int32_t n) { *ptr += n; }
inline uint64_t GetSystemCycle() {
    if (++Test::cycleCalls > 10000U) throw std::runtime_error("peer loop failed to make reorder progress");
    return Test::traceClock += 200U;
}
template <bool> void SyncAll() {}
}
namespace MegaMoeUrma {
using namespace AscendC;
constexpr uint32_t kPackStartupRounds = 2U;
constexpr uint32_t kPackUbBase = 128U * 1024U;
constexpr uint32_t kPackUbBytes = 58880U;
constexpr uint32_t kSignalCopyBytes = 32U;
constexpr uint32_t kDispatchQp = 0U, kCombineQp = 1U;
enum class Phase { Barrier, Dispatch, Combine };
struct Timer {
    void Begin(Phase) {}
    void End(Phase, uint64_t = 0U) {}
    uint64_t TraceBegin() { return ++Test::traceClock; }
    void TraceEnd(TraceEvent event, uint64_t start, const TraceMeta&) {
        Test::traces.push_back({event, start, ++Test::traceClock});
    }
};
inline uint32_t PeerForOwner(uint32_t rank, uint32_t owner) { return owner < rank ? owner : owner + 1U; }
inline int32_t* ComputeFlag(uint64_t address, uint32_t index) { return Test::Flag(address, index); }
inline int32_t* PackReadyFlag(const Arguments& args, uint32_t peer, uint32_t cv) {
    return ComputeFlag(args.packReady, peer * args.config.maxPeerChunks + cv);
}
inline Chunk LoadPackChunk(const Arguments& args, uint32_t round, uint32_t peer) {
    Test::round = round;
    return reinterpret_cast<const PackRound*>(args.packRounds)[round].chunks[peer];
}
inline void PackOneRound(const Arguments&, uint32_t round, uint32_t owner, uint32_t workers, Timer&) {
    assert(owner == Test::owner && workers == kAic);
    Test::packRounds.push_back(round);
    Test::lastPackRound = round;
    Test::lastPackSerial = ++Test::serial;
    // A real six-slot Pack owns this entire phase-private range. Poison it
    // even for an empty peer, exposing persistent communication-state aliases.
    std::memset(Test::ub.data() + kPackUbBase, 0xB7, kPackUbBytes);
}
inline void MarkPackWorkerComplete(const Arguments& args) {
    ++*ComputeFlag(args.localFlags, kLocalFlagPackDone);
}
inline void WaitComputeFlag(uint64_t address, uint32_t index, uint32_t expected) {
    assert(*ComputeFlag(address, index) >= int32_t(expected));
}
inline void WaitStartAtLeast(uint8_t* address, uint64_t expected) {
    assert(*reinterpret_cast<uint64_t*>(address) >= expected);
}
class OwnedReorder {
public:
    OwnedReorder(const Arguments& args, uint32_t owner, Timer& timer) : timer_(timer) {
        assert(&args == Test::args && owner == Test::owner);
        Test::CompleteReorder();
    }
    bool Ready() const {
        if (Test::reorderRemaining && ++Test::readyChecks < Test::readyThreshold) return false;
        return Test::reorderRemaining != 0U &&
            (!Test::deferReorderUntilFinished || Test::finishedPublications != 0U);
    }
    bool DrainReady() {
        ++Test::drainCalls;
        Test::lastDrainSerial = ++Test::serial;
        // Reorder MTE2 may overwrite BOTH low 64KiB slots while Combine
        // pending/submitted state must survive until its later quiet/flags.
        std::memset(Test::ub.data(), 0xD3, kPackUbBase);
        if (!Ready()) return false;
        const uint64_t start = timer_.TraceBegin();
        Test::traceClock += 20U;
        if (Test::enableInitialReorder || Test::markOnFirstDrain) {
            for (uint32_t index = 0; index < Test::args->config.fragmentCount; ++index) {
                const auto& fragment = reinterpret_cast<const Fragment*>(Test::args->fragments)[index];
                *ComputeFlag(Test::args->gmm2Done, index) = ComputeTileCount(fragment.rows, Test::args->config.hidden);
            }
        }
        Test::reorderCompleted += Test::reorderRemaining;
        Test::reorderRemaining = 0U;
        if (Test::unblockFlag) *Test::unblockFlag = 7;
        Test::CompleteReorder();
        timer_.TraceEnd(TraceEvent::Gmm2Reorder, start, {});
        return true;
    }
    bool Done() const { return Test::reorderRemaining == 0U; }
private:
    Timer& timer_;
};
inline void Put(uint8_t*, uint64_t offset, uint8_t*, uint32_t bytes, uint32_t peer, uint32_t qp = kDispatchQp) {
    assert(peer == Test::peer && bytes != 0U);
    if (qp == kDispatchQp) return;
    assert(qp == kCombineQp && !Test::finishedPublications);
    const auto* chunks = reinterpret_cast<const Chunk*>(Test::args->recvChunks);
    uint32_t found = UINT32_MAX;
    for (uint32_t i = 0; i < Test::args->config.recvChunkCount; ++i) {
        if (offset == Test::args->config.returnYOffset + chunks[i].remoteRow * Test::args->config.hidden * 2U) {
            assert(bytes == chunks[i].rows * Test::args->config.hidden * 2U);
            assert(*ComputeFlag(Test::args->gmm2Done, i) == int32_t(ComputeTileCount(chunks[i].rows, Test::args->config.hidden)));
            found = i; break;
        }
    }
    assert(found != UINT32_MAX && ++Test::combineSends.at(found) == 1U);
    Test::combinePending.push_back(found);
}
inline void Quiet(uint32_t peer, uint32_t qp) {
    assert(peer == Test::peer);
    assert(Test::lastDrainSerial > Test::lastQuietSerial && "every quiet must first check paired reorder");
    Test::lastQuietSerial = ++Test::serial;
    if (qp == kDispatchQp) {
        assert(Test::lastPackRound == Test::round + kPackStartupRounds);
        assert(Test::lastDrainSerial > Test::lastPackSerial && "Pack has priority before Dispatch quiet");
        ++Test::dispatchQuiets;
    } else {
        assert(qp == kCombineQp && !Test::combinePending.empty());
        ++Test::combineQuiets;
        Test::combineBatches.push_back(Test::combinePending);
        for (auto index : Test::combinePending) Test::quieted.at(index) = true;
        Test::combinePending.clear();
    }
}
inline void PublishDone(uint8_t*, uint64_t offset, uint32_t peer, uint64_t epoch,
                        LocalTensor<uint64_t> signal, uint32_t event) {
    assert(peer == Test::peer && epoch == Test::args->config.epoch && event == EVENT_ID0);
    assert(signal.address >= kPackUbBase + kPackUbBytes);
    signal.SetValue(0U, epoch);
    const auto& config = Test::args->config;
    const uint64_t finished = config.barrierOffset + (config.world + config.rank) * kFlagBytes;
    if (offset == finished) {
        for (auto count : Test::combinePublications) assert(count == 1U);
        assert(Test::combinePending.empty() && ++Test::finishedPublications == 1U);
        // The peer only acknowledges after receiving our Finished. This
        // catches waiting for incoming/reorder before publishing our side.
        *reinterpret_cast<uint64_t*>(Test::args->symmetric + config.barrierOffset +
                                    (config.world + peer) * kFlagBytes) = epoch;
        return;
    }
    for (uint32_t cv = 0; cv < Test::combinePublications.size(); ++cv) {
        const uint64_t expected = config.combineReadyOffset + (config.rank * config.maxPeerChunks + cv) * kFlagBytes;
        if (offset == expected) {
            assert(Test::quieted.at(cv) && ++Test::combinePublications.at(cv) == 1U);
            return;
        }
    }
    for (uint32_t cv = 0; cv < Test::dispatchPublications.size(); ++cv) {
        const uint64_t expected = config.dispatchReadyOffset + (config.rank * config.maxPeerChunks + cv) * kFlagBytes;
        if (offset == expected) {
            assert(Test::dispatchQuiets > cv && ++Test::dispatchPublications.at(cv) == 1U);
            return;
        }
    }
    throw std::runtime_error("published an unexpected or corrupted CV offset");
}
} // namespace MegaMoeUrma
"""


DRIVER = r"""
using namespace MegaMoeUrma;
void CheckWaitAndReorder(TraceEvent event, uint32_t expectedWaits) {
    uint32_t waits = 0, reorders = 0;
    for (const auto& row : Test::traces) {
        if (row.event == TraceEvent::Gmm2Reorder) ++reorders;
        if (row.event != event) continue;
        ++waits;
        for (const auto& other : Test::traces) {
            if (other.event != TraceEvent::Gmm2Reorder) continue;
            assert(row.end <= other.start || other.end <= row.start);
        }
    }
    assert(waits == expectedWaits && reorders == 1U);
}
int main(int argc, char** argv) {
    using namespace MegaMoeUrma;
    assert(argc == 5);
    const uint32_t world = std::stoul(argv[1]), owner = std::stoul(argv[2]);
    const std::string scenario = argv[3];
    const uint32_t combineCount = std::stoul(argv[4]);
    const uint32_t dispatchCount = scenario == "early" ? 4U : scenario == "empty-rounds" ? 3U : 0U;
    Arguments args{};
    args.config.world = world; args.config.rank = world - 1U;
    args.config.hidden = 3584U; args.config.maxPeerChunks = kMaxPeerChunks;
    args.config.epoch = 7U; args.config.packRoundCount = dispatchCount;
    args.config.barrierOffset = 0U; args.config.dispatchReadyOffset = 8192U;
    args.config.combineReadyOffset = 2U << 20U; args.config.returnYOffset = 4U << 20U;
    args.config.receiveXOffset = 8U << 20U; args.config.receiveScaleOffset = 12U << 20U;
    std::vector<uint64_t> symmetric(4096U);
    std::array<int32_t, 5U * 32U> local{};
    std::vector<int32_t> ready(world * kMaxPeerChunks * 32U, 256);
    std::vector<int32_t> compute(std::max(combineCount, 1U) * 32U);
    std::vector<Fragment> fragments(combineCount);
    std::vector<uint32_t> indices(combineCount);
    std::vector<Chunk> sends, receives;
    std::vector<PackRound> packs(dispatchCount);
    Test::args = &args; Test::owner = owner; Test::peer = PeerForOwner(args.config.rank, owner);
    for (uint32_t cv = 0; cv < dispatchCount; ++cv) {
        Chunk chunk{Test::peer, cv, cv * 10U, cv * 10U, 3U, 0U, 0U, 0U};
        if (scenario == "empty-rounds") chunk.rows = 0U;
        else sends.push_back(chunk);
        packs[cv].chunks[Test::peer] = chunk;
    }
    for (uint32_t cv = 0; cv < combineCount; ++cv) {
        const uint32_t rows = cv % 3U + 1U;
        receives.push_back({Test::peer, cv, cv * 10U, cv * 10U, rows, cv, 1U, 0U});
        fragments[cv].rows = rows;
        indices[cv] = cv;
        if (scenario == "tail") compute[cv * 32U] = ComputeTileCount(rows, args.config.hidden);
    }
    args.symmetric = Test::Ptr(symmetric.data()); args.localFlags = Test::Ptr(local.data());
    args.packReady = Test::Ptr(ready.data()); args.gmm2Done = Test::Ptr(compute.data());
    args.fragments = Test::Ptr(fragments.data()); args.chunkGroupIndices = Test::Ptr(indices.data());
    args.packRounds = Test::Ptr(packs.data()); args.sendChunks = Test::Ptr(sends.data());
    args.recvChunks = Test::Ptr(receives.data()); args.config.sendChunkCount = sends.size();
    args.config.recvChunkCount = receives.size(); args.config.fragmentCount = fragments.size();
    *reinterpret_cast<uint64_t*>(args.symmetric + Test::peer * kFlagBytes) = args.config.epoch;
    local[kLocalFlagPackDone * 32U] = kAic - 1U;
    local[kLocalFlagDispatchDone * 32U] = world - (owner == 0U ? 2U : 1U);
    local[kLocalFlagAicDone * 32U] = kAic;
    local[kLocalFlagReorderDone * 32U] = kAic - 1U;
    Test::combineSends.resize(combineCount); Test::combinePublications.resize(combineCount);
    Test::quieted.resize(combineCount); Test::dispatchPublications.resize(sends.size());
    Test::deferReorderUntilFinished = scenario == "tail" || scenario == "empty" || scenario == "empty-rounds";
    Test::enableInitialReorder = !Test::deferReorderUntilFinished;
    Test::reorderRemaining = scenario == "no-owned-work" ? 0U : std::max(combineCount, 3U);
    const auto expectedReorders = Test::reorderRemaining;
    if (scenario == "no-owned-work") {
        for (uint32_t cv = 0; cv < combineCount; ++cv)
            compute[cv * 32U] = ComputeTileCount(fragments[cv].rows, args.config.hidden);
    }
    Timer timer;
    if (scenario == "wait-trace") {
        Test::readyThreshold = 40U;
        int32_t waited = 0;
        Test::unblockFlag = &waited;
        OwnedReorder reorder(args, owner, timer);
        WaitWithReorder(&waited, int32_t(7), reorder, timer, TraceEvent::WaitDispatch, {});
        assert(Test::readyChecks >= 40U && Test::cycleCalls != 0U);
        // Forty unsuccessful polls produce one pre-work wait, one work row,
        // and one trailing wait: never one wait event per poll.
        CheckWaitAndReorder(TraceEvent::WaitDispatch, 2U);
        assert(Test::traces.size() == 3U);
        assert(Test::traces[0].event == TraceEvent::WaitDispatch);
        assert(Test::traces[1].event == TraceEvent::Gmm2Reorder);
        assert(Test::traces[2].event == TraceEvent::WaitDispatch);
        return 0;
    }
    if (scenario == "waiting") Test::readyThreshold = 2U;
    CommunicatePeer(args, owner, timer);
    assert(Test::reorderCompleted == expectedReorders && Test::reorderRemaining == 0U);
    assert(Test::reorderDoneMarks == 1U && local[kLocalFlagReorderDone * 32U] == int32_t(kAic));
    assert(local[kLocalFlagIncomingDone * 32U] == (owner == 0U ? 2 : 1));
    assert(Test::finishedPublications == 1U && Test::combinePending.empty());
    assert(Test::packRounds.size() == dispatchCount);
    for (uint32_t round = 0; round < dispatchCount; ++round) assert(Test::packRounds[round] == round + kPackStartupRounds);
    for (auto count : Test::combineSends) assert(count == 1U);
    for (auto count : Test::combinePublications) assert(count == 1U);
    for (auto count : Test::dispatchPublications) assert(count == 1U);
    if (combineCount) {
        assert(!Test::combineBatches.empty() && Test::combineBatches.back().size() == 1U);
        assert(Test::combineBatches.back().front() == combineCount - 1U);
    }
    if (scenario == "early" && combineCount >= 5U) {
        assert(Test::combineBatches.size() == 3U);
        assert((Test::combineBatches.front() == std::vector<uint32_t>{0U, 1U}));
    }
    if (scenario == "unready") assert(Test::drainCalls != 0U && Test::reorderCompleted != 0U);
    if (scenario == "waiting") {
        assert(Test::cycleCalls != 0U);
        CheckWaitAndReorder(TraceEvent::WaitCombineReady, 1U);
    }
    // No re-counting or extra publication on repeated observation.
    bool peerCounted = true, selfCounted = owner == 0U;
    assert(ObserveIncomingCompletion(args, owner, peerCounted, selfCounted));
    assert(local[kLocalFlagIncomingDone * 32U] == (owner == 0U ? 2 : 1));
}
"""


@pytest.fixture(scope="module")
def reorder_comm_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for production communication checks")
    root = Path(__file__).resolve().parents[3]
    headers = root / "experiments/megamoe/csrc/kernels/mega_moe/urma"
    kernel = (headers / "kernel.hpp").read_text(encoding="utf-8")
    # Compile the real peer loop and all its production local helpers. Only
    # Process() (hardware launch routing) is outside this host harness.
    kernel = kernel[kernel.index("namespace MegaMoeUrma {") : kernel.index("__aicore__ inline void Process")]
    tail = (headers / "tail_completion.hpp").read_text(encoding="utf-8")
    tail = tail[tail.index("namespace MegaMoeUrma {") :]
    temporary = tmp_path_factory.mktemp("urma_reorder_comm")
    source = temporary / "comm.cpp"
    source.write_text(FAKE_BACKEND + "\n" + tail + "\n" + kernel + "\n}\n" + DRIVER, encoding="utf-8")
    binary = temporary / ("comm.exe" if os.name == "nt" else "comm")
    compiled = subprocess.run(
        [*command, "-std=c++17", "-O1", "-UNDEBUG", "-I" + str(headers), str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return binary


@pytest.mark.parametrize("world", [4, 8, 16])
@pytest.mark.parametrize("last_owner", [False, True])
@pytest.mark.parametrize(
    ("scenario", "combine_count"),
    [("early", 5), ("unready", 5), ("tail", 1), ("empty", 0), ("empty-rounds", 0), ("no-owned-work", 5)],
)
def test_actual_peer_loop_progress_publications_and_ub_isolation(
    reorder_comm_binary, world, last_owner, scenario, combine_count
):
    result = subprocess.run(
        [str(reorder_comm_binary), str(world), str(world - 2 if last_owner else 0), scenario, str(combine_count)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_peer_loop_full_capacity_pending_metadata_survives_reorder(reorder_comm_binary):
    result = subprocess.run(
        [str(reorder_comm_binary), "16", "0", "early", "512"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(("scenario", "combine_count"), [("wait-trace", 0), ("waiting", 5)])
def test_actual_wait_scopes_exclude_reorder_and_coalesce_idle_polls(reorder_comm_binary, scenario, combine_count):
    result = subprocess.run(
        [str(reorder_comm_binary), "16", "0", scenario, str(combine_count)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
