"""Run the actual peer loop against delayed producers and an ordered-QP model.

The production scheduler and tail protocol are compiled unchanged. External
compute, DMA, and network completion are simulated; this is no NPU timing test.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


BACKEND = r"""
#include "abi.hpp"
#include "compute_contract.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#define __aicore__
#define __gm__
namespace Test {
using namespace MegaMoeUrma;
struct Operation {
    uint32_t kind, cv;
    const uint8_t* source;
    std::vector<uint8_t> data;
};
struct State {
    const Arguments* args = nullptr;
    uint32_t owner = 0, peer = 0, ticks = 0, quiets = 0, markers = 0, quietTick = 0;
    uint32_t dispatchQuiets = 0, dispatchMarkers = 0;
    uint32_t serial = 0, finishSerial = 0, quietSerial = 0, markerSerial = 0, selfSerial = 0;
    uint32_t incomingSerial = 0, dispatchCount = 0, remoteCount = 0, selfCount = 0, producer = 0;
    bool lateSelf = false;
    bool partialSeen = false;
    std::array<uint8_t, kRequiredUbBytes> ub{};
    std::vector<uint32_t> dispatchPuts, dispatchFlags, combinePuts, combineFlags;
    std::vector<uint32_t> copyCounts, selfFlags, sendOrder, packRounds, production;
    std::vector<Operation> queue;
};
inline State state;
inline uint64_t Ptr(const void* ptr) { return reinterpret_cast<uint64_t>(ptr); }
inline int32_t* Flag(uint64_t address, uint32_t index) {
    return reinterpret_cast<int32_t*>(address + uint64_t(index) * kFlagBytes);
}
inline uint64_t* Incoming() {
    const auto& a = *state.args;
    return reinterpret_cast<uint64_t*>(a.symmetric + a.config.barrierOffset +
                                      (a.config.world + state.peer) * kFlagBytes);
}
inline void Advance() {
    auto& s = state;
    if (++s.ticks > 100000U) throw std::runtime_error("communication loop made no progress");
    if ((!s.lateSelf || s.finishSerial != 0U) && s.ticks % 4U == 0U && s.producer < s.production.size()) {
        const auto index = s.production[s.producer++];
        *Flag(s.args->gmm2Done, index) = 14;
    }
    // Remote acknowledgement deliberately follows the local final quiet.
    // This makes send-complete marker placement observable independently of
    // incoming completion, including delayed self output and empty peers.
    if (s.quiets && s.ticks > s.quietTick + 8U && *Incoming() == 0U) {
        assert(s.remoteCount == 0U || s.markerSerial > s.quietSerial);
        s.incomingSerial = ++s.serial;
        *Incoming() = s.args->config.epoch;
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
        assert(uint64_t(offset) + uint64_t(length) * sizeof(T) <= Test::state.ub.size());
    }
    void SetValue(uint32_t index, T value) {
        assert(index < count);
        std::memcpy(Test::state.ub.data() + address + index * sizeof(T), &value, sizeof(T));
    }
    T GetValue(uint32_t index) const {
        assert(index < count); T value;
        std::memcpy(&value, Test::state.ub.data() + address + index * sizeof(T), sizeof(T)); return value;
    }
};
template <typename T> T ReadGmByPassDCache(T* pointer) { return *pointer; }
inline void WriteGmByPassDCache(uint64_t* pointer, uint64_t value) {
    using MegaMoeUrma::kFlagBytes;
    auto& s = Test::state; const auto& c = s.args->config;
    assert(value == c.epoch);
    const uint64_t first = s.args->symmetric + c.combineReadyOffset + c.rank * c.maxPeerChunks * kFlagBytes;
    assert(Test::Ptr(pointer) >= first && (Test::Ptr(pointer) - first) % kFlagBytes == 0U);
    const uint32_t cv = (Test::Ptr(pointer) - first) / kFlagBytes;
    assert(cv < s.selfCount && ++s.selfFlags[cv] == 1U);
    const auto* chunks = reinterpret_cast<const MegaMoeUrma::Chunk*>(s.args->recvChunks);
    bool found = false;
    for (uint32_t i = 0; i < c.recvChunkCount; ++i)
        if (chunks[i].peer == c.rank && chunks[i].cv == cv) {
            const auto range = reinterpret_cast<const MegaMoeUrma::PeerReorderRange*>(s.args->peerReorderRanges)[i];
            assert(s.copyCounts[i] == range.spanCount); found = true;
        }
    assert(found); *pointer = value;
}
inline void AtomicAdd(int32_t* pointer, int32_t value) { *pointer += value; }
inline uint64_t GetSystemCycle() { Test::Advance(); return uint64_t(Test::state.ticks) * 200U; }
template <bool> void SyncAll() {}
}
namespace MegaMoeUrma {
using namespace AscendC;
constexpr uint32_t kPackStartupRounds = 2U;
constexpr uint32_t kPackUbBase = 128U * 1024U;
constexpr uint32_t kPackUbBytes = 58880U;
constexpr uint32_t kSignalCopyBytes = 32U;
enum class Phase { Barrier, Dispatch, Combine };
struct Timer {
    void Begin(Phase) {}
    void End(Phase, uint64_t = 0U, bool = false) {}
    uint64_t TraceBegin() { return ++Test::state.serial; }
    void TraceEnd(TraceEvent, uint64_t, const TraceMeta&) { ++Test::state.serial; }
    void MarkSendComplete(Phase phase) {
        auto& s = Test::state;
        if (phase == Phase::Dispatch) {
            assert(++s.dispatchMarkers == s.dispatchQuiets && !s.finishSerial); return;
        }
        assert(phase == Phase::Combine && s.quiets == 1U && s.finishSerial < s.quietSerial);
        assert(*Test::Incoming() == 0U && ++s.markers == 1U);
        s.markerSerial = ++s.serial;
    }
};
inline uint32_t PeerForOwner(uint32_t rank, uint32_t owner) { return owner < rank ? owner : owner + 1U; }
inline int32_t* ComputeFlag(uint64_t address, uint32_t index) { return Test::Flag(address, index); }
inline int32_t* PackReadyFlag(const Arguments& a, uint32_t peer, uint32_t cv) {
    return ComputeFlag(a.packReady, peer * a.config.maxPeerChunks + cv);
}
inline Chunk LoadPackChunk(const Arguments& a, uint32_t round, uint32_t peer) {
    return reinterpret_cast<const PackRound*>(a.packRounds)[round].chunks[peer];
}
inline void PackOneRound(const Arguments&, uint32_t round, uint32_t owner, uint32_t workers, Timer&) {
    assert(owner == Test::state.owner && workers == kAic);
    Test::state.packRounds.push_back(round);
    std::memset(Test::state.ub.data() + kPackUbBase, 0xB7, kPackUbBytes);
}
inline void MarkPackWorkerComplete(const Arguments& a) { ++*ComputeFlag(a.localFlags, kLocalFlagPackDone); }
inline void PublishDone(uint8_t*, uint64_t, uint32_t, uint64_t, LocalTensor<uint64_t>, uint32_t) {
    throw std::runtime_error("payload readiness must use SO, not MTE publication");
}
inline const Chunk& RemoteChunk(uint32_t cv) {
    const auto& s = Test::state;
    const auto* chunks = reinterpret_cast<const Chunk*>(s.args->recvChunks);
    for (uint32_t i = 0; i < s.args->config.recvChunkCount; ++i)
        if (chunks[i].peer == s.peer && chunks[i].cv == cv) return chunks[i];
    throw std::runtime_error("unexpected remote CV");
}
inline void VerifyChunk(const Chunk& chunk, const uint8_t* source) {
    const auto& s = Test::state; const auto& a = *s.args;
    const auto* chunks = reinterpret_cast<const Chunk*>(a.recvChunks);
    const uint32_t index = &chunk - chunks;
    const auto range = reinterpret_cast<const PeerReorderRange*>(a.peerReorderRanges)[index];
    const auto* spans = reinterpret_cast<const PeerReorderSpan*>(a.peerReorderSpans);
    uint32_t row = 0U;
    for (uint32_t i = 0; i < range.spanCount; ++i) {
        const auto& span = spans[range.spanBegin + i];
        const uint64_t bytes = uint64_t(span.rows) * a.config.hidden * 2U;
        const auto* expected = reinterpret_cast<const uint8_t*>(a.gmm2Contiguous + span.sourceRow * a.config.hidden * 2U);
        assert(std::memcmp(source + uint64_t(row) * a.config.hidden * 2U, expected, bytes) == 0);
        row += span.rows;
    }
    assert(row == chunk.rows);
}
// Model only DMA here; execute the real readiness/bitmap scanner below.
struct PeerReorderPipeline {
    const Arguments* args;
    Chunk chunk;
    uint32_t index;
    void Init(const Arguments& a, const Chunk& c) {
        auto& s = Test::state; args = &a; chunk = c;
        assert(*ComputeFlag(a.localFlags, kLocalFlagDispatchDone) == int32_t(a.config.world));
        assert(c.peer == s.peer || (c.peer == a.config.rank && c.cv % ActivationWorkers(a.config.world) == s.owner));
        const auto* chunks = reinterpret_cast<const Chunk*>(a.recvChunks);
        index = 0U;
        while (index < a.config.recvChunkCount && (chunks[index].peer != c.peer || chunks[index].cv != c.cv)) ++index;
        assert(index < a.config.recvChunkCount);
    }
    void Copy(const PeerReorderSpan& span) {
        const auto& a = *args; auto& s = Test::state;
        assert(*ComputeFlag(a.gmm2Done, span.fragmentIndex) == 14 && "whole M-group N direction must be ready");
        const auto range = reinterpret_cast<const PeerReorderRange*>(a.peerReorderRanges)[index];
        const auto* spans = reinterpret_cast<const PeerReorderSpan*>(a.peerReorderSpans);
        for (uint32_t j = 0U; j < range.spanCount; ++j)
            if (*ComputeFlag(a.gmm2Done, spans[range.spanBegin + j].fragmentIndex) < 14) s.partialSeen = true;
        std::memset(s.ub.data(), 0xD3, kPackUbBase);
        const uint64_t destination = chunk.peer == a.config.rank ? a.symmetric + a.config.returnYOffset : a.gmm2Output;
        std::memcpy(reinterpret_cast<void*>(destination + span.destinationRow * a.config.hidden * 2U),
                    reinterpret_cast<const void*>(a.gmm2Contiguous + span.sourceRow * a.config.hidden * 2U),
                    uint64_t(span.rows) * a.config.hidden * 2U);
        ++s.copyCounts[index];
        if (chunk.peer == a.config.rank && !s.selfSerial) s.selfSerial = ++s.serial;
    }
    void Finish() {}
};
inline void Put(uint8_t*, uint64_t offset, uint8_t* source, uint32_t bytes, uint32_t peer) {
    auto& s = Test::state; const auto& a = *s.args; const auto& c = a.config;
    assert(peer == s.peer && bytes && !s.finishSerial && s.quiets == 0U);
    uint32_t kind = 0U, cv = UINT32_MAX;
    if (offset >= c.receiveXOffset) {
        kind = 0U;
        const bool scale = offset >= c.receiveScaleOffset;
        const uint64_t width = scale ? c.hidden / 32U : c.hidden;
        const uint64_t base = scale ? c.receiveScaleOffset : c.receiveXOffset;
        cv = (offset - base) / (8U * width);
        assert(cv < s.dispatchCount && bytes == (cv + 1U) * width);
        const auto expected = (scale ? a.expandedScale : a.expandedX) + cv * 8U * width;
        assert(Test::Ptr(source) == expected && ++s.dispatchPuts[cv] <= 2U);
    } else {
        kind = 1U;
        assert(*ComputeFlag(a.localFlags, kLocalFlagDispatchDone) == int32_t(c.world));
        cv = (offset - c.returnYOffset) / (8U * c.hidden * 2U);
        const auto& chunk = RemoteChunk(cv);
        assert(offset == c.returnYOffset + chunk.remoteRow * c.hidden * 2U);
        assert(bytes == chunk.rows * c.hidden * 2U);
        assert(Test::Ptr(source) == a.gmm2Output + chunk.localRow * c.hidden * 2U);
        assert(++s.combinePuts.at(cv) == 1U);
        VerifyChunk(chunk, source); s.sendOrder.push_back(cv);
    }
    s.queue.push_back({kind, cv, source, std::vector<uint8_t>(source, source + bytes)});
}
inline void PublishOrdered(uint8_t*, uint64_t offset, uint32_t peer, uint64_t epoch) {
    auto& s = Test::state; const auto& c = s.args->config;
    assert(peer == s.peer && epoch == c.epoch && s.quiets == 0U && !s.finishSerial);
    const uint64_t finish = c.barrierOffset + (c.world + c.rank) * kFlagBytes;
    if (offset == finish) {
        assert(std::all_of(s.combineFlags.begin(), s.combineFlags.end(), [](auto n) { return n == 1U; }));
        assert(std::all_of(s.dispatchFlags.begin(), s.dispatchFlags.end(), [](auto n) { return n == 1U; }));
        s.finishSerial = ++s.serial; s.queue.push_back({4U, 0U, nullptr, {}}); return;
    }
    const bool combine = offset >= c.combineReadyOffset;
    const uint64_t base = combine ? c.combineReadyOffset : c.dispatchReadyOffset;
    const uint64_t ready = (offset - base) / kFlagBytes;
    assert(ready / c.maxPeerChunks == c.rank && (offset - base) % kFlagBytes == 0U);
    const uint32_t cv = ready % c.maxPeerChunks;
    if (combine) {
        assert(s.combinePuts.at(cv) == 1U && ++s.combineFlags.at(cv) == 1U);
    } else {
        assert(s.dispatchPuts.at(cv) == 2U && ++s.dispatchFlags.at(cv) == 1U);
    }
    s.queue.push_back({combine ? 3U : 2U, cv, nullptr, {}});
}
inline void Quiet(uint32_t peer) {
    auto& s = Test::state;
    assert(peer == s.peer && !s.queue.empty());
    const bool final = s.finishSerial != 0U;
    if (final) {
        assert(++s.quiets == 1U && s.queue.back().kind == 4U);
        // Self copies belong to the activation workers now, so this owner must
        // retire its QP without waiting for them.
    } else {
        assert(s.queue.size() == 3U && s.queue.back().kind == 2U);
        assert(s.queue.back().cv == s.dispatchQuiets++);
    }
    std::vector<uint32_t> dispatchDelivered(s.dispatchCount), combineDelivered(s.remoteCount);
    for (const auto& op : s.queue) {
        if (op.kind <= 1U) {
            // Consume at the relevant quiet, catching source overwrites.
            assert(std::memcmp(op.source, op.data.data(), op.data.size()) == 0);
            ++(op.kind ? combineDelivered : dispatchDelivered)[op.cv];
        } else if (op.kind == 2U) assert(dispatchDelivered[op.cv] == 2U);
        else if (op.kind == 3U) assert(combineDelivered[op.cv] == 1U);
    }
    s.queue.clear();
    if (final) { s.quietSerial = ++s.serial; s.quietTick = s.ticks; }
}
} // namespace MegaMoeUrma
"""


DRIVER = r"""
using namespace MegaMoeUrma;
void Run(uint32_t world, uint32_t owner, const std::string& scenario, bool timerEnabled) {
    Test::state = Test::State{};
    auto& s = Test::state;
    Arguments a{}; auto& c = a.config;
    c.world = world; c.rank = world / 2U; c.hidden = 3584U; c.maxPeerChunks = kMaxPeerChunks;
    c.epoch = 7U; c.packRoundCount = 4U;
    c.barrierOffset = 0U; c.dispatchReadyOffset = 8192U; c.combineReadyOffset = 2U << 20U;
    c.returnYOffset = 4U << 20U; c.receiveXOffset = 64U << 20U; c.receiveScaleOffset = 80U << 20U;
    s.args = &a; s.owner = owner; s.peer = PeerForOwner(c.rank, owner);
    s.remoteCount = scenario == "empty" || scenario == "self-only" ? 0U : scenario == "capacity" ? 512U : 5U;
    s.selfCount = scenario == "empty" ? 0U : world + 1U;
    s.dispatchCount = scenario == "empty" || scenario == "self-only" ? 0U : 2U;
    s.lateSelf = scenario == "late-self" || scenario == "self-only";
    s.dispatchPuts.resize(s.dispatchCount); s.dispatchFlags.resize(s.dispatchCount);
    s.combinePuts.resize(s.remoteCount); s.combineFlags.resize(s.remoteCount); s.selfFlags.resize(s.selfCount);
    std::vector<uint64_t> symmetric((c.returnYOffset + uint64_t(s.selfCount + 1U) * 8U * c.hidden * 2U) / 8U);
    std::array<int32_t, 16U * 32U> local{};
    std::vector<int32_t> packReady(world * kMaxPeerChunks * 32U, 256);
    std::vector<Fragment> fragments;
    std::vector<Chunk> chunks, sends;
    std::vector<PeerReorderRange> ranges;
    std::vector<PeerReorderSpan> spans;
    std::vector<uint32_t> indices;
    std::vector<PackRound> packs(c.packRoundCount);
    uint64_t receiveRows = 0U, sourceRows = 0U;
    // Include fully-ready unrelated peers: the real loop must never consume
    // another communication owner's rows even if they are easiest to copy.
    for (uint32_t peer = 0; peer < world; ++peer) {
        const uint32_t count = peer == c.rank ? s.selfCount : peer == s.peer ? s.remoteCount : 2U;
        for (uint32_t cv = 0; cv < count; ++cv) {
            const uint32_t rows = cv % 3U + 2U;
            chunks.push_back({peer, cv, receiveRows, cv * 8U, rows, uint32_t(indices.size()), 2U, 0U});
            ranges.push_back({uint32_t(spans.size()), 2U});
            for (uint32_t part = 0; part < 2U; ++part) {
                const uint32_t length = part ? rows - 1U : 1U;
                const uint32_t group = fragments.size();
                fragments.push_back({0U, 0U, sourceRows, length, 0U, group * 5U, group * 14U});
                indices.push_back(group);
                spans.push_back({sourceRows, (peer == c.rank ? cv * 8U : receiveRows) + part, group, length});
                sourceRows += length;
            }
            receiveRows += rows;
        }
    }
    s.copyCounts.resize(chunks.size());
    std::vector<uint16_t> scratch(sourceRows * c.hidden), output(receiveRows * c.hidden, 0U);
    std::vector<uint8_t> x(32U * c.hidden, 0x4B), scale(32U * (c.hidden / 32U), 0xC1);
    for (uint64_t i = 0; i < scratch.size(); ++i) scratch[i] = (i * 43U + 13U) % 65534U + 1U;
    std::vector<int32_t> compute(fragments.size() * 32U, 0U);
    for (uint32_t group = 0; group < fragments.size(); ++group) compute[group * 32U] = 14;
    for (uint32_t round = 0; round < c.packRoundCount; ++round) {
        packs[round].chunks[s.peer] = {s.peer, round, round * 8U, round * 8U,
                                     round < s.dispatchCount ? round + 1U : 0U, 0U, 0U, 0U};
        if (round < s.dispatchCount) sends.push_back(packs[round].chunks[s.peer]);
    }
    const bool reverse = scenario == "reverse" || scenario == "capacity";
    // Two independent producer groups per CV, each missing its final N tile.
    // Releasing one group allows partial reorder, but not the CV's PUT.
    if (reverse) {
        for (auto it = chunks.rbegin(); it != chunks.rend(); ++it) {
            if (it->peer != s.peer) continue;
            for (uint32_t j = it->fragmentCount; j-- > 0U;) {
                const auto group = indices[it->fragmentBegin + j];
                compute[group * 32U] = 13; s.production.push_back(group);
            }
        }
    }
    if (s.lateSelf) {
        for (const auto& chunk : chunks) {
            if (chunk.peer != c.rank || chunk.cv % (world - 1U) != owner) continue;
            for (uint32_t j = 0; j < chunk.fragmentCount; ++j) {
                const auto group = indices[chunk.fragmentBegin + j];
                compute[group * 32U] = 13; s.production.push_back(group);
            }
        }
    }
    a.symmetric = Test::Ptr(symmetric.data()); a.localFlags = Test::Ptr(local.data());
    a.packReady = Test::Ptr(packReady.data()); a.gmm2Done = Test::Ptr(compute.data());
    a.gmm2Contiguous = Test::Ptr(scratch.data()); a.gmm2Output = Test::Ptr(output.data());
    a.expandedX = Test::Ptr(x.data()); a.expandedScale = Test::Ptr(scale.data());
    a.fragments = Test::Ptr(fragments.data()); a.chunkGroupIndices = Test::Ptr(indices.data());
    a.recvChunks = Test::Ptr(chunks.data()); a.sendChunks = Test::Ptr(sends.data());
    a.peerReorderRanges = Test::Ptr(ranges.data()); a.peerReorderSpans = Test::Ptr(spans.data());
    a.packRounds = Test::Ptr(packs.data()); a.timer = timerEnabled ? 1U : 0U;
    c.recvChunkCount = chunks.size(); c.sendChunkCount = sends.size(); c.fragmentCount = fragments.size();
    *reinterpret_cast<uint64_t*>(a.symmetric + s.peer * kFlagBytes) = c.epoch;
    local[kLocalFlagPackDone * 32U] = kAic - 1U;
    local[kLocalFlagDispatchDone * 32U] = world - (owner == 0U ? 2U : 1U);
    local[kLocalFlagAicDone * 32U] = kAic;
    // The activation workers own self copies, so their aggregate is complete
    // before this owner reaches the tail; the peer loop must not touch it.
    local[kLocalFlagReorderDone * 32U] = int32_t(ActivationWorkers(world));
    local[kLocalFlagIncomingDone * 32U] = world - (owner == 0U ? 2U : 1U);
    Timer timer;
    CommunicatePeer(a, owner, timer);
    assert(s.quiets == 1U && s.finishSerial && s.finishSerial < s.quietSerial);
    assert(s.dispatchQuiets == s.dispatchCount && s.dispatchMarkers == s.dispatchCount);
    if (reverse) assert(s.partialSeen && "ready later spans must not wait for the first span");
    assert(s.markers == (s.remoteCount != 0U) && s.incomingSerial > s.quietSerial);
    if (s.remoteCount) assert(s.quietSerial < s.markerSerial && s.markerSerial < s.incomingSerial);
    // Self copies are the activation workers' duty: this loop must leave them
    // untouched and must not report the aggregate reorder completion.
    assert(s.selfSerial == 0U);
    assert(local[kLocalFlagReorderDone * 32U] == int32_t(ActivationWorkers(world)));
    assert(local[kLocalFlagIncomingDone * 32U] == int32_t(world));
    assert(s.packRounds.size() == c.packRoundCount);
    for (uint32_t j = 0; j < c.packRoundCount; ++j) assert(s.packRounds[j] == j + kPackStartupRounds);
    for (uint32_t cv = 0; cv < s.remoteCount; ++cv) {
        assert(s.combinePuts[cv] == 1U && s.combineFlags[cv] == 1U);
        if (reverse) assert(s.sendOrder[cv] == s.remoteCount - 1U - cv);
    }
    for (uint32_t cv = 0; cv < s.selfCount; ++cv) assert(s.selfFlags[cv] == 0U);
    for (uint32_t i = 0; i < chunks.size(); ++i) {
        const auto& chunk = chunks[i];
        const bool own = chunk.peer == s.peer;
        assert(s.copyCounts[i] == (own ? ranges[i].spanCount : 0U));
        if (chunk.peer == c.rank && own)
            VerifyChunk(chunk, reinterpret_cast<const uint8_t*>(a.symmetric + c.returnYOffset +
                                                               chunk.remoteRow * c.hidden * 2U));
    }
}

// The activation workers own the self copies now. Each one spreads the same CV
// slice statically, publishes the local per-CV ready flag and reports one
// aggregate reorder completion.
void RunSelf(uint32_t world, uint32_t worker, bool timerEnabled, std::vector<uint32_t>& selfCoverage) {
    Test::state = Test::State{};
    auto& s = Test::state;
    Arguments a{}; auto& c = a.config;
    c.world = world; c.rank = world / 2U; c.hidden = 3584U; c.maxPeerChunks = kMaxPeerChunks;
    c.epoch = 7U; c.barrierOffset = 0U; c.combineReadyOffset = 2U << 20U; c.returnYOffset = 4U << 20U;
    const uint32_t selfCount = 6U;
    s.selfCount = selfCount;
    s.owner = worker;
    s.args = &a;
    std::vector<Chunk> chunks;
    std::vector<PeerReorderRange> ranges;
    std::vector<PeerReorderSpan> spans;
    std::vector<uint32_t> indices;
    std::vector<Fragment> fragments;
    uint32_t receiveRows = 0U;
    for (uint32_t cv = 0; cv < selfCount; ++cv) {
        const uint32_t rows = cv % 2U + 1U;
        chunks.push_back({c.rank, cv, receiveRows, cv * 8U, rows, uint32_t(spans.size()), rows, 0U});
        ranges.push_back({uint32_t(spans.size()), rows});
        for (uint32_t part = 0; part < rows; ++part) {
            const uint32_t group = uint32_t(indices.size());
            fragments.push_back({0U, 0U, receiveRows + part, 1U, 0U, group * 5U, group * 14U});
            indices.push_back(group);
            spans.push_back({receiveRows + part, cv * 8U + part, group, 1U});
        }
        receiveRows += rows;
    }
    s.copyCounts.resize(chunks.size());
    s.selfFlags.resize(selfCount);
    std::vector<uint16_t> scratch(uint64_t(receiveRows) * c.hidden);
    for (uint64_t i = 0; i < scratch.size(); ++i) scratch[i] = uint16_t((i * 43U + 13U) % 65534U + 1U);
    std::vector<int32_t> compute(indices.size() * 32U, 0U);
    for (uint32_t group = 0; group < indices.size(); ++group) compute[group * 32U] = 14;
    std::vector<uint64_t> symmetric((c.returnYOffset + uint64_t(selfCount + 1U) * 8U * c.hidden * 2U) / 8U);
    std::array<int32_t, 16U * 32U> local{};
    local[kLocalFlagDispatchDone * 32U] = int32_t(world);
    a.symmetric = Test::Ptr(symmetric.data()); a.localFlags = Test::Ptr(local.data());
    a.gmm2Done = Test::Ptr(compute.data()); a.gmm2Contiguous = Test::Ptr(scratch.data());
    a.recvChunks = Test::Ptr(chunks.data()); a.peerReorderRanges = Test::Ptr(ranges.data());
    a.peerReorderSpans = Test::Ptr(spans.data()); a.timer = timerEnabled ? 1U : 0U;
    c.recvChunkCount = chunks.size(); c.fragmentCount = fragments.size();
    Timer timer;
    ReorderSelfSpans(a, worker, ActivationWorkers(world), timer);
    assert(local[kLocalFlagReorderDone * 32U] == 1);
    for (uint32_t cv = 0; cv < selfCount; ++cv) {
        const bool own = cv % ActivationWorkers(world) == worker;
        assert(s.selfFlags[cv] == uint32_t(own));
        selfCoverage[cv] += s.selfFlags[cv];
        for (uint32_t i = 0; i < chunks.size(); ++i) {
            if (chunks[i].cv != cv) continue;
            assert(s.copyCounts[i] == (own ? ranges[i].spanCount : 0U));
            if (own) VerifyChunk(chunks[i], reinterpret_cast<const uint8_t*>(a.symmetric + c.returnYOffset +
                                                                           chunks[i].remoteRow * c.hidden * 2U));
        }
    }
}
int main(int argc, char** argv) {
    assert(argc == 4);
    const uint32_t world = std::stoul(argv[1]);
    const std::string scenario = argv[2];
    const bool timer = std::stoul(argv[3]) != 0U;
    for (uint32_t owner = 0; owner < world - 1U; ++owner) Run(world, owner, scenario, timer);
    std::vector<uint32_t> selfCoverage(6U, 0U);
    for (uint32_t worker = 0; worker < ActivationWorkers(world); ++worker)
        RunSelf(world, worker, timer, selfCoverage);
    for (auto count : selfCoverage) assert(count == 1U);
}
"""


@pytest.fixture(scope="module")
def so_peer_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for the communication-loop test")
    headers = Path(__file__).resolve().parents[1] / "csrc/kernels/mega_moe/urma"
    tail = (headers / "tail_completion.hpp").read_text(encoding="utf-8")
    tail = tail[tail.index("namespace MegaMoeUrma {") :]
    progress = (headers / "peer_reorder_progress.hpp").read_text(encoding="utf-8")
    progress = progress[progress.index("namespace MegaMoeUrma {") :]
    kernel = (headers / "kernel.hpp").read_text(encoding="utf-8")
    kernel = kernel[kernel.index("namespace MegaMoeUrma {") : kernel.index("__aicore__ inline void Process(")]
    temporary = tmp_path_factory.mktemp("urma_so_comm")
    source = temporary / "so_comm.cpp"
    source.write_text(BACKEND + "\n" + progress + "\n" + tail + "\n" + kernel + "\n}\n" + DRIVER, encoding="utf-8")
    binary = temporary / ("so_comm.exe" if os.name == "nt" else "so_comm")
    result = subprocess.run(
        [*command, "-std=c++17", "-O1", "-UNDEBUG", "-I" + str(headers), str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("world", [4, 8, 16])
@pytest.mark.parametrize("scenario", ["ready", "reverse", "late-self", "empty", "self-only"])
@pytest.mark.parametrize("timer_enabled", [False, True])
def test_peer_loop_orders_full_rows_so_and_one_final_quiet(so_peer_binary, world, scenario, timer_enabled):
    result = subprocess.run(
        [str(so_peer_binary), str(world), scenario, str(int(timer_enabled))],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_full_capacity_metadata_survives_every_reorder_dma(so_peer_binary):
    result = subprocess.run([str(so_peer_binary), "16", "capacity", "0"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
