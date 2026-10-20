// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "compute_contract.hpp"
#include "kernel_operator.h"

namespace MegaMoeUrma {

// A peer owner retains these buffers while submitting PUTs. Offset zero is
// reserved for transport.hpp's independent 32-byte remote publication buffer.
struct ReadyMonitorUb {
    __aicore__ static constexpr uint32_t Align(uint32_t bytes)
    {
        return (bytes + 31U) / 32U * 32U;
    }
    static constexpr uint32_t dependenciesOffset = 256U;
    static constexpr uint32_t dependenciesBytes = (kMaxReadyOwnerEdges * sizeof(ReadyDependency) + 31U) / 32U * 32U;
    static constexpr uint32_t edgesOffset = dependenciesOffset + dependenciesBytes;
    static constexpr uint32_t edgesBytes = (kMaxReadyOwnerEdges * sizeof(ReadyEdge) + 31U) / 32U * 32U;
    static constexpr uint32_t countsOffset = edgesOffset + edgesBytes;
    static constexpr uint32_t countsBytes = (kMaxReadyOwnerTokens * sizeof(uint32_t) + 31U) / 32U * 32U;
    static constexpr uint32_t pendingOffset = countsOffset + countsBytes;
    static constexpr uint32_t signalOffset = pendingOffset + countsBytes;
    static constexpr uint32_t end = signalOffset + 32U;
};
static_assert(ReadyMonitorUb::end <= kRequiredUbBytes);

// Exactly one communication owner maintains a token. A completed dependency
// contributes all its TopK multiplicities once; only the final epoch is exposed
// to reducers. The private CSR/counts have no other Scalar/MTE/Vector writers.
class ReadyMonitor {
public:
    __aicore__ inline void Init(const Arguments& args, uint32_t owner)
    {
        using namespace AscendC;
        const auto* descriptor = reinterpret_cast<__gm__ ReadyOwner*>(args.reduceReadyOwners) + owner;
        tokenBegin_ = descriptor->tokenBegin;
        tokenCount_ = descriptor->tokenCount;
        dependencyCount_ = descriptor->dependencyCount;
        const uint32_t dependencyBegin = descriptor->dependencyBegin;
        const uint32_t edgeBegin = descriptor->edgeBegin;
        const uint32_t edgeCount = descriptor->edgeCount;
        dependencies_ = LocalTensor<uint64_t>(TPosition::VECCALC, ReadyMonitorUb::dependenciesOffset,
                                              ReadyMonitorUb::dependenciesBytes / sizeof(uint64_t));
        edges_ = LocalTensor<uint32_t>(TPosition::VECCALC, ReadyMonitorUb::edgesOffset,
                                       ReadyMonitorUb::edgesBytes / sizeof(uint32_t));
        counts_ = LocalTensor<uint32_t>(TPosition::VECCALC, ReadyMonitorUb::countsOffset,
                                        ReadyMonitorUb::countsBytes / sizeof(uint32_t));
        pending_ = LocalTensor<uint32_t>(TPosition::VECCALC, ReadyMonitorUb::pendingOffset,
                                         ReadyMonitorUb::countsBytes / sizeof(uint32_t));
        signal_ = LocalTensor<uint64_t>(TPosition::VECCALC, ReadyMonitorUb::signalOffset, 4U);
        GlobalTensor<uint64_t> dependencies;
        GlobalTensor<uint32_t> edges;
        dependencies.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(args.reduceReadyDependencies) +
                                     static_cast<uint64_t>(dependencyBegin) * 2U);
        edges.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(args.reduceReadyEdges) + edgeBegin);
        if (dependencyCount_ != 0U) {
            DataCopy(dependencies_, dependencies,
                     ReadyMonitorUb::Align(dependencyCount_ * sizeof(ReadyDependency)) / sizeof(uint64_t));
            DataCopy(edges_, edges, ReadyMonitorUb::Align(edgeCount * sizeof(ReadyEdge)) / sizeof(uint32_t));
            SetFlag<HardEvent::MTE2_S>(kEvent);
            WaitFlag<HardEvent::MTE2_S>(kEvent);
        }
        for (uint32_t i = 0U; i < tokenCount_; ++i) counts_.SetValue(i, 0U);
        signal_.SetValue(0U, args.config.epoch);
        signal_.SetValue(1U, 0U);
        signal_.SetValue(2U, 0U);
        signal_.SetValue(3U, 0U);
        published_ = 0U;
        SetFlag<HardEvent::MTE3_S>(kEvent);
    }

    // One finite pass. Every unresolved dependency is probed at most once.
    // No wait for incoming CVs or local GMM2 is allowed before outgoing quiet.
    __aicore__ inline uint32_t RefreshOnce(const Arguments& args)
    {
        using namespace AscendC;
        uint32_t pendingCount = 0U;
        constexpr uint32_t selfTiles = (3584U + kComputeTileN - 1U) / kComputeTileN;
        for (uint32_t i = 0U; i < dependencyCount_; ++i) {
            const uint64_t span = dependencies_.GetValue(i * 2U + 1U);
            if (span == 0U) continue;  // This dependency was already consumed.
            const uint64_t dependency = dependencies_.GetValue(i * 2U);
            const uint64_t index = dependency & ~kReduceSelfDependency;
            bool ready;
            if ((dependency & kReduceSelfDependency) != 0U) {
                auto* flag = reinterpret_cast<__gm__ int32_t*>(args.gmm2Done + index * kFlagBytes);
                ready = ReadGmByPassDCache(flag) >= static_cast<int32_t>(selfTiles);
            } else {
                auto* flag = reinterpret_cast<__gm__ uint64_t*>(args.symmetric + args.config.combineReadyOffset +
                                                                index * kFlagBytes);
                ready = ReadGmByPassDCache(flag) == args.config.epoch;
            }
            if (!ready) continue;
            const uint32_t begin = static_cast<uint32_t>(span);
            const uint32_t count = static_cast<uint32_t>(span >> 32U);
            for (uint32_t edge = begin; edge < begin + count; ++edge) {
                const uint32_t contribution = edges_.GetValue(edge);
                const uint32_t token = contribution & 0xffffU;
                const uint32_t total = counts_.GetValue(token) + (contribution >> 16U);
                counts_.SetValue(token, total);
                if (total == 8U) pending_.SetValue(pendingCount++, tokenBegin_ + token);
            }
            dependencies_.SetValue(i * 2U + 1U, 0U);
        }
        if (pendingCount != 0U) {
            // Recycle the previous S->MTE3 gate only after MTE3 consumed it.
            // Return its credit before writes, without draining output DMA.
            WaitFlag<HardEvent::MTE3_S>(kEvent);
            SetFlag<HardEvent::S_MTE3>(kEvent);
            WaitFlag<HardEvent::S_MTE3>(kEvent);
            SetFlag<HardEvent::MTE3_S>(kEvent);
            for (uint32_t i = 0U; i < pendingCount; ++i) {
                GlobalTensor<uint64_t> target;
                target.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(
                    args.tokenReady + static_cast<uint64_t>(pending_.GetValue(i)) * kFlagBytes));
                // The source epoch buffer stays immutable for this invocation.
                DataCopy(target, signal_, 4U);
            }
            published_ += pendingCount;
        }
        return pendingCount;
    }

    __aicore__ inline bool Done() const
    {
        return published_ == tokenCount_;
    }
    __aicore__ inline uint32_t TokenBegin() const
    {
        return tokenBegin_;
    }
    __aicore__ inline uint32_t TokenCount() const
    {
        return tokenCount_;
    }

    __aicore__ inline void Finish(const Arguments& args)
    {
        while (!Done()) {
            if (RefreshOnce(args) == 0U) {
                const uint64_t start = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - start < 200U) {
                }
            }
        }
        // Balance the gate credit; Process's existing PIPE_ALL drains the
        // immutable signal buffer before this engine releases its UB.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvent);
    }

private:
    static constexpr AscendC::TEventID kEvent = static_cast<AscendC::TEventID>(1U);
    uint32_t tokenBegin_ = 0U;
    uint32_t tokenCount_ = 0U;
    uint32_t dependencyCount_ = 0U;
    uint32_t published_ = 0U;
    AscendC::LocalTensor<uint64_t> dependencies_;
    AscendC::LocalTensor<uint32_t> edges_;
    AscendC::LocalTensor<uint32_t> counts_;
    AscendC::LocalTensor<uint32_t> pending_;
    AscendC::LocalTensor<uint64_t> signal_;
};
}  // namespace MegaMoeUrma
