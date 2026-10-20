// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

// Host-only inversion of the actual token-major dependency descriptors.
// Kept independent of torch/CANN for execution in CPU regression tests.
#include "abi.hpp"

#include <map>
#include <stdexcept>
#include <vector>

namespace MegaMoeUrma {
struct ReduceReadyPlan {
    std::vector<ReadyOwner> owners;
    std::vector<ReadyDependency> dependencies;
    std::vector<ReadyEdge> edges;
};

inline ReduceReadyPlan BuildReduceReadyPlan(const uint64_t* tokenDependencies, const Config& config)
{
    if (tokenDependencies == nullptr || config.bs == 0U || config.bs > 8192U || config.topk != 8U ||
        config.rank >= kWorld || config.maxPeerChunks == 0U || config.maxPeerChunks > 512U) {
        throw std::invalid_argument("Invalid URMA token-ready plan dimensions");
    }
    ReduceReadyPlan plan;
    plan.owners.reserve(kCommunicationWorkers);
    const uint32_t quotient = config.bs / kCommunicationWorkers;
    const uint32_t remainder = config.bs % kCommunicationWorkers;
    uint32_t tokenBegin = 0U;
    for (uint32_t worker = 0U; worker < kCommunicationWorkers; ++worker) {
        ReadyOwner owner{};
        owner.tokenBegin = tokenBegin;
        owner.tokenCount = quotient + (worker < remainder ? 1U : 0U);
        owner.dependencyBegin = static_cast<uint32_t>(plan.dependencies.size());
        owner.edgeBegin = static_cast<uint32_t>(plan.edges.size());
        std::map<uint64_t, std::vector<ReadyEdge>> dependencies;
        for (uint32_t localToken = 0U; localToken < owner.tokenCount; ++localToken) {
            std::map<uint64_t, uint16_t> multiplicities;
            const uint64_t firstRoute = static_cast<uint64_t>(tokenBegin + localToken) * config.topk;
            for (uint32_t slot = 0U; slot < config.topk; ++slot) {
                const uint64_t dependency = tokenDependencies[firstRoute + slot];
                const uint64_t index = dependency & ~kReduceSelfDependency;
                const bool self = (dependency & kReduceSelfDependency) != 0U;
                if (self ? index >= config.fragmentCount
                         : (index >= static_cast<uint64_t>(kWorld) * config.maxPeerChunks ||
                            index / config.maxPeerChunks == config.rank)) {
                    throw std::invalid_argument("Invalid URMA token-ready dependency descriptor");
                }
                ++multiplicities[dependency];
            }
            uint32_t total = 0U;
            for (const auto& [dependency, multiplicity] : multiplicities) {
                dependencies[dependency].push_back({static_cast<uint16_t>(localToken), multiplicity});
                total += multiplicity;
            }
            if (total != config.topk) {
                throw std::logic_error("URMA token-ready contributions must sum to topk");
            }
        }
        for (const auto& [dependency, edges] : dependencies) {
            plan.dependencies.push_back({dependency, static_cast<uint32_t>(plan.edges.size()) - owner.edgeBegin,
                                         static_cast<uint32_t>(edges.size())});
            plan.edges.insert(plan.edges.end(), edges.begin(), edges.end());
        }
        owner.dependencyCount = static_cast<uint32_t>(plan.dependencies.size()) - owner.dependencyBegin;
        owner.edgeCount = static_cast<uint32_t>(plan.edges.size()) - owner.edgeBegin;
        if (owner.tokenCount > kMaxReadyOwnerTokens || owner.edgeCount > kMaxReadyOwnerEdges) {
            throw std::logic_error("URMA token-ready owner exceeds its UB contract");
        }
        plan.owners.push_back(owner);
        // Each owner's independent DMA range begins and ends on a 32B boundary.
        // Value-initialized padding is excluded from the logical counts above.
        plan.dependencies.resize((plan.dependencies.size() + 1U) / 2U * 2U);
        plan.edges.resize((plan.edges.size() + 7U) / 8U * 8U);
        tokenBegin += owner.tokenCount;
    }
    return plan;
}
}  // namespace MegaMoeUrma
