// Copyright (c) 2026, SimpleBright_Man
#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace deepep::megamoe::hierarchy {
constexpr uint32_t kWorld = 16;
constexpr uint32_t kGroup = 4;
struct RankPlan {
    std::vector<uint32_t> token_keys;
    std::vector<uint32_t> edge_keys;
    std::vector<int32_t> token_slots;
};
struct RoutePlan {
    std::array<RankPlan, kWorld> ranks;
    std::vector<uint16_t> consumers;
    uint32_t batch;
    uint32_t topk;
};
inline uint32_t Proxy(uint32_t group, uint32_t source)
{
    return group * kGroup + source % kGroup;
}
inline uint32_t ProxySlot(uint32_t key, uint32_t batch)
{
    return (key / batch / kGroup) * batch + key % batch;
}

inline RoutePlan Build(const int32_t* routes, uint32_t batch, uint32_t topk, uint32_t local_experts)
{
    if (!routes || batch == 0 || batch > 8192 || topk == 0 || topk > 32 || local_experts == 0 || local_experts > 128 ||
        topk > local_experts * kWorld) {
        throw std::invalid_argument("unsupported hierarchical routing shape");
    }
    RoutePlan plan{};
    plan.batch = batch;
    plan.topk = topk;
    plan.consumers.resize(kWorld * batch);
    for (auto& rank : plan.ranks) rank.token_slots.assign(kWorld * batch, -1);
    for (uint32_t key = 0; key < kWorld * batch; ++key) {
        for (uint32_t slot = 0; slot < topk; ++slot) {
            uint32_t edge = key * topk + slot;
            int32_t expert = routes[edge];
            if (expert < 0 || static_cast<uint32_t>(expert) >= kWorld * local_experts)
                throw std::invalid_argument("route expert is out of range");
            for (uint32_t previous = 0; previous < slot; ++previous)
                if (routes[key * topk + previous] == expert)
                    throw std::invalid_argument("duplicate experts within one token are unsupported");
            uint32_t destination = static_cast<uint32_t>(expert) / local_experts;
            auto& rank = plan.ranks[destination];
            rank.edge_keys.push_back(edge);
            plan.consumers[key] |= static_cast<uint16_t>(1U << destination);
            if (rank.token_slots[key] == -1) {
                rank.token_slots[key] = static_cast<int32_t>(rank.token_keys.size());
                rank.token_keys.push_back(key);
            }
        }
    }
    return plan;
}
}  // namespace deepep::megamoe::hierarchy
