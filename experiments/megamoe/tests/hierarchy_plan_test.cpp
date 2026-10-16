// Copyright (c) 2026, SimpleBright_Man
#include "hierarchy_plan.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>

namespace {
void Require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
void Check(const std::vector<int32_t>& routes, uint32_t batch, uint32_t topk, uint32_t local)
{
    using namespace deepep::megamoe::hierarchy;
    auto plan = Build(routes.data(), batch, topk, local);
    std::vector<uint32_t> coverage(routes.size());
    for (uint32_t rank = 0; rank < 16; ++rank) {
        const auto& mine = plan.ranks[rank];
        std::set<uint32_t> oracle;
        for (uint32_t edge = 0; edge < routes.size(); ++edge)
            if (static_cast<uint32_t>(routes[edge]) / local == rank) oracle.insert(edge / topk);
        Require(std::vector<uint32_t>(oracle.begin(), oracle.end()) == mine.token_keys, "token dedup/order");
        uint32_t previous = 0;
        for (uint32_t slot = 0; slot < mine.token_keys.size(); ++slot) {
            auto key = mine.token_keys[slot];
            Require(mine.token_slots[key] == static_cast<int32_t>(slot), "directory roundtrip");
            uint32_t source = key / batch, proxy = Proxy(rank / 4, source);
            Require(proxy / 4 == rank / 4 && proxy % 4 == source % 4, "proxy topology");
            auto proxy_slot = ProxySlot(key, batch);
            Require(proxy_slot < 4 * batch, "proxy capacity");
            Require((proxy_slot / batch) * 4 + proxy % 4 == source && proxy_slot % batch == key % batch,
                    "proxy slot identity");
        }
        for (uint32_t edge : mine.edge_keys) {
            Require(edge >= previous && edge < routes.size(), "route edge order");
            Require(static_cast<uint32_t>(routes[edge]) / local == rank, "edge destination");
            Require(mine.token_slots[edge / topk] >= 0, "fanout input missing");
            ++coverage[edge];
            previous = edge;
        }
    }
    for (auto count : coverage) Require(count == 1, "lost or repeated expert edge");
    for (uint32_t key = 0; key < 16 * batch; ++key) {
        uint16_t expected_mask = 0;
        double direct = 0;
        double rank_partial[16]{};
        for (uint32_t k = 0; k < topk; ++k) {
            uint32_t expert = routes[key * topk + k], destination = expert / local;
            expected_mask |= static_cast<uint16_t>(1U << destination);
            // Include signed and unequal weights: payload dedup must not discard contributions.
            double weight = (static_cast<int>((key + k) % 17) - 8) / 16.0;
            double expert_output = static_cast<double>(expert + 1) / 8.0;
            direct += expert_output * weight;
            rank_partial[destination] += expert_output * weight;
        }
        double hierarchical = 0;
        for (uint32_t group = 0; group < 4; ++group) {
            double group_partial = 0;
            for (uint32_t consumer = group * 4; consumer < group * 4 + 4; ++consumer)
                group_partial += rank_partial[consumer];
            hierarchical += group_partial;
        }
        Require(expected_mask == plan.consumers[key], "consumer mask incomplete");
        Require(direct == hierarchical, "weighted reverse reduction");
    }
}
}  // namespace

int main()
{
    using namespace deepep::megamoe::hierarchy;
    uint32_t cases = 0;
    std::mt19937 random(20261016);
    for (uint32_t batch : {1U, 3U, 17U, 128U}) {
        for (uint32_t topk : {1U, 2U, 8U, 32U}) {
            std::vector<int32_t> routes(16U * batch * topk);
            for (uint32_t key = 0; key < 16 * batch; ++key) {
                std::set<int32_t> chosen;
                for (uint32_t slot = 0; slot < topk; ++slot) {
                    int32_t expert;
                    do {
                        expert = static_cast<int32_t>(random() % 384);
                    } while (!chosen.insert(expert).second);
                    routes[key * topk + slot] = expert;
                }
            }
            Check(routes, batch, topk, 24);
            ++cases;
            // All edges target only rank 7; the remaining 15 ranks have empty queues.
            if (topk <= 24) {
                for (uint32_t edge = 0; edge < routes.size(); ++edge) routes[edge] = 7 * 24 + edge % topk;
                Check(routes, batch, topk, 24);
                ++cases;
            }
        }
    }
    for (uint32_t batch : {2048U, 4096U, 8192U}) {
        std::vector<int32_t> routes(16U * batch * 8U);
        for (uint32_t edge = 0; edge < routes.size(); ++edge) routes[edge] = (edge % 16) * 24 + (edge / 16) % 24;
        Check(routes, batch, 8, 24);
        auto plan = Build(routes.data(), batch, 8, 24);
        uint64_t cross = 0;
        for (uint32_t key = 0; key < 16 * batch; ++key)
            for (uint32_t group = 0; group < 4; ++group)
                if (group != key / batch / 4 && ((plan.consumers[key] >> (group * 4)) & 15U)) ++cross;
        Require(cross == 16ULL * batch * 3 / 2, "balanced cross dedup must be 25% of direct");
        ++cases;
    }
    std::vector<int32_t> bad(32, 0);
    bool rejected = false;
    try {
        Build(bad.data(), 1, 2, 24);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(rejected, "duplicate route should fail");
    bad.assign(16, 384);
    rejected = false;
    try {
        Build(bad.data(), 1, 1, 24);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(rejected, "out-of-range route should fail");
    std::cout << "hierarchy_plan: " << cases << " route/forward/reverse oracle cases passed\n";
}
