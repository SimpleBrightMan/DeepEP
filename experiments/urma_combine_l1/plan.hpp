#pragma once
#include "layout.hpp"
#include "../megamoe/csrc/kernels/mega_moe/urma/expert_plan.hpp"
namespace Demo {
inline MegaMoeUrma::ExpertPlan MakePlan(uint32_t bs,uint32_t world,uint32_t rank,uint32_t hidden=5120,uint32_t intermediate=2304,uint32_t topk=6) {
    std::array<int64_t,MegaMoeUrma::kWorld*384> counts{};
    uint32_t local=384/world;uint64_t expanded=uint64_t(bs)*topk;
    // Exactly benchmark.inputs' balanced stream, including per-source offsets.
    for(uint32_t source=0;source<world;++source)
        for(uint64_t flat=source*expanded;flat<(source+1)*expanded;++flat)
            ++counts[source*384+(flat%world)*local+(flat/world)%local];
    MegaMoeUrma::Config c{};
    c.rank=rank;c.world=world;c.bs=bs;c.topk=topk;c.hidden=hidden;c.intermediate=intermediate;
    c.localExperts=local;c.cvRows=256;c.maxPeerChunks=(expanded+255)/256;c.gmm1AheadMGroups=12;
    auto p=MegaMoeUrma::BuildExpertPlan(counts.data(),c,expanded);
    return p;
}
}
