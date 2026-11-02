#pragma once
#include "layout.hpp"
#include "../megamoe/csrc/kernels/mega_moe/urma/expert_plan.hpp"
namespace Demo {
inline MegaMoeUrma::ExpertPlan MakePlan(uint32_t bs,uint32_t world,uint32_t rank) {
    std::array<int64_t,MegaMoeUrma::kWorld*384> counts{};
    uint32_t local=384/world;uint64_t expanded=uint64_t(bs)*8;
    // Exactly benchmark.inputs' balanced stream, including per-source offsets.
    for(uint32_t source=0;source<world;++source)
        for(uint64_t flat=source*expanded;flat<(source+1)*expanded;++flat)
            ++counts[source*384+(flat%world)*local+(flat/world)%local];
    MegaMoeUrma::Config c{};
    c.rank=rank;c.world=world;c.bs=bs;c.topk=8;c.hidden=3584;c.intermediate=1280;
    c.localExperts=local;c.cvRows=256;c.maxPeerChunks=(expanded+255)/256;c.gmm1AheadMGroups=12;
    auto p=MegaMoeUrma::BuildExpertPlan(counts.data(),c,expanded);
    // Keep production batch boundaries and global tile ownership, alternate
    // GMM1/GMM2 per batch without lookahead/activation dependencies.
    p.computeTasks.clear();
    for(uint32_t b=0;b<p.computeBatches.size();++b) {
        p.computeTasks.push_back({MegaMoeUrma::ComputeBatchStage::Gmm1,b});
        p.computeTasks.push_back({MegaMoeUrma::ComputeBatchStage::Gmm2,b});
    }
    p.config.computeTaskCount=p.computeTasks.size();return p;
}
}
