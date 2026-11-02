#pragma once
#include "../megamoe/csrc/kernels/mega_moe/urma/compute.hpp"
namespace Demo {
using namespace MegaMoeUrma;
template<bool Gmm1,bool Fused>
__aicore__ inline auto MakeConfig(const MegaMoeImpl::ProblemShape& shape,const MegaMoeImpl::BlockJobContext& job) {
    if constexpr(Gmm1)return UrmaGmmConfigFor<Fused>::BuildGmm1ProblemConfig(shape,job,kComputeTileM);
    else return UrmaGmmConfig::BuildGmm2ProblemConfig(shape,job,kComputeTileM);
}
template<bool Gmm1,bool Fused,class Gmm>
__aicore__ inline void ComputeFragment(const Arguments& a,uint32_t block,uint32_t index,uint32_t first,Gmm& gmm) {
    const auto f=LoadFragment(a,index);
    const auto shape=FragmentShape(a,f);
    const MegaMoeImpl::BlockJobContext job{block,kAic};
    auto config=MakeConfig<Gmm1,Fused>(shape,job);
    MegaMoeImpl::GmmKernel::BlockScheduler scheduler(
        {config.m,config.schedulerN,config.k},
        MegaMoeImpl::GmmKernel::BlockScheduler::Params{
            Te::MakeCoord(static_cast<int64_t>(config.tileM),static_cast<int64_t>(config.tileN))});
    if constexpr(Gmm1) {
        gmm.UpdateProblem(config,Gmm1Addresses(a,f));
        gmm.SetRowIO(FragmentRowIO<UrmaInputRows>(a,f));
    } else {
        gmm.UpdateProblem(config,Gmm2Addresses(a,f));
        gmm.SetRowIO(FragmentRowIO<UrmaOutputRows>(a,f));
    }
    gmm.Begin();
    for(uint32_t tile=first;tile<scheduler.GetTileNum();tile+=kAic) {
        const auto coord=scheduler.GetBlockCoord(tile);
        const auto extent=scheduler.GetBlockShape(coord);
        const uint32_t m=Get<MegaMoeImpl::M_VALUE>(coord),n=Get<MegaMoeImpl::N_VALUE>(coord);
        const uint32_t rows=Get<MegaMoeImpl::M_VALUE>(extent),cols=Get<MegaMoeImpl::N_VALUE>(extent);
        if constexpr(g_coreType==AscendC::AIC)gmm.Compute(m,n,rows,cols);
        else gmm.Dequantize(n,rows,cols);
    }
}
__aicore__ inline void RunCompute(const Arguments& a,uint32_t block) {
    UrmaGmm1 g1;UrmaGmm2 g2;
    bool active=false,first=false;
    for(uint32_t step=0;step<a.config.computeTaskCount;++step) {
        const auto task=LoadComputeTask(a,step);const auto range=LoadComputeBatch(a,task.batchIndex);
        const bool isG1=task.stage==ComputeBatchStage::Gmm1;
        for(uint32_t index=range.begin;index<range.end;++index) {
            const auto f=LoadFragment(a,index);
            const auto count=isG1?FragmentGmm1Tiles(a,f):FragmentGmm2Tiles(a,f);
            const auto owned=FirstOwnedComputeTile(isG1?f.gmm1TileBegin:f.gmm2TileBegin,block,kAic);
            if(owned>=count)continue;
            if(active && first!=isG1) {
                if(first)g1.End();else g2.End();
                CompleteComputeContext();active=false;
            }
            if(isG1) {
                if(UseFusedGateUp(a.config))ComputeFragment<true,true>(a,block,index,owned,g1);
                else ComputeFragment<true,false>(a,block,index,owned,g1);
            } else ComputeFragment<false,false>(a,block,index,owned,g2);
            first=isG1;active=true;
        }
    }
    if(active) {
        if(first)g1.End();else g2.End();
        CompleteComputeContext();
    }
}
}
