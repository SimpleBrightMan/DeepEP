#pragma once
#include "../megamoe/csrc/kernels/mega_moe/urma/compute.hpp"
#include "../megamoe/csrc/kernels/mega_moe/urma/shared_completion.hpp"
#include "completion_policy.hpp"
namespace Demo {
using namespace MegaMoeUrma;
using SharedGmm1=MegaMoeImpl::GmmKernel::CatlassGmm1<UrmaProblemConfig,false,MegaMoeImpl::GmmKernel::ContiguousRowIO>;
using SharedGmm2=MegaMoeImpl::GmmKernel::CatlassGmm2<UrmaProblemConfig,false,MegaMoeImpl::GmmKernel::ContiguousRowIO>;
__aicore__ inline Arguments SharedView(const Arguments& a) {
    auto v=a;v.config.intermediate=a.sharedIntermediate;
    v.weight1=a.sharedWeight1;v.weight2=a.sharedWeight2;v.weightScale1=a.sharedScale1;v.weightScale2=a.sharedScale2;
    v.gmm1Output=a.sharedGmm1;v.activation=a.sharedActivation;v.activationScale=a.sharedActivationScale;v.gmm2Output=a.sharedOutput;return v;
}
__aicore__ inline Fragment SharedFragment(const Arguments& a,uint32_t group) {
    const uint32_t row=group*256,rows=a.config.bs-row<256?a.config.bs-row:256;
    return {0,0,row,rows,0,uint64_t(group)*(a.sharedIntermediate/256),uint64_t(group)*(a.config.hidden/256)};
}
template<bool Gmm1,bool Fused>
__aicore__ inline auto MakeConfig(const MegaMoeImpl::ProblemShape& shape,const MegaMoeImpl::BlockJobContext& job) {
    if constexpr(Gmm1)return UrmaGmmConfigFor<Fused>::BuildGmm1ProblemConfig(shape,job,kComputeTileM);
    else return UrmaGmmConfig::BuildGmm2ProblemConfig(shape,job,kComputeTileM);
}
template<bool Notify,bool Gmm1,bool Fused,class Gmm,class Pending>
__aicore__ inline void ComputeFragment(const Arguments& a,uint32_t block,uint32_t index,uint32_t first,Gmm& gmm,Pending& pending) {
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
        if constexpr(g_coreType==AscendC::AIC) {
            TraceMeta meta;
            if constexpr(Notify)meta=ComputeTileTraceMeta(FragmentTraceMeta(f,index),m,n,rows,cols);
            gmm.Compute(m,n,rows,cols);
            if constexpr(Notify)pending.AfterSubmit(Gmm1,(Gmm1?f.gmm1TileBegin:f.gmm2TileBegin)+tile,meta);
        } else gmm.Dequantize(n,rows,cols);
    }
}
template<bool Notify,bool First>
__aicore__ inline void RunIndependentShared(const Arguments& a,uint32_t block) {
    if(!a.sharedExperts)return;
    auto v=SharedView(a);std::conditional_t<First,SharedGmm1,SharedGmm2> gmm;bool active=false;
    NoopTimer disabled;
    SharedOutputBackend<NoopTimer> backend{a,disabled};
    OptionalTileCompletion<Notify && First && g_coreType==AscendC::AIC,SharedOutputBackend<NoopTimer>> pending(backend);
    for(uint32_t group=0;group<(a.config.bs+255)/256;++group) {
        const auto f=SharedFragment(a,group);const uint32_t count=First?a.sharedIntermediate/256:a.config.hidden/256;
        const auto first=FirstOwnedComputeTile(uint64_t(group)*count,block,kAic);if(first>=count)continue;
        const MegaMoeImpl::BlockJobContext job{block,kAic};
        auto config=MakeConfig<First,false>(FragmentShape(v,f),job);
        auto addr=First?Gmm1Addresses(v,f):Gmm2Addresses(v,f);
        if constexpr(First){addr.aGlobal=reinterpret_cast<GM_ADDR>(a.x+f.rowBegin*a.config.hidden);addr.aScaleGlobal=reinterpret_cast<GM_ADDR>(a.xScale+f.rowBegin*(a.config.hidden/32));}
        else addr.gmm2OutGlobal=reinterpret_cast<GM_ADDR>(a.sharedOutput+f.rowBegin*a.config.hidden*2);
        gmm.UpdateProblem(config,addr);gmm.Begin();active=true;
        for(uint32_t tile=first;tile<count;tile+=kAic) {
            if constexpr(g_coreType==AscendC::AIC) {
                TraceMeta meta;
                if constexpr(Notify && First){meta=FragmentTraceMeta(f,group);meta.expert=kExperts;meta.tileN=tile*256;meta.columns=256;}
                gmm.Compute(0,tile*256,f.rows,256);
                if constexpr(Notify && First)pending.AfterSubmit(true,f.gmm1TileBegin+tile,meta);
            } else gmm.Dequantize(tile*256,f.rows,256);
        }
    }
    if constexpr(Notify && First && g_coreType==AscendC::AIC)pending.Flush();
    if(active){gmm.End(Notify && First && g_coreType==AscendC::AIC);CompleteComputeContext();}
}
template<bool Notify>
__aicore__ inline void RunCompute(const Arguments& a,uint32_t block) {
    RunIndependentShared<Notify,true>(a,block);
    UrmaGmm1 g1;UrmaGmm2 g2;
    NoopTimer disabled;
    TileOutputBackend<NoopTimer> backend{a,disabled};
    OptionalTileCompletion<Notify && g_coreType==AscendC::AIC,TileOutputBackend<NoopTimer>> pending(backend);
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
                if constexpr(Notify && g_coreType==AscendC::AIC)pending.BeforeStage(isG1);
                if(first)g1.End(Notify && g_coreType==AscendC::AIC);else g2.End(Notify && g_coreType==AscendC::AIC);
                CompleteComputeContext();active=false;
            }
            if(isG1) {
                if(UseFusedGateUp(a.config))ComputeFragment<Notify,true,true>(a,block,index,owned,g1,pending);
                else ComputeFragment<Notify,true,false>(a,block,index,owned,g1,pending);
            } else ComputeFragment<Notify,false,false>(a,block,index,owned,g2,pending);
            first=isG1;active=true;
        }
    }
    if constexpr(Notify && g_coreType==AscendC::AIC)pending.Flush();
    if(active) {
        if(first)g1.End(Notify && g_coreType==AscendC::AIC);else g2.End(Notify && g_coreType==AscendC::AIC);
        CompleteComputeContext();
    }
    RunIndependentShared<Notify,false>(a,block);
}
}
