#pragma once
#include "compute.hpp"
#include "shared_contract.hpp"
#include "shared_add.hpp"
#include "tail_completion.hpp"
namespace MegaMoeUrma {
using SharedGmm1 = MegaMoeImpl::GmmKernel::CatlassGmm1<UrmaProblemConfig,false,MegaMoeImpl::GmmKernel::ContiguousRowIO>;
using SharedGmm2 = MegaMoeImpl::GmmKernel::CatlassGmm2<UrmaProblemConfig,false,MegaMoeImpl::GmmKernel::ContiguousRowIO>;
// Contiguous output makes both classes select MmadA8W4Mx<..., unitFlag=true>.
__aicore__ inline Arguments SharedView(const Arguments& a) {
    Arguments v=a;v.config.intermediate=a.sharedIntermediate;
    v.weight1=a.sharedWeight1;v.weight2=a.sharedWeight2;v.weightScale1=a.sharedScale1;v.weightScale2=a.sharedScale2;
    v.gmm1Output=a.sharedGmm1;v.activation=a.sharedActivation;v.activationScale=a.sharedActivationScale;
    v.gmm2Output=a.sharedOutput;return v;
}
__aicore__ inline Fragment SharedFragment(const Arguments& a,uint32_t group) {
    const uint32_t row=group*256U,rows=a.config.bs-row<256U?a.config.bs-row:256U;
    return {0,0,row,rows,0,uint64_t(group)*(a.sharedIntermediate/256),uint64_t(group)*(a.config.hidden/256)};
}
template<class T> struct SharedOutputBackend {
    const Arguments& a;T& timer;
    __aicore__ inline void Arm(){SetFlag<HardEvent::FIX_S>(EVENT_ID1);}
    __aicore__ inline void Complete(bool,uint64_t tile,const TraceMeta& m){
        const auto start=timer.TraceBegin();WaitFlag<HardEvent::FIX_S>(EVENT_ID1);
        timer.TraceEnd(TraceEvent::SharedGmm1Drain,start,m);
        WriteGmByPassDCache(ComputeFlag(a.sharedGmm1Done,tile),1);
    }
};
template<bool First,class T>
__aicore__ inline void RunSharedGmm(const Arguments& a,uint32_t core,T& timer) {
    auto v=SharedView(a);std::conditional_t<First,SharedGmm1,SharedGmm2> gmm;
    SharedOutputBackend<T> backend{a,timer};TileCompletion<SharedOutputBackend<T>> pending(backend);
    bool active=false;
    for(uint32_t group=0;group<(a.config.bs+255)/256;++group) {
        const auto f=SharedFragment(a,group);
        const uint32_t count=First?a.sharedIntermediate/256:a.config.hidden/256;
        const uint32_t first=FirstOwnedComputeTile(uint64_t(group)*count,core,kAic);
        if(first>=count)continue;
        const MegaMoeImpl::BlockJobContext job{core,kAic};
        auto config=UrmaGmmConfig::BuildGmm1ProblemConfig(FragmentShape(v,f),job,kComputeTileM);
        // Both builders have the same ProblemConfig type; shared gate/up stays planar.
        if constexpr(!First)config=UrmaGmmConfig::BuildGmm2ProblemConfig(FragmentShape(v,f),job,kComputeTileM);
        auto addr=First?Gmm1Addresses(v,f):Gmm2Addresses(v,f);
        if constexpr(First) {
            addr.aGlobal=reinterpret_cast<GM_ADDR>(a.x+f.rowBegin*a.config.hidden);
            addr.aScaleGlobal=reinterpret_cast<GM_ADDR>(a.xScale+f.rowBegin*(a.config.hidden/32));
        } else {
            addr.gmm2OutGlobal=reinterpret_cast<GM_ADDR>(a.sharedOutput+f.rowBegin*a.config.hidden*2);
            if constexpr(g_coreType==AIC)
                WaitComputeFlag(a.sharedActivationDone,group,(a.sharedIntermediate/256)*((f.rows+127)/128));
        }
        gmm.UpdateProblem(config,addr);gmm.Begin();active=true;
        for(uint32_t tile=first;tile<count;tile+=kAic) {
            TraceMeta meta=FragmentTraceMeta(f,group);meta.expert=kExperts;
            meta.tileN=tile*256;meta.columns=256;
            const auto start=timer.TraceBegin();
            if constexpr(g_coreType==AIC)gmm.Compute(0,tile*256,f.rows,256);
            else gmm.Dequantize(tile*256,f.rows,256);
            timer.TraceEnd(First?TraceEvent::SharedGmm1Tile:TraceEvent::SharedGmm2Tile,start,meta);
            if constexpr(First && g_coreType==AIC)pending.AfterSubmit(true,f.gmm1TileBegin+tile,meta);
        }
    }
    if constexpr(First && g_coreType==AIC)pending.Flush();
    if(active) {
        const auto start=timer.TraceBegin();
        gmm.End(First && g_coreType==AIC);CompleteComputeContext();
        timer.TraceEnd(First?TraceEvent::SharedGmm1Drain:TraceEvent::SharedGmm2Drain,start);
    }
}

struct SharedWaitScope {
    Timer& timer;bool& waiting;uint64_t& start;const TraceMeta& meta;TraceEvent event;
    __aicore__ inline void Before() {
        if(waiting){timer.TraceEnd(event,start,meta);waiting=false;}
    }
};
class SharedActivationState {
public:
    __aicore__ inline SharedActivationState(const Arguments& a,uint32_t worker,Timer& t):a_(a),worker_(worker),timer_(t) {
        total_=SharedTaskCount(a.config.bs,a.sharedIntermediate);
        owned_=SharedOwnedTasks(total_,worker);
    }
    __aicore__ inline bool Done()const{return finished_==owned_;}
    template<class BeforeWork> __aicore__ inline bool TryOne(BeforeWork& before) {
        for(uint32_t j=0;j<owned_;++j) {
            const uint32_t local=(cursor_+j)%owned_;
            if(done_[local/64]&(uint64_t{1}<<(local%64)))continue;
            const uint32_t task=worker_+local*kAic;
            const auto assignment=SharedTaskAt(task,a_.sharedIntermediate);
            const uint32_t group=assignment.group,tile=assignment.tile,slice=assignment.slice;
            const auto f=SharedFragment(a_,group);
            if(slice*128>=f.rows){Mark(local);continue;}
            if(!ReadGmByPassDCache(ComputeFlag(a_.sharedGmm1Done,f.gmm1TileBegin+tile)))continue;
            before.Before(); // End a prior idle bar before recording actual shared work.
            auto v=SharedView(a_);auto addr=Gmm1Addresses(v,f);addr.activationToGmm2Flag=nullptr;
            UrmaActivation op;
            op.Init({.yGmAddr=reinterpret_cast<GM_ADDR>(a_.sharedActivation),
                     .yScaleGmAddr=reinterpret_cast<GM_ADDR>(a_.sharedActivationScale),.clampLimit=FLT_MAX});
            op.UpdateGlobalAddr({static_cast<int64_t>(f.rowBegin*v.config.intermediate),
                static_cast<int64_t>(f.rowBegin*(v.config.intermediate/32)),0,0,0,0});
            const MegaMoeImpl::BlockJobContext job{worker_,kAic};
            auto config=UrmaGmmConfig::BuildGmm1ProblemConfig(FragmentShape(v,f),job,kComputeTileM);
            const auto layouts=UrmaGmmConfig::BuildLayouts(config);
            auto gmC=Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ bfloat16_t*>(addr.gmm1OutGlobal)),layouts.c);
            op.UpdateNextProblem({config.m,config.outputN,config.k,0});config.tileM=128;
            const uint32_t m=slice*128,rows=f.rows-m<128?f.rows-m:128;
            const MegaMoeImpl::GmmKernel::BlockScheduler::BlockShape shape{rows,256,config.k};
            auto meta=ComputeTileTraceMeta(FragmentTraceMeta(f,group),m,tile*256,rows,256);meta.expert=kExperts;
            const auto start=timer_.TraceBegin();
            MegaMoeImpl::GmmKernel::Gmm1Aiv1EpilogueTileA8W4<bfloat16_t,UrmaGmmConfig::MakeLayoutC,true>(
                op,gmC,addr,config,shape,m,tile*256);
            AtomicAdd(ComputeFlag(a_.sharedActivationDone,group),1);
            timer_.TraceEnd(TraceEvent::SharedActivation,start,meta);
            Mark(local);cursor_=(local+1)%owned_;return true;
        }
        return false;
    }
private:
    __aicore__ inline void Mark(uint32_t i){done_[i/64]|=uint64_t{1}<<(i%64);++finished_;}
    const Arguments& a_;uint32_t worker_;Timer& timer_;
    uint64_t done_[3]{}; // max ceil(32 groups * 64 N tiles * 2 slices / 28)=147
    uint32_t total_=0,owned_=0,finished_=0,cursor_=0;
};

struct SharedIncoming {
    const Arguments& a;uint32_t core;bool peer=false,self=false;
    __aicore__ inline void Poll(){if(core<CommunicationWorkers(a.config.world))ObserveIncomingCompletion(a,core,peer,self);}
};

__aicore__ inline void RunActivationShared(const Arguments& args, uint32_t worker, Timer& timer)
{
    const uint32_t activationWorkers = ActivationWorkers(args.config.world);
    const uint32_t communicationWorkers = CommunicationWorkers(args.config.world);
    SharedActivationState shared(args,worker+communicationWorkers,timer);
    UrmaActivation epilogue;
    // Match main MegaMoE's unclamped SwiGLU. The epilogue default is zero,
    // which clamps every up value to zero and makes the entire FFN zero.
    epilogue.Init({.yGmAddr = reinterpret_cast<GM_ADDR>(args.activation),
                   .yScaleGmAddr = reinterpret_cast<GM_ADDR>(args.activationScale),
                   .clampLimit = FLT_MAX});
    uint32_t nextGather = worker;
    uint32_t nextActivation = 0U;
    uint32_t activationTile = 0U;
    uint32_t nextPackRound = kPackStartupRounds;
    PackBatch pack(args, timer);
    bool packFinished = false;
    bool activationPrepared = false;
    bool idle = false;
    uint64_t idleStart = 0U;
    TraceEvent idleEvent = TraceEvent::WaitGmm1;
    TraceMeta idleMeta;
    while (nextGather < args.config.fragmentCount || nextActivation < args.config.fragmentCount ||
           nextPackRound < args.config.packRoundCount || !shared.Done()) {
        bool progress = false;
        // Skip groups without this worker's activation tile, but never skip a
        // scale gather merely because its owner has no activation in that group.
        while (nextActivation < args.config.fragmentCount) {
            const auto fragment = LoadFragment(args, nextActivation);
            const uint32_t tileCount = FragmentGmm1Tiles(args, fragment);
            if (!activationPrepared) {
                activationTile = FirstOwnedComputeTile(fragment.gmm1TileBegin, worker, activationWorkers);
                activationPrepared = true;
            }
            if (activationTile >= tileCount) {
                ++nextActivation;
                activationPrepared = false;
                continue;
            }
            if (ReadGmByPassDCache(ComputeFlag(args.gmm1Done, fragment.gmm1TileBegin + activationTile)) != 0) {
                if (idle) {
                    timer.TraceEnd(idleEvent, idleStart, idleMeta);
                    idle = false;
                }
                pack.Close(); // Activation reuses DMA event IDs.
                if (UseFusedGateUp(args.config)) {
                    RunReadyActivationTile<true>(args, worker, fragment, nextActivation, activationTile, epilogue, timer);
                } else {
                    RunReadyActivationTile<false>(args, worker, fragment, nextActivation, activationTile, epilogue, timer);
                }
                activationTile += activationWorkers;
                progress = true;
            }
            break;
        }
        if (!progress && nextGather < args.config.fragmentCount) {
            const auto fragment = LoadFragment(args, nextGather);
            if (AreGroupInputsReady(args, fragment)) {
                if (idle) {
                    timer.TraceEnd(idleEvent, idleStart, idleMeta);
                    idle = false;
                }
                pack.Close(); // Scale gather reuses DMA event IDs.
                GatherInputScale(args, fragment, nextGather, timer);
                nextGather += activationWorkers;
                progress = true;
            }
        }
        // Only fall back to Pack when neither activation nor a scale gather
        // ran. Consecutive Pack quotas retain events/cursor; close them only
        // before actual activation/gather work or when all Pack is complete.
        if (!progress && nextPackRound < args.config.packRoundCount) {
            if (idle) {
                timer.TraceEnd(idleEvent, idleStart, idleMeta);
                idle = false;
            }
            PackOneRound(args, nextPackRound, worker + communicationWorkers, kAic, pack);
            ++nextPackRound;
            progress = true;
        }
        if (!packFinished && nextPackRound >= args.config.packRoundCount) {
            pack.Close();
            MarkPackWorkerComplete(args);
            packFinished = true;
        }
        if (!progress && !shared.Done()) {
            pack.Close();
            SharedWaitScope before{timer,idle,idleStart,idleMeta,idleEvent};
            progress=shared.TryOne(before);
            if(!progress && nextGather>=args.config.fragmentCount && nextActivation>=args.config.fragmentCount) {
                const auto start=GetSystemCycle();while(GetSystemCycle()-start<64U){}
            }
        }
        if (!progress && (nextGather < args.config.fragmentCount || nextActivation < args.config.fragmentCount)) {
            if (!idle) {
                idleStart = timer.TraceBegin();
                idle = true;
                const bool waitingActivation = nextActivation < args.config.fragmentCount;
                const uint32_t index = waitingActivation ? nextActivation : nextGather;
                idleEvent = waitingActivation ? TraceEvent::WaitGmm1 : TraceEvent::WaitDispatch;
                idleMeta = FragmentTraceMeta(LoadFragment(args, index), index);
            }
            const uint64_t backoff = GetSystemCycle();
            while (GetSystemCycle() - backoff < 64U) {
            }
        }
    }
    if (idle) timer.TraceEnd(idleEvent, idleStart, idleMeta);
    pack.Close();
    // Empty compute workloads and <=2 Pack rounds can skip the loop entirely.
    if (!packFinished) MarkPackWorkerComplete(args);
}

} // namespace MegaMoeUrma
