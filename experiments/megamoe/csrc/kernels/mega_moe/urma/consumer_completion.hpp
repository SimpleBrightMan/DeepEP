// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "abi.hpp"
#include "owned_compute_cursor.hpp"
#include "compute_contract.hpp"
#include "lazy_compute_routes.hpp"
#include "timer.hpp"
#include "routed_fix_channel.hpp"

namespace MegaMoeUrma {
struct DeviceRoutePlanView {
    const Arguments& a;
    __aicore__ inline uint32_t TaskCount()const{return a.config.computeTaskCount;}
    __aicore__ inline ComputeBatchTask Task(uint32_t i)const {
        const auto* p=reinterpret_cast<const __gm__ ComputeBatchTask*>(a.computeTasks)+i;
        return {p->stage,p->batchIndex};
    }
    __aicore__ inline ComputeBatch Batch(uint32_t i)const {
        const auto* p=reinterpret_cast<const __gm__ ComputeBatch*>(a.computeBatches)+i;
        return {p->fragmentBegin,p->fragmentCount};
    }
    __aicore__ inline uint32_t Tiles(bool g1)const{return ComputeTileCount(kComputeTileM,g1?a.config.intermediate:a.config.hidden);}
    __aicore__ inline Fragment Group(uint32_t group)const{return RuntimeFragment(a,group);}
};
class OwnedRouteTileCursor {
public:
    __aicore__ OwnedRouteTileCursor(const Arguments& a,uint32_t core):plan_{a},sequence_(plan_,core){}
    __aicore__ inline bool Next(RoutedCompletionItem& item){return sequence_.Next(item);}
private:
    DeviceRoutePlanView plan_;OwnedTileSequence<DeviceRoutePlanView> sequence_;
};
__aicore__ inline __gm__ int32_t* RoutedIssued(const Arguments& a,uint32_t core) {
    return reinterpret_cast<__gm__ int32_t*>(a.gmmIssued+uint64_t(core)*kFlagBytes);
}

class RoutedCompletionProducer {
public:
    __aicore__ RoutedCompletionProducer(const Arguments& a,uint32_t core):a_(a),core_(core){}
    __aicore__ inline void Acquire(){RoutedFixChannel::Acquire(sequence_);}
    __aicore__ inline void Commit(){
        RoutedFixChannel::Publish(sequence_);
        // Issued only means the FIX full has been enqueued. The consumer must
        // wait for that hardware event before setting any compute-ready flag.
        AscendC::WriteGmByPassDCache(RoutedIssued(a_,core_),static_cast<int32_t>(++sequence_));
    }
    template<class TimerT> __aicore__ inline void Finish(TimerT& timer) {
        const auto start=timer.TraceBegin();RoutedFixChannel::DrainProducer();
        // AIC final completion remains visible in the unified GMM envelope.
        timer.TraceEnd(TraceEvent::Gmm2Drain,start);
    }
private:
    const Arguments& a_;uint32_t core_,sequence_=0U;
};

class RoutedCompletionConsumer {
public:
    __aicore__ RoutedCompletionConsumer(const Arguments& a,uint32_t core):a_(a),core_(core),cursor_(a,core){}
    __aicore__ inline void Init(){RoutedFixChannel::InitConsumer();}
    // Caller closes Pack/scale/Reorder DMA lifetimes before invoking this.
    // Route summary must be acquired before using cursor/Done/TryCompleteOne.
    __aicore__ inline bool Done(){Prepare();return exhausted_;}
    __aicore__ inline bool HasIssued(){
        Prepare();if(exhausted_)return false;
        if(consumed_>=issued_)issued_=static_cast<uint32_t>(AscendC::ReadGmByPassDCache(RoutedIssued(a_,core_)));
        return consumed_<issued_;
    }
    template<class TimerT> __aicore__ inline bool TryCompleteOne(RoutedCompletionItem& item,TimerT& timer) {
        if(!HasIssued())return false;
        const auto start=timer.TraceBegin();RoutedFixChannel::Wait(consumed_);
        item=next_;
        auto* done=reinterpret_cast<__gm__ int32_t*>((item.gmm1?a_.gmm1Done:a_.gmm2MatmulDone)+item.logicalTile*kFlagBytes);
        AscendC::WriteGmByPassDCache(done,1);
        if(!item.gmm1)AscendC::AtomicAdd(reinterpret_cast<__gm__ int32_t*>(a_.gmm2Done+uint64_t(item.group)*kFlagBytes),1);
        // Credits protect event reuse, not GM storage: every tile has its own
        // immutable output range, so activation/Reorder may execute later.
        RoutedFixChannel::Release(consumed_++);prepared_=false;
        TraceMeta m;m.fragment=item.group;m.expert=item.fragment.expert;
        m.row=item.fragment.rowBegin;m.rows=item.fragment.rows;m.tileN=uint64_t(item.tile)*256U;m.columns=256U;
        timer.TraceEnd(item.gmm1?TraceEvent::Gmm1TileDrain:TraceEvent::Gmm2Drain,start,m);
        return true;
    }
private:
    __aicore__ inline void Prepare(){if(!prepared_&&!exhausted_){exhausted_=!cursor_.Next(next_);prepared_=!exhausted_;}}
    const Arguments& a_;uint32_t core_,consumed_=0U,issued_=0U;
    OwnedRouteTileCursor cursor_;RoutedCompletionItem next_{};bool prepared_=false,exhausted_=false;
};
} // namespace MegaMoeUrma
