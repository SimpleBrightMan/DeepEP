// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "reorder.hpp"
#include <c_api/asc_simd.h>

namespace MegaMoeUrma {
static_assert(120U*1024U+kComputeTileM*kComputeTileN*sizeof(uint16_t)<=kRequiredUbBytes);
// AIV0 owns GMM2's result mailbox; GMM1 still targets AIV1. B supply uses
// flags4..11, context retirement2, GMM1/AIV1 uses12/14. These use13/15.
struct Aiv0Gmm2Channel {
    static constexpr uint32_t Full=13U,Empty=15U;
    __aicore__ static inline void Open(){asc_sync_intra_arrive(PIPE_MTE3,Empty);}
    __aicore__ static inline void Acquire(){asc_sync_intra_wait(PIPE_FIX,Empty);}
    __aicore__ static inline void Publish(){asc_sync_intra_arrive(PIPE_FIX,Full);}
    __aicore__ static inline void Wait(){
        asc_sync_intra_wait(PIPE_MTE2,Full);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(static_cast<AscendC::TEventID>(0));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(static_cast<AscendC::TEventID>(0));
    }
    __aicore__ static inline void Release(){asc_sync_intra_arrive(PIPE_MTE3,Empty);}
};

template<class TimerT>
class Aiv0Reorder {
public:
    __aicore__ Aiv0Reorder(const Arguments& a,TimerT& t):a_(a),timer_(t){}
    // Only call after the prologue shrank/drained its old GMM1 UB layout.
    __aicore__ inline void BeginStage(){if(!active_){Aiv0Gmm2Channel::Open();active_=true;}}
    __aicore__ inline void AfterSupply(const Fragment& f,uint32_t group,uint32_t tile){
        // One tile of weight supply ahead: never try to supply a third while
        // the first result owns UB and the second result occupies L0C.
        if(pending_)Consume(true);
        fragment_=f;group_=group;tile_=tile;pending_=true;
    }
    __aicore__ inline void EndStage(){
        if(pending_)Consume(false);
        // Do NOT leave an EMPTY behind: GMM1 may now reuse this UB region.
        // The next GMM2 stage grants a new credit only after shrinking again.
        active_=false;
    }
private:
    __aicore__ inline void Consume(bool release){
        const auto start=timer_.TraceBegin();Aiv0Gmm2Channel::Wait();
        TraceMeta meta;meta.fragment=group_;meta.expert=fragment_.expert;
        meta.row=fragment_.rowBegin;meta.rows=fragment_.rows;meta.tileN=tile_*256U;meta.columns=256U;
        timer_.TraceEnd(TraceEvent::Gmm2Drain,start,meta);
        ReorderGmm2Tile<120U*1024U>(a_,fragment_,group_,tile_,timer_);
        if(release)Aiv0Gmm2Channel::Release();
        pending_=false;
    }
    const Arguments& a_;TimerT& timer_;Fragment fragment_{};
    uint32_t group_=0U,tile_=0U;bool active_=false,pending_=false;
};
} // namespace MegaMoeUrma
