#pragma once
#include "compute.hpp"
namespace MegaMoeUrma {
template<class T> struct SharedOutputBackend {
    const Arguments& a;T& timer;
    __aicore__ inline void Arm(){SetFlag<HardEvent::FIX_S>(EVENT_ID1);}
    __aicore__ inline void Complete(bool,uint64_t tile,const TraceMeta& m){
        const auto start=timer.TraceBegin();WaitFlag<HardEvent::FIX_S>(EVENT_ID1);
        timer.TraceEnd(TraceEvent::SharedGmm1Drain,start,m);
        WriteGmByPassDCache(ComputeFlag(a.sharedGmm1Done,tile),1);
    }
};
}
