// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "device_routes.hpp"
#include "pack_round.hpp"

namespace MegaMoeUrma {
__aicore__ inline __gm__ int32_t* RouteSummaryFlag(const Arguments& a) {
    return reinterpret_cast<__gm__ int32_t*>(a.localFlags+kLocalFlagRouteSummaryDone*kFlagBytes);
}
__aicore__ inline bool RouteSummaryReady(const Arguments& a) {
    return AscendC::ReadGmByPassDCache(RouteSummaryFlag(a))==1;
}
__aicore__ inline void AcquireRouteSummary(Arguments& a) {
    // The producer never changes outgoing's cache lines. Only invalidate the
    // newly published receive portion, including lines fetched speculatively.
    constexpr uint32_t offset=offsetof(DeviceRouteInfo,incoming);
    RouteFlush(a.routeInfo+offset,sizeof(DeviceRouteInfo)-offset);
    RouteReloadConfig(a);
    RouteFence<AscendC::HardEvent::S_MTE2>();
}
__aicore__ inline void WaitRouteSummary(Arguments& a,Timer& timer) {
    timer.CoarseBegin(CoarsePhase::RouteWait);
    const uint64_t start=timer.TraceBegin();
    while(!RouteSummaryReady(a)) {
        const uint64_t tick=AscendC::GetSystemCycle();
        while(AscendC::GetSystemCycle()-tick<64U){}
    }
    AcquireRouteSummary(a);
    timer.TraceEnd(TraceEvent::WaitRouteSummary,start);
    timer.CoarseEnd(CoarsePhase::RouteWait);
}

// Only helper0 calls this. Cache positive observations and return immediately
// at the first missing source so it can resume its static Pack quota.
__aicore__ inline bool TryRouteInboxReady(const Arguments& a,uint32_t& readyMask) {
    for(uint32_t peer=0;peer<a.config.world;++peer) {
        const uint32_t bit=uint32_t{1}<<peer;
        if(readyMask&bit)continue;
        auto* flag=reinterpret_cast<__gm__ uint64_t*>(a.symmetric+a.config.routeReadyOffset+peer*kFlagBytes);
        if(AscendC::ReadGmByPassDCache(flag)!=a.config.epoch)return false;
        readyMask|=bit;
    }
    return true;
}

// Called after the existing startup quotas. Pack assignments never change:
// this loop advances the SAME nextPackRound cursor used by normal activation.
// It releases low UB and all Pack events before the owner builds the summary.
__aicore__ inline void PrepareActivationRoutes(Arguments& a,uint32_t helper,
    uint32_t& nextPackRound,PackBatch& pack,Timer& timer)
{
    const uint32_t packOwner=helper+CommunicationWorkers(a.config.world);
    uint32_t readyMask=0U;
    while(!RouteSummaryReady(a)) {
        if(helper==0U && TryRouteInboxReady(a,readyMask)) {
            pack.Close();
            timer.CoarseBegin(CoarsePhase::RouteSummary);
            const uint64_t start=timer.TraceBegin();
            RouteSummary(a);
            // Complete table DMA and Config publication before the local
            // release flag. No all-core synchronization is introduced.
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::WriteGmByPassDCache(RouteSummaryFlag(a),int32_t{1});
            timer.TraceEnd(TraceEvent::RouteSummary,start);
            timer.CoarseEnd(CoarsePhase::RouteSummary);
            break;
        }
        if(nextPackRound<a.config.packRoundCount) {
            PackOneRound(a,nextPackRound++,packOwner,kAic,pack);
        } else {
            pack.Close();
            const uint64_t tick=AscendC::GetSystemCycle();
            while(AscendC::GetSystemCycle()-tick<64U){}
        }
    }
    // No UB payload is touched here; non-owner helpers may retain their Pack
    // ring until the first actual activation/scale-gather task closes it.
    WaitRouteSummary(a,timer);
}
} // namespace MegaMoeUrma
