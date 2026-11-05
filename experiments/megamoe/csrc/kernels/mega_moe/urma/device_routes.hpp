// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "device_route_plan.hpp"
#include "transport.hpp"
#include "timer.hpp"
#include "kernel_operator.h"

namespace MegaMoeUrma {
using namespace AscendC;
constexpr uint32_t kRouteBlockItems = ((kMaxTokensPerRank * 8U + kLocalWorkers - 1U) / kLocalWorkers + 63U) / 64U * 64U;
constexpr uint32_t kRouteInfoUb = 96U * 1024U;
static_assert(kRouteInfoUb + sizeof(DeviceRouteInfo) < kRequiredUbBytes);

template<HardEvent E> __aicore__ inline void RouteFence() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
__aicore__ inline void RouteLoad(uint64_t gm, uint32_t ub, uint32_t bytes) {
    GlobalTensor<uint8_t> src; src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gm));
    LocalTensor<uint8_t> dst(TPosition::VECCALC,ub,bytes);
    DataCopyPad(dst,src,{1U,bytes,0U,0U,0U},{false,0U,0U,0U});
    RouteFence<HardEvent::MTE2_S>();
}
__aicore__ inline void RouteSave(uint64_t gm, uint32_t ub, uint32_t bytes) {
    if (!bytes) return;
    GlobalTensor<uint8_t> dst; dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gm));
    LocalTensor<uint8_t> src(TPosition::VECCALC,ub,bytes);
    RouteFence<HardEvent::S_MTE3>();
    DataCopyPad(dst,src,{1U,bytes,0U,0U,0U});
    RouteFence<HardEvent::MTE3_S>();
}
// Descriptor writes are contiguous metadata, never scalar token payload I/O.
// Each array has one writer; flush complete cache lines before the local join.
template<class T> __aicore__ inline void RouteStore(uint64_t address,uint32_t index,const T& value) {
    auto* dst=reinterpret_cast<__gm__ uint32_t*>(address+uint64_t(index)*sizeof(T));
    const auto* src=reinterpret_cast<const uint32_t*>(&value);
    for(uint32_t i=0;i<sizeof(T)/4U;++i)dst[i]=src[i];
}
__aicore__ inline void RouteFlush(uint64_t address,uint64_t bytes) {
    GlobalTensor<uint32_t> gm;gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    for(uint64_t p=0;p<bytes;p+=128U)
        DataCacheCleanAndInvalid<uint32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(gm[p/4U]);
}
struct DeviceRouteOutput {
    const Arguments& a;
    __aicore__ inline uint32_t Gmm1Tiles(uint32_t rows,const Config& c) {
#if MEGAMOE_URMA_ROUTE_REORDER
        return c.intermediate/256U;
#else
        return Gmm1ComputeTileCount(rows,c);
#endif
    }
    __aicore__ inline void FragmentAt(uint32_t i,const Fragment& v){RouteStore(a.fragments,i,v);}
    __aicore__ inline void Segment(uint32_t i,const RowSegment& v){RouteStore(a.segments,i,v);}
    __aicore__ inline void Send(uint32_t i,const Chunk& v){RouteStore(a.sendChunks,i,v);}
    __aicore__ inline void Receive(uint32_t i,const Chunk& v){RouteStore(a.recvChunks,i,v);}
    __aicore__ inline void Link(uint32_t i,uint32_t v){RouteStore(a.chunkGroupIndices,i,v);}
    __aicore__ inline void Batch(uint32_t i,const ComputeBatch& v){RouteStore(a.computeBatches,i,v);}
    __aicore__ inline void Task(uint32_t i,const ComputeBatchTask& v){RouteStore(a.computeTasks,i,v);}
    __aicore__ inline void PackChunk(uint32_t cv,uint32_t p,const Chunk& v){
        RouteStore(a.packRounds+uint64_t(cv)*sizeof(PackRound),p,v);
    }
    __aicore__ inline void PackRows(uint32_t cv,uint64_t v){
        RouteStore(a.packRounds+uint64_t(cv)*sizeof(PackRound)+kWorld*sizeof(Chunk),0,v);
    }
    __aicore__ inline void ReorderSpan(uint32_t i,uint64_t src,uint64_t dst,uint32_t group,uint32_t rows){
#if MEGAMOE_URMA_ROUTE_REORDER
        RouteStore(a.peerReorderSpans,i,PeerReorderSpan{src,dst,group,rows});
#endif
    }
    __aicore__ inline void ReorderRange(uint32_t i,uint32_t first,uint32_t count){
#if MEGAMOE_URMA_ROUTE_REORDER
        RouteStore(a.peerReorderRanges,i,PeerReorderRange{first,count});
#endif
    }
    __aicore__ inline void Finish(const Config& c,bool wave12) {
        RouteFlush(a.fragments,uint64_t(c.fragmentCount)*sizeof(Fragment));
        RouteFlush(a.segments,uint64_t(c.routeSegmentCount)*sizeof(RowSegment));
        RouteFlush(a.sendChunks,uint64_t(c.sendChunkCount)*sizeof(Chunk));
        RouteFlush(a.recvChunks,uint64_t(c.recvChunkCount)*sizeof(Chunk));
        RouteFlush(a.chunkGroupIndices,uint64_t(c.routeLinkCount)*sizeof(uint32_t));
        RouteFlush(a.computeBatches,uint64_t(wave12?(c.fragmentCount+11U)/12U:c.localExperts)*sizeof(ComputeBatch));
        RouteFlush(a.computeTasks,uint64_t(c.computeTaskCount)*sizeof(ComputeBatchTask));
        RouteFlush(a.packRounds,uint64_t(c.packRoundCount)*sizeof(PackRound));
#if MEGAMOE_URMA_ROUTE_REORDER
        RouteFlush(a.peerReorderRanges,uint64_t(c.recvChunkCount)*sizeof(PeerReorderRange));
        RouteFlush(a.peerReorderSpans,uint64_t(c.routeLinkCount)*sizeof(PeerReorderSpan));
#endif
        RouteStore(a.routeConfig,0,c);RouteFlush(a.routeConfig,sizeof(Config));
    }
};

// Two passes over a worker's bounded contiguous input slice. Vector compare /
// compact produces a stable per-expert list; only the selected local UB indices
// are handled by scalar. Token payloads and full topk tensors never leave Device.
__aicore__ inline void RouteLocalPermutation(const Arguments& a,uint32_t worker,bool emit) {
    constexpr uint32_t n=kRouteBlockItems;
    constexpr uint32_t idsAt=0U,positionsAt=4U*n,selectedAt=8U*n,maskAt=12U*n;
    constexpr uint32_t orderAt=16U*n,inverseAt=24U*n,dependenciesAt=32U*n;
    static_assert(40U*n<=kRouteInfoUb);
    const uint32_t total=a.config.bs*a.config.topk;
    const uint32_t begin=uint64_t(total)*worker/kLocalWorkers;
    const uint32_t end=uint64_t(total)*(worker+1U)/kLocalWorkers,valid=end-begin;
    LocalTensor<int32_t> ids(TPosition::VECCALC,idsAt,n),positions(TPosition::VECCALC,positionsAt,n);
    LocalTensor<int32_t> selected(TPosition::VECCALC,selectedAt,n);
    LocalTensor<uint8_t> mask(TPosition::VECCALC,maskAt,n*4U);
    LocalTensor<uint64_t> ordered(TPosition::VECCALC,orderAt,n),inverse(TPosition::VECCALC,inverseAt,n);
    LocalTensor<uint64_t> dependencies(TPosition::VECCALC,dependenciesAt,n);
    if(valid) {
        if(a.topkIdBytes==4U)RouteLoad(a.topkIds+uint64_t(begin)*4U,idsAt,valid*4U);
        else {
            // Legal IDs fit in 32 bits. Compact low words, without a tensor
            // cast kernel or scalar loads from GM.
            RouteLoad(a.topkIds+uint64_t(begin)*8U,inverseAt,valid*8U);
            LocalTensor<int32_t> raw(TPosition::VECCALC,inverseAt,2U*n);
            auto bits=mask.ReinterpretCast<uint32_t>();
            for(uint32_t i=0;i<(2U*valid+31U)/32U;++i)bits.SetValue(i,0x55555555U);
            RouteFence<HardEvent::S_V>();
            uint64_t count=0;
            GatherMask(ids,raw,bits,true,2U*valid,{1,1,0,0},count);
            RouteFence<HardEvent::V_S>();
        }
        RouteFence<HardEvent::S_V>();
        CreateVecIndex(positions,static_cast<int32_t>(begin),valid);
    }
    auto* hist=reinterpret_cast<__gm__ uint32_t*>(a.routeHistogram)+worker*kExperts;
    auto* info=reinterpret_cast<__ubuf__ DeviceRouteInfo*>(static_cast<uint64_t>(kRouteInfoUb));
    if(emit)RouteLoad(a.routeInfo,kRouteInfoUb,sizeof(DeviceRouteInfo));
    for(uint32_t e=0;e<kExperts;++e) {
        uint64_t count=0U;
        if(valid) {
            CompareScalar(mask,ids,static_cast<int32_t>(e),CMPMODE::EQ,valid);
            GatherMask(selected,positions,mask.ReinterpretCast<uint32_t>(),true,valid,{1,1,0,0},count);
            RouteFence<HardEvent::V_S>();
        }
        if(!emit)hist[e]=count;
        else if(count) {
            const uint32_t offset=ReadGmByPassDCache(hist+e);
            const uint32_t peer=e/a.config.localExperts;
            const uint32_t peerBase=info->outgoing[peer*a.config.localExperts];
            for(uint32_t i=0;i<count;++i) {
                const uint32_t flat=selected.GetValue(i),expanded=offset+i;
                ordered.SetValue(i,flat);inverse.SetValue(flat-begin,expanded);
                uint64_t dep=uint64_t(peer)*a.config.maxPeerChunks+(expanded-peerBase)/a.config.cvRows;
#if !MEGAMOE_URMA_ROUTE_REORDER
                if(peer==a.config.rank) {
                    const uint32_t local=e%a.config.localExperts;
                    const uint32_t expertRow=info->sources[local][peer]+expanded-info->outgoing[e];
                    dep=kReduceSelfDependency|(info->groupBase[local]+expertRow/256U);
                }
#endif
                dependencies.SetValue(flat-begin,dep);
            }
            RouteSave(a.order+uint64_t(offset)*8U,orderAt,count*8U);
        }
    }
    if(emit) {
        RouteSave(a.reduceIndex+uint64_t(begin)*8U,inverseAt,valid*8U);
        RouteSave(a.reduceDependencies+uint64_t(begin)*8U,dependenciesAt,valid*8U);
    } else RouteFlush(a.routeHistogram+worker*kExperts*4U,kExperts*4U);
}

__aicore__ inline void RoutePrefix(const Arguments& a) {
    // The 56 block histograms fit in UB. No scalar token loop or GM atomic
    // histogram contention; only small count/prefix metadata is processed here.
    RouteLoad(a.routeHistogram,0U,kRouteHistogramBytes);
    LocalTensor<int32_t> histogram(TPosition::VECCALC,0U,kLocalWorkers*kExperts);
    auto* r=reinterpret_cast<__ubuf__ DeviceRouteInfo*>(static_cast<uint64_t>(kRouteInfoUb));
    RouteFence<HardEvent::S_V>();
    for(uint32_t w=1U;w<kLocalWorkers;++w)
        Add(histogram[w*kExperts],histogram[w*kExperts],histogram[(w-1U)*kExperts],kExperts);
    RouteFence<HardEvent::V_S>();
    LocalTensor<int32_t> prefix(TPosition::VECCALC,kRouteHistogramBytes,kExperts);
    uint32_t total=0U;
    for(uint32_t e=0;e<kExperts;++e) {
        r->outgoing[e]=total;
        const uint32_t first=total;
        prefix.SetValue(e,total);
        total+=histogram.GetValue((kLocalWorkers-1U)*kExperts+e);
        RouteStore(a.routeCounts,e,uint64_t(total-first));
        RouteStore(a.routeOffsets,e,uint64_t(first));
    }
    r->outgoing[kExperts]=total;RouteStore(a.routeOffsets,kExperts,uint64_t(total));
    RouteFlush(a.routeCounts,kExperts*8U);RouteFlush(a.routeOffsets,(kExperts+1U)*8U);
    RouteFence<HardEvent::S_V>();
    for(uint32_t w=kLocalWorkers-1U;w>0U;--w)
        Add(histogram[w*kExperts],histogram[(w-1U)*kExperts],prefix,kExperts);
    Adds(histogram,prefix,int32_t{0},kExperts);
    RouteFence<HardEvent::V_S>();
    RouteSave(a.routeHistogram,0U,kRouteHistogramBytes);
    RouteSave(a.routeInfo,kRouteInfoUb,(kExperts+1U)*4U);
}

__aicore__ inline void RouteExchange(const Arguments& a,uint32_t worker) {
    if(worker>=a.config.world)return;
    RouteLoad(a.routeInfo,kRouteInfoUb,(kExperts+1U)*4U);
    const auto* r=reinterpret_cast<__ubuf__ DeviceRouteInfo*>(static_cast<uint64_t>(kRouteInfoUb));
    const uint32_t le=a.config.localExperts,base=r->outgoing[worker*le];
    LocalTensor<uint64_t> ret(TPosition::VECCALC,0U,4U);ret.SetValue(0,base);
    LocalTensor<uint32_t> counts(TPosition::VECCALC,8U,le+1U);
    for(uint32_t e=0;e<=le;++e)counts.SetValue(e,r->outgoing[worker*le+e]-base);
    auto* sym=reinterpret_cast<__gm__ uint8_t*>(a.symmetric);
    const uint64_t target=reinterpret_cast<uint64_t>(aclshmem_ptr(
        sym+a.config.routeInboxOffset+uint64_t(a.config.rank)*kRouteMailboxBytes,worker));
    RouteSave(target,0U,8U+(le+1U)*4U);
    LocalTensor<uint64_t> flag(TPosition::VECCALC,1024U,4U);
    PublishDone(sym,a.config.routeReadyOffset+uint64_t(a.config.rank)*kFlagBytes,worker,a.config.epoch,flag,EVENT_ID0);
}

__aicore__ inline void RouteCompile(Arguments& a) {
    RouteLoad(a.routeInfo,kRouteInfoUb,(kExperts+1U)*4U);
    RouteLoad(a.symmetric+a.config.routeInboxOffset,0U,a.config.world*kRouteMailboxBytes);
    auto* r=reinterpret_cast<__ubuf__ DeviceRouteInfo*>(static_cast<uint64_t>(kRouteInfoUb));
    LocalTensor<uint32_t> inbox(TPosition::VECCALC,0U,a.config.world*kRouteMailboxBytes/4U);
    for(uint32_t s=0;s<a.config.world;++s) {
        const uint32_t at=s*kRouteMailboxBytes/4U;
        r->returnBase[s]=uint64_t(inbox.GetValue(at))|(uint64_t(inbox.GetValue(at+1U))<<32U);
        for(uint32_t e=0;e<=a.config.localExperts;++e)r->incoming[s][e]=inbox.GetValue(at+2U+e);
    }
    DeviceRouteOutput out{a};
    const bool wave12=a.config.world==4U && (!MEGAMOE_URMA_ROUTE_REORDER || a.sharedExperts!=0U);
    CompileDeviceRoutes(a.config,*r,out,wave12,MEGAMOE_URMA_ROUTE_REORDER);
    RouteSave(a.routeInfo,kRouteInfoUb,sizeof(DeviceRouteInfo));
    out.Finish(a.config,wave12);
}

__aicore__ inline void PrepareDeviceRoutes(Arguments& a,Timer& timer) {
    timer.CoarseBegin(CoarsePhase::Route);
    const uint64_t start=timer.TraceBegin();
    const uint32_t block=GetBlockIdx()/GetTaskRation();
    if constexpr(g_coreType==AIV) {
        const uint32_t worker=LocalWorkerIndex(block,GetSubBlockIdx());
        // A next invocation may overwrite mailboxes only after every rank has
        // left its previous invocation. Separate monotonic epoch slots survive
        // the data-path's per-invocation SO resets, including empty peer edges.
        if(worker==0U) {
            LocalTensor<uint64_t> flag(TPosition::VECCALC,0U,4U);
            RankBarrier(reinterpret_cast<__gm__ uint8_t*>(a.symmetric),a.config.routeEntryOffset,
                        a.config.rank,a.config.epoch,flag,EVENT_ID0,a.config.world);
        }
    }
    SyncAll<false>();
    if constexpr(g_coreType==AIV)RouteLocalPermutation(a,LocalWorkerIndex(block,GetSubBlockIdx()),false);
    SyncAll<false>();
    if constexpr(g_coreType==AIV)if(block==0U&&GetSubBlockIdx()==0U)RoutePrefix(a);
    SyncAll<false>();
    if constexpr(g_coreType==AIV)RouteExchange(a,LocalWorkerIndex(block,GetSubBlockIdx()));
    SyncAll<false>();
    if constexpr(g_coreType==AIV)if(block==0U&&GetSubBlockIdx()==0U) {
        for(uint32_t s=0;s<a.config.world;++s)
            WaitStartAtLeast(reinterpret_cast<__gm__ uint8_t*>(a.symmetric+a.config.routeReadyOffset+s*kFlagBytes),a.config.epoch);
        RouteCompile(a);
    }
    SyncAll<false>();
    // AIC also consumes the exact same Device-generated descriptor bounds.
    auto* dst=reinterpret_cast<uint32_t*>(&a.config);
    auto* src=reinterpret_cast<__gm__ uint32_t*>(a.routeConfig);
    for(uint32_t i=0;i<sizeof(Config)/4U;++i)dst[i]=ReadGmByPassDCache(src+i);
    if constexpr(g_coreType==AIV)RouteLocalPermutation(a,LocalWorkerIndex(block,GetSubBlockIdx()),true);
    PipeBarrier<PIPE_ALL>();
    SyncAll<false>();
    timer.TraceEnd(TraceEvent::RoutePrepare,start);
    timer.CoarseEnd(CoarsePhase::Route);
}
} // namespace MegaMoeUrma
