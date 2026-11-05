// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once
#include "device_route_plan.hpp"
#include "parallel_route_plan.hpp"
#include "transport.hpp"
#include "timer.hpp"
#include "kernel_operator.h"
#include "route_compact_vf.hpp"
#include "route_emit_vf.hpp"

namespace MegaMoeUrma {
using namespace AscendC;
constexpr uint32_t kRouteBlockItems = ((kMaxTokensPerRank * 8U + kLocalWorkers - 1U) / kLocalWorkers + 63U) / 64U * 64U;
constexpr uint32_t kRouteInfoUb = 96U * 1024U;
static_assert(kRouteInfoUb + sizeof(DeviceRouteInfo) < kRequiredUbBytes);

// This region survives Prefix (low 85.5KiB), Exchange (low 1KiB), and
// Compile (low inbox plus RouteInfo at 96KiB). It is released before Pack.
// For n=2368: CSR 128..137.25KiB, counts 137.25..138.75KiB,
// begin 138.75..140.254KiB, padded order64 140.28125..167.78125KiB.
struct RouteCsrUb {
    static constexpr uint32_t n = kRouteBlockItems;
    static constexpr uint32_t csrAt = 128U * 1024U;
    static constexpr uint32_t countsAt = csrAt + n * sizeof(uint32_t);
    static constexpr uint32_t beginAt = countsAt + kExperts * sizeof(uint32_t);
    static constexpr uint32_t orderAt = (beginAt + (kExperts + 1U) * sizeof(uint32_t) + 31U) / 32U * 32U;
    // Each expert's int64 source starts on a 32B boundary. At most three
    // padding entries per expert, even for fully skewed/duplicate routing.
    static constexpr uint32_t orderItems = n + 3U * kExperts;
    static constexpr uint32_t end = orderAt + orderItems * sizeof(uint64_t);
    static constexpr uint32_t inverseAt = 0U;
    static constexpr uint32_t dependenciesAt = n * sizeof(uint64_t);
    static constexpr uint32_t workerBaseAt = 2U * n * sizeof(uint64_t);
};
static_assert(kRouteInfoUb + sizeof(DeviceRouteInfo) <= RouteCsrUb::csrAt);
static_assert(RouteCsrUb::end <= kRequiredUbBytes);
static_assert(RouteCsrUb::workerBaseAt + kExperts * sizeof(uint32_t) <= kRouteInfoUb);

struct RouteEmitUb {
    static constexpr uint32_t keptIdsAt=RouteCsrUb::end; // 167.78125KiB
    static constexpr uint32_t paddedBeginAt=(keptIdsAt+kRouteBlockItems*sizeof(uint32_t)+31U)/32U*32U;
    static constexpr uint32_t end=paddedBeginAt+kExperts*sizeof(uint32_t); // 178.53125KiB
};
static_assert(kExperts==kRouteEmitExperts);
static_assert(RouteEmitUb::keptIdsAt%32U==0U && RouteEmitUb::paddedBeginAt%32U==0U);
static_assert(RouteEmitUb::end<=kRequiredUbBytes);

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
// Caller provides one S_MTE3 edge before the batch and one MTE3_S edge after
// it. Every UB source remains immutable until that final completion edge.
__aicore__ inline void RouteEnqueueSave(uint64_t gm,uint32_t ub,uint32_t bytes) {
    if(bytes==0U)return;
    GlobalTensor<uint8_t> dst;dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gm));
    LocalTensor<uint8_t> src(TPosition::VECCALC,ub,bytes);
    DataCopyPad(dst,src,{1U,bytes,0U,0U,0U});
}
// Descriptor writes are contiguous metadata, never scalar token payload I/O.
// Each array has one writer; flush complete cache lines before the local join.
template<class T> __aicore__ inline void RouteStore(uint64_t address,uint32_t index,const T& value) {
    auto* dst=reinterpret_cast<__gm__ uint32_t*>(address+uint64_t(index)*sizeof(T));
    const auto* src=reinterpret_cast<const uint32_t*>(&value);
    for(uint32_t i=0;i<sizeof(T)/4U;++i)dst[i]=src[i];
}
__aicore__ inline void RouteFlush(uint64_t address,uint64_t bytes) {
    // DCache lines are 64B on Ascend 950; a 128B signal/mailbox slot is NOT
    // the cache-line size. Missing alternate lines leaves dirty count stores
    // alive across the DMA rewrite to prefixes and corrupts route addresses.
    if(bytes==0U)return;
    constexpr uint64_t lineBytes=64U;
    for(uint64_t p=address&~(lineBytes-1U);p<address+bytes;p+=lineBytes) {
        GlobalTensor<uint32_t> line;line.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(p));
        DataCacheCleanAndInvalid<uint32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(line);
    }
}
// Compile has already copied the inbox into RouteInfo before constructing
// this writer. The low 40KiB is dead scratch until emit; RouteInfo at 96KiB
// and the live CSR at 128KiB+ are untouched.
struct RouteDescriptorUb {
    static constexpr uint32_t stride = 4096U;
    // 3840 is divisible by 32 and every ordinary descriptor size (4/8/24/40).
    // Flushes never split a typed record, and every full DMA is 32B aligned.
    static constexpr uint32_t batchBytes = 3840U;
    // PackChunk x16 + PackRows = one 648B PackRound. Four complete rounds are
    // 2592B, divisible by 32; the mixed 40B/8B append stream cannot overflow.
    static constexpr uint32_t packBatchBytes = 4U * sizeof(PackRound);
    static constexpr uint32_t end = 10U * stride;
};
static_assert(RouteDescriptorUb::batchBytes % 32U == 0U);
static_assert(RouteDescriptorUb::batchBytes % sizeof(Fragment) == 0U);
static_assert(RouteDescriptorUb::batchBytes % sizeof(RowSegment) == 0U);
static_assert(RouteDescriptorUb::batchBytes % sizeof(Chunk) == 0U);
static_assert(RouteDescriptorUb::batchBytes % sizeof(ComputeBatch) == 0U);
static_assert(RouteDescriptorUb::batchBytes % sizeof(ComputeBatchTask) == 0U);
static_assert(RouteDescriptorUb::packBatchBytes % 32U == 0U);
static_assert(RouteDescriptorUb::packBatchBytes <= RouteDescriptorUb::stride);
static_assert(RouteDescriptorUb::end <= kRouteInfoUb);
#if MEGAMOE_URMA_ROUTE_REORDER
static_assert(RouteDescriptorUb::batchBytes % sizeof(PeerReorderSpan) == 0U);
static_assert(RouteDescriptorUb::batchBytes % sizeof(PeerReorderRange) == 0U);
#endif

template<uint32_t UbOffset, uint32_t CapacityBytes = RouteDescriptorUb::batchBytes>
class RouteDescriptorDmaStream {
public:
    __aicore__ explicit RouteDescriptorDmaStream(uint64_t address) : address_(address) {}

    // CompileDeviceRoutes writes each table in increasing, contiguous byte
    // order. Keep the supplied offset for a buffer's first record; no hidden
    // producer-side read, atomic append, lookup or descriptor reorder.
    template<class T>
    __aicore__ inline void Append(uint64_t byteOffset,const T& value) {
        static_assert(sizeof(T) % sizeof(uint32_t) == 0U);
        static_assert(UbOffset % 32U == 0U && CapacityBytes % 32U == 0U);
        if(usedBytes_ == 0U) firstByte_ = byteOffset;
        LocalTensor<uint32_t> buffer(TPosition::VECCALC,UbOffset,CapacityBytes/4U);
        const auto* words = reinterpret_cast<const uint32_t*>(&value);
        for(uint32_t word = 0U; word < sizeof(T)/4U; ++word)
            buffer.SetValue(usedBytes_/4U + word,words[word]);
        usedBytes_ += sizeof(T);
        if(usedBytes_ == CapacityBytes) Flush();
    }

    __aicore__ inline void Flush() {
        if(usedBytes_ == 0U) return;
        // RouteSave does S_MTE3 -> DataCopyPad -> MTE3_S. The buffer is only
        // reused after that completion edge; at most one DMA is outstanding.
        RouteSave(address_ + firstByte_,UbOffset,usedBytes_);
        usedBytes_ = 0U;
    }

private:
    uint64_t address_;
    uint64_t firstByte_ = 0U;
    uint32_t usedBytes_ = 0U;
};

struct DeviceRouteOutput {
    const Arguments& a;
    RouteDescriptorDmaStream<0U * RouteDescriptorUb::stride> fragments;
    RouteDescriptorDmaStream<1U * RouteDescriptorUb::stride> segments;
    RouteDescriptorDmaStream<2U * RouteDescriptorUb::stride> sends;
    RouteDescriptorDmaStream<3U * RouteDescriptorUb::stride> receives;
    RouteDescriptorDmaStream<4U * RouteDescriptorUb::stride> links;
    RouteDescriptorDmaStream<5U * RouteDescriptorUb::stride> batches;
    RouteDescriptorDmaStream<6U * RouteDescriptorUb::stride> tasks;
    RouteDescriptorDmaStream<7U * RouteDescriptorUb::stride,RouteDescriptorUb::packBatchBytes> pack;
#if MEGAMOE_URMA_ROUTE_REORDER
    RouteDescriptorDmaStream<8U * RouteDescriptorUb::stride> reorderRanges;
    RouteDescriptorDmaStream<9U * RouteDescriptorUb::stride> reorderSpans;
#endif

    __aicore__ explicit DeviceRouteOutput(const Arguments& args)
        : a(args), fragments(args.fragments), segments(args.segments),
          sends(args.sendChunks), receives(args.recvChunks), links(args.chunkGroupIndices),
          batches(args.computeBatches), tasks(args.computeTasks), pack(args.packRounds)
#if MEGAMOE_URMA_ROUTE_REORDER
          , reorderRanges(args.peerReorderRanges), reorderSpans(args.peerReorderSpans)
#endif
    {}

    __aicore__ inline uint32_t Gmm1Tiles(uint32_t rows,const Config& c) {
#if MEGAMOE_URMA_ROUTE_REORDER
        return c.intermediate/256U;
#else
        return Gmm1ComputeTileCount(rows,c);
#endif
    }
    __aicore__ inline void FragmentAt(uint32_t i,const Fragment& v){fragments.Append(uint64_t(i)*sizeof(v),v);}
    __aicore__ inline void Segment(uint32_t i,const RowSegment& v){segments.Append(uint64_t(i)*sizeof(v),v);}
    __aicore__ inline void Send(uint32_t i,const Chunk& v){sends.Append(uint64_t(i)*sizeof(v),v);}
    __aicore__ inline void Receive(uint32_t i,const Chunk& v){receives.Append(uint64_t(i)*sizeof(v),v);}
    __aicore__ inline void Link(uint32_t i,uint32_t v){links.Append(uint64_t(i)*sizeof(v),v);}
    __aicore__ inline void Batch(uint32_t i,const ComputeBatch& v){batches.Append(uint64_t(i)*sizeof(v),v);}
    __aicore__ inline void Task(uint32_t i,const ComputeBatchTask& v){tasks.Append(uint64_t(i)*sizeof(v),v);}
    __aicore__ inline void PackChunk(uint32_t cv,uint32_t peer,const Chunk& v){
        pack.Append(uint64_t(cv)*sizeof(PackRound)+uint64_t(peer)*sizeof(Chunk),v);
    }
    __aicore__ inline void PackRows(uint32_t cv,uint64_t v){
        pack.Append(uint64_t(cv)*sizeof(PackRound)+kWorld*sizeof(Chunk),v);
    }
    __aicore__ inline void ReorderSpan(uint32_t i,uint64_t src,uint64_t dst,uint32_t group,uint32_t rows){
#if MEGAMOE_URMA_ROUTE_REORDER
        const PeerReorderSpan value{src,dst,group,rows};
        reorderSpans.Append(uint64_t(i)*sizeof(value),value);
#endif
    }
    __aicore__ inline void ReorderRange(uint32_t i,uint32_t first,uint32_t count){
#if MEGAMOE_URMA_ROUTE_REORDER
        const PeerReorderRange value{first,count};
        reorderRanges.Append(uint64_t(i)*sizeof(value),value);
#endif
    }
    __aicore__ inline void FlushTables() {
        fragments.Flush();segments.Flush();sends.Flush();receives.Flush();
        links.Flush();batches.Flush();tasks.Flush();pack.Flush();
#if MEGAMOE_URMA_ROUTE_REORDER
        reorderRanges.Flush();reorderSpans.Flush();
#endif
    }
    __aicore__ inline void Finish(const Config& c,bool wave12) {
        (void)wave12;FlushTables();
        // Publish the dynamic config only after every descriptor DMA has
        // completed. Config itself retains the established Scalar/DCCI path.
        RouteStore(a.routeConfig,0,c);RouteFlush(a.routeConfig,sizeof(Config));
    }
};

// One input load and one VF expert-filtering pass. The first invocation leaves
// a compact stable expert list in high UB; the emit invocation consumes that
// list after worker prefixes and incoming metadata have been prepared.
__aicore__ inline void RouteLocalPermutation(const Arguments& a,uint32_t worker,bool emit) {
    constexpr uint32_t n=kRouteBlockItems;
    const uint32_t total=a.config.bs*a.config.topk;
    const uint32_t begin=uint64_t(total)*worker/kLocalWorkers;
    const uint32_t end=uint64_t(total)*(worker+1U)/kLocalWorkers,valid=end-begin;
    const bool vectorEmit=a.config.cvRows!=0U && (a.config.cvRows&(a.config.cvRows-1U))==0U;
    LocalTensor<uint32_t> csr(TPosition::VECCALC,RouteCsrUb::csrAt,n);
    LocalTensor<uint32_t> counts(TPosition::VECCALC,RouteCsrUb::countsAt,kExperts);
    LocalTensor<uint32_t> starts(TPosition::VECCALC,RouteCsrUb::beginAt,kExperts+1U);
    if(!emit) {
        constexpr uint32_t idsAt=0U,maskAt=12U*n,rawAt=24U*n;
        static_assert(rawAt+8U*n<=kRouteInfoUb);
        LocalTensor<int32_t> ids(TPosition::VECCALC,idsAt,n);
        LocalTensor<uint8_t> mask(TPosition::VECCALC,maskAt,n*4U);
        if(valid) {
            if(a.topkIdBytes==4U)RouteLoad(a.topkIds+uint64_t(begin)*4U,idsAt,valid*4U);
            else {
                RouteLoad(a.topkIds+uint64_t(begin)*8U,rawAt,valid*8U);
                LocalTensor<int32_t> raw(TPosition::VECCALC,rawAt,2U*n);
                auto bits=mask.ReinterpretCast<uint32_t>();
                for(uint32_t i=0;i<(2U*valid+31U)/32U;++i)bits.SetValue(i,0x55555555U);
                RouteFence<HardEvent::S_V>();
                uint64_t count=0U;
                GatherMask(ids,raw,bits,true,2U*valid,{1,1,0,0},count);
                RouteFence<HardEvent::V_S>();
            }
        }
        // The VF creates global flat indices internally and appends exactly
        // 'valid' entries across all expert buckets. Even empty input invokes
        // it so all 384 counts are defined as zero for this invocation.
        RouteFence<HardEvent::S_V>();
        if(vectorEmit && valid) {
            LocalTensor<int32_t> keptIds(TPosition::VECCALC,RouteEmitUb::keptIdsAt,n);
            Adds(keptIds,ids,int32_t{0},valid);
        }
        asc_vf_call<RouteCompactCsrVF>(
            reinterpret_cast<__ubuf__ int32_t*>(ids.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(csr.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int32_t*>(counts.GetPhyAddr()),valid,begin);
        RouteFence<HardEvent::V_S>();
        uint32_t cursor=0U;
        if(vectorEmit) {
            LocalTensor<uint32_t> paddedBegin(TPosition::VECCALC,RouteEmitUb::paddedBeginAt,kExperts);
            uint32_t paddedCursor=0U;
            for(uint32_t e=0;e<kExperts;++e) {
                paddedCursor=(paddedCursor+3U)&~3U;
                starts.SetValue(e,cursor);paddedBegin.SetValue(e,paddedCursor);
                const uint32_t count=counts.GetValue(e);
                cursor+=count;paddedCursor+=count;
            }
        } else {
            for(uint32_t e=0;e<kExperts;++e) {
                starts.SetValue(e,cursor);
                cursor+=counts.GetValue(e);
            }
        }
        starts.SetValue(kExperts,cursor);
        // One aligned DMA publishes the whole count row. No dirty Scalar GM
        // histogram lines survive into the coordinator's prefix rewrite.
        RouteSave(a.routeHistogram+uint64_t(worker)*kExperts*sizeof(uint32_t),
                  RouteCsrUb::countsAt,kExperts*sizeof(uint32_t));
        return;
    }

    RouteLoad(a.routeInfo,kRouteInfoUb,sizeof(DeviceRouteInfo));
    RouteLoad(a.routeHistogram+uint64_t(worker)*kExperts*sizeof(uint32_t),
              RouteCsrUb::workerBaseAt,kExperts*sizeof(uint32_t));
    auto* info=reinterpret_cast<__ubuf__ DeviceRouteInfo*>(static_cast<uint64_t>(kRouteInfoUb));
    LocalTensor<uint32_t> workerBase(TPosition::VECCALC,RouteCsrUb::workerBaseAt,kExperts);
    LocalTensor<uint64_t> ordered(TPosition::VECCALC,RouteCsrUb::orderAt,RouteCsrUb::orderItems);
    LocalTensor<uint64_t> inverse(TPosition::VECCALC,RouteCsrUb::inverseAt,n);
    LocalTensor<uint64_t> dependencies(TPosition::VECCALC,RouteCsrUb::dependenciesAt,n);
    uint32_t orderCursor=0U;
    if(vectorEmit) {
        uint32_t cvShift=0U;
        for(uint32_t rows=a.config.cvRows;rows>1U;rows>>=1U)++cvShift;
        LocalTensor<uint32_t> keptIds(TPosition::VECCALC,RouteEmitUb::keptIdsAt,n);
        LocalTensor<uint32_t> paddedBegin(TPosition::VECCALC,RouteEmitUb::paddedBeginAt,kExperts);
        LocalTensor<uint32_t> routeInfo(TPosition::VECCALC,kRouteInfoUb,sizeof(DeviceRouteInfo)/sizeof(uint32_t));
        RouteFence<HardEvent::S_V>();
        asc_vf_call<RouteEmitCsrVF>(
            reinterpret_cast<__ubuf__ uint32_t*>(csr.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(keptIds.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(starts.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(paddedBegin.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(workerBase.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(routeInfo.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(ordered.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(inverse.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(dependencies.GetPhyAddr()),valid,begin,cvShift,
            a.config.world,a.config.localExperts,a.config.rank,a.config.maxPeerChunks);
        RouteFence<HardEvent::V_S>();
    } else {
    // Non-power-of-two CV retains the established exact Scalar division path.
    for(uint32_t e=0;e<kExperts;++e) {
        // Padded expert segments keep every MTE3 UB source 32B-aligned.
        orderCursor=(orderCursor+3U)&~3U;
        const uint32_t count=counts.GetValue(e);
        if(count) {
            const uint32_t localBegin=starts.GetValue(e),offset=workerBase.GetValue(e);
            const uint32_t peer=e/a.config.localExperts;
            const uint32_t peerBase=info->outgoing[peer*a.config.localExperts];
            for(uint32_t i=0;i<count;++i) {
                const uint32_t flat=csr.GetValue(localBegin+i),expanded=offset+i;
                ordered.SetValue(orderCursor+i,flat);inverse.SetValue(flat-begin,expanded);
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
        }
        orderCursor+=count;
    }
    }
    // Finish every producer before starting this batch. Do not modify any of
    // these buffers again until the one final MTE3_S completes all writes.
    RouteFence<HardEvent::S_MTE3>();
    orderCursor=0U;
    for(uint32_t e=0;e<kExperts;++e) {
        orderCursor=(orderCursor+3U)&~3U;
        const uint32_t count=counts.GetValue(e);
        if(count)RouteEnqueueSave(a.order+uint64_t(workerBase.GetValue(e))*sizeof(uint64_t),
            RouteCsrUb::orderAt+orderCursor*sizeof(uint64_t),count*sizeof(uint64_t));
        orderCursor+=count;
    }
    RouteEnqueueSave(a.reduceIndex+uint64_t(begin)*sizeof(uint64_t),RouteCsrUb::inverseAt,valid*sizeof(uint64_t));
    RouteEnqueueSave(a.reduceDependencies+uint64_t(begin)*sizeof(uint64_t),RouteCsrUb::dependenciesAt,valid*sizeof(uint64_t));
    RouteFence<HardEvent::MTE3_S>();
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

// Prefix/count workspace is allocated after the public RouteInfo payload.
// One writer per 64B line is NOT assumed: parallel writers below use only DMA.
__aicore__ inline uint64_t RouteGroupPrefixAddress(const Arguments& a) {
    return a.routeInfo+RouteGroupPrefixOffset();
}
__aicore__ inline uint64_t RouteLinkPrefixAddress(const Arguments& a) {
    return a.routeInfo+RouteLinkPrefixOffset(RouteGroupCapacity(a.config.world*a.config.sourceRowCapacity,a.config.localExperts));
}
__aicore__ inline void RouteReloadConfig(Arguments& a) {
    // All callers are AFTER the corresponding existing local join. The sole
    // writer has completed its 64B clean/invalidate before that join. Invalidate
    // each consumer's old clean cache lines ONCE, then read the immutable
    // snapshot normally. Per-word ReadGmByPassDCache prevents adjacent words
    // from benefiting from this ordinary cached read path.
    // Consumers never Scalar-write routeConfig, so this clean cannot write
    // stale consumer data over the leader's publication.
    RouteFlush(a.routeConfig,sizeof(Config));
    auto* dst=reinterpret_cast<uint32_t*>(&a.config);
    const volatile __gm__ uint32_t* src=reinterpret_cast<const volatile __gm__ uint32_t*>(a.routeConfig);
    static_assert(sizeof(Config)%sizeof(uint32_t)==0U);
    for(uint32_t i=0;i<sizeof(Config)/sizeof(uint32_t);++i)dst[i]=src[i];
}
__aicore__ inline void RouteParallelSummary(Arguments& a) {
    RouteLoad(a.routeInfo,kRouteInfoUb,(kExperts+1U)*4U);
    RouteLoad(a.symmetric+a.config.routeInboxOffset,0U,a.config.world*kRouteMailboxBytes);
    auto* r=reinterpret_cast<__ubuf__ DeviceRouteInfo*>(static_cast<uint64_t>(kRouteInfoUb));
    LocalTensor<uint32_t> inbox(TPosition::VECCALC,0U,a.config.world*kRouteMailboxBytes/4U);
    for(uint32_t source=0;source<a.config.world;++source) {
        const uint32_t at=source*kRouteMailboxBytes/4U;
        r->returnBase[source]=uint64_t(inbox.GetValue(at))|(uint64_t(inbox.GetValue(at+1U))<<32U);
        for(uint32_t e=0;e<=a.config.localExperts;++e)r->incoming[source][e]=inbox.GetValue(at+2U+e);
    }
    BuildRouteSummary(a.config,*r,a.config.world==4U);
    RouteSave(a.routeInfo,kRouteInfoUb,sizeof(DeviceRouteInfo));
    RouteStore(a.routeConfig,0,a.config);RouteFlush(a.routeConfig,sizeof(Config));
}
__aicore__ inline void RouteParallelCounts(const Arguments& a,uint32_t worker) {
    RouteLoad(a.routeInfo,kRouteInfoUb,sizeof(DeviceRouteInfo));
    auto* r=reinterpret_cast<__ubuf__ DeviceRouteInfo*>(static_cast<uint64_t>(kRouteInfoUb));
    LocalTensor<uint32_t> counts(TPosition::VECCALC,0U,2048U);
    const auto groups=RouteWork(a.config.fragmentCount,worker);
    for(uint32_t g=groups.first;g<groups.last;++g)counts.SetValue(g-groups.first,CountGroupSegments(a.config,*r,g));
    RouteSave(RouteGroupPrefixAddress(a)+uint64_t(groups.first)*4U,0U,(groups.last-groups.first)*4U);
    const auto cvs=RouteWork(a.config.recvChunkCount,worker);
    for(uint32_t cv=cvs.first;cv<cvs.last;++cv)counts.SetValue(cv-cvs.first,CountCvLinks(a.config,*r,cv));
    RouteSave(RouteLinkPrefixAddress(a)+uint64_t(cvs.first)*4U,0U,(cvs.last-cvs.first)*4U);
}
__aicore__ inline uint32_t RouteCountPrefix(uint64_t gm,uint32_t count) {
    // At most 16*1024 words=64KiB. This bounded coordinator scan is a
    // metadata-only first version; workers perform the expensive descriptor
    // generation. It never accesses or rescans topk inputs.
    LocalTensor<uint32_t> values(TPosition::VECCALC,0U,16384U);
    if(count)RouteLoad(gm,0U,count*4U);
    uint32_t total=0U;
    for(uint32_t i=0;i<count;++i){total+=values.GetValue(i);values.SetValue(i,total);}
    RouteSave(gm+4U,0U,count*4U);
    values.SetValue(0U,0U);RouteSave(gm,0U,4U);
    return total;
}
__aicore__ inline void RouteParallelPrefix(Arguments& a) {
    a.config.routeSegmentCount=RouteCountPrefix(RouteGroupPrefixAddress(a),a.config.fragmentCount);
    a.config.routeLinkCount=RouteCountPrefix(RouteLinkPrefixAddress(a),a.config.recvChunkCount);
    RouteStore(a.routeConfig,0,a.config);RouteFlush(a.routeConfig,sizeof(Config));
}
struct RouteParallelPrefixView {
    uint32_t firstGroup,firstCv;
    LocalTensor<uint32_t> groups,cvs;
    __aicore__ RouteParallelPrefixView(uint32_t g,uint32_t cv)
        : firstGroup(g),firstCv(cv),groups(TPosition::VECCALC,48U*1024U,2048U),cvs(TPosition::VECCALC,56U*1024U,2048U){}
    __aicore__ inline uint32_t Group(uint32_t i){return groups.GetValue(i-firstGroup);}
    __aicore__ inline uint32_t Link(uint32_t i){return cvs.GetValue(i-firstCv);}
};
__aicore__ inline void RouteParallelEmit(const Arguments& a,uint32_t worker) {
    const auto groups=RouteWork(a.config.fragmentCount,worker),cvs=RouteWork(a.config.recvChunkCount,worker);
    RouteLoad(RouteGroupPrefixAddress(a)+uint64_t(groups.first)*4U,48U*1024U,(groups.last-groups.first+1U)*4U);
    RouteLoad(RouteLinkPrefixAddress(a)+uint64_t(cvs.first)*4U,56U*1024U,(cvs.last-cvs.first+1U)*4U);
    // The coordinator's prefix scan used only low UB. RouteInfo remains
    // live at 96KiB on every worker from Summary/Counts, including worker0.
    auto* r=reinterpret_cast<__ubuf__ DeviceRouteInfo*>(static_cast<uint64_t>(kRouteInfoUb));
    RouteParallelPrefixView prefix(groups.first,cvs.first);DeviceRouteOutput out{a};
    EmitRouteGroups(a.config,*r,prefix,out,groups);
    EmitRouteReceives(a.config,*r,prefix,out,cvs,false);
    EmitRouteSends(a.config,*r,out,RouteWork(a.config.sendChunkCount,worker));
    EmitRoutePack(a.config,*r,out,RouteWork(a.config.packRoundCount,worker));
    if(worker==0U)EmitRouteTasks(a.config,*r,out,a.config.world==4U);
    out.FlushTables(); // no concurrent Scalar Config publication
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
#if MEGAMOE_URMA_ROUTE_REORDER
        RouteCompile(a); // retain the existing serial unitflag descriptor path
#else
        RouteParallelSummary(a);
#endif
    }
#if !MEGAMOE_URMA_ROUTE_REORDER
    SyncAll<false>(); // summary ready
    RouteReloadConfig(a);
    if constexpr(g_coreType==AIV)RouteParallelCounts(a,LocalWorkerIndex(block,GetSubBlockIdx()));
    SyncAll<false>(); // independently counted table lengths ready
    if constexpr(g_coreType==AIV)if(block==0U&&GetSubBlockIdx()==0U)RouteParallelPrefix(a);
    SyncAll<false>(); // disjoint output offsets ready
    RouteReloadConfig(a);
    if constexpr(g_coreType==AIV)RouteParallelEmit(a,LocalWorkerIndex(block,GetSubBlockIdx()));
#endif
    SyncAll<false>();
    // AIC also consumes the exact same Device-generated descriptor bounds.
    RouteReloadConfig(a);
    if constexpr(g_coreType==AIV)RouteLocalPermutation(a,LocalWorkerIndex(block,GetSubBlockIdx()),true);
    PipeBarrier<PIPE_ALL>();
    SyncAll<false>();
    timer.TraceEnd(TraceEvent::RoutePrepare,start);
    timer.CoarseEnd(CoarsePhase::Route);
}
} // namespace MegaMoeUrma
