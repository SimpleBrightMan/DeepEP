#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#endif
#include "basic_api/kernel_basic_intf.h"
#include "layout.hpp"
#if defined(__NPU_DEVICE__)
#include "kernel_operator.h"
#include "shmem.h"
// Use the production transport verbatim, including QP0 RO PUT and SO bitmap.
#include "../megamoe/csrc/kernels/mega_moe/urma/transport.hpp"
#endif
using namespace AscendC;

extern "C" __global__ __vector__ void combine_l1_barrier_kernel() {
#if defined(__NPU_DEVICE__)
    aclshmemx_sync_vec_all();
#endif
}
extern "C" void launch_barrier(void* stream) {
    combine_l1_barrier_kernel<<<1,nullptr,stream>>>();
}

__schedmode__(1) __global__ __mix__(1,2) void combine_l1_kernel(Demo::Args a) {
#if defined(__NPU_DEVICE__)
    InitSocState();
    util_set_ffts_config(a.ffts);
    SetSysWorkspaceForce(reinterpret_cast<GM_ADDR>(a.workspace));
    const uint32_t block=GetBlockIdx()/GetTaskRation();
    uint64_t start=0,end=0,bytes=0;
    uint32_t record=UINT32_MAX;
    SyncAll<false>();
    if constexpr(g_coreType==AIC) {
        if(a.l1Enable) {
            GlobalTensor<uint8_t> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a.l1Source));
            gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            LocalTensor<uint8_t> l1(TPosition::A1,0,Demo::copyBytes*Demo::slots);
            const uint64_t chunks=(a.l1Bytes+Demo::copyBytes-1)/Demo::copyBytes;
            uint32_t cursor=0;
            start=GetSystemCycle();
            for(uint32_t repeat=0;repeat<a.l1Repeat;++repeat) {
                for(uint64_t chunk=block;chunk<chunks;chunk+=Demo::cores) {
                    const uint64_t offset=chunk*Demo::copyBytes;
                    const uint32_t n=a.l1Bytes-offset<Demo::copyBytes?a.l1Bytes-offset:Demo::copyBytes;
                    // One MTE2 stream, no consumer: same-pipe stores are ordered.
                    // No per-copy Scalar wait in the unthrottled path.
                    DataCopy(l1[(cursor%Demo::slots)*Demo::copyBytes],gm[offset],n);
                    bytes+=n;++cursor;
                    if(a.gapCycles && cursor%Demo::slots==0) {
                        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                        const uint64_t t=GetSystemCycle();
                        while(GetSystemCycle()-t<a.gapCycles) {}
                    }
                }
            }
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            end=GetSystemCycle();record=block;
        }
    } else {
        if(GetSubBlockIdx()==1 && block<a.world-1) {
            const uint32_t peer=MegaMoeUrma::PeerForOwner(a.rank,block);
            const uint32_t rows=a.bs*a.topk/a.world;
            const uint64_t rowBytes=uint64_t(a.hidden)*2;
            const uint64_t slot=uint64_t(rows)*rowBytes;
            const uint32_t cvs=(rows+Demo::cvRows-1)/Demo::cvRows;
            auto* dst=reinterpret_cast<__gm__ uint8_t*>(a.destination);
            auto* src=reinterpret_cast<__gm__ uint8_t*>(a.source)+uint64_t(peer)*slot;
            start=GetSystemCycle();
            for(uint32_t cv=0;cv<cvs;++cv) {
                const uint32_t row=cv*Demo::cvRows;
                const uint32_t n=rows-row<Demo::cvRows?rows-row:Demo::cvRows;
                MegaMoeUrma::Put(dst,uint64_t(a.rank)*slot+row*rowBytes,src+row*rowBytes,n*rowBytes,peer);
                MegaMoeUrma::PublishCombineRange(reinterpret_cast<__gm__ uint8_t*>(a.flags),0,cvs,a.rank,peer,cv,1);
            }
            // Exactly the main branch's one final QP quiet, not one per CV.
            MegaMoeUrma::Quiet(peer);
            end=GetSystemCycle();bytes=slot;record=Demo::cores+block;
        }
    }
    // Keep timing writes out of EVERY worker's measured interval.
    SyncAll<false>();
    if(record!=UINT32_MAX) {
        auto* p=reinterpret_cast<__gm__ uint64_t*>(a.records)+uint64_t(record)*Demo::words;
        p[0]=start;p[1]=end;p[2]=bytes;
        GlobalTensor<uint64_t> g;g.SetGlobalBuffer(p);
        DataCacheCleanAndInvalid<uint64_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(g);
    }
#endif
}
extern "C" void launch_demo(uint32_t blocks,uint32_t ub,void* stream,const Demo::Args* args) {
    combine_l1_kernel<<<blocks,ub,stream>>>(*args);
}
