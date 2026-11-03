#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#endif
#include "basic_api/kernel_basic_intf.h"
#include "layout.hpp"
#if defined(__NPU_DEVICE__)
#include "kernel_operator.h"
#include "shmem.h"
// Use production QP0 RO PUT/quiet verbatim. This demo never publishes SO flags.
#include "../megamoe/csrc/kernels/mega_moe/urma/transport.hpp"
#include "compute.hpp"
#include "../megamoe/csrc/kernels/mega_moe/urma/communication_schedule.hpp"
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
    uint64_t start=0,dispatchEnd=0,combineStart=0,end=0,dbytes=0,cbytes=0;
    uint32_t record=UINT32_MAX;
    SyncAll<false>();
    if constexpr(g_coreType==AIC) {
        if(a.computeEnable) {
            start=GetSystemCycle();
            Demo::RunCompute(a.compute,block);
            end=GetSystemCycle();record=block;
        }
    } else if(GetSubBlockIdx()==0) {
        if(a.computeEnable) {
            start=GetSystemCycle();
            Demo::RunCompute(a.compute,block);
            end=GetSystemCycle();record=Demo::cores+block;
        }
    } else if(block<a.world-1) {
        const uint32_t peer=MegaMoeUrma::PeerForOwner(a.rank,block);
        const uint32_t rows=a.bs*a.topk/a.world,scales=a.hidden/32;
        const uint64_t slot=uint64_t(rows)*a.hidden*2,slotX=slot/2,slotS=uint64_t(rows)*scales;
        const uint32_t cvs=(rows+Demo::cvRows-1)/Demo::cvRows;
        auto* source=reinterpret_cast<__gm__ uint8_t*>(a.source)+uint64_t(peer)*slot;
        auto* sourceScale=reinterpret_cast<__gm__ uint8_t*>(a.sourceScale)+uint64_t(peer)*slotS;
        auto* dispatch=reinterpret_cast<__gm__ uint8_t*>(a.dispatchDestination);
        auto* combine=reinterpret_cast<__gm__ uint8_t*>(a.destination);
        start=GetSystemCycle();
        uint32_t step=0,packRound=2;
        for(uint32_t cv=0;cv<cvs;) {
            const uint32_t count=MegaMoeUrma::DispatchBatchLimit(step,cvs-cv,packRound>=a.compute.config.packRoundCount);
            const uint32_t row=cv*Demo::cvRows,n=rows-row<count*Demo::cvRows?rows-row:count*Demo::cvRows;
            MegaMoeUrma::Put(dispatch,uint64_t(a.rank)*slotX+uint64_t(row)*a.hidden,
                source+uint64_t(row)*a.hidden,n*a.hidden,peer);
            MegaMoeUrma::Put(dispatch,uint64_t(a.world)*slotX+uint64_t(a.rank)*slotS+uint64_t(row)*scales,
                sourceScale+uint64_t(row)*scales,n*scales,peer);
            cv+=count;++step;
            packRound=MegaMoeUrma::DispatchPackAfterLimit(count,cv,step,cvs,packRound,a.compute.config.packRoundCount);
        }
        dispatchEnd=GetSystemCycle();
        combineStart=GetSystemCycle();
        for(uint32_t cv=0;cv<cvs;) {
            // Production merge rule with all CVs ready: contiguous, one flag
            // word, <=256MiB, and leave the final unsent CV separate.
            const uint32_t remaining=cvs-cv;
            uint32_t count=remaining>1?remaining-1:1;
            count=count<64U-cv%64U?count:64U-cv%64U;
            const uint32_t byteLimit=MegaMoeUrma::kMaxPutBytes/(Demo::cvRows*a.hidden*2);
            count=count<byteLimit?count:byteLimit;
            const uint32_t row=cv*Demo::cvRows,n=rows-row<count*Demo::cvRows?rows-row:count*Demo::cvRows;
            MegaMoeUrma::Put(combine,uint64_t(a.rank)*slot+uint64_t(row)*a.hidden*2,
                source+uint64_t(row)*a.hidden*2,n*a.hidden*2,peer);
            cv+=count;
        }
        // No SO flags, no ready protocol and no inter-stage quiet. Only this
        // final quiet certifies completion of both streams before buffer reuse.
        MegaMoeUrma::Quiet(peer);
        end=GetSystemCycle();dbytes=slotX+slotS;cbytes=slot;record=2*Demo::cores+block;
    }
    SyncAll<false>(); // Outside every measured interval.
    if(record!=UINT32_MAX) {
        auto* p=reinterpret_cast<__gm__ uint64_t*>(a.records)+uint64_t(record)*Demo::words;
        p[0]=start;p[1]=end;p[2]=dbytes;p[3]=cbytes;p[4]=dispatchEnd;p[5]=combineStart;
        GlobalTensor<uint64_t> g;g.SetGlobalBuffer(p);
        DataCacheCleanAndInvalid<uint64_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(g);
    }
#endif
}
extern "C" void launch_demo(uint32_t blocks,uint32_t ub,void* stream,const Demo::Args* args) {
    combine_l1_kernel<<<blocks,ub,stream>>>(*args);
}
