// RoutePrepare experiment: contiguous CSR traversal, UB gather/scatter emit.
// No integer vector division: the caller enables this only for power-of-two CV.
// Validated in the EP8 RoutePrepare-only laboratory; see ROUTE_ONLY_LAB.md.
#pragma once
#include "kernel_operator.h"
#include "device_route_plan.hpp"
#include <cstddef>

namespace MegaMoeUrma {
constexpr uint32_t kRouteEmitExperts=384U;

// Reg::Scatter uses ELEMENT indices (uint32 here), not byte offsets. Both
// writes retain the same 32B-aligned base. Physical offsets are row*8 and
// row*8+4; divided by sizeof(uint32_t), indices are row*2 and row*2+1.
// Do not use base+1 for the high word: that would break base alignment.
// Reference: CANN 9.2 Reg Scatter / kernel_reg_compute_datacopy_intf.h.
__simd_callee__ inline void RouteScatterU64Words(
    __ubuf__ uint32_t* base,
    AscendC::Reg::RegTensor<uint32_t>& low,
    AscendC::Reg::RegTensor<uint32_t>& high,
    AscendC::Reg::RegTensor<uint32_t>& rows,
    AscendC::Reg::MaskReg& active)
{
    constexpr int16_t wordBytesShift=2;
    constexpr int16_t rowBytesShift=3;
    AscendC::Reg::RegTensor<uint32_t> wordIndex;
    AscendC::Reg::ShiftLefts(wordIndex,rows,static_cast<int16_t>(rowBytesShift-wordBytesShift),active);
    AscendC::Reg::Scatter(base,low,wordIndex,active);
    AscendC::Reg::Adds(wordIndex,wordIndex,uint32_t{4}>>wordBytesShift,active);
    AscendC::Reg::Scatter(base,high,wordIndex,active);
}

// Every flat ID occurs once even when top-k repeats an expert. Thus active
// inverse/dep scatter destinations are unique; no UB atomics are required.
__simd_vf__ inline void RouteEmitCsrVF(
    __ubuf__ uint32_t* csr,
    __ubuf__ uint32_t* originalIds,
    __ubuf__ uint32_t* starts,
    __ubuf__ uint32_t* paddedBegin,
    __ubuf__ uint32_t* workerBase,
    __ubuf__ uint32_t* routeInfo,
    __ubuf__ uint32_t* orderWords,
    __ubuf__ uint32_t* inverseWords,
    __ubuf__ uint32_t* dependencyWords,
    uint32_t itemCount,uint32_t globalBegin,uint32_t cvShift,
    uint32_t world,uint32_t localExperts,uint32_t rank,uint32_t maxPeerChunks)
{
    constexpr uint32_t lanes=AscendC::GetVecLen()/sizeof(uint32_t);
    constexpr uint32_t outgoingWords=offsetof(DeviceRouteInfo,outgoing)/sizeof(uint32_t);
    constexpr uint32_t sourcesWords=offsetof(DeviceRouteInfo,sources)/sizeof(uint32_t);
    constexpr uint32_t groupBaseWords=offsetof(DeviceRouteInfo,groupBase)/sizeof(uint32_t);
    static_assert(offsetof(DeviceRouteInfo,outgoing)%sizeof(uint32_t)==0U);
    static_assert(offsetof(DeviceRouteInfo,sources)%sizeof(uint32_t)==0U);
    static_assert(offsetof(DeviceRouteInfo,groupBase)%sizeof(uint32_t)==0U);
    auto full=AscendC::Reg::CreateMask<uint32_t,AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<uint32_t> zero,beginReg,flat,localFlat,expert;
    AscendC::Reg::RegTensor<uint32_t> ordinal,offset,expanded,orderRow;
    AscendC::Reg::RegTensor<uint32_t> peerRow,remoteLow,selfLow,depLow,tagHigh;
    AscendC::Reg::RegTensor<uint32_t> peer,localExpert,candidate,thresholdReg;
#if !MEGAMOE_URMA_ROUTE_REORDER
    AscendC::Reg::RegTensor<uint32_t> selfTag;
    AscendC::Reg::Duplicate(selfTag,0x80000000U,full);
#endif
    AscendC::Reg::RegTensor<int32_t> sequenceSigned;
    AscendC::Reg::Duplicate(zero,0U,full);
    AscendC::Reg::Duplicate(beginReg,globalBegin,full);
    for(uint32_t first=0U;first<itemCount;first+=lanes) {
        uint32_t remaining=itemCount-first;
        auto active=AscendC::Reg::UpdateMask<uint32_t>(remaining);
        AscendC::Reg::LoadAlign(flat,csr+first);
        AscendC::Reg::Sub(localFlat,flat,beginReg,active);
        AscendC::Reg::Gather(expert,originalIds,localFlat,active);
        AscendC::Reg::Arange(sequenceSigned,static_cast<int32_t>(first));
        auto& sequence=reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(sequenceSigned);
        AscendC::Reg::Gather(offset,starts,expert,active);
        AscendC::Reg::Sub(ordinal,sequence,offset,active);

        AscendC::Reg::Gather(offset,workerBase,expert,active);
        AscendC::Reg::Add(expanded,offset,ordinal,active);
        AscendC::Reg::Gather(offset,paddedBegin,expert,active);
        AscendC::Reg::Add(orderRow,offset,ordinal,active);
        RouteScatterU64Words(orderWords,flat,zero,orderRow,active);
        RouteScatterU64Words(inverseWords,expanded,zero,localFlat,active);

        // EP is 4/8/16. Binary threshold decomposition computes both quotient
        // and remainder exactly; no vector Div or floating reciprocal.
        AscendC::Reg::Duplicate(peer,0U,full);
        AscendC::Reg::Adds(localExpert,expert,0U,active);
        for(uint32_t bit=world/2U;bit!=0U;bit>>=1U) {
            const uint32_t threshold=bit*localExperts;
            AscendC::Reg::MaskReg take;
            AscendC::Reg::Compares<uint32_t,AscendC::CMPMODE::GE>(take,localExpert,threshold,active);
            AscendC::Reg::Duplicate(thresholdReg,threshold,full);
            AscendC::Reg::Sub(candidate,localExpert,thresholdReg,active);
            AscendC::Reg::Select(localExpert,candidate,localExpert,take);
            AscendC::Reg::Adds(candidate,peer,bit,active);
            AscendC::Reg::Select(peer,candidate,peer,take);
        }
        // Always use the 32B-aligned RouteInfo base. sources/groupBase member
        // pointers are NOT 32B aligned, so their word offsets belong in indices.
        AscendC::Reg::Muls(candidate,peer,localExperts,active);
        AscendC::Reg::Adds(candidate,candidate,outgoingWords,active);
        AscendC::Reg::Gather(offset,routeInfo,candidate,active);
        AscendC::Reg::Sub(peerRow,expanded,offset,active);
        AscendC::Reg::ShiftRights(remoteLow,peerRow,static_cast<int16_t>(cvShift),active);
        AscendC::Reg::Muls(offset,peer,maxPeerChunks,active);
        AscendC::Reg::Add(remoteLow,remoteLow,offset,active);

#if !MEGAMOE_URMA_ROUTE_REORDER
        AscendC::Reg::MaskReg self;
        AscendC::Reg::Compares<uint32_t,AscendC::CMPMODE::EQ>(self,peer,rank,active);
        AscendC::Reg::Muls(candidate,localExpert,uint32_t{kWorld+1U},active);
        AscendC::Reg::Adds(candidate,candidate,sourcesWords+rank,active);
        AscendC::Reg::Gather(offset,routeInfo,candidate,self);
        AscendC::Reg::Add(selfLow,offset,expanded,self);
        AscendC::Reg::Adds(candidate,expert,outgoingWords,active);
        AscendC::Reg::Gather(offset,routeInfo,candidate,self);
        AscendC::Reg::Sub(selfLow,selfLow,offset,self);
        AscendC::Reg::ShiftRights(selfLow,selfLow,static_cast<int16_t>(8),self);
        AscendC::Reg::Adds(candidate,localExpert,groupBaseWords,active);
        AscendC::Reg::Gather(offset,routeInfo,candidate,self);
        AscendC::Reg::Add(selfLow,selfLow,offset,self);
        AscendC::Reg::Select(depLow,selfLow,remoteLow,self);
        AscendC::Reg::Select(tagHigh,selfTag,zero,self);
        RouteScatterU64Words(dependencyWords,depLow,tagHigh,localFlat,active);
#else
        // unitflag self return is protected by the self CV's reorder flag.
        RouteScatterU64Words(dependencyWords,remoteLow,zero,localFlat,active);
#endif
    }
}
} // namespace MegaMoeUrma
