#include <cassert>
#include <initializer_list>
#include "common/mega_moe_aiv1_ub_layout.hpp"
using namespace MegaMoeImpl;
int main()
{
    for(unsigned ep : {8U,32U,64U,128U})
     for(unsigned h : {1024U,1088U,7168U,8192U})
      for(unsigned route : {1U,32U,256U,4096U,16384U})
       for(unsigned slots : {1U,2U,6U}) {
        auto align=[](uint64_t n){return (n+31U)/32U*32U;};
        const unsigned e=384U/ep, bytes=(h+255U)/256U*256U+align((h+31U)/32U);
        const auto p=MakeAiv1ResidentUbLayout(h,ep,e,route,slots,bytes,2U,false);
        const uint64_t base=64U*256U*2U*2U;
        assert(p.dispatchBase==base);
        assert(p.routeIndicesBase==base+align(uint64_t(ep)*e*4U));
        assert(p.dispatchCopyBase==p.routeIndicesBase+align(uint64_t(route)*4U));
        assert(p.expertCountsBase==p.dispatchCopyBase+uint64_t(slots)*bytes);
        assert(p.dispatchMetaBase==p.expertCountsBase+align(uint64_t(e)*4U));
        assert(p.dispatchEnd==p.dispatchMetaBase+uint64_t(slots)*8U*4U);
        assert(p.combineRowsBase==align(p.dispatchEnd));
        assert(p.combineMetaBase==p.combineRowsBase+2U*align(uint64_t(h)*2U));
       }
    for (unsigned ep : {8,32,64,128})
        for (unsigned bs : {1,32,96,256,512,1024,2048,4096,65536}) {
            assert(!UseGmm1CvForLaunch(bs,6,ep,384/ep,4096));
            const auto gm = MakeAiv1ResidentUbLayout(7168,ep,384/ep,4096,6,7392,2,false);
            assert(gm.valid && gm.dispatchBase==Aiv1ResidentUbLayout::GmActivationBytes);
            assert(gm.dispatchEnd<=gm.combineRowsBase && gm.combineRowsBase<=gm.combineMetaBase);
            assert(gm.end<=256*1024 && gm.combineRowCount==2);
        }
    assert(!UseGmm1CvForLaunch(0,0,0,0,0));
}
