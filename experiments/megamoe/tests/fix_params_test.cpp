#ifdef NDEBUG
#undef NDEBUG
#endif
#include "fix_to_paired_ub.hpp"
#include <iostream>
struct Layout { uint32_t operator()(uint32_t coord)const{return coord;} };
struct L0C {
    using Element=float;
    uint32_t mStride,origin;
    auto stride()const{return std::make_tuple(std::make_tuple(2U,1U),std::make_tuple(1U,2U*mStride));}
    auto layout()const{return Layout{};}
    auto coord()const{return origin;}
    auto data()const{return AscendC::LocalTensor<float>(AscendC::TPosition::CO1,4096U,65536U);}
};
int main() {
    uint32_t cases=0;
    for(uint32_t m:{1U,17U,128U,256U})for(uint32_t n:{16U,128U,256U})
    for(uint32_t flag:{0U,2U,3U})for(uint32_t origin:{0U,64U}) {
        const uint32_t stride=(m+15U)/16U*16U;
        MegaMoeImpl::GmmKernel::StoreUb(L0C{stride,origin},m,n,flag);
        const auto& p=AscendC::observed;
        assert(p.mSize==m && p.nSize==n && p.srcStride==stride && p.dstStride==256);
        assert(p.quantPre==::QuantMode_t::QF322BF16_PRE && p.deqScalar==0x42800000ULL);
        assert(!p.reluEn && p.unitFlag==flag && !p.dualDstCtl && p.subBlockId==1);
        assert(AscendC::dstAddress==0 && AscendC::srcAddress==4096U+origin*4U);
        ++cases;
    }
    std::cout<<"paired UB fix parameters cases="<<cases<<" passed\n";
}
