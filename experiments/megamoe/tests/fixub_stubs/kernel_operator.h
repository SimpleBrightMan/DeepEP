#pragma once
#include <cstdint>
#include <cassert>
#define __aicore__
struct bfloat16_t { uint16_t bits; };
enum class QuantMode_t { QF322BF16_PRE };
namespace AscendC {
enum class TPosition { CO1, VECCALC };
enum class CO2Layout { ROW_MAJOR };
struct FixpipeConfig { CO2Layout format; bool isToUB; };
template<CO2Layout> struct FixpipeParamsC310 {
    uint32_t mSize{},nSize{},srcStride{},dstStride{};
    QuantMode_t quantPre{};uint64_t deqScalar{};
    bool reluEn{};uint8_t unitFlag{},dualDstCtl{},subBlockId{};
};
template<class T> struct LocalTensor {
    TPosition position;uint64_t address;uint32_t elements;
    LocalTensor(TPosition p,uint64_t a,uint32_t n):position(p),address(a),elements(n){}
    auto operator[](uint32_t i)const{return LocalTensor(position,address+uint64_t(i)*sizeof(T),elements-i);}
};
inline FixpipeParamsC310<CO2Layout::ROW_MAJOR> observed;
inline uint64_t srcAddress{},dstAddress{};
template<class D,class S,const FixpipeConfig& C>
void Fixpipe(LocalTensor<D> dst,LocalTensor<S> src,const FixpipeParamsC310<C.format>& p) {
    static_assert(C.isToUB && C.format==CO2Layout::ROW_MAJOR);
    assert(src.position==TPosition::CO1 && dst.position==TPosition::VECCALC);
    observed=p;srcAddress=src.address;dstAddress=dst.address;
}
}
