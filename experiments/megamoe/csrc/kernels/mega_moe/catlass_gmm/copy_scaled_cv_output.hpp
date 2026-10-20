// Copyright (c) 2026, Lu Lu
#ifndef MEGAMOE_COPY_SCALED_CV_OUTPUT_HPP
#define MEGAMOE_COPY_SCALED_CV_OUTPUT_HPP
#include "catlass/gemm/tile/ascend950/copy_l0c_to_ub.hpp"
namespace MegaMoeImpl::GmmKernel {
// 保留本实现的scale=64补偿、BF16舍入、unit flag与父L0C stride，仅改FIX目的存储层级。
template <class Base> struct ScaledCvOutputCopy : Base {
    template <class TensorC> struct CopyL0CToDst {
        struct Params { float scale = 1.0F; } params;
        template <class TensorL0C>
        __aicore__ inline void operator()(const TensorC& dst, const TensorL0C& src, uint8_t unitFlag = 0)
        {
            if constexpr (TensorC::position == AscendC::TPosition::VECCALC) {
                using Out = typename TensorC::Element;
                using In = typename TensorL0C::Element;
                AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> p;
                p.nSize = tla::get<1>(dst.shape());
                p.mSize = tla::get<0>(dst.shape());
                p.srcStride = tla::get<1,1>(src.stride()) / tla::get<0,0>(src.stride());
                p.dstStride = tla::get<0>(dst.stride());
                p.quantPre = Catlass::Gemm::Tile::CopyL0CToDstQuantMode<Catlass::Arch::Ascend950,
                    In,Out,Catlass::Gemm::Tile::ScaleGranularity::PER_TENSOR>::VALUE;
                p.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&params.scale));
                p.reluEn = false;
                p.unitFlag = unitFlag;
                p.dualDstCtl = 0;
                p.subBlockId = true;
                AscendC::Fixpipe<Out,In,CFG_ROW_MAJOR_UB>(
                    dst.data()[dst.layout()(dst.coord())],src.data()[src.layout()(src.coord())],p);
            } else {
                typename Base::template CopyL0CToDst<TensorC> copy;
                copy.params.scale = params.scale;
                copy(dst,src,unitFlag);
            }
        }
    };
};
}
#endif
