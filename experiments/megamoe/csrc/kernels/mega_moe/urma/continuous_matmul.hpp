// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "../catlass_gmm/mega_moe_grouped_matmul.hpp"

namespace MegaMoeUrma {

// One physical pipeline owns every L1/L0 slot and AIC/AIV0 credit for both
// projections. Updating the expert or stage must never reset those indices.
// Only GMM1 gathers A; both stages use one continuous Unit Flag output.
struct ContinuousComputeRows : MegaMoeImpl::GmmKernel::SegmentedRowIO<RowSegment, true, false> {
    bool gather = true;

    template <class TensorL1>
    __aicore__ inline void Gather(TensorL1& dst, uint32_t tileBegin, uint32_t tileRows, uint32_t columnBegin,
                                  uint32_t columns) const
    {
        if (gather) {
            MegaMoeImpl::GmmKernel::SegmentedRowIO<RowSegment, true, false>::Gather(dst, tileBegin, tileRows,
                                                                                    columnBegin, columns);
            return;
        }
        AscendC::GlobalTensor<uint8_t> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(inputX));
        AscendC::Nd2NzParams params{};
        params.ndNum = 1;
        params.nValue = tileRows;
        params.dValue = columns;
        params.srcDValue = inputStride;
        params.dstNzC0Stride = tla::get<1, 1>(dst.stride()) / 32U;
        params.dstNzNStride = 1;
        AscendC::DataCopy(dst.data().template ReinterpretCast<uint8_t>(),
                          input[static_cast<uint64_t>(tileBegin) * inputStride + columnBegin], params);
    }
};

template <class Config>
class ContinuousMatmul
    : private MegaMoeImpl::GmmKernel::detail::A8W4MatmulPipeline<Config, false, ContinuousComputeRows> {
    using Pipeline = MegaMoeImpl::GmmKernel::detail::A8W4MatmulPipeline<Config, false, ContinuousComputeRows>;
    bool gateUp_ = false;
    uint32_t halfN_ = 0U;

public:
    using Pipeline::Begin;
    using Pipeline::End;
    using Pipeline::SetRowIO;

    template <class Problem>
    __aicore__ inline void UpdateProblem(const Problem& config, const MegaMoeImpl::GMMAddrInfo& addr, bool gmm1)
    {
        // URMA uses planar gate/up, never fused weight columns.
        static_assert(!Problem::FUSED_GATE_UP);
        Pipeline::UpdateProblem(config, addr, gmm1 ? addr.gmm1OutGlobal : addr.gmm2OutGlobal);
        gateUp_ = gmm1;
        halfN_ = config.outputN;
    }

    __aicore__ inline void Compute(uint32_t m, uint32_t n, uint32_t rows, uint32_t columns)
    {
        Pipeline::ComputeProjection(m, n, rows, columns, n);
        if (gateUp_) Pipeline::ComputeProjection(m, n + halfN_, rows, columns, n + halfN_);
    }

    __aicore__ inline void Dequantize(uint32_t n, uint32_t rows, uint32_t columns)
    {
        Pipeline::DequantizeWeight(n, rows, columns);
        if (gateUp_) Pipeline::DequantizeWeight(n + halfN_, rows, columns);
    }
};

}  // namespace MegaMoeUrma
