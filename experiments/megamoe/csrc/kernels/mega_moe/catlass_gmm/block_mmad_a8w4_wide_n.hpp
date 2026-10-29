/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * third-party/cann.LICENSE for the full text of the License.
 */
// Modified by SimpleBright_Man 2026

#pragma once

// Isolated M128/N512 L1+L0C block with N256 MMAD subtiles. Do not change the existing M256 block:
// its event order, buffer addresses and generated hot path remain independent.
#include "resource.hpp"
#include "a8w4_k_plan.hpp"
#include "segmented_row_io.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Block {
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_,
          class ElementC_, class ElementBias_, class TileCopy_,
          class TileMmad_ =
              Gemm::Tile::TileMmadTla<typename DispatchPolicy_::ArchTag, ElementA_, typename TileCopy_::LayoutTagL1A>>
struct MegaMoeBlockMmadA8W4WideN {
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementMxScaleA = typename TileCopy::ElementMxScaleA;
    using ElementMxScaleB = typename TileCopy::ElementMxScaleB;
    using ElementL0A = typename helper::GetL0Element<ElementA, true>::Element;
    using ElementL0B = typename helper::GetL0Element<ElementB, true>::Element;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL1MxScaleA = typename TileCopy::LayoutTagL1MxScaleA;
    using LayoutTagL1MxScaleB = typename TileCopy::LayoutTagL1MxScaleB;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using TileMmad = TileMmad_;

    static constexpr uint32_t L1A_STAGES = 2U, L1B_STAGES = 2U;
    static constexpr uint32_t L0A_STAGES = 2U, L0B_STAGES = 2U, L0C_STAGES = 1U;
    static constexpr uint32_t MAX_SCALE_K = 4096U;
    static_assert(tla::get<0>(L1TileShape_{}) == 128 && tla::get<1>(L1TileShape_{}) == 512 &&
                  tla::get<2>(L1TileShape_{}) == 256);
    static_assert(tla::get<0>(L0TileShape_{}) == 128 && tla::get<1>(L0TileShape_{}) == 256 &&
                  tla::get<2>(L0TileShape_{}) == 128);
    static_assert(!DispatchPolicy::ENABLE_UNIT_FLAG && !TileCopy::HAS_BIAS);
    static_assert(SizeOfBits<ElementA>::value == 8 && SizeOfBits<ElementB>::value == 8 &&
                  SizeOfBits<ElementL0A>::value == 8 && SizeOfBits<ElementL0B>::value == 8);
    static_assert(2U * 128U * 128U <= ArchTag::L0A_SIZE);
    static_assert(2U * 128U * 256U <= ArchTag::L0B_SIZE);
    static_assert(128U * 512U * sizeof(ElementAccumulator) <= ArchTag::L0C_SIZE);
    static_assert(464U * 1024U <= ArchTag::L1_SIZE);
    static_assert(128U * (MAX_SCALE_K / MX_SCALE_GROUP_NUM) * sizeof(ElementMxScaleA) <= 16U * 1024U);
    static_assert(512U * (MAX_SCALE_K / MX_SCALE_GROUP_NUM) * sizeof(ElementMxScaleB) <= 64U * 1024U);

    CATLASS_DEVICE explicit MegaMoeBlockMmadA8W4WideN(MegaMoeResource& resource,
                                                     uint32_t base = 0U, bool activate = true)
    {
        // KiB: B[0,128),B[128,256),A[256,320),A[320,384),SA[384,400),SB[400,464).
        for (uint32_t i = 0; i < 2; ++i) {
            a1_[i] = resource.l1Buf.template GetBufferByByte<ElementA>(base + (256U + i * 64U) * 1024U);
            b1_[i] = resource.l1Buf.template GetBufferByByte<ElementB>(base + i * 128U * 1024U);
            a0_[i] = resource.l0ABuf.template GetBufferByByte<ElementL0A>(i * 16U * 1024U);
            b0_[i] = resource.l0BBuf.template GetBufferByByte<ElementL0B>(i * 32U * 1024U);
        }
        sa1_ = resource.l1Buf.template GetBufferByByte<ElementMxScaleA>(base + 384U * 1024U);
        sb1_ = resource.l1Buf.template GetBufferByByte<ElementMxScaleB>(base + 400U * 1024U);
        c0_ = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        if (activate) Activate();
    }

    CATLASS_DEVICE void Activate()
    {
        if (active_) return;
        a1slot_ = b1slot_ = a0slot_ = b0slot_ = 0;
        for (uint32_t i = 0; i < 2; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(i);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2U + i);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(4);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        active_ = true;
    }

    CATLASS_DEVICE void Deactivate()
    {
        if (!active_) return;
        for (uint32_t i = 0; i < 2; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(i);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2U + i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(4);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        active_ = false;
    }
    CATLASS_DEVICE ~MegaMoeBlockMmadA8W4WideN() { Deactivate(); }

    template <class TensorA, class TensorC, class TensorMxScaleA = EmptyClass, class TensorMxScaleB = EmptyClass,
              class TensorBias = EmptyClass, class RowIO = MegaMoeImpl::GmmKernel::ContiguousRowIO>
    CATLASS_DEVICE void operator()(TensorA& tensorA, TensorC& tensorC, GemmCoord const& shape,
                                   TensorMxScaleA const& tensorMxScaleA = {}, TensorMxScaleB const& tensorMxScaleB = {},
                                   bool allowDynamicB = false, TensorBias const& tensorBias = {},
                                   RowIO const& rowIO = {}, uint32_t rowBegin = 0U, uint32_t columnBegin = 0U)
    {
        typename TileCopy::template CopyGmToL1MxScaleA<TensorMxScaleA> copySA;
        typename TileCopy::template CopyGmToL1MxScaleB<TensorMxScaleB> copySB;
        typename TileCopy::template CopyL0CToDst<TensorC> copyC;
        (void)allowDynamicB; (void)tensorBias;
        const uint32_t m = shape.m(), n = shape.n(), k = shape.k();
        // Preserve the old N256 A-window heuristic and K256 B window. With
        // m<=128 this A window fits 64KiB. Fixed model K is 3584 or 1280.
        const uint32_t ka = MegaMoeImpl::GmmKernel::MakeA8W4KPlan(m, 256U, false).ka;
        constexpr uint32_t kb = 256U, k0 = 128U, n0 = 256U;
        const uint32_t scales = CeilDiv<MX_BASEK_FACTOR>(k) * 2U;
        auto sa = tla::MakeTensor(sa1_, tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagL1MxScaleA, false>(m, scales), Arch::PositionL1{});
        auto sb = tla::MakeTensor(sb1_, tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagL1MxScaleB, true>(scales, n), Arch::PositionL1{});
        auto sga = GetTile(tensorMxScaleA, tla::MakeCoord(0, 0), tla::MakeShape(m, scales));
        auto sgb = GetTile(tensorMxScaleB, tla::MakeCoord(0, 0), tla::MakeShape(scales, n));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(4);
        copySA(sa, sga); copySB(sb, sgb);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(4);
        auto c = tla::MakeTensor(c0_, tla::MakeLayoutL0C(m, n), Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        for (uint32_t kw = 0; kw < k; kw += kb) {
            const uint32_t bCount = Min(kb, k - kw);
            const uint32_t aBegin = kw / ka * ka, aCount = Min(ka, k - aBegin);
            auto a1 = tla::MakeTensor(a1_[a1slot_], tla::MakeLayout<ElementA, LayoutTagL1A>(m, RoundUp<MX_BASEK_FACTOR>(aCount)), Arch::PositionL1{});
            if (kw == aBegin) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(a1slot_);
                if constexpr (RowIO::kGatherInput) {
                    rowIO.Gather(a1, rowBegin, m, aBegin, aCount);
                } else {
                    typename TileCopy::template CopyGmToL1A<TensorA> copyA;
                    auto ag = GetTile(tensorA, tla::MakeCoord(0, aBegin), tla::MakeShape(m, aCount));
                    copyA(a1, ag);
                }
                InitZeroInL1A(a1, tla::MakeShape(m, aCount));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(a1slot_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(a1slot_);
            }
            auto b1 = tla::MakeTensor(b1_[b1slot_], tla::MakeLayout<ElementB, LayoutTagL1B>(kb, n), Arch::PositionL1{});
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(6U + b1slot_);
            for (uint32_t ko = 0; ko < bCount; ko += k0) {
                const uint32_t kc = Min(k0, bCount - ko); // fixed URMA K values are multiples of K128
                auto a = tla::MakeTensor(a0_[a0slot_], tla::MakeLayout<ElementL0A, LayoutTagL0A>(m, kc), Arch::PositionL0A{});
                auto aTile = GetTile(a1, tla::MakeCoord(0, kw - aBegin + ko), tla::MakeShape(m, kc));
                auto saTile = GetTile(sa, tla::MakeCoord(0, (kw + ko) / MX_SCALE_GROUP_NUM), tla::MakeShape(m, kc / MX_SCALE_GROUP_NUM));
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(a0slot_);
                copyA0_(a, aTile, saTile);
                // K outside, N inside: the SAME L0A operand serves gate/up
                // (or two GMM2 output channel blocks) before it can be freed.
                for (uint32_t no = 0; no < n; no += n0) {
                    const uint32_t nc = Min(n0, n - no);
                    auto b = tla::MakeTensor(b0_[b0slot_], tla::MakeLayout<ElementL0B, LayoutTagL0B>(kc, nc), Arch::PositionL0B{});
                    auto bTile = GetTile(b1, tla::MakeCoord(ko, no), tla::MakeShape(kc, nc));
                    auto sbTile = GetTile(sb, tla::MakeCoord((kw + ko) / MX_SCALE_GROUP_NUM, no), tla::MakeShape(kc / MX_SCALE_GROUP_NUM, nc));
                    // TileMmadTla passes .data() directly to Mmad and does not
                    // apply a Tensor coordinate. Rebase the L0C pointer here;
                    // a coordinate-only GetTile would overwrite gate with up.
                    const auto cOffset = c.layout()(tla::MakeCoord(0, no));
                    auto cTile = tla::MakeTensor(c.data()[cOffset], tla::MakeLayoutL0C(m, nc), Arch::PositionL0C{});
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2U + b0slot_);
                    copyB0_(b, bTile, sbTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(2U + b0slot_);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(2U + b0slot_);
                    mma_(cTile, a, b, m, nc, kc, kw == 0U && ko == 0U, uint8_t{0});
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2U + b0slot_);
                    b0slot_ = (b0slot_ + 1U) % 2U;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(a0slot_);
                a0slot_ = (a0slot_ + 1U) % 2U;
            }
            if (kw + bCount == k || (kw + bCount) % ka == 0U) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(a1slot_);
                a1slot_ = (a1slot_ + 1U) % 2U;
            }
            AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(8U + b1slot_);
            b1slot_ = (b1slot_ + 1U) % 2U;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(4);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        copyC.params.scale = 64;
        if constexpr (RowIO::kScatterOutput)
            rowIO.Scatter(copyC, tensorC, c, rowBegin, m, columnBegin, n);
        else
            copyC(tensorC, c);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    }

protected:

    template <class TensorL1, class Shape>
    CATLASS_DEVICE void InitZeroInL1A(TensorL1& tensorL1, Shape actualShape)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<typename TensorL1::Element>::value;
        const uint32_t mL1Actual = tla::get<0>(actualShape);
        const uint32_t kL1Actual = tla::get<1>(actualShape);

        uint32_t alignKL1 = RoundUp<MX_BASEK_FACTOR>(kL1Actual);
        uint32_t validKL1 = kL1Actual;
        if constexpr (tla::detail::iszN<typename TensorL1::Element, typename TensorL1::Layout>::value) {
            validKL1 = RoundUp<ELE_NUM_PER_C0>(kL1Actual);
        }
        uint32_t padKL1 = alignKL1 - validKL1;
        if (padKL1 > 0) {
            AscendC::InitConstValueParams<uint16_t> initConstValueParams;

            if constexpr (tla::detail::iszN<typename TensorL1::Element, typename TensorL1::Layout>::value) {
                initConstValueParams.repeatTimes = 1;
                initConstValueParams.blockNum = mL1Actual;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = 0;
            } else {
                initConstValueParams.repeatTimes = CeilDiv<ELE_NUM_PER_C0>(mL1Actual);
                initConstValueParams.blockNum = padKL1;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = tla::get<0, 1>(tensorL1.stride()) / ELE_NUM_PER_C0 - padKL1;
            }

            auto offset = tensorL1.layout()(tla::MakeCoord(0, validKL1));
            auto srcUint16 = tensorL1.data()[offset].template ReinterpretCast<uint16_t>();
            InitConstValue(srcUint16, initConstValueParams);
        }
    }

    template <class TensorL1, class Shape>
    CATLASS_DEVICE void InitZeroInL1B(TensorL1& tensorL1, Shape actualShape)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<typename TensorL1::Element>::value;
        const uint32_t kL1Actual = tla::get<0>(actualShape);
        const uint32_t nL1Actual = tla::get<1>(actualShape);

        uint32_t alignKL1 = RoundUp<MX_BASEK_FACTOR>(kL1Actual);
        uint32_t validKL1 = kL1Actual;
        if constexpr (tla::detail::isnZ<typename TensorL1::Element, typename TensorL1::Layout>::value) {
            validKL1 = RoundUp<ELE_NUM_PER_C0>(kL1Actual);
        }
        uint32_t padKL1 = alignKL1 - validKL1;
        if (padKL1 > 0) {
            AscendC::InitConstValueParams<uint16_t> initConstValueParams;

            if constexpr (tla::detail::iszN<typename TensorL1::Element, typename TensorL1::Layout>::value) {
                initConstValueParams.repeatTimes = CeilDiv<ELE_NUM_PER_C0>(nL1Actual);
                initConstValueParams.blockNum = padKL1;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = tla::get<1, 1>(tensorL1.stride()) / ELE_NUM_PER_C0 - padKL1;
            } else {
                initConstValueParams.repeatTimes = 1;
                initConstValueParams.blockNum = nL1Actual;
                initConstValueParams.initValue = 0;
                initConstValueParams.dstGap = 0;
            }

            auto offset = tensorL1.layout()(tla::MakeCoord(validKL1, 0));
            auto srcUint16 = tensorL1.data()[offset].template ReinterpretCast<uint16_t>();
            InitConstValue(srcUint16, initConstValueParams);
        }
    }



private:
    AscendC::LocalTensor<ElementA> a1_[2];
    AscendC::LocalTensor<ElementB> b1_[2];
    AscendC::LocalTensor<ElementMxScaleA> sa1_;
    AscendC::LocalTensor<ElementMxScaleB> sb1_;
    AscendC::LocalTensor<ElementL0A> a0_[2];
    AscendC::LocalTensor<ElementL0B> b0_[2];
    AscendC::LocalTensor<ElementAccumulator> c0_;
    uint32_t a1slot_=0, b1slot_=0, a0slot_=0, b0slot_=0;
    bool active_=false;
    TileMmad mma_;
    CopyL1ToL0A copyA0_;
    CopyL1ToL0B copyB0_;
};
#endif
} // namespace Catlass::Gemm::Block
