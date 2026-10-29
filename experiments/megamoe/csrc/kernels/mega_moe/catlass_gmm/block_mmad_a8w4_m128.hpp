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

// An isolated small-M implementation. Do not change the existing M256 block:
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
          class ElementC_, class ElementBias_, class TileCopy_, uint32_t ScaleK_,
          class TileMmad_ =
              Gemm::Tile::TileMmadTla<typename DispatchPolicy_::ArchTag, ElementA_, typename TileCopy_::LayoutTagL1A>>
struct MegaMoeBlockMmadA8W4M128 {
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

    static constexpr uint32_t L1A_STAGES = 3U, L1B_STAGES = DispatchPolicy::L1B_STAGES;
    static constexpr uint32_t L0A_STAGES = 2U, L0B_STAGES = 2U, L0C_STAGES = 2U;
    static constexpr uint32_t SCALE_STAGES = 2U;
    static constexpr uint32_t L0_TILE_K = 128U;
    static constexpr uint32_t L1A_TILE_SIZE = 64U * 1024U;
    static constexpr uint32_t L0A_TILE_SIZE = 128U * 128U;
    static constexpr uint32_t L0B_TILE_SIZE = 128U * 256U;
    static constexpr uint32_t L0C_TILE_SIZE = 128U * 256U * sizeof(ElementAccumulator);
    // Each stage has its own full-K scale window: 3584 for GMM1, 1280 for GMM2.
    // Data K remains L1 K256 / L0 K128; only scale reservation is compacted.
    static constexpr uint32_t MAX_SCALE_K = ScaleK_;
    static_assert(tla::get<0>(L1TileShape_{}) == 128 && tla::get<1>(L1TileShape_{}) == 256 &&
                  tla::get<2>(L1TileShape_{}) == 256);
    static_assert(tla::get<0>(L0TileShape_{}) == 128 && tla::get<1>(L0TileShape_{}) == 256 &&
                  tla::get<2>(L0TileShape_{}) == L0_TILE_K);
    static_assert(!DispatchPolicy::ENABLE_UNIT_FLAG && !TileCopy::HAS_BIAS);
    static_assert(DispatchPolicy::L1A_STAGES == 3 && (L1B_STAGES == 3 || L1B_STAGES == 4) &&
                  DispatchPolicy::L0A_STAGES == 2 && DispatchPolicy::L0B_STAGES == 2 &&
                  DispatchPolicy::L0C_STAGES == 2 && DispatchPolicy::L1_SCALE_FACTOR_K == 16);
    static_assert(SizeOfBits<ElementA>::value == 8 && SizeOfBits<ElementB>::value == 8 &&
                  SizeOfBits<ElementL0A>::value == 8 && SizeOfBits<ElementL0B>::value == 8);
    static_assert(2U * L0A_TILE_SIZE <= ArchTag::L0A_SIZE && 2U * L0B_TILE_SIZE <= ArchTag::L0B_SIZE);
    static constexpr uint32_t SCALE_A_BYTES = 128U * (MAX_SCALE_K / MX_SCALE_GROUP_NUM) * sizeof(ElementMxScaleA);
    static constexpr uint32_t SCALE_B_BYTES = 256U * (MAX_SCALE_K / MX_SCALE_GROUP_NUM) * sizeof(ElementMxScaleB);
    static constexpr uint32_t SCALE_A_BASE = (192U + L1B_STAGES * 64U) * 1024U;
    static constexpr uint32_t SCALE_B_BASE = SCALE_A_BASE + 2U * SCALE_A_BYTES;
    static constexpr uint32_t L1_END = SCALE_B_BASE + 2U * SCALE_B_BYTES;
    static_assert((L1B_STAGES == 3 && ScaleK_ == 3584) || (L1B_STAGES == 4 && ScaleK_ == 1280));
    static_assert(L1_END == (L1B_STAGES == 3 ? 468U : 478U) * 1024U);
    static_assert(2U * L0C_TILE_SIZE <= ArchTag::L0C_SIZE && L1_END <= ArchTag::L1_SIZE);
    static_assert(128U * (MAX_SCALE_K / MX_SCALE_GROUP_NUM) * sizeof(ElementMxScaleA) <= 16U * 1024U);
    static_assert(256U * (MAX_SCALE_K / MX_SCALE_GROUP_NUM) * sizeof(ElementMxScaleB) <= 32U * 1024U);

    CATLASS_DEVICE explicit MegaMoeBlockMmadA8W4M128(MegaMoeResource& resource,
                                                     uint32_t l1BufAddrStart = 0U, bool activate = true)
    {
        for (uint32_t i = 0U; i < L1A_STAGES; ++i)
            l1A_[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + i * L1A_TILE_SIZE);
        for (uint32_t i = 0U; i < L1B_STAGES; ++i)
            l1B_[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + (192U + i * 64U) * 1024U);
        for (uint32_t i = 0U; i < 2U; ++i) {
            scaleA_[i] = resource.l1Buf.template GetBufferByByte<ElementMxScaleA>(l1BufAddrStart + SCALE_A_BASE + i * SCALE_A_BYTES);
            scaleB_[i] = resource.l1Buf.template GetBufferByByte<ElementMxScaleB>(l1BufAddrStart + SCALE_B_BASE + i * SCALE_B_BYTES);
            l0A_[i] = resource.l0ABuf.template GetBufferByByte<ElementL0A>(i * L0A_TILE_SIZE);
            l0B_[i] = resource.l0BBuf.template GetBufferByByte<ElementL0B>(i * L0B_TILE_SIZE);
            l0C_[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(i * L0C_TILE_SIZE);
        }
        if (activate) Activate();
    }

    CATLASS_DEVICE void Activate()
    {
        if (active_) return;
        l1ABase_ = l1BSlot_ = l0ABase_ = l0BSlot_ = l0CSlot_ = scaleSlot_ = 0U;
        for (uint32_t i = 0U; i < L1A_STAGES; ++i)
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        for (uint32_t i = 0U; i < L0A_STAGES; ++i)
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(i);
        for (uint32_t i = 0U; i < 2U; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(4U + i); // scale pair free
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2U + i);    // L0 B free
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(i);         // L0 C free
        }
        active_ = true;
    }

    CATLASS_DEVICE void Deactivate()
    {
        if (!active_) return;
        for (uint32_t i = 0U; i < L1A_STAGES; ++i)
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        for (uint32_t i = 0U; i < L0A_STAGES; ++i)
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(i);
        for (uint32_t i = 0U; i < 2U; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(4U + i);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2U + i);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(i);
        }
        active_ = false;
    }

    CATLASS_DEVICE ~MegaMoeBlockMmadA8W4M128() { Deactivate(); }

    template <class TensorA, class TensorC, class TensorMxScaleA = EmptyClass, class TensorMxScaleB = EmptyClass,
              class TensorBias = EmptyClass, class RowIO = MegaMoeImpl::GmmKernel::ContiguousRowIO>
    CATLASS_DEVICE void operator()(TensorA& tensorA, TensorC& tensorC, GemmCoord const& shape,
                                   TensorMxScaleA const& tensorMxScaleA = {}, TensorMxScaleB const& tensorMxScaleB = {},
                                   bool allowDynamicB = false, TensorBias const& tensorBias = {},
                                   RowIO const& rowIO = {}, uint32_t rowBegin = 0U, uint32_t columnBegin = 0U)
    {
        using CopyScaleA = typename TileCopy::template CopyGmToL1MxScaleA<TensorMxScaleA>;
        using CopyScaleB = typename TileCopy::template CopyGmToL1MxScaleB<TensorMxScaleB>;
        using CopyOutput = typename TileCopy::template CopyL0CToDst<TensorC>;
        CopyScaleA copyScaleA;
        CopyScaleB copyScaleB;
        CopyOutput copyOutput;
        (void)tensorBias;
        const uint32_t m = shape.m(), n = shape.n(), k = shape.k();
        const auto plan = MegaMoeImpl::GmmKernel::MakeA8W4KPlan(m, n, allowDynamicB);
        const uint32_t ka = plan.ka, kb = plan.kb;
        const uint32_t aWindows = CeilDiv(k, ka);
        const uint32_t l0Tiles = CeilDiv(k, L0_TILE_K);
        const uint32_t scaleCount = CeilDiv<MX_BASEK_FACTOR>(k) * 2U;
        const auto scaleLayoutA = tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagL1MxScaleA, false>(m, scaleCount);
        const auto scaleLayoutB = tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagL1MxScaleB, true>(scaleCount, n);
        auto scaleA = tla::MakeTensor(scaleA_[scaleSlot_], scaleLayoutA, Arch::PositionL1{});
        auto scaleB = tla::MakeTensor(scaleB_[scaleSlot_], scaleLayoutB, Arch::PositionL1{});
        auto tileSA = GetTile(tensorMxScaleA, tla::MakeCoord(0, 0), tla::MakeShape(m, scaleCount));
        auto tileSB = GetTile(tensorMxScaleB, tla::MakeCoord(0, 0), tla::MakeShape(scaleCount, n));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(4U + scaleSlot_);
        copyScaleA(scaleA, tileSA);
        copyScaleB(scaleB, tileSB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(4U + scaleSlot_);

        // First fill three independent A windows on MTE2, without waiting for B.
        // Refill happens only after all L0A reads of the old window are queued.
        for (uint32_t window = 0U; window < Min(L1A_STAGES, aWindows); ++window)
            PrefetchL1A(tensorA, rowIO, rowBegin, m, k, ka, window);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(4U + scaleSlot_);

        // Two K128 operands are staged ahead. Refill a slot only AFTER the
        // MMAD that consumes its old contents has been submitted below.
        for (uint32_t tile = 0U; tile < Min(L0A_STAGES, l0Tiles); ++tile)
            PrefetchL0A(tensorA, scaleA, rowIO, rowBegin, m, k, ka, aWindows, tile);

        auto layoutC = tla::MakeLayoutL0C(m, n);
        auto output = tla::MakeTensor(l0C_[l0CSlot_], layoutC, Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CSlot_);
        for (uint32_t tile = 0U; tile < l0Tiles; ++tile) {
            const uint32_t kStart = tile * L0_TILE_K;
            const uint32_t actualK = RoundUp<MX_BASEK_FACTOR>(Min(L0_TILE_K, k - kStart));
            const uint32_t bOffset = kStart % kb;
            const uint32_t bActual = Min(kb, k - (kStart / kb) * kb);
            auto bL1 = tla::MakeTensor(l1B_[l1BSlot_],
                                      tla::MakeLayout<ElementB, LayoutTagL1B>(kb, n), Arch::PositionL1{});
            if (bOffset == 0U) {
                if (bActual % MX_BASEK_FACTOR != 0U) {
                    AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(6U + l1BSlot_);
                    InitZeroInL1B(bL1, tla::MakeShape(bActual, n));
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(6U);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(6U);
                } else {
                    AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(6U + l1BSlot_);
                }
            }
            const uint32_t aSlot = (l0ABase_ + tile) % L0A_STAGES;
            auto aL0 = tla::MakeTensor(l0A_[aSlot],
                                      tla::MakeLayout<ElementL0A, LayoutTagL0A>(m, actualK), Arch::PositionL0A{});
            auto bL0 = tla::MakeTensor(l0B_[l0BSlot_],
                                      tla::MakeLayout<ElementB, LayoutTagL0B>(actualK, n), Arch::PositionL0B{});
            auto bTile = GetTile(bL1, tla::MakeCoord(bOffset, 0), tla::MakeShape(actualK, n));
            auto sbTile = GetTile(scaleB, tla::MakeCoord(kStart / MX_SCALE_GROUP_NUM, 0),
                                  tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualK), n));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2U + l0BSlot_);
            copyL1ToL0B_(bL0, bTile, sbTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(2U + l0BSlot_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(aSlot);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(2U + l0BSlot_);
            tileMmad_(output, aL0, bL0, m, n, actualK, tile == 0U, uint8_t{0});
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(aSlot);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2U + l0BSlot_);
            l0BSlot_ = (l0BSlot_ + 1U) % 2U;
            if (bOffset + actualK >= bActual) {
                AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(12U + l1BSlot_);
                l1BSlot_ = (l1BSlot_ + 1U) % L1B_STAGES;
            }
            if (tile + L0A_STAGES < l0Tiles)
                PrefetchL0A(tensorA, scaleA, rowIO, rowBegin, m, k, ka, aWindows, tile + L0A_STAGES);
        }
        // Every scale read is on MTE1 and precedes this release. No scale is
        // overwritten merely because one A/B data slot has become available.
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(4U + scaleSlot_);
        scaleSlot_ = (scaleSlot_ + 1U) % 2U;
        l0ABase_ = (l0ABase_ + l0Tiles) % L0A_STAGES;
        l1ABase_ = (l1ABase_ + aWindows) % L1A_STAGES;

        // Ordinary NZ C slots: release only after ALL FIX stores, including
        // every remote/self scatter segment. No UnitFlag and no scalar drain.
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CSlot_);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CSlot_);
        copyOutput.params.scale = 64;
        if constexpr (RowIO::kScatterOutput)
            rowIO.Scatter(copyOutput, tensorC, output, rowBegin, m, columnBegin, n);
        else
            copyOutput(tensorC, output);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CSlot_);
        l0CSlot_ = (l0CSlot_ + 1U) % 2U;
    }

protected:
    template <class TensorA, class RowIO>
    CATLASS_DEVICE void PrefetchL1A(TensorA& source, const RowIO& rowIO, uint32_t rowBegin,
                                    uint32_t m, uint32_t k, uint32_t ka, uint32_t window)
    {
        const uint32_t slot = (l1ABase_ + window) % L1A_STAGES;
        const uint32_t begin = window * ka;
        const uint32_t count = Min(ka, k - begin);
        auto dst = tla::MakeTensor(l1A_[slot],
                                  tla::MakeLayout<ElementA, LayoutTagL1A>(m, RoundUp<MX_BASEK_FACTOR>(count)),
                                  Arch::PositionL1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
        if constexpr (RowIO::kGatherInput) {
            rowIO.Gather(dst, rowBegin, m, begin, count);
        } else {
            typename TileCopy::template CopyGmToL1A<TensorA> copy;
            auto src = GetTile(source, tla::MakeCoord(0, begin), tla::MakeShape(m, count));
            copy(dst, src);
        }
        InitZeroInL1A(dst, tla::MakeShape(m, count));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
    }

    template <class TensorA, class TensorScale, class RowIO>
    CATLASS_DEVICE void PrefetchL0A(TensorA& source, TensorScale& scales, const RowIO& rowIO, uint32_t rowBegin,
                                    uint32_t m, uint32_t k, uint32_t ka, uint32_t windows, uint32_t tile)
    {
        const uint32_t begin = tile * L0_TILE_K;
        const uint32_t actualK = RoundUp<MX_BASEK_FACTOR>(Min(L0_TILE_K, k - begin));
        const uint32_t window = begin / ka;
        const uint32_t aStart = window * ka;
        const uint32_t count = Min(ka, k - aStart);
        const uint32_t l1Slot = (l1ABase_ + window) % L1A_STAGES;
        const uint32_t l0Slot = (l0ABase_ + tile) % L0A_STAGES;
        if (begin == aStart)
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1Slot);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Slot);
        auto l1 = tla::MakeTensor(l1A_[l1Slot],
                                 tla::MakeLayout<ElementA, LayoutTagL1A>(m, RoundUp<MX_BASEK_FACTOR>(count)),
                                 Arch::PositionL1{});
        auto l0 = tla::MakeTensor(l0A_[l0Slot],
                                 tla::MakeLayout<ElementL0A, LayoutTagL0A>(m, actualK), Arch::PositionL0A{});
        auto a = GetTile(l1, tla::MakeCoord(0, begin - aStart), tla::MakeShape(m, actualK));
        auto sa = GetTile(scales, tla::MakeCoord(0, begin / MX_SCALE_GROUP_NUM),
                          tla::MakeShape(m, CeilDiv<MX_SCALE_GROUP_NUM>(actualK)));
        copyL1ToL0A_(l0, a, sa);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Slot);
        if (begin + actualK >= aStart + count) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1Slot);
            if (window + L1A_STAGES < windows)
                PrefetchL1A(source, rowIO, rowBegin, m, k, ka, window + L1A_STAGES);
        }
    }

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
    AscendC::LocalTensor<ElementA> l1A_[L1A_STAGES];
    AscendC::LocalTensor<ElementB> l1B_[L1B_STAGES];
    AscendC::LocalTensor<ElementMxScaleA> scaleA_[2];
    AscendC::LocalTensor<ElementMxScaleB> scaleB_[2];
    AscendC::LocalTensor<ElementL0A> l0A_[L0A_STAGES];
    AscendC::LocalTensor<ElementL0B> l0B_[2];
    AscendC::LocalTensor<ElementAccumulator> l0C_[2];
    uint32_t l1ABase_ = 0U, l1BSlot_ = 0U, l0ABase_ = 0U, l0BSlot_ = 0U, l0CSlot_ = 0U, scaleSlot_ = 0U;
    bool active_ = false;
    TileMmad tileMmad_;
    CopyL1ToL0A copyL1ToL0A_;
    CopyL1ToL0B copyL1ToL0B_;
};
#endif
} // namespace Catlass::Gemm::Block
