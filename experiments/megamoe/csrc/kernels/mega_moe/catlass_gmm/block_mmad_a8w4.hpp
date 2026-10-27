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

#ifndef MEGA_MOE_CATLASS_BLOCK_MMAD_A8W4_HPP
#define MEGA_MOE_CATLASS_BLOCK_MMAD_A8W4_HPP

#include "resource.hpp"
#include "a8w4_k_plan.hpp"
#include "../common/gmm1_async_completion.hpp"
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

// Based on CATLASS example 74. Only AIV0 produces the complete L1 B tile.
template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_,
          class ElementC_, class ElementBias_, class TileCopy_,
          class TileMmad_ =
              Gemm::Tile::TileMmadTla<typename DispatchPolicy_::ArchTag, ElementA_, typename TileCopy_::LayoutTagL1A>>
struct MegaMoeBlockMmadA8W4 {
public:
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using Buffers = MegaMoeImpl::GmmKernel::A8W4BufferPlan;
    using TileCopy = TileCopy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutA = typename TileCopy::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopy::LayoutB;
    using ElementMxScaleA = typename TileCopy::ElementMxScaleA;
    using ElementMxScaleB = typename TileCopy::ElementMxScaleB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy::LayoutC;
    using ElementBias = ElementBias_;
    using ElementL0A = typename helper::GetL0Element<ElementA, true>::Element;
    using ElementL0B = typename helper::GetL0Element<ElementB, true>::Element;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyL1ToBT = typename TileCopy::CopyL1ToBT;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    static constexpr bool HAS_BIAS = TileCopy::HAS_BIAS;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL1MxScaleA = typename TileCopy::LayoutTagL1MxScaleA;
    using LayoutTagL1MxScaleB = typename TileCopy::LayoutTagL1MxScaleB;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static_assert(tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
                  "L1TileShape must be tla::tuple and static!");
    static_assert(tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
                  "L0TileShape must be tla::tuple and static!");

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr bool ENABLE_L1_RESIDENT = DispatchPolicy::ENABLE_L1_RESIDENT;
    static constexpr uint32_t L1A_STAGES = DispatchPolicy::L1A_STAGES;
    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;

    static constexpr uint32_t L1_SCALE_FACTOR_K = DispatchPolicy::L1_SCALE_FACTOR_K;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    // L1 tile size
    // 收紧 A 容量以增加独立 B 槽；ka 必须服从共享容量约束。
    static constexpr uint32_t L1A_TILE_SIZE = Buffers::ABytes;
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * SizeOfBits<ElementB>::value / 8;
    static constexpr uint32_t L1SCALEA_TILE_SIZE =
        L1_TILE_M * L1_TILE_K / MX_SCALE_GROUP_NUM * sizeof(ElementMxScaleA) * L1_SCALE_FACTOR_K;
    static constexpr uint32_t L1SCALEB_TILE_SIZE =
        L1_TILE_N * L1_TILE_K / MX_SCALE_GROUP_NUM * sizeof(ElementMxScaleB) * L1_SCALE_FACTOR_K;
    static constexpr uint32_t L1_USED_SIZE = L1A_TILE_SIZE * L1A_STAGES + L1B_TILE_SIZE * L1B_STAGES +
                                             L1SCALEA_TILE_SIZE * Buffers::ScaleStages +
                                             L1SCALEB_TILE_SIZE * Buffers::ScaleStages;
    static_assert(L1_USED_SIZE == 512U * 1024U && L1SCALEA_TILE_SIZE == Buffers::ScaleBytes &&
                  L1SCALEB_TILE_SIZE == Buffers::ScaleBytes);
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * SizeOfBits<ElementL0A>::value / 8;
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * SizeOfBits<ElementL0B>::value / 8;
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    static_assert(L1_TILE_M == 256 && L1_TILE_N == 256 && L1_TILE_K == 256 &&
                      L1A_STAGES == Buffers::AStages && L1B_STAGES == Buffers::BStages &&
                      L1_SCALE_FACTOR_K == 8 && SizeOfBits<ElementA>::value == 8 && SizeOfBits<ElementB>::value == 8 &&
                      !HAS_BIAS,
                  "A8W4 requires A64KiB x3, B64KiB x4, scale16KiB x2 per operand and no bias");

    // Check L0C_STAGES
    static_assert(!(ENABLE_UNIT_FLAG && L0C_STAGES != 1), "L0C_STAGES must be 1 when UnitFlag is true!");

    // Check LayoutC
    static_assert(tla::detail::isRowMajor<LayoutC>::value ||
                      ((std::is_same_v<ElementC, half> || std::is_same_v<ElementC, bfloat16_t> ||
                        std::is_same_v<ElementC, float>) &&
                       tla::detail::iszN<ElementC, LayoutC>::value),
                  "LayoutC only supports zN in half or bfloat16 or float, RowMajor in all dtype yet!");

    // Check L1TileShape::K and L0TileShape::K
    static_assert(L1_TILE_K % MX_BASEK_FACTOR == 0 && L0_TILE_K % MX_BASEK_FACTOR == 0,
                  "L1TileShape::K and L0TileShape::K must be multiples of 64!");

    // Check L1TileShape
    static_assert(L1_USED_SIZE <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static_assert(L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
                  "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");
    static_assert(L0_TILE_K <= L1_TILE_K, "L0TileShape::K cannot exceed L1TileShape::K");

    static_assert((!HAS_BIAS && (L1A_STAGES + L1B_STAGES) <= 8) || (HAS_BIAS && (L1A_STAGES + L1B_STAGES) <= 7),
                  "L1 Buffer overflow: Exceeds the supported range of EVENT(0~7)");

    static_assert((!HAS_BIAS && (L0A_STAGES + L0B_STAGES) <= 8) || (HAS_BIAS && (L0A_STAGES + L0B_STAGES) <= 7),
                  "L0 Buffer overflow: Exceeds the supported range of EVENT_ID(0~7)");

    static constexpr auto L1A_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{});
    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});
    static constexpr auto L1SCALEA_LAYOUT = tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagL1MxScaleA, false>(
        tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K / MX_SCALE_GROUP_NUM * L1_SCALE_FACTOR_K>{});
    static constexpr auto L1SCALEB_LAYOUT = tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagL1MxScaleB, true>(
        tla::Int<L1_TILE_K / MX_SCALE_GROUP_NUM * L1_SCALE_FACTOR_K>{}, tla::Int<L1_TILE_N>{});
    static constexpr auto L1BIAS_LAYOUT = tla::MakeLayout(tla::Int<L1_TILE_N>{});
    static constexpr auto L0BIAS_LAYOUT = tla::MakeLayout(tla::Int<L0_TILE_N>{});

    // When enabling L1 resident mode, restore the pointer and coordinates that record the last state
    // to the initial state. if tow blockmmad instances need to be consecutively invoked at the kernel layer,
    // RestoreStatus() must be inserted between them.
    CATLASS_DEVICE
    void RestoreStatus()
    {
        for (uint32_t i = 0; i < L1A_STAGES; ++i) {
            lastAddrA[i] = nullptr;
            lastCoordA[i] = MatrixCoord{0U, 0U};
        }
        for (uint32_t i = 0; i < L1B_STAGES; ++i) {
            lastAddrB[i] = nullptr;
            lastCoordB[i] = MatrixCoord{0U, 0U};
        }
        l1MxScaleAListId = 0;
        l1MxScaleBListId = 0;
    }

    /// Construct
    CATLASS_DEVICE
    MegaMoeBlockMmadA8W4(MegaMoeResource& resource, uint32_t l1BufAddrStart = 0, bool activate = true)
    {
        // L1: A[0,192KiB), B[192,448KiB), SA[448,480KiB), SB[480,512KiB).
        // MTE1_MTE2: A 使用 0..2，scale 使用 3..4；padding 的 7 独立。
        uint32_t l1Offset = l1BufAddrStart;
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1BufAddrStart + Buffers::BBase + i * Buffers::BBytes);
            l1Offset += L1B_TILE_SIZE;
            // Assign event ID for each stages
            l1BEventList[i] = i + L1A_STAGES;
        }
        for (uint32_t i = 0; i < L1A_STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1ATensorList[i] =
                resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + Buffers::ABase + i * L1A_TILE_SIZE);
            l1Offset += L1A_TILE_SIZE;
            // Assign event ID for each stages
            l1AEventList[i] = i;
        }
        for (uint32_t i = 0; i < L0A_STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementL0A>(L0A_TILE_SIZE * i);
            // Assign event ID for each stages
            l0AEventList[i] = i;
        }
        for (uint32_t i = 0; i < L0B_STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementL0B>(L0B_TILE_SIZE * i);
            //  Assign event ID for each stages
            l0BEventList[i] = L0A_STAGES + i;  // 同一A需被两个N半块消费，B独立释放。
        }
        if constexpr (!ENABLE_UNIT_FLAG) {
            for (uint32_t i = 0; i < L0C_STAGES; i++) {
                l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * i);
                l0CEventList[i] = i;
            }
        } else {
            l0CTensorList[l0CListId] =
                resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * l0CListId);
        }
        if constexpr (HAS_BIAS) {
            l1BiasTensor = resource.l1Buf.template GetBufferByByte<uint8_t>(l1Offset);
            l1Offset += L1_TILE_N * sizeof(ElementBias);
            l0BiasTensor = resource.btBuf.template GetBufferByByte<ElementAccumulator>(0);
        }
        for (uint32_t i = 0; i < Buffers::ScaleStages; i++) {
            l1MxScaleATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementMxScaleA>(
                l1BufAddrStart + Buffers::ScaleABase + i * Buffers::ScaleBytes);
            l1Offset += L1SCALEA_TILE_SIZE;
        }
        for (uint32_t i = 0; i < Buffers::ScaleStages; i++) {
            l1MxScaleBTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementMxScaleB>(
                l1BufAddrStart + Buffers::ScaleBBase + i * Buffers::ScaleBytes);
            l1Offset += L1SCALEB_TILE_SIZE;
        }

        if (activate) {
            Activate();
        }
    }

    // Buffer views/event IDs are initialized once; stages sharing this hardware
    // still acquire/release the event tokens at each problem boundary.
    CATLASS_DEVICE
    void Activate()
    {
        if (active_) {
            return;
        }
        l1AListId = l1BListId = l0AListId = l0BListId = l0CListId = 0;
        RestoreStatus();
        if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
            AscendC::SetMMLayoutTransform(true);
        }
        for (uint32_t i = 0; i < L1A_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        for (uint32_t i = 0; i < L1B_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        if constexpr (!ENABLE_UNIT_FLAG) {
            for (uint32_t i = 0; i < L0C_STAGES; ++i) {
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
            }
        }
        if constexpr (HAS_BIAS) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
        }
        active_ = true;
    }

    // 调用者保证 next 不持有硬件资源；只交接同一物理布局的游标，不重新投放 credit。
    // 已发出的 DMA/MMAD/FIX 仍由原事件依赖保护，next 从准确的下一槽继续。
    CATLASS_DEVICE
    void MovePipelineTo(MegaMoeBlockMmadA8W4& next)
    {
        static_assert(ENABLE_UNIT_FLAG && L0C_STAGES == 1 && !ENABLE_L1_RESIDENT && !HAS_BIAS,
                      "Handoff requires unit-flag L0C ownership and no resident operand cache");
        next.l1AListId = l1AListId;
        next.l1BListId = l1BListId;
        next.l0AListId = l0AListId;
        next.l0BListId = l0BListId;
        next.l0CListId = l0CListId;
        next.l1MxScaleAListId = l1MxScaleAListId;
        next.l1MxScaleBListId = l1MxScaleBListId;
        next.active_ = active_;
        active_ = false;
    }

    CATLASS_DEVICE
    void Deactivate()
    {
        if (!active_) {
            return;
        }
        if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
            AscendC::SetMMLayoutTransform(false);
        }
        for (uint32_t i = 0; i < L1A_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        if constexpr (!ENABLE_UNIT_FLAG) {
            for (uint32_t i = 0; i < L0C_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
            }
        }
        if constexpr (HAS_BIAS) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
        }
        active_ = false;
    }

    CATLASS_DEVICE
    ~MegaMoeBlockMmadA8W4()
    {
        Deactivate();
    }


    /// Perform a block-scoped matrix multiply-accumulate
    template <class TensorA, class TensorC, class TensorMxScaleA = EmptyClass, class TensorMxScaleB = EmptyClass,
              class TensorBias = EmptyClass, class InputReady = EmptyClass>
    CATLASS_DEVICE void operator()(TensorA& tensorA, TensorC& tensorC, GemmCoord const& actualShape,
                                   TensorMxScaleA const& tensorMxScaleA = {}, TensorMxScaleB const& tensorMxScaleB = {},
                                   bool allowDynamicB = false, TensorBias const& tensorBias = {},
                                   InputReady inputReady = {})
    {
        // Check L1TileShape
        if constexpr (HAS_BIAS) {
            static constexpr uint32_t BIAS_BUF_SIZE = L0_TILE_N * sizeof(ElementAccumulator);
            static constexpr uint32_t L1BIAS_SIZE = L1_TILE_N * sizeof(ElementBias);
            static_assert(BIAS_BUF_SIZE <= ArchTag::BIAS_SIZE,
                          "BIAS_BUF_SIZE exceeding the BT space! Reduce L0_TILE_N");
            static_assert(L1_USED_SIZE + L1BIAS_SIZE <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");
        }

        // datacopy tile
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1MxScaleA = typename TileCopy::template CopyGmToL1MxScaleA<TensorMxScaleA>;
        using CopyGmToL1MxScaleB = typename TileCopy::template CopyGmToL1MxScaleB<TensorMxScaleB>;
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorC>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1MxScaleA copyGmToL1MxScaleA;
        CopyGmToL1MxScaleB copyGmToL1MxScaleB;
        CopyL0CToDst copyL0CToDst;

        uint32_t l1ScaleTileK = L1_TILE_K * L1_SCALE_FACTOR_K;

        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();

        uint32_t mL1Actual = mBlockActual;
        if constexpr (std::is_same_v<ArchTag, Arch::AtlasA2>) {
            // Avoid using the gemv mode in mmad
            if (mL1Actual == 1) {
                mL1Actual = 16;
            }
        }
        uint32_t nL1Actual = nBlockActual;

        auto layoutInL0C = tla::MakeLayoutL0C(mL1Actual, nL1Actual);
        auto tensorL0C = tla::MakeTensor(l0CTensorList[l0CListId], layoutInL0C, Arch::PositionL0C{});
        auto tensorL0Bias = tla::MakeTensor(l0BiasTensor, L0BIAS_LAYOUT, Arch::PositionBias{});

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
        }

        const auto kPlan = MegaMoeImpl::GmmKernel::MakeA8W4KPlan(mBlockActual);
        const uint32_t ka = kPlan.ka;
        const uint32_t kb = kPlan.kb;
        const uint32_t kL1Loop = CeilDiv(kBlockActual, kb);
        // 最后K衔接：一个N半块完成累加就发FIX，不等另一半的MMAD提交。
        // 回调只负责输出；保持scale=64、父L0C stride和每半块的unit flag。
        auto emitHalf = [&](const auto& accumulator, uint32_t column, uint32_t columns) __aicore__ {
            auto destination = GetTile(tensorC, tla::MakeCoord(0U, column),
                                       tla::MakeShape(mBlockActual, columns));
            typename TileCopy::template CopyL0CToDst<decltype(destination)> copy;
            copy.params.scale = 64;
            copy(destination, accumulator, 0b11);
        };
        for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; ++kL1Idx) {
            const uint32_t kOffset = kL1Idx * kb;
            const uint32_t kL1Actual = Min(kb, kBlockActual - kOffset);
            const uint32_t aStart = kOffset / ka * ka;
            const uint32_t aActual = Min(ka, kBlockActual - aStart);
            // Scale buffers have their own lifetime: neither A nor B window release
            // permits overwriting scales that are still used by this scale-K window.
            const uint32_t scaleStart = kOffset / l1ScaleTileK * l1ScaleTileK;
            const uint32_t scaleActual = Min(l1ScaleTileK, kBlockActual - scaleStart);
            const uint32_t scaleCount = CeilDiv<MX_BASEK_FACTOR>(scaleActual) * 2U;
            auto scaleLayoutA =
                tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagL1MxScaleA, false>(mL1Actual, scaleCount);
            auto scaleLayoutB =
                tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagL1MxScaleB, true>(scaleCount, nL1Actual);
            if (kOffset == scaleStart) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1MxScaleAListId]);
                auto tensorL1MxScaleA =
                    tla::MakeTensor(l1MxScaleATensorList[l1MxScaleAListId], scaleLayoutA, Arch::PositionL1{});
                auto tensorL1MxScaleB =
                    tla::MakeTensor(l1MxScaleBTensorList[l1MxScaleBListId], scaleLayoutB, Arch::PositionL1{});
                auto tileScaleA = GetTile(tensorMxScaleA, tla::MakeCoord(0, scaleStart / MX_SCALE_GROUP_NUM),
                                          tla::MakeShape(mBlockActual, scaleCount));
                auto tileScaleB = GetTile(tensorMxScaleB, tla::MakeCoord(scaleStart / MX_SCALE_GROUP_NUM, 0),
                                          tla::MakeShape(scaleCount, nBlockActual));
                if constexpr (std::is_same_v<InputReady,EmptyClass>) {
                    copyGmToL1MxScaleA(tensorL1MxScaleA, tileScaleA);
                    copyGmToL1MxScaleB(tensorL1MxScaleB, tileScaleB);
                } else {
                    // B scale 与输入ready无关，先提交MTE2；A及A scale仍必须在原ready边界之后。
                    copyGmToL1MxScaleB(tensorL1MxScaleB, tileScaleB);
                    if (kL1Idx==0U) inputReady();
                    copyGmToL1MxScaleA(tensorL1MxScaleA, tileScaleA);
                }
            }

            // A窗口随M切换K512/K256，均不超过独立64KiB槽。
            auto tensorL1A =
                tla::MakeTensor(l1ATensorList[l1AListId],
                                tla::MakeLayout<ElementA, LayoutTagL1A>(mL1Actual, RoundUp<MX_BASEK_FACTOR>(aActual)),
                                Arch::PositionL1{});
            // 首窗显式加载；后续A窗已在上一窗提交，不能重复消耗同一empty credit。
            if (kL1Idx == 0U) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                auto tensorTileA = GetTile(tensorA, tla::MakeCoord(0, aStart), tla::MakeShape(mBlockActual, aActual));
                copyGmToL1A(tensorL1A, tensorTileA);
                InitZeroInL1A(tensorL1A, tla::MakeShape(mL1Actual, aActual));
            }

            const uint32_t bChunks = CeilDiv(nL1Actual, kPlan.bColumns);
            for (uint32_t chunk = 0; chunk < bChunks; ++chunk) {
                const uint32_t slot = (l1BListId + chunk) % L1B_STAGES;
                const uint32_t columns = Min(kPlan.bColumns, nL1Actual - chunk * kPlan.bColumns);
                auto paddedB = tla::MakeTensor(l1BTensorList[slot],
                    tla::MakeLayout<ElementB, LayoutTagL1B>(kb, columns), Arch::PositionL1{});
                // Only an incomplete 64-K MX group needs MTE2 padding after AIV0's store.
                // Aligned blocks wait on MTE1, leaving MTE2 free to prefetch A/scales.
                if (kL1Actual % MX_BASEK_FACTOR != 0U) {
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_MTE2>(AIV_SYNC_AIC_FLAG + slot);
                    InitZeroInL1B(paddedB, tla::MakeShape(kL1Actual, columns));
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
                } else {
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_MTE1>(AIV_SYNC_AIC_FLAG + slot);
                }
            }
            // One fence covers scale and A transfers at each K-window boundary.
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            auto tensorL1MxScaleA =
                tla::MakeTensor(l1MxScaleATensorList[l1MxScaleAListId], scaleLayoutA, Arch::PositionL1{});
            auto tensorL1MxScaleB =
                tla::MakeTensor(l1MxScaleBTensorList[l1MxScaleBListId], scaleLayoutB, Arch::PositionL1{});
            const uint32_t l0kOffset = kOffset - scaleStart;
            const uint32_t kL0Loop = CeilDiv<L0_TILE_K>(kL1Actual);

            // 当前A/scale的MTE2->MTE1完成已在上方排入；再预取下一A窗，
            // 使其可与本窗MTE1/MMAD重叠。必须放在当前完成通知之后，
            // 否则下一槽的empty等待可能挡住当前消费。容量/槽号及最后读后归还规则不变。
            const uint32_t nextK = kOffset + kL1Actual;
            if (nextK < kBlockActual && nextK % ka == 0U) {
                const uint32_t nextASlot = (l1AListId + 1U) % L1A_STAGES;
                const uint32_t nextAActual = Min(ka, kBlockActual - nextK);
                auto nextA = tla::MakeTensor(l1ATensorList[nextASlot],
                    tla::MakeLayout<ElementA, LayoutTagL1A>(mL1Actual, RoundUp<MX_BASEK_FACTOR>(nextAActual)),
                    Arch::PositionL1{});
                auto nextTile = GetTile(tensorA, tla::MakeCoord(0, nextK),
                                        tla::MakeShape(mBlockActual, nextAActual));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[nextASlot]);
                copyGmToL1A(nextA, nextTile);
                InitZeroInL1A(nextA, tla::MakeShape(mL1Actual, nextAActual));
            }

            if (mL1Actual <= 128U) {
                MmadPairedB(tensorL1A, tensorL1MxScaleA, tensorL1MxScaleB, tensorL0C,
                            mL1Actual, nL1Actual, kb, kL1Actual, kOffset - aStart, l0kOffset,
                            kL1Idx == 0U, kL1Idx + 1U == kL1Loop, emitHalf);
            } else {
                auto tensorL1B = tla::MakeTensor(
                    l1BTensorList[l1BListId], tla::MakeLayout<ElementB, LayoutTagL1B>(kb, nL1Actual), Arch::PositionL1{});
                for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? L0_TILE_K : (kL1Actual - kL0Idx * L0_TILE_K);

                    // mmad k must be multiples of 64 when mx type
                    kL0Actual = RoundUp<MX_BASEK_FACTOR>(kL0Actual);

                    // Locate the current tile on L0A
                    auto layoutAInL0 = tla::MakeLayout<ElementL0A, LayoutTagL0A>(mL1Actual, kL0Actual);
                    auto tensorL0A = tla::MakeTensor(l0ATensorList[l0AListId], layoutAInL0, Arch::PositionL0A{});

                    // Locate the current tile of matrix A on L1
                    auto tensorTileL1A = GetTile(tensorL1A, tla::MakeCoord(0, kOffset - aStart + kL0Idx * L0_TILE_K),
                                                 tla::MakeShape(mL1Actual, kL0Actual));

                    // Locate the current tile of matrix mxScaleA on L1
                    auto tensorTileL1MxScaleA =
                        GetTile(tensorL1MxScaleA, tla::MakeCoord(0, (l0kOffset + kL0Idx * L0_TILE_K) / MX_SCALE_GROUP_NUM),
                                tla::MakeShape(mL1Actual, CeilDiv<MX_SCALE_GROUP_NUM>(kL0Actual)));

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);

                    // Load current tile from L1 to L0A
                    copyL1ToL0A(tensorL0A, tensorTileL1A, tensorTileL1MxScaleA);

                    // Locate the current tile on L0B
                    auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, nL1Actual);
                    auto tensorL0B = tla::MakeTensor(l0BTensorList[l0BListId], layoutBInL0, Arch::PositionL0B{});

                    // Locate the current tile of matrix mxScaleB on L1
                    auto tensorTileL1B =
                        GetTile(tensorL1B, tla::MakeCoord(kL0Idx * L0_TILE_K, 0), tla::MakeShape(kL0Actual, nL1Actual));

                    // Locate the current tile of matrix mxScaleB on L1
                    auto tensorTileL1MxScaleB =
                        GetTile(tensorL1MxScaleB, tla::MakeCoord((l0kOffset + kL0Idx * L0_TILE_K) / MX_SCALE_GROUP_NUM, 0),
                                tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(kL0Actual), nL1Actual));

                    // Wait for mmad finished
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);

                    // Load current tile from L1 to L0B
                    copyL1ToL0B(tensorL0B, tensorTileL1B, tensorTileL1MxScaleB);

                    bool initC = ((kL1Idx == 0) && (kL0Idx == 0));
                    if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
                        if (initC) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                            auto l1Bias = l1BiasTensor.template ReinterpretCast<ElementBias>();
                            auto tensorL1Bias = tla::MakeTensor(l1Bias, L1BIAS_LAYOUT, Arch::PositionL1{});
                            auto tensorTileL1Bias = GetTile(tensorL1Bias, tla::MakeCoord(0), tla::MakeShape(nL1Actual));
                            // Load bias to l0 biastable
                            copyL1ToBT(tensorL0Bias, tensorTileL1Bias);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
                        }
                    }

                    // Notify to do mmad
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);

                    // If the unit flag is enabled, the unit flag is set according to the calculation progress
                    uint8_t unitFlag = 0b00;
                    if constexpr (ENABLE_UNIT_FLAG) {
                        if ((kL1Idx == kL1Loop - 1) && (kL0Idx == kL0Loop - 1)) {
                            unitFlag = 0b11;
                        } else {
                            unitFlag = 0b10;
                        }
                    }
                    if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
                        if (initC) {
                            tileMmad(tensorL0C, tensorL0A, tensorL0B, tensorL0Bias, mL1Actual, nL1Actual, kL0Actual, initC,
                                     unitFlag);
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                        } else {
                            tileMmad(tensorL0C, tensorL0A, tensorL0B, mL1Actual, nL1Actual, kL0Actual, initC, unitFlag);
                        }
                    } else {
                        tileMmad(tensorL0C, tensorL0A, tensorL0B, mL1Actual, nL1Actual, kL0Actual, initC, unitFlag);
                    }

                    // Notify to move the next L0B tile
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;

                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
                }

            }
            if (nextK == kBlockActual || nextK % ka == 0U) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                l1AListId = (l1AListId + 1) % L1A_STAGES;
            }
            for (uint32_t chunk = 0; chunk < bChunks; ++chunk) {
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_MTE1>(
                    AIC_SYNC_AIV_FLAG + (l1BListId + chunk) % L1B_STAGES);
            }
            l1BListId = (l1BListId + bChunks) % L1B_STAGES;
            if (nextK == kBlockActual || nextK % l1ScaleTileK == 0U) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1MxScaleAListId]);
                l1MxScaleAListId = (l1MxScaleAListId + 1) % Buffers::ScaleStages;
                l1MxScaleBListId = (l1MxScaleBListId + 1) % Buffers::ScaleStages;
            }
        }
        if constexpr (ENABLE_UNIT_FLAG) {
            if (mL1Actual <= 128U) return;  // 两个N半块已在最后K中各提交一次FIX。
        }
        // copy block out
        copyL0CToDst.params.scale = 64;
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
            copyL0CToDst(tensorC, tensorL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
        } else {
            copyL0CToDst(tensorC, tensorL0C, 0b11);
        }
    }

protected:

    // M<=128 时，32KiB L0A 可容纳 M×256，32KiB L0B 可容纳 256×128。
    // 同一 A 供两个 N 半块使用；只有最后一半消费后才归还 A credit。
    // UnitFlag 是按 L0C 存储单元生效：每个 N 半块各自在最后 K 发布完成，
    // 整块 FIX 才能等待并消费两半；只给第二半 UPDATE 会让第一半永远不就绪。
    template <class A, class SA, class SB, class C, class Emit>
    CATLASS_DEVICE void MmadPairedB(A& a, SA& sa, SB& sb, C& c, uint32_t m, uint32_t n,
                                   uint32_t bStrideK, uint32_t kActual, uint32_t aOffset, uint32_t scaleOffset,
                                   bool firstKWindow, bool lastKWindow, Emit& emit)
    {
        static_assert(!HAS_BIAS && L0A_TILE_SIZE >= 128U * 256U && L0B_TILE_SIZE >= 256U * 128U,
                      "split-N requires 32KiB operand slots and no bias");
        for (uint32_t k = 0; k < kActual; k += 256U) {
            const uint32_t kk = RoundUp<MX_BASEK_FACTOR>(Min(256U, kActual - k));
            auto l0a = tla::MakeTensor(l0ATensorList[l0AListId],
                tla::MakeLayout<ElementL0A, LayoutTagL0A>(m, kk), Arch::PositionL0A{});
            auto ta = GetTile(a, tla::MakeCoord(0, aOffset + k), tla::MakeShape(m, kk));
            auto tsa = GetTile(sa, tla::MakeCoord(0, (scaleOffset + k) / MX_SCALE_GROUP_NUM),
                               tla::MakeShape(m, CeilDiv<MX_SCALE_GROUP_NUM>(kk)));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
            copyL1ToL0A(l0a, ta, tsa);
            for (uint32_t no = 0; no < n; no += 128U) {
                const uint32_t nn = Min(128U, n - no);
                auto l0b = tla::MakeTensor(l0BTensorList[l0BListId],
                    tla::MakeLayout<ElementB, LayoutTagL0B>(kk, nn), Arch::PositionL0B{});
                const uint32_t bSlot = (l1BListId + no / 128U) % L1B_STAGES;
                auto b = tla::MakeTensor(l1BTensorList[bSlot],
                    tla::MakeLayout<ElementB, LayoutTagL1B>(bStrideK, nn), Arch::PositionL1{});
                auto tb = GetTile(b, tla::MakeCoord(k, 0), tla::MakeShape(kk, nn));
                auto tsb = GetTile(sb, tla::MakeCoord((scaleOffset + k) / MX_SCALE_GROUP_NUM, no),
                                   tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(kk), nn));
                // GetTile 只记录逻辑 coord，而 TileMmad 直接使用 data()。
                // L0C 必须显式偏移，否则第二半会覆盖第一半。
                // unit-flag路径每半块独立FIX；保持64KiB槽间距，不把M64补算成M128。
                // 非unit-flag路径仍需后续整块copy，保留原父layout，不能只改写入地址。
                const auto cOffset = ENABLE_UNIT_FLAG ? no / 128U * Buffers::SmallL0CHalfElements
                                                     : c.layout()(tla::MakeCoord(0, no));
                auto tc = tla::MakeTensor(c.data()[cOffset],
                                          tla::MakeLayoutL0C(m, nn), Arch::PositionL0C{});
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                copyL1ToL0B(l0b, tb, tsb);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);
                const bool last = lastKWindow && k + kk >= kActual;
                const uint8_t unitFlag = ENABLE_UNIT_FLAG ? (last ? 0b11 : 0b10) : 0b00;
                tileMmad(tc, l0a, l0b, m, nn, kk, firstKWindow && k == 0U, unitFlag);
                if constexpr (ENABLE_UNIT_FLAG) {
                    if (last) emit(tc, no, nn);
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                l0BListId = (l0BListId + 1U) % L0B_STAGES;
            }
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
            l0AListId = (l0AListId + 1U) % L0A_STAGES;
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

    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementMxScaleA> l1MxScaleATensorList[Buffers::ScaleStages];
    AscendC::LocalTensor<ElementMxScaleB> l1MxScaleBTensorList[Buffers::ScaleStages];
    AscendC::LocalTensor<ElementL0A> l0ATensorList[L0A_STAGES];
    AscendC::LocalTensor<ElementL0B> l0BTensorList[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];
    AscendC::LocalTensor<uint8_t> l1BiasTensor;
    AscendC::LocalTensor<ElementAccumulator> l0BiasTensor;

    // Multi-stage event id list
    int32_t l1AEventList[L1A_STAGES];
    int32_t l1BEventList[L1B_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    int32_t l0CEventList[L0C_STAGES];

    __gm__ typename AscendC::GlobalTensor<ElementA>::PrimType* lastAddrA[L1A_STAGES];
    __gm__ typename AscendC::GlobalTensor<ElementB>::PrimType* lastAddrB[L1B_STAGES];
    MatrixCoord lastCoordA[L1A_STAGES];
    MatrixCoord lastCoordB[L1B_STAGES];

    bool active_ = false;

    // The id of current stage
    uint32_t l1AListId{0};
    uint32_t l1BListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
    uint32_t l0CListId{0};

    uint32_t l1MxScaleAListId{0};  // scale id only used in Mmad3
    uint32_t l1MxScaleBListId{0};

    constexpr static uint8_t AIC_SYNC_AIV_MODE = 4;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = Buffers::BFull;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = Buffers::BEmpty;

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL1ToBT copyL1ToBT;
};

#endif  // (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

}  // namespace Catlass::Gemm::Block

#endif  // MEGA_MOE_CATLASS_BLOCK_MMAD_A8W4_HPP
