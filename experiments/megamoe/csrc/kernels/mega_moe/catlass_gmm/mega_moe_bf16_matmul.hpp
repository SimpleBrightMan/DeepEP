// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

/*!
 * \file mega_moe_bf16_matmul.hpp
 * \brief 非持有型 CATLASS Ascend950 BF16xBF16->BF16 矩阵乘封装（纯 AIC 执行，不占用 AIV0，不创建 TPipe）。
 */

#ifndef MEGA_MOE_CATLASS_BF16_MATMUL_HPP
#define MEGA_MOE_CATLASS_BF16_MATMUL_HPP

#include "resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace MegaMoeImpl::GmmKernel {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_,
          class ElementC_, class TileCopy_>
struct MegaMoeBlockMmadBf16 {
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;
    using LayoutC = typename TileCopy::LayoutC;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, ElementA, LayoutTagL1A>;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t L1A_STAGES = DispatchPolicy::L1A_STAGES;
    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape_{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape_{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape_{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape_{});

    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);

    static_assert(ENABLE_UNIT_FLAG && L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
                  "MegaMoeBlockMmadBf16 requires unit-flag L0C and matching M/N tile shapes");
    static_assert(L1A_TILE_SIZE * L1A_STAGES + L1B_TILE_SIZE * L1B_STAGES <= ArchTag::L1_SIZE, "L1 overflow");
    static_assert(L0A_TILE_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE, "L0A overflow");
    static_assert(L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "L0B overflow");

    static constexpr auto L1A_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{});
    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});

    CATLASS_DEVICE MegaMoeBlockMmadBf16(Catlass::Gemm::Block::MegaMoeResource& resource, uint32_t l1BufAddrStart = 0)
    {
        if constexpr (g_coreType == AscendC::AIC) {
            AscendC::SetHF32Mode(false);
            if constexpr (tla::detail::isRowMajor<LayoutC>::value) {
                AscendC::SetMMLayoutTransform(true);
            }
            const uint32_t l1AOffset = l1BufAddrStart;
            const uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * L1A_STAGES;
            for (uint32_t i = 0; i < L1A_STAGES; ++i) {
                l1ATensorList_[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_TILE_SIZE * i);
                l1AEventList_[i] = static_cast<int32_t>(i);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList_[i]);
            }
            for (uint32_t i = 0; i < L1B_STAGES; ++i) {
                l1BTensorList_[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);
                l1BEventList_[i] = static_cast<int32_t>(i + L1A_STAGES);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[i]);
            }
            for (uint32_t i = 0; i < L0A_STAGES; ++i) {
                l0ATensorList_[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_TILE_SIZE * i);
                l0AEventList_[i] = static_cast<int32_t>(i);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; ++i) {
                l0BTensorList_[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * i);
                l0BEventList_[i] = static_cast<int32_t>(i + L0A_STAGES);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[i]);
            }
            l0CTensor_ = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        }
    }

    CATLASS_DEVICE ~MegaMoeBlockMmadBf16()
    {
        if constexpr (g_coreType == AscendC::AIC) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            if constexpr (tla::detail::isRowMajor<LayoutC>::value) {
                AscendC::SetMMLayoutTransform(false);
            }
            for (uint32_t i = 0; i < L1A_STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList_[i]);
            }
            for (uint32_t i = 0; i < L1B_STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[i]);
            }
            for (uint32_t i = 0; i < L0A_STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[i]);
            }
        }
    }

    template <class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE void operator()(TensorA& tensorA, TensorB& tensorB, TensorC& tensorC,
                                   const Catlass::GemmCoord& actualShape)
    {
        if constexpr (g_coreType != AscendC::AIC) {
            return;
        }
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorC>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyL0CToDst copyL0CToDst;

        const uint32_t mActual = actualShape.m();
        const uint32_t nActual = actualShape.n();
        const uint32_t kActual = actualShape.k();

        auto tensorL0C = tla::MakeTensor(l0CTensor_, tla::MakeLayoutL0C(mActual, nActual), Catlass::Arch::PositionL0C{});
        uint32_t kL1Actual = kActual < L1_TILE_K ? kActual : L1_TILE_K;

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList_[l1AListId_]);
        auto tensorL1A = tla::MakeTensor(l1ATensorList_[l1AListId_], L1A_LAYOUT, Catlass::Arch::PositionL1{});
        auto tileA0 = tla::GetTile(tensorA, tla::MakeCoord(0U, 0U), tla::MakeShape(mActual, kL1Actual));
        copyGmToL1A(tensorL1A, tileA0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList_[l1AListId_]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1BListId_]);
        auto tensorL1B = tla::MakeTensor(l1BTensorList_[l1BListId_], L1B_LAYOUT, Catlass::Arch::PositionL1{});
        auto tileB0 = tla::GetTile(tensorB, tla::MakeCoord(0U, 0U), tla::MakeShape(kL1Actual, nActual));
        copyGmToL1B(tensorL1B, tileB0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1BListId_]);

        const uint32_t kL1Loop = (kActual + L1_TILE_K - 1U) / L1_TILE_K;
        for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; ++kL1Idx) {
            const uint32_t l1ANext = (l1AListId_ + 1U) % L1A_STAGES;
            const uint32_t l1BNext = (l1BListId_ + 1U) % L1B_STAGES;
            uint32_t kL1Next = 0U;
            if (kL1Idx + 1U < kL1Loop) {
                const uint32_t nextIdx = kL1Idx + 1U;
                kL1Next = (nextIdx + 1U < kL1Loop) ? L1_TILE_K : (kActual - nextIdx * L1_TILE_K);
                auto nextL1A = tla::MakeTensor(l1ATensorList_[l1ANext], L1A_LAYOUT, Catlass::Arch::PositionL1{});
                auto nextL1B = tla::MakeTensor(l1BTensorList_[l1BNext], L1B_LAYOUT, Catlass::Arch::PositionL1{});
                auto nextTileA = tla::GetTile(tensorA, tla::MakeCoord(0U, nextIdx * L1_TILE_K),
                                              tla::MakeShape(mActual, kL1Next));
                auto nextTileB = tla::GetTile(tensorB, tla::MakeCoord(nextIdx * L1_TILE_K, 0U),
                                              tla::MakeShape(kL1Next, nActual));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList_[l1ANext]);
                copyGmToL1A(nextL1A, nextTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList_[l1ANext]);

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1BNext]);
                copyGmToL1B(nextL1B, nextTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1BNext]);
            }

            auto curL1A = tla::MakeTensor(l1ATensorList_[l1AListId_], L1A_LAYOUT, Catlass::Arch::PositionL1{});
            auto curL1B = tla::MakeTensor(l1BTensorList_[l1BListId_], L1B_LAYOUT, Catlass::Arch::PositionL1{});
            const uint32_t kL0Loop = (kL1Actual + L0_TILE_K - 1U) / L0_TILE_K;
            for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; ++kL0Idx) {
                const uint32_t kL0Actual =
                    (kL0Idx + 1U < kL0Loop) ? L0_TILE_K : (kL1Actual - kL0Idx * L0_TILE_K);
                auto tensorL0A = tla::MakeTensor(l0ATensorList_[l0AListId_],
                                                 tla::MakeLayout<ElementA, LayoutTagL0A>(mActual, kL0Actual),
                                                 Catlass::Arch::PositionL0A{});
                auto subL1A = tla::GetTile(curL1A, tla::MakeCoord(0U, kL0Idx * L0_TILE_K),
                                           tla::MakeShape(mActual, kL0Actual));
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[l0AListId_]);
                if (kL0Idx == 0U) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList_[l1AListId_]);
                }
                copyL1ToL0A_(tensorL0A, subL1A);
                if (kL0Idx + 1U == kL0Loop) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList_[l1AListId_]);
                }

                auto tensorL0B = tla::MakeTensor(l0BTensorList_[l0BListId_],
                                                 tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, nActual),
                                                 Catlass::Arch::PositionL0B{});
                auto subL1B = tla::GetTile(curL1B, tla::MakeCoord(kL0Idx * L0_TILE_K, 0U),
                                           tla::MakeShape(kL0Actual, nActual));
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[l0BListId_]);
                if (kL0Idx == 0U) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList_[l1BListId_]);
                }
                copyL1ToL0B_(tensorL0B, subL1B);
                if (kL0Idx + 1U == kL0Loop) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList_[l1BListId_]);
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
                const bool initC = (kL1Idx == 0U && kL0Idx == 0U);
                const uint8_t unitFlag = (kL1Idx + 1U == kL1Loop && kL0Idx + 1U == kL0Loop) ? 0b11 : 0b10;
                tileMmad_(tensorL0C, tensorL0A, tensorL0B, mActual, nActual, kL0Actual, initC, unitFlag);

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList_[l0BListId_]);
                l0BListId_ = (l0BListId_ + 1U) % L0B_STAGES;
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList_[l0AListId_]);
                l0AListId_ = (l0AListId_ + 1U) % L0A_STAGES;
            }
            l1AListId_ = l1ANext;
            l1BListId_ = l1BNext;
            kL1Actual = kL1Next;
        }
        copyL0CToDst(tensorC, tensorL0C, 0b11);
    }

private:
    AscendC::LocalTensor<ElementA> l1ATensorList_[L1A_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList_[L1B_STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList_[L0A_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList_[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor_;
    int32_t l1AEventList_[L1A_STAGES]{};
    int32_t l1BEventList_[L1B_STAGES]{};
    int32_t l0AEventList_[L0A_STAGES]{};
    int32_t l0BEventList_[L0B_STAGES]{};
    uint32_t l1AListId_{0};
    uint32_t l1BListId_{0};
    uint32_t l0AListId_{0};
    uint32_t l0BListId_{0};
    TileMmad tileMmad_;
    CopyL1ToL0A copyL1ToL0A_;
    CopyL1ToL0B copyL1ToL0B_;
};

// A[M, K] RowMajor BF16 * B[N, K] RowMajor (即 [K, N] ColumnMajor) BF16 -> C[M, N] RowMajor BF16
class CatlassBf16Matmul {
    using Arch = Catlass::Arch::Ascend950;
    using Row = Catlass::layout::RowMajor;
    using Col = Catlass::layout::ColumnMajor;
    using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, bfloat16_t, Row, bfloat16_t, Col, bfloat16_t, Row,
                                                            void>;
    using Policy = Catlass::Gemm::MmadPingpong<Arch, true, false, 1, false, 2, 2, 2, 2>;
    using L1 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<128>>;
    using L0 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<64>>;
    using Block = MegaMoeBlockMmadBf16<Policy, L1, L0, bfloat16_t, bfloat16_t, bfloat16_t, TileCopy>;

    Catlass::Gemm::Block::MegaMoeResource resource_;
    Block block_;

public:
    __aicore__ inline CatlassBf16Matmul() : block_(resource_) {}

    __aicore__ inline void ComputeTile(GM_ADDR aBase, GM_ADDR bBase, GM_ADDR cBase, uint32_t fullM, uint32_t fullN,
                                       uint32_t fullK, uint32_t mLoc, uint32_t nLoc, uint32_t mActual, uint32_t nActual)
    {
        if constexpr (g_coreType != AscendC::AIC) {
            return;
        }
        AscendC::GlobalTensor<bfloat16_t> gmA, gmB, gmC;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(aBase));
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(bBase));
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cBase));
        auto tensorA = tla::MakeTensor(gmA, tla::MakeLayout<bfloat16_t, Row>(fullM, fullK), Catlass::Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, tla::MakeLayout<bfloat16_t, Col>(fullK, fullN), Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gmC, tla::MakeLayout<bfloat16_t, Row>(fullM, fullN), Catlass::Arch::PositionGM{});
        auto tileA = tla::GetTile(tensorA, tla::MakeCoord(mLoc, 0U), tla::MakeShape(mActual, fullK));
        auto tileB = tla::GetTile(tensorB, tla::MakeCoord(0U, nLoc), tla::MakeShape(fullK, nActual));
        auto tileC = tla::GetTile(tensorC, tla::MakeCoord(mLoc, nLoc), tla::MakeShape(mActual, nActual));
        block_(tileA, tileB, tileC, Catlass::GemmCoord{mActual, nActual, fullK});
    }
};

#endif  // CATLASS_ARCH == 3510

}  // namespace MegaMoeImpl::GmmKernel

#endif  // MEGA_MOE_CATLASS_BF16_MATMUL_HPP
