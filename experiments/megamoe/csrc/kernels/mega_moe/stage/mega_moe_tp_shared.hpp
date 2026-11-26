// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

/*!
 * \file mega_moe_tp_shared.hpp
 * \brief TP MegaMoE 共享专家（不做 TP，只算本卡 bs）执行流水：支持全 BF16 与全 FP8 (MXFP8) 两种精度。
 *        GMM1 与 GMM2 100% 在 AIC 执行（不占 AIV0）；SwiGLU 在 Phase 2 由 AIV1 优先级循环按 tile 消费。
 */

#ifndef MEGA_MOE_TP_SHARED_HPP
#define MEGA_MOE_TP_SHARED_HPP

#include <c_api/asc_simd.h>
#include "../common/mega_moe_tp_layout.hpp"
#include "../common/mega_moe_utils.hpp"
#include "../common/ascend_timer_v2_device.hpp"
#include "../catlass_gmm/mega_moe_bf16_matmul.hpp"
#include "../catlass_gmm/block_mmad_mx.hpp"
#include "mega_moe_gmm1_activation.hpp"

namespace MegaMoeImpl {

using namespace AscendC;

namespace GmmKernel {

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
// 纯 AIC 执行的 MXFP8 RowMajor A * NZ MXFP8 B -> BF16 RowMajor C 封装（不占用 AIV0）
class CatlassMxFp8NzMatmul {
    using Arch = Catlass::Arch::Ascend950;
    using Row = Catlass::layout::RowMajor;
    using Col = Catlass::layout::ColumnMajor;
    using BLayout = Catlass::layout::nZ;
    using SALayout = decltype(tla::MakeMxScaleLayout<fp8_e8m0_t, Row, false>(uint32_t{}, uint32_t{}));
    using SBLayout = decltype(tla::MakeMxScaleLayout<fp8_e8m0_t, Col, true>(uint32_t{}, uint32_t{}));
    using TileCopy = MxStageTileCopy<Catlass::Gemm::Tile::PackedMxTileCopyTla<
        Arch, fp8_e4m3fn_t, Row, fp8_e4m3fn_t, BLayout, fp8_e8m0_t, SALayout, fp8_e8m0_t, SBLayout, bfloat16_t, Row,
        void>>;
    using Policy = Catlass::Gemm::MmadMx<Arch, true, 2, 1, false, 2, 2, 2, 2>;
    using L1 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
    using L0 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<128>>;
    using Block =
        Catlass::Gemm::Block::MegaMoeBlockMmadMx<Policy, L1, L0, fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, void, TileCopy>;

    Catlass::Gemm::Block::MegaMoeResource resource_;
    Block block_;

public:
    __aicore__ inline CatlassMxFp8NzMatmul() : block_(resource_) {}
    __aicore__ inline ~CatlassMxFp8NzMatmul()
    {
        if constexpr (g_coreType == AscendC::AIC) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
        }
    }

    __aicore__ inline void ComputeTile(GM_ADDR aBase, GM_ADDR bBase, GM_ADDR saBase, GM_ADDR sbBase, GM_ADDR cBase,
                                       uint32_t fullM, uint32_t fullN, uint32_t fullK, uint32_t mLoc, uint32_t nLoc,
                                       uint32_t mActual, uint32_t nActual)
    {
        if constexpr (g_coreType != AscendC::AIC) {
            return;
        }
        const uint32_t scaleK = (fullK + 63U) / 64U * 2U;
        AscendC::GlobalTensor<fp8_e4m3fn_t> gmA, gmB;
        AscendC::GlobalTensor<fp8_e8m0_t> gmSA, gmSB;
        AscendC::GlobalTensor<bfloat16_t> gmC;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e4m3fn_t*>(aBase));
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e4m3fn_t*>(bBase));
        gmSA.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e8m0_t*>(saBase));
        gmSB.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e8m0_t*>(sbBase));
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cBase));

        auto av = tla::MakeTensor(gmA, tla::MakeLayout<fp8_e4m3fn_t, Row>(fullM, fullK), Catlass::Arch::PositionGM{});
        auto bv = tla::MakeTensor(gmB, tla::MakeLayout<fp8_e4m3fn_t, BLayout>(fullK, fullN), Catlass::Arch::PositionGM{});
        auto sav = tla::MakeTensor(gmSA, tla::MakeMxScaleLayout<fp8_e8m0_t, Row, false>(fullM, scaleK),
                                   Catlass::Arch::PositionGM{});
        auto sbv = tla::MakeTensor(gmSB, tla::MakeMxScaleLayout<fp8_e8m0_t, Col, true>(scaleK, fullN),
                                   Catlass::Arch::PositionGM{});
        auto cv = tla::MakeTensor(gmC, tla::MakeLayout<bfloat16_t, Row>(fullM, fullN), Catlass::Arch::PositionGM{});

        auto tileA = tla::GetTile(av, tla::MakeCoord(mLoc, 0U), tla::MakeShape(mActual, fullK));
        auto tileB = tla::GetTile(bv, tla::MakeCoord(0U, nLoc), tla::MakeShape(fullK, nActual));
        auto tileSA = tla::GetTile(sav, tla::MakeCoord(mLoc, 0U), tla::MakeShape(mActual, scaleK));
        auto tileSB = tla::GetTile(sbv, tla::MakeCoord(0U, nLoc), tla::MakeShape(scaleK, nActual));
        auto tileC = tla::GetTile(cv, tla::MakeCoord(mLoc, nLoc), tla::MakeShape(mActual, nActual));
        block_(tileA, tileB, tileC, Catlass::GemmCoord{mActual, nActual, fullK}, tileSA, tileSB);
    }
};
#endif

}  // namespace GmmKernel

struct TpSharedSwiGluCursor {
    uint32_t expert = 0U;
    uint32_t nextTile = 0U;
    uint32_t tileCount = 0U;
    bool gmm1Ready = false;
    bool done = false;
};

// Phase 1 AIC 独占执行 Shared GMM1（与 64 AIV 的 AllGather + RouteTable 完全重叠）
__aicore__ inline void ProcessTpSharedGmm1(const MegaMoeTpTilingData& tiling, uint32_t rankId, uint32_t blockIdx,
                                           GM_ADDR xGm, GM_ADDR xScalesGm, GM_ADDR sharedWeight1,
                                           GM_ADDR sharedWeightScales1, const TpWorkspaceInfo& workspace,
                                           AscendTimerDevice& timer)
{
    if constexpr (g_coreType != AIC) {
        return;
    }
    if (tiling.sharedExpertNum == 0U) {
        return;
    }

    const uint32_t fullM = tiling.bs;
    const uint32_t fullN = tiling.sharedGmm1OutDim;
    const uint32_t fullK = tiling.h;
    const uint32_t mTiles = Ops::Base::CeilDiv(fullM, L1_TILE_M_256);
    const uint32_t nTiles = Ops::Base::CeilDiv(fullN, L1_TILE_N);
    const uint32_t totalTiles = mTiles * nTiles;
    const uint32_t aicNum = tiling.aicNum;
    const auto mode = static_cast<SharedPrecisionMode>(tiling.sharedPrecisionMode);
    uint32_t startBlock = 0U;

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
    if (mode == SharedPrecisionMode::BF16) {
        GmmKernel::CatlassBf16Matmul bf16Mmad;
        GM_ADDR aBase = (xScalesGm == nullptr) ? xGm : workspace.sharedBf16InputPtr;
        for (uint32_t s = 0U; s < tiling.sharedExpertNum; ++s) {
            GM_ADDR bBase = sharedWeight1 + static_cast<uint64_t>(s) * fullN * fullK * sizeof(bfloat16_t);
            GM_ADDR cBase = workspace.sharedGmm1OutPtr + static_cast<uint64_t>(s) * fullM * fullN * sizeof(bfloat16_t);
            timer.SetWave(AscendTimer::NO_WAVE);
            timer.SetSlice(static_cast<int>(s), 0U, fullM);
            const uint32_t firstTile = (blockIdx + aicNum - startBlock) % aicNum;
            for (uint32_t loopIdx = firstTile; loopIdx < totalTiles; loopIdx += aicNum) {
                const uint32_t mLoc = (loopIdx / nTiles) * L1_TILE_M_256;
                const uint32_t nLoc = (loopIdx % nTiles) * L1_TILE_N;
                const uint32_t mActual = Min(L1_TILE_M_256, fullM - mLoc);
                const uint32_t nActual = Min(L1_TILE_N, fullN - nLoc);
                AscendTimerTileScope tileTimer(timer.Enabled() ? &timer : nullptr, AscendTimer::AIC_SHARED_GMM1, mLoc,
                                               nLoc, mActual, L1_TILE_N);
                bf16Mmad.ComputeTile(aBase, bBase, cBase, fullM, fullN, fullK, mLoc, nLoc, mActual, nActual);
            }
            timer.ClearSlice();
            startBlock = (startBlock + totalTiles) % aicNum;
        }
    } else if (mode == SharedPrecisionMode::FP8_MX) {
        GmmKernel::CatlassMxFp8NzMatmul mxMmad;
        const uint64_t scaleK = static_cast<uint64_t>(CalcTpMxScaleStrideBytes(fullK));
        GM_ADDR aBase = workspace.agXDataPtr + static_cast<uint64_t>(rankId) * fullM * fullK;
        GM_ADDR saBase = workspace.agXScalePtr + static_cast<uint64_t>(rankId) * fullM * scaleK;
        for (uint32_t s = 0U; s < tiling.sharedExpertNum; ++s) {
            GM_ADDR bBase = sharedWeight1 + static_cast<uint64_t>(s) * fullN * fullK;
            GM_ADDR sbBase = sharedWeightScales1 + static_cast<uint64_t>(s) * fullN * scaleK;
            GM_ADDR cBase = workspace.sharedGmm1OutPtr + static_cast<uint64_t>(s) * fullM * fullN * sizeof(bfloat16_t);
            timer.SetWave(AscendTimer::NO_WAVE);
            timer.SetSlice(static_cast<int>(s), 0U, fullM);
            const uint32_t firstTile = (blockIdx + aicNum - startBlock) % aicNum;
            for (uint32_t loopIdx = firstTile; loopIdx < totalTiles; loopIdx += aicNum) {
                const uint32_t mLoc = (loopIdx / nTiles) * L1_TILE_M_256;
                const uint32_t nLoc = (loopIdx % nTiles) * L1_TILE_N;
                const uint32_t mActual = Min(L1_TILE_M_256, fullM - mLoc);
                const uint32_t nActual = Min(L1_TILE_N, fullN - nLoc);
                AscendTimerTileScope tileTimer(timer.Enabled() ? &timer : nullptr, AscendTimer::AIC_SHARED_GMM1, mLoc,
                                               nLoc, mActual, L1_TILE_N);
                mxMmad.ComputeTile(aBase, bBase, saBase, sbBase, cBase, fullM, fullN, fullK, mLoc, nLoc, mActual,
                                   nActual);
            }
            timer.ClearSlice();
            startBlock = (startBlock + totalTiles) % aicNum;
        }
    }
#endif

    SetFlag<HardEvent::FIX_S>(0);
    WaitFlag<HardEvent::FIX_S>(0);
    asc_sync_data_barrier(DSB_DDR);
    WriteGmByPassDCache(
        reinterpret_cast<__gm__ int32_t*>(workspace.flagSharedGmm1DonePtr) + static_cast<uint64_t>(blockIdx) * INT_CACHELINE,
        1);
}

// BF16 共享专家的单个 (256, 256) SwiGLU tile 执行：复用低 48 KiB 激活常驻 UB（受 Aiv1ActivationUb 保护）
__aicore__ inline void RunTpSharedBf16SwiGluTile(GM_ADDR gmm1OutBase, GM_ADDR actOutBase, uint32_t fullM,
                                                 uint32_t interN, uint32_t mLoc, uint32_t nLoc, uint32_t mActual,
                                                 uint32_t nActual, float clampLimit)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    constexpr uint32_t kSubRows = 32U;
    constexpr uint32_t kTileCols = L1_TILE_N;  // 256
    constexpr uint32_t kBufElems = kSubRows * kTileCols;
    constexpr uint32_t kBufBytes = kBufElems * sizeof(bfloat16_t);  // 16 KiB

    LocalTensor<bfloat16_t> gateUb(TPosition::VECIN, 0U, kBufElems);
    LocalTensor<bfloat16_t> upUb(TPosition::VECIN, kBufBytes, kBufElems);
    LocalTensor<bfloat16_t> outUb(TPosition::VECOUT, 2U * kBufBytes, kBufElems);

    GlobalTensor<bfloat16_t> srcGm, dstGm;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gmm1OutBase));
    dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(actOutBase));

    const uint32_t gmm1Stride = interN * ACTIVATION_N_HALF;  // 2 * I_shared
    const uint32_t copyBytes = nActual * static_cast<uint32_t>(sizeof(bfloat16_t));
    const uint32_t srcRowGapBytes = (gmm1Stride - nActual) * static_cast<uint32_t>(sizeof(bfloat16_t));
    const uint32_t ubDstGapBlocks = (kTileCols - nActual) * static_cast<uint32_t>(sizeof(bfloat16_t)) / ALIGN_32;
    const uint32_t dstRowGapBytes = (interN - nActual) * static_cast<uint32_t>(sizeof(bfloat16_t));

    SyncFuncStatic<HardEvent::S_MTE2, 0>();
    for (uint32_t row = 0U; row < mActual; row += kSubRows) {
        Aiv1ActivationUb::Acquire();
        const uint32_t cnt = Min(kSubRows, mActual - row);
        const uint64_t r0 = mLoc + row;
        DataCopyPad(gateUb, srcGm[r0 * gmm1Stride + nLoc],
                    {static_cast<uint16_t>(cnt), copyBytes, srcRowGapBytes, ubDstGapBlocks, 0U}, {false, 0U, 0U, 0U});
        DataCopyPad(upUb, srcGm[r0 * gmm1Stride + interN + nLoc],
                    {static_cast<uint16_t>(cnt), copyBytes, srcRowGapBytes, ubDstGapBlocks, 0U}, {false, 0U, 0U, 0U});
        SyncFuncStatic<HardEvent::MTE2_V, 0>();

        Activation::GatedActivationTileContext<bfloat16_t> ctx{};
        ctx.gate = reinterpret_cast<__ubuf__ bfloat16_t*>(gateUb.GetPhyAddr());
        ctx.up = reinterpret_cast<__ubuf__ bfloat16_t*>(upUb.GetPhyAddr());
        ctx.output = reinterpret_cast<__ubuf__ bfloat16_t*>(outUb.GetPhyAddr());
        ctx.topkWeights = nullptr;
        ctx.inputRowStrideElements = kTileCols;
        ctx.outputRowStrideElements = kTileCols;
        ctx.rowLoopCount = static_cast<uint16_t>(cnt);
        ctx.fullVectorLoopCount = static_cast<uint16_t>(nActual / Activation::VECTOR_LENGTH_FP32);
        const uint32_t tailElems = nActual % Activation::VECTOR_LENGTH_FP32;
        ctx.gateTail = ctx.gate;
        ctx.upTail = ctx.up;
        ctx.outputTail = ctx.output;
        ctx.additionalPaddingOutput = ctx.output;
        if (tailElems > 0U) {
            ctx.needTailVectorCompute = 1U;
            ctx.tailComputeMaskElementCount = tailElems;
            ctx.tailStoreMaskElementCount = tailElems;
            const uint32_t tailCol = ctx.fullVectorLoopCount * Activation::VECTOR_LENGTH_FP32;
            ctx.gateTail = ctx.gate + tailCol;
            ctx.upTail = ctx.up + tailCol;
            ctx.outputTail = ctx.output + tailCol;
        }
        SetCtrlSpr<60, 60>(0);
        Activation::RunSwiGLU<bfloat16_t, false, false>(ctx, Activation::SwiGluParams{clampLimit});

        SyncFuncStatic<HardEvent::V_MTE3, 0>();
        DataCopyPad(dstGm[r0 * interN + nLoc], outUb,
                    {static_cast<uint16_t>(cnt), copyBytes, ubDstGapBlocks, dstRowGapBytes, 0U});
        Aiv1ActivationUb::Release();
    }
}

// Phase 2 AIV1 非阻塞消费 1 个共享专家 SwiGLU tile
template <typename SharedEpilogueOp>
__aicore__ inline bool TryRunTpSharedSwiGlu(const MegaMoeTpTilingData& tiling, uint32_t blockIdx,
                                            const TpWorkspaceInfo& workspace, SharedEpilogueOp& sharedEpilogueOp,
                                            TpSharedSwiGluCursor& cursor, AscendTimerDevice& timer)
{
    if constexpr (g_coreType == AIC) {
        return false;
    }
    if (cursor.done || tiling.sharedExpertNum == 0U) {
        cursor.done = true;
        return false;
    }
    const uint32_t aicNum = tiling.aicNum;
    if (!cursor.gmm1Ready) {
        auto* flags = reinterpret_cast<__gm__ int32_t*>(workspace.flagSharedGmm1DonePtr);
        for (uint32_t core = 0U; core < aicNum; ++core) {
            if (ReadGmByPassDCache(flags + static_cast<uint64_t>(core) * INT_CACHELINE) < 1) {
                return false;
            }
        }
        cursor.gmm1Ready = true;
        cursor.nextTile = blockIdx;
        const uint32_t interN = tiling.sharedGmm1OutDim / ACTIVATION_N_HALF;
        cursor.tileCount =
            Ops::Base::CeilDiv(tiling.bs, L1_TILE_M_256) * Ops::Base::CeilDiv(interN, static_cast<uint32_t>(L1_TILE_N));
    }

    const uint32_t fullM = tiling.bs;
    const uint32_t gmm1N = tiling.sharedGmm1OutDim;
    const uint32_t interN = gmm1N / ACTIVATION_N_HALF;
    const uint32_t nTiles = Ops::Base::CeilDiv(interN, static_cast<uint32_t>(L1_TILE_N));
    const auto mode = static_cast<SharedPrecisionMode>(tiling.sharedPrecisionMode);

    if (cursor.nextTile < cursor.tileCount) {
        const uint32_t s = cursor.expert;
        const uint32_t mLoc = (cursor.nextTile / nTiles) * L1_TILE_M_256;
        const uint32_t nLoc = (cursor.nextTile % nTiles) * L1_TILE_N;
        const uint32_t mActual = Min(L1_TILE_M_256, fullM - mLoc);
        const uint32_t nActual = Min(L1_TILE_N, interN - nLoc);

        GM_ADDR gmm1OutBase =
            workspace.sharedGmm1OutPtr + static_cast<uint64_t>(s) * fullM * gmm1N * sizeof(bfloat16_t);

        timer.SetWave(AscendTimer::NO_WAVE);
        timer.SetSlice(static_cast<int>(s), 0U, fullM);
        if (mode == SharedPrecisionMode::BF16) {
            GM_ADDR actOutBase =
                workspace.sharedActDataPtr + static_cast<uint64_t>(s) * fullM * interN * sizeof(bfloat16_t);
            AscendTimerTileScope tileTimer(timer.Enabled() ? &timer : nullptr, AscendTimer::AIV_SHARED_SWIGLU, mLoc,
                                           nLoc, mActual, L1_TILE_N);
            RunTpSharedBf16SwiGluTile(gmm1OutBase, actOutBase, fullM, interN, mLoc, nLoc, mActual, nActual,
                                      tiling.clampLimit);
        } else {
            const uint64_t scaleInter = static_cast<uint64_t>(CalcTpMxScaleStrideBytes(interN));
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                static_cast<int64_t>(static_cast<uint64_t>(s) * fullM * interN),
                static_cast<int64_t>(static_cast<uint64_t>(s) * fullM * scaleInter),
                0L, 0L, 0L, 0L};
            sharedEpilogueOp.UpdateGlobalAddr(vecBaseOffset);
            GMMAddrInfo addr{};
            addr.gmm1OutGlobal = gmm1OutBase;
            addr.timer = timer.Enabled() ? &timer : nullptr;
            ProblemShape shape{static_cast<int64_t>(fullM), static_cast<int64_t>(gmm1N),
                               static_cast<int64_t>(tiling.h), 0};
            BlockJobContext job{blockIdx, aicNum};
            RunReadyGmm1ActivationTileA8W4<__fp8e4m3, __fp8e4m3, bfloat16_t, fp8_e8m0_t, fp8_e8m0_t, L1_TILE_M_256,
                                           false>(sharedEpilogueOp, shape, addr, job, cursor.nextTile, 0U,
                                                  AscendTimer::AIV_SHARED_SWIGLU, true);
        }
        timer.ClearSlice();

        cursor.nextTile += aicNum;
        if (cursor.nextTile >= cursor.tileCount) {
            ++cursor.expert;
            cursor.nextTile = blockIdx;
        }
    } else {
        cursor.expert = tiling.sharedExpertNum;
    }

    if (cursor.expert >= tiling.sharedExpertNum) {
        SetFlag<HardEvent::MTE3_S>(static_cast<TEventID>(Aiv1AsyncEvents::ActivationDone));
        WaitFlag<HardEvent::MTE3_S>(static_cast<TEventID>(Aiv1AsyncEvents::ActivationDone));
        asc_sync_data_barrier(DSB_DDR);
        WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t*>(workspace.flagSharedActDonePtr) +
                                static_cast<uint64_t>(blockIdx) * INT_CACHELINE,
                            1);
        cursor.done = true;
    }
    return true;
}

// Phase 3 AIC 独占执行 Shared GMM2（与 64 AIV 的 Unpermute+Send 及 Recv+Reduce 完全并行，逐 256 行通知）
__aicore__ inline void ProcessTpSharedGmm2(const MegaMoeTpTilingData& tiling, uint32_t blockIdx, GM_ADDR sharedWeight2,
                                           GM_ADDR sharedWeightScales2, const TpWorkspaceInfo& workspace,
                                           AscendTimerDevice& timer)
{
    if constexpr (g_coreType != AIC) {
        return;
    }
    if (tiling.sharedExpertNum == 0U) {
        return;
    }

    WaitForAllCoreFlags(workspace.flagSharedActDonePtr, tiling.aicNum);
    PipeBarrier<PIPE_ALL>();

    const uint32_t fullM = tiling.bs;
    const uint32_t fullN = tiling.h;
    const uint32_t fullK = tiling.sharedGmm1OutDim / ACTIVATION_N_HALF;
    const uint32_t mTiles = Ops::Base::CeilDiv(fullM, L1_TILE_M_256);
    const uint32_t nTiles = Ops::Base::CeilDiv(fullN, L1_TILE_N);
    const uint32_t totalTiles = mTiles * nTiles;
    const uint32_t aicNum = tiling.aicNum;
    const auto mode = static_cast<SharedPrecisionMode>(tiling.sharedPrecisionMode);
    uint32_t startBlock = 0U;

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
    if (mode == SharedPrecisionMode::BF16) {
        GmmKernel::CatlassBf16Matmul bf16Mmad;
        for (uint32_t s = 0U; s < tiling.sharedExpertNum; ++s) {
            GM_ADDR aBase = workspace.sharedActDataPtr + static_cast<uint64_t>(s) * fullM * fullK * sizeof(bfloat16_t);
            GM_ADDR bBase = sharedWeight2 + static_cast<uint64_t>(s) * fullN * fullK * sizeof(bfloat16_t);
            GM_ADDR cBase = workspace.sharedGmm2OutPtr + static_cast<uint64_t>(s) * fullM * fullN * sizeof(bfloat16_t);
            __gm__ int32_t* cntBase = reinterpret_cast<__gm__ int32_t*>(workspace.sharedGmm2TileCounterPtr) +
                                      static_cast<uint64_t>(s) * mTiles * INT_CACHELINE;

            timer.SetWave(AscendTimer::NO_WAVE);
            timer.SetSlice(static_cast<int>(s), 0U, fullM);
            const uint32_t firstTile = (blockIdx + aicNum - startBlock) % aicNum;
            for (uint32_t loopIdx = firstTile; loopIdx < totalTiles; loopIdx += aicNum) {
                const uint32_t mTile = loopIdx / nTiles;
                const uint32_t nTile = loopIdx % nTiles;
                const uint32_t mLoc = mTile * L1_TILE_M_256;
                const uint32_t nLoc = nTile * L1_TILE_N;
                const uint32_t mActual = Min(L1_TILE_M_256, fullM - mLoc);
                const uint32_t nActual = Min(L1_TILE_N, fullN - nLoc);
                AscendTimerTileScope tileTimer(timer.Enabled() ? &timer : nullptr, AscendTimer::AIC_SHARED_GMM2, mLoc,
                                               nLoc, mActual, L1_TILE_N);
                bf16Mmad.ComputeTile(aBase, bBase, cBase, fullM, fullN, fullK, mLoc, nLoc, mActual, nActual);
                SetFlag<HardEvent::FIX_S>(0);
                WaitFlag<HardEvent::FIX_S>(0);
                AtomicAdd(cntBase + static_cast<uint64_t>(mTile) * INT_CACHELINE, 1);
            }
            timer.ClearSlice();
            startBlock = (startBlock + totalTiles) % aicNum;
        }
    } else if (mode == SharedPrecisionMode::FP8_MX) {
        GmmKernel::CatlassMxFp8NzMatmul mxMmad;
        const uint64_t scaleK = static_cast<uint64_t>(CalcTpMxScaleStrideBytes(fullK));
        for (uint32_t s = 0U; s < tiling.sharedExpertNum; ++s) {
            GM_ADDR aBase = workspace.sharedActDataPtr + static_cast<uint64_t>(s) * fullM * fullK;
            GM_ADDR saBase = workspace.sharedActScalePtr + static_cast<uint64_t>(s) * fullM * scaleK;
            GM_ADDR bBase = sharedWeight2 + static_cast<uint64_t>(s) * fullN * fullK;
            GM_ADDR sbBase = sharedWeightScales2 + static_cast<uint64_t>(s) * fullN * scaleK;
            GM_ADDR cBase = workspace.sharedGmm2OutPtr + static_cast<uint64_t>(s) * fullM * fullN * sizeof(bfloat16_t);
            __gm__ int32_t* cntBase = reinterpret_cast<__gm__ int32_t*>(workspace.sharedGmm2TileCounterPtr) +
                                      static_cast<uint64_t>(s) * mTiles * INT_CACHELINE;

            timer.SetWave(AscendTimer::NO_WAVE);
            timer.SetSlice(static_cast<int>(s), 0U, fullM);
            const uint32_t firstTile = (blockIdx + aicNum - startBlock) % aicNum;
            for (uint32_t loopIdx = firstTile; loopIdx < totalTiles; loopIdx += aicNum) {
                const uint32_t mTile = loopIdx / nTiles;
                const uint32_t nTile = loopIdx % nTiles;
                const uint32_t mLoc = mTile * L1_TILE_M_256;
                const uint32_t nLoc = nTile * L1_TILE_N;
                const uint32_t mActual = Min(L1_TILE_M_256, fullM - mLoc);
                const uint32_t nActual = Min(L1_TILE_N, fullN - nLoc);
                AscendTimerTileScope tileTimer(timer.Enabled() ? &timer : nullptr, AscendTimer::AIC_SHARED_GMM2, mLoc,
                                               nLoc, mActual, L1_TILE_N);
                mxMmad.ComputeTile(aBase, bBase, saBase, sbBase, cBase, fullM, fullN, fullK, mLoc, nLoc, mActual,
                                   nActual);
                SetFlag<HardEvent::FIX_S>(0);
                WaitFlag<HardEvent::FIX_S>(0);
                AtomicAdd(cntBase + static_cast<uint64_t>(mTile) * INT_CACHELINE, 1);
            }
            timer.ClearSlice();
            startBlock = (startBlock + totalTiles) % aicNum;
        }
    }
#endif
}

}  // namespace MegaMoeImpl

#endif  // MEGA_MOE_TP_SHARED_HPP
