// Copyright (c) 2026, Lu Lu
// Modified by SimpleBrightMan 2026

#pragma once

#include <cstdint>
#include "kernel_operator.h"
#include "kernels/mega_moe_urma/mega_moe_urma_tiling.hpp"
#include "kernels/mega_moe_urma/common/mega_moe_compute_contract.hpp"
#include "kernels/mega_moe_common/epilogue/quant/mx_quant_compute_scale.hpp"
#include "kernels/mega_moe_common/epilogue/quant/mxfp8_quant.hpp"

// On-the-fly in-kernel BF16 -> MXFP8 input quantization (and MXFP8 -> BF16
// dequantization when prequantized FP8 input is paired with a BF16 shared expert).
// Runs across all S::kVectorCores AIVs inside PrepareDeviceRoutes before SyncAll #1.

namespace MegaMoeUrma {

template <class S>
__aicore__ inline void QuantizeInputBf16ToMxfp8(const Arguments& a, uint32_t worker)
{
    using namespace AscendC;
    const uint64_t bs = a.config.bs;
    const uint64_t rowBegin = LocalRowBegin(bs, worker, S::kVectorCores);
    const uint64_t ownedRows = LocalOwnedRows(bs, worker, S::kVectorCores);
    if (ownedRows == 0U) return;

    constexpr uint32_t kHidden = S::kHidden;
    constexpr uint32_t kScaleCols = kHidden / 32U;
    // Ping-pong double-buffered chunks: 2 x 60 KiB = 120 KiB UB (< 248 KiB).
    // Overlaps MTE2 load of chunk (i+1) with Vector quant + MTE3 store of chunk i.
    constexpr uint32_t kMaxChunkRows = 16384U / kHidden;
    static_assert(kMaxChunkRows >= 1U && kHidden % 256U == 0U);

    constexpr uint32_t kBufStride = 60U * 1024U;
    constexpr uint32_t kBf16Off = 0U;
    constexpr uint32_t kFp8Off = 32U * 1024U;
    constexpr uint32_t kMaxExpOff = 48U * 1024U;
    constexpr uint32_t kScaleOff = 52U * 1024U;
    constexpr uint32_t kRecipOff = 56U * 1024U;
    static_assert(2U * kBufStride <= kRequiredUbBytes);

    GlobalTensor<bfloat16_t> srcGm;
    GlobalTensor<uint8_t> dstX, dstScale;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a.rawBf16X));
    dstX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a.quantX));
    dstScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a.quantXScale));

    LocalTensor<bfloat16_t> bf16Ub[2] = {
        LocalTensor<bfloat16_t>(TPosition::VECCALC, kBf16Off, kMaxChunkRows * kHidden),
        LocalTensor<bfloat16_t>(TPosition::VECCALC, kBufStride + kBf16Off, kMaxChunkRows * kHidden)};
    LocalTensor<uint8_t> fp8Ub[2] = {
        LocalTensor<uint8_t>(TPosition::VECCALC, kFp8Off, kMaxChunkRows * kHidden),
        LocalTensor<uint8_t>(TPosition::VECCALC, kBufStride + kFp8Off, kMaxChunkRows * kHidden)};
    LocalTensor<uint16_t> maxExpUb[2] = {
        LocalTensor<uint16_t>(TPosition::VECCALC, kMaxExpOff, kMaxChunkRows * kScaleCols),
        LocalTensor<uint16_t>(TPosition::VECCALC, kBufStride + kMaxExpOff, kMaxChunkRows * kScaleCols)};
    LocalTensor<uint8_t> scaleUb[2] = {
        LocalTensor<uint8_t>(TPosition::VECCALC, kScaleOff, kMaxChunkRows * kScaleCols),
        LocalTensor<uint8_t>(TPosition::VECCALC, kBufStride + kScaleOff, kMaxChunkRows * kScaleCols)};
    LocalTensor<uint16_t> recipUb[2] = {
        LocalTensor<uint16_t>(TPosition::VECCALC, kRecipOff, kMaxChunkRows * kScaleCols),
        LocalTensor<uint16_t>(TPosition::VECCALC, kBufStride + kRecipOff, kMaxChunkRows * kScaleCols)};

    SetCtrlSpr<60, 60>(0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);

    uint64_t offset = 0U;
    uint32_t ping = 0U;
    while (offset < ownedRows) {
        const uint32_t rows = static_cast<uint32_t>(
            ownedRows - offset < kMaxChunkRows ? ownedRows - offset : kMaxChunkRows);
        const uint32_t elems = rows * kHidden;
        const uint32_t scaleCount = rows * kScaleCols;
        const uint16_t dataLoops = static_cast<uint16_t>(elems / 256U);
        const uint16_t scaleLoops = static_cast<uint16_t>((scaleCount + 127U) / 128U);
        const uint64_t gmRow = rowBegin + offset;
        const TEventID ev = ping ? EVENT_ID1 : EVENT_ID0;

        WaitFlag<HardEvent::V_MTE2>(ev);
        DataCopy(bf16Ub[ping], srcGm[gmRow * kHidden], elems);
        SetFlag<HardEvent::MTE2_V>(ev);

        WaitFlag<HardEvent::MTE2_V>(ev);
        WaitFlag<HardEvent::MTE3_V>(ev);
        auto* bf16Ptr = reinterpret_cast<__ubuf__ bfloat16_t*>(bf16Ub[ping].GetPhyAddr());
        auto* fp8Ptr = reinterpret_cast<__ubuf__ int8_t*>(fp8Ub[ping].GetPhyAddr());
        auto* maxExpPtr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpUb[ping].GetPhyAddr());
        auto* scalePtr = reinterpret_cast<__ubuf__ uint16_t*>(scaleUb[ping].GetPhyAddr());
        auto* recipPtr = reinterpret_cast<__ubuf__ uint16_t*>(recipUb[ping].GetPhyAddr());
        UrmaMegaMoeImpl::MxQuant::ComputeGroupMaxExp(bf16Ptr, maxExpPtr, elems, dataLoops);
        PipeBarrier<PIPE_V>();
        UrmaMegaMoeImpl::MxQuant::ComputeMxScale<fp8_e4m3fn_t>(maxExpPtr, scalePtr, recipPtr, scaleCount, scaleLoops);
        PipeBarrier<PIPE_V>();
        UrmaMegaMoeImpl::MxQuant::QuantizeMxFp8Data<fp8_e4m3fn_t>(bf16Ptr, recipPtr, fp8Ptr, elems, dataLoops);
        SetFlag<HardEvent::V_MTE2>(ev);
        SetFlag<HardEvent::V_MTE3>(ev);

        WaitFlag<HardEvent::V_MTE3>(ev);
        DataCopy(dstX[gmRow * kHidden], fp8Ub[ping], elems);
        DataCopyPad(dstScale[gmRow * kScaleCols], scaleUb[ping], {1U, scaleCount, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE3_V>(ev);

        offset += rows;
        ping ^= 1U;
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline float DecodeFp8E4M3ToFloat(uint8_t code)
{
    const uint32_t sign = static_cast<uint32_t>(code & 0x80U) << 24U;
    const uint32_t exp = (code >> 3U) & 0x0FU;
    const uint32_t mant = code & 0x07U;
    uint32_t bits = 0U;
    if (exp == 0U) {
        if (mant != 0U) {
            union {
                float f;
                uint32_t u;
            } v{static_cast<float>(mant) * (1.0f / 512.0f)};
            bits = sign | v.u;
        } else {
            bits = sign;
        }
    } else {
        bits = sign | ((exp + 120U) << 23U) | (mant << 20U);
    }
    union {
        uint32_t u;
        float f;
    } out{bits};
    return out.f;
}

__aicore__ inline float DecodeE8M0ToFloat(uint8_t code)
{
    if (code == 0U) return 0.0f;
    union {
        uint32_t u;
        float f;
    } out{static_cast<uint32_t>(code) << 23U};
    return out.f;
}

template <class S>
__aicore__ inline void DequantizeInputMxfp8ToBf16(const Arguments& a, uint32_t worker)
{
    using namespace AscendC;
    const uint64_t bs = a.config.bs;
    const uint64_t rowBegin = LocalRowBegin(bs, worker, S::kVectorCores);
    const uint64_t ownedRows = LocalOwnedRows(bs, worker, S::kVectorCores);
    if (ownedRows == 0U) return;

    constexpr uint32_t kHidden = S::kHidden;
    constexpr uint32_t kScaleCols = kHidden / 32U;
    constexpr uint32_t kFp8UbOffset = 0U;
    constexpr uint32_t kScaleUbOffset = 16U * 1024U;
    constexpr uint32_t kFp32UbOffset = 32U * 1024U;
    constexpr uint32_t kBf16UbOffset = 64U * 1024U;

    GlobalTensor<uint8_t> srcX, srcScale;
    GlobalTensor<bfloat16_t> dstBf16;
    srcX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a.x));
    srcScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a.xScale));
    dstBf16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a.rawBf16X));

    LocalTensor<uint8_t> fp8Ub(TPosition::VECCALC, kFp8UbOffset, kHidden);
    LocalTensor<uint8_t> scaleUb(TPosition::VECCALC, kScaleUbOffset, kScaleCols);
    LocalTensor<float> fp32Ub(TPosition::VECCALC, kFp32UbOffset, kHidden);
    LocalTensor<bfloat16_t> bf16Ub(TPosition::VECCALC, kBf16UbOffset, kHidden);

    for (uint64_t r = 0U; r < ownedRows; ++r) {
        const uint64_t gmRow = rowBegin + r;
        DataCopyPad(fp8Ub, srcX[gmRow * kHidden], {1U, kHidden, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        DataCopyPad(scaleUb, srcScale[gmRow * kScaleCols], {1U, kScaleCols, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t b = 0U; b < kScaleCols; ++b) {
            const float sc = DecodeE8M0ToFloat(scaleUb.GetValue(b));
            const uint32_t base = b * 32U;
            for (uint32_t i = 0U; i < 32U; ++i) {
                fp32Ub.SetValue(base + i, DecodeFp8E4M3ToFloat(fp8Ub.GetValue(base + i)) * sc);
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Cast(bf16Ub, fp32Ub, RoundMode::CAST_RINT, kHidden);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(dstBf16[gmRow * kHidden], bf16Ub,
                    {1U, static_cast<uint32_t>(kHidden * sizeof(bfloat16_t)), 0U, 0U, 0U});
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

}  // namespace MegaMoeUrma
