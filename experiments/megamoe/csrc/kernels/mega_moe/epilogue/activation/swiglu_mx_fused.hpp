// Adapted from DeepSeek's register-resident SwiGLU/quant VF structure.
// Copyright (c) 2026 DeepSeek. MIT License:
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#pragma once
#include <c_api/asc_simd.h>

namespace MegaMoeImpl::Activation {
// 仅迁移寄存器驻留/融合结构，不采用DS的FP32 amax scale和前置路由加权。
// 保留本工程的SwiGLU→BF16 RN→最大指数MX→BF16乘scale→FP8 RN契约。
__simd_vf__ inline void SwiGluMxFused(__ubuf__ bfloat16_t* input, __ubuf__ int8_t* output,
                                    __ubuf__ uint16_t* outputScale, uint32_t rows, float clamp)
{
    const auto groupMask = asc_create_mask_b32(PAT_VL32);
    const auto pairMask = asc_create_mask_b32(PAT_VL64);
    vector_bfloat16_t zeroBf16;
    asc_duplicate_scalar(zeroBf16, 0);
    vector_uint32_t zero, nanScale, nanReciprocal;
    asc_duplicate_scalar(zero, uint32_t{0});
    asc_duplicate_scalar(nanScale, uint32_t{255});
    asc_duplicate_scalar(nanReciprocal, uint32_t{0x7f810000});
    vector_int32_t range;
    vector_uint32_t highGroup;
    asc_arange(range, 0);
    asc_add_scalar(highGroup, reinterpret_cast<vector_uint32_t&>(range), uint32_t{32}, groupMask);
    vector_uint32_t exponentMask, reciprocalBias;
    asc_duplicate_scalar(exponentMask, uint32_t{0x7f800000});
    asc_duplicate_scalar(reciprocalBias, uint32_t{0x7f000000});
    auto gatePtr = input;
    auto upPtr = input + 128;
    auto out = reinterpret_cast<__ubuf__ uint8_t*>(output);
    // stage槽包含完整128行，奇数尾只读取分配内的padding行；写回仍只提交有效rows。
    for (uint32_t m = 0; m < rows; m += 2) {
        vector_float gate[4], up[4], value[4];
        #pragma unroll
        for (uint32_t r = 0; r < 2; ++r) {
            vector_bfloat16_t gateBf16, upBf16;
            asc_loadalign_postupdate(gateBf16, gatePtr, 256);
            asc_loadalign_postupdate(upBf16, upPtr, 256);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(gate[r*2]),
                      reinterpret_cast<vector_bfloat16_t&>(gate[r*2+1]), zeroBf16, gateBf16);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(up[r*2]),
                      reinterpret_cast<vector_bfloat16_t&>(up[r*2+1]), zeroBf16, upBf16);
        }
        #pragma unroll
        for (uint32_t p = 0; p < 4; ++p) {
            asc_min_scalar(gate[p], gate[p], clamp, pairMask);
            asc_min_scalar(up[p], up[p], clamp, pairMask);
            asc_max_scalar(up[p], up[p], -clamp, pairMask);
            asc_mul_scalar(value[p], gate[p], -1.0f, pairMask);
            asc_exp(value[p], value[p], pairMask);
            asc_add_scalar(value[p], value[p], 1.0f, pairMask);
            asc_div(value[p], gate[p], value[p], pairMask);
            asc_mul(value[p], value[p], up[p], pairMask);
            vector_bfloat16_t rounded;
            asc_float2bfloat16_rn(rounded, value[p], pairMask, ASC_POSITION_EVEN);
            asc_bfloat162float(value[p], rounded, pairMask, ASC_POSITION_EVEN);
        }
        #pragma unroll
        for (uint32_t p = 0; p < 4; ++p) {
            vector_uint32_t exponent[2];
            asc_and(exponent[0], reinterpret_cast<vector_uint32_t&>(value[p]), exponentMask, pairMask);
            asc_gather(exponent[1], exponent[0], highGroup);
            #pragma unroll
            for (uint32_t g = 0; g < 2; ++g) {
                asc_reduce_max(exponent[g], exponent[g], groupMask);
                asc_duplicate(exponent[g], exponent[g], pairMask);
            }
            asc_select(exponent[0], exponent[0], exponent[1], groupMask);
            const auto nonzero = asc_ge_scalar(exponent[0], uint32_t{1}, pairMask);
            const auto finite = asc_le_scalar(exponent[0], uint32_t{0x7f000000}, pairMask);
            asc_max_scalar(exponent[0], exponent[0], uint32_t{0x04000000}, pairMask);
            asc_add_scalar(reinterpret_cast<vector_int32_t&>(exponent[0]),
                           reinterpret_cast<vector_int32_t&>(exponent[0]), -0x04000000, pairMask);
            vector_uint32_t inv, scale;
            asc_sub(inv, reciprocalBias, exponent[0], pairMask);
            asc_select(inv, inv, nanReciprocal, finite);
            asc_select(inv, inv, zero, nonzero);
            asc_mul(value[p], value[p], reinterpret_cast<vector_float&>(inv), pairMask);
            // 原路径是在BF16中乘scale；显式保留该舍入（含下溢），不能只保留FP32乘法。
            vector_bfloat16_t scaledBf16;
            asc_float2bfloat16_rn(scaledBf16, value[p], pairMask, ASC_POSITION_EVEN);
            asc_bfloat162float(value[p], scaledBf16, pairMask, ASC_POSITION_EVEN);
            vector_fp8_e4m3fn_t fp8;
            asc_float2e4m3_rn_sat(fp8, value[p], pairMask, ASC_DISPERSE_FIRST_QUARTER);
            asc_storealign_pack_quarter(reinterpret_cast<__ubuf__ uint32_t*>(out),
                                       reinterpret_cast<vector_uint32_t&>(fp8), pairMask);
            out += 64;
            asc_shiftright_scalar(scale, exponent[0], 23, pairMask);
            asc_select(scale, scale, nanScale, finite);
            asc_select(scale, scale, zero, nonzero);
            asc_gather(exponent[1], scale, highGroup);
            vector_uint8_t pair, unused;
            asc_intlv(pair, unused, reinterpret_cast<vector_uint8_t&>(scale),
                      reinterpret_cast<vector_uint8_t&>(exponent[1]));
            // 保留工程原有row-major compact scale布局：每行两个UE8M0 pair。
            asc_storealign_1st(reinterpret_cast<__ubuf__ int16_t*>(outputScale) +
                              (m + p/2)*2 + p%2, reinterpret_cast<vector_int16_t&>(pair), 0);
        }
    }
}
} // namespace MegaMoeImpl::Activation
