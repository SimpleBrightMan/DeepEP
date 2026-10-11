// Copyright (c) 2026, Lu Lu
// BF16 register unpack/pack adapted from DeepGEMM-Ascend simd/mega_moe.hpp.
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#ifndef MEGA_MOE_UNPERMUTE_FUSED_HPP
#define MEGA_MOE_UNPERMUTE_FUSED_HPP
#if defined(__DAV_C310__)
#include <c_api/asc_simd.h>

namespace MegaMoeImpl {

// 与原Cast/Muls/Add一致：BF16精确展开，逐项FP32乘后再加，最后仅一次BF16舍入。
// 借鉴DeepGEMM的寄存器内归并；不能照搬其无权重sum，因为本路径尚未应用top-k权重。
__simd_vf__ inline void WeightedUnpermuteVf(__ubuf__ bfloat16_t* inputs, __ubuf__ float* weights,
                                           uint32_t topK, uint32_t channels)
{
    const auto mask16 = asc_create_mask_b16(PAT_ALL);
    const auto mask32 = asc_create_mask_b32(PAT_ALL);
    vector_bfloat16_t zero;
    asc_duplicate_scalar(zero, 0);
    auto output = inputs;
    for (uint32_t channel = 0; channel < channels; channel += 128U) {
        vector_float sum0, sum1;
        auto input = inputs + channel;
        auto weight = weights;
        for (uint32_t item = 0; item < topK; ++item) {
            vector_bfloat16_t value;
            vector_float low, high, scale;
            asc_loadalign_postupdate(value, input, channels);
            asc_loadalign_brc_postupdate(scale, weight, 1);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(low),
                      reinterpret_cast<vector_bfloat16_t&>(high), zero, value);
            asc_mul(low, low, scale, mask32);
            asc_mul(high, high, scale, mask32);
            if (item == 0U) {
                sum0 = low;
                sum1 = high;
            } else {
                asc_add(sum0, sum0, low, mask32);
                asc_add(sum1, sum1, high, mask32);
            }
        }
        vector_bfloat16_t out0, out1, packed, unused;
        asc_float2bfloat16_rn(out0, sum0, mask32, ASC_POSITION_EVEN);
        asc_float2bfloat16_rn(out1, sum1, mask32, ASC_POSITION_EVEN);
        asc_deintlv(packed, unused, out0, out1);
        // 只覆盖已读完的第一行当前chunk；不覆盖后续channel或其他输入。
        asc_storealign_postupdate(output, packed, 128U, mask16);
    }
}
}  // namespace MegaMoeImpl
#endif
#endif
