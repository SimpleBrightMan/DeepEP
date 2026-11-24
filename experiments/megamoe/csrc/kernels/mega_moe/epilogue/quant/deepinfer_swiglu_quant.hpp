// DeepInfer compatibility: FP32 SwiGLU -> MXFP8, without an intermediate BF16 rounding.
// Scale rounding follows DeepGEMM-Ascend simd/mega_moe.hpp (MIT), adapted to separate
// gate/up UB tiles and compact row-major SF. Scheduling and completion remain with the caller.
#ifndef MEGAMOE_DEEPINFER_SWIGLU_QUANT_HPP
#define MEGAMOE_DEEPINFER_SWIGLU_QUANT_HPP
#if defined(__DAV_C310__)
#include <c_api/asc_simd.h>

namespace MegaMoeImpl {
template <bool ApplyRouteWeight>
__simd_vf__ inline void DeepInferSwiGluQuant(
    __ubuf__ bfloat16_t* gateInput, __ubuf__ bfloat16_t* upInput,
    __ubuf__ uint8_t* output, __ubuf__ int16_t* scales, __ubuf__ float* routeWeights,
    uint32_t rows, uint32_t columns, uint32_t inputStride, uint32_t outputStride, float clamp)
{
    const vector_bool groupMask = asc_create_mask_b32(PAT_VL32);
    const vector_bool pairMask = asc_create_mask_b32(PAT_VL64);
    vector_bfloat16_t zero;
    asc_duplicate_scalar(zero, 0);
    vector_int32_t arange;
    vector_uint32_t highGroup, exponentMask, invBias;
    asc_arange(arange, 0);
    asc_add_scalar(highGroup, reinterpret_cast<vector_uint32_t&>(arange), 32, groupMask);
    asc_duplicate_scalar(exponentMask, 0x7f800000u);
    asc_duplicate_scalar(invBias, 0x7f000000u);
    for (uint32_t row = 0; row < rows; ++row) {
        vector_float routeWeight;
        if constexpr (ApplyRouteWeight) {
            auto pointer = routeWeights + row * 8U + 3U;
            asc_loadalign_brc_postupdate(routeWeight, pointer, 1);
        }
        for (uint32_t col = 0; col < columns; col += 128U) {
            vector_bfloat16_t gateBf16, upBf16;
            vector_float gate[2], up[2], value[2];
            asc_loadalign(gateBf16, gateInput + row * inputStride + col);
            asc_loadalign(upBf16, upInput + row * inputStride + col);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(gate[0]),
                      reinterpret_cast<vector_bfloat16_t&>(gate[1]), zero, gateBf16);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(up[0]),
                      reinterpret_cast<vector_bfloat16_t&>(up[1]), zero, upBf16);
            #pragma unroll
            for (uint32_t pair = 0; pair < 2; ++pair) {
                asc_min_scalar(gate[pair], gate[pair], clamp, pairMask);
                asc_min_scalar(up[pair], up[pair], clamp, pairMask);
                asc_max_scalar(up[pair], up[pair], -clamp, pairMask);
                asc_exp_sub(value[pair], reinterpret_cast<vector_float&>(zero), gate[pair], pairMask);
                asc_add_scalar(value[pair], value[pair], 1.0f, pairMask);
                asc_div(value[pair], gate[pair], value[pair], pairMask);
                asc_mul(value[pair], value[pair], up[pair], pairMask);
                if constexpr (ApplyRouteWeight)
                    asc_mul(value[pair], value[pair], routeWeight, pairMask);
                vector_uint32_t bits[2];
                asc_abs(reinterpret_cast<vector_float&>(bits[0]), value[pair], pairMask);
                asc_gather(bits[1], bits[0], highGroup);
                #pragma unroll
                for (uint32_t group = 0; group < 2; ++group) {
                    asc_reduce_max(bits[group], bits[group], groupMask);
                    asc_duplicate(bits[group], bits[group], pairMask);
                }
                asc_select(bits[0], bits[0], bits[1], groupMask);
                // ceil_pow2(max(amax,1e-4)/448)，不能仅取BF16最大指数再减8。
                asc_max_scalar(bits[0], bits[0], 0x38d1b717u, pairMask);
                asc_add_scalar(bits[0], bits[0], 0x001fffffu, pairMask);
                asc_and(bits[0], bits[0], exponentMask, pairMask);
                asc_add_scalar(reinterpret_cast<vector_int32_t&>(bits[0]),
                               reinterpret_cast<vector_int32_t&>(bits[0]), -0x04000000, pairMask);
                asc_min_scalar(bits[0], bits[0], 0x7f000000u, pairMask);
                vector_uint32_t invScale;
                asc_sub(invScale, invBias, bits[0], pairMask);
                asc_mul(value[pair], value[pair], reinterpret_cast<vector_float&>(invScale), pairMask);
                vector_fp8_e4m3fn_t quantized;
                asc_float2e4m3_rn_sat(quantized, value[pair], pairMask, ASC_DISPERSE_FIRST_QUARTER);
                asc_storealign_pack_quarter(
                    reinterpret_cast<__ubuf__ uint32_t*>(output + row * outputStride + col + pair * 64U),
                    reinterpret_cast<vector_uint32_t&>(quantized), pairMask);
                asc_shiftright_scalar(bits[0], bits[0], 23, pairMask);
                asc_gather(bits[1], bits[0], highGroup);
                vector_uint8_t packed, unused;
                asc_intlv(packed, unused, reinterpret_cast<vector_uint8_t&>(bits[0]),
                          reinterpret_cast<vector_uint8_t&>(bits[1]));
                asc_storealign_1st(scales + row * (outputStride / 64U) + col / 64U + pair,
                                   reinterpret_cast<vector_int16_t&>(packed), 0);
            }
        }
    }
}
} // namespace MegaMoeImpl
#endif
#endif
