#include "cuda.h"
#include "cuda_runtime.h"
#include "dtype_defs.h"
#include "general_specs.h"
#include "parallalism_and_tiling.h"
#include "simulation_constants.h"
#include <iostream>

using namespace std;

#ifndef COMMON_FUNCS
#define COMMON_FUNCS

__host__ __device__ inline void PACK_32_8(int &dst_32, uint8_t src_8, int pos)
{
    dst_32 = dst_32 | ((src_8) << (pos << 3));
}

__host__ __device__ inline int PACK_32_8s(uint8_t src_8[4])
{
    int packed = 0;
    return packed | ((int)src_8[0]) | (((int)src_8[1]) << 8) | (((int)src_8[2]) << 16) | (((int)src_8[3]) << 24);
}

__host__ __device__ inline int PACK_32_8s(uint8_t src_8_0, uint8_t src_8_1, uint8_t src_8_2, uint8_t src_8_3)
{
    int packed = 0;
    return packed | ((int)src_8_0) | (((int)src_8_1) << 8) | (((int)src_8_2) << 16) | (((int)src_8_3) << 24);
}

__host__ __device__ inline int8_t EXTRACT_8_32(int src_32, int pos) { return (int8_t)((src_32 & (((unsigned int)255) << (pos * 8))) >> (pos * 8)); }

fms_dt inline __device__ get_fms_val(fms_dt *ifms,
                                     const int index_h, const int index_w,
                                     const int dw_ifm_height,
                                     const int dw_ifm_width,
                                     const int index_in_ifms,
                                     const fms_dt packed_ifm_zp)
{

    if (index_h >= 0 &&
        index_h < dw_ifm_height &&
        index_w >= 0 && index_w < dw_ifm_width)
    {
        return ifms[index_in_ifms];
    }
    else
    {
        return packed_ifm_zp;
    }
}

inline __device__ void get_fms_vals(fms_dt *ifms,
                                    const int index_h, const int index_w,
                                    const int dw_ifm_height,
                                    const int dw_ifm_width,
                                    const int index_in_ifms,
                                    const fms_dt ifm_zp,
                                    const int val_to_val_offset,
                                    fms_dt &val0,
                                    fms_dt &val1,
                                    fms_dt &val2,
                                    fms_dt &val3)
{

    if (index_h >= 0 &&
        index_h < dw_ifm_height &&
        index_w >= 0 && index_w < dw_ifm_width)
    {
        val0 = ifms[index_in_ifms];
        val1 = ifms[index_in_ifms + val_to_val_offset];
        val2 = ifms[index_in_ifms + 2 * val_to_val_offset];
        val3 = ifms[index_in_ifms + 3 * val_to_val_offset];
    }
    else
    {
        val0 = ifm_zp;
        val1 = ifm_zp;
        val2 = ifm_zp;
        val3 = ifm_zp;
    }
}

__host__ __device__ inline int8_t clamp(int16_t val)
{

    if (val > QUANTIZATION_MAX)
    {
        return QUANTIZATION_MAX;
    }
    if (val < QUANTIZATION_MIN)
    {
        return QUANTIZATION_MIN;
    }
    return (int8_t)val;
}

__host__ __device__ inline int8_t quant_relu6(pss_dt pss, const fused_scales_dt fused_scale,
                                              const biases_dt fused_zp,
                                              const fms_dt ofms_zp,
                                              const int relu_threshold)
{
#if COMPARE_WITH_CUDNN
    if (pss <= 0)
    {
        return 0;
    }
    else if (pss > 6)
    {
        return 6;
    }
    else
    {
        return pss;
    }
#else
    pss += fused_zp;

    if (pss <= 0)
    {
        return ofms_zp;
    }

    float scaled_pss = pss * fused_scale;
    if (scaled_pss <= relu_threshold)
    {
        scaled_pss += ofms_zp;
        scaled_pss += 0.5 - (scaled_pss < 0);
        return clamp((int16_t)scaled_pss);
    }
    return clamp((int16_t)(ofms_zp + relu_threshold));
#endif
}

__host__ __device__ inline int8_t quant_no_activation(pss_dt pss, const fused_scales_dt fused_scale,
                                                      const biases_dt fused_zp,
                                                      const fms_dt ofms_zp)
{
#if COMPARE_WITH_CUDNN
    if (pss <= 0)
    {
        return 0;
    }
    else if (pss > 6)
    {
        return 6;
    }
    else
    {
        return pss;
    }
#else
    pss += fused_zp;
    float scaled_pss = pss * fused_scale;
    scaled_pss += ofms_zp;
    scaled_pss += 0.5 - (scaled_pss < 0);
    return clamp((int16_t)scaled_pss);
#endif
}

#endif