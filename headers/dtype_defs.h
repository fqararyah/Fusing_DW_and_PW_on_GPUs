
#include <cinttypes>
#include "simulation_constants.h"

#ifndef DTYPE_DEFS
#define DTYPE_DEFS

using namespace std;

#if DATA_TYPE == FLOAT_DTYPE
typedef float weights_dt;
typedef float fms_dt;
typedef float pss_dt;
#elif DATA_TYPE == INT8_DTYPE
typedef int weights_dt;
typedef int fms_dt;
typedef int pss_dt; // partial sums
#endif
typedef int64_t norm_act_pss_dt;
typedef int64_t layer_0_norm_act_pss_dt;
typedef double pss_f_dt;    //+ 16
typedef int64_t dw_pss_dt;  // partial sums
typedef double dw_pss_f_dt; // + 16
typedef int64_t first_conv_pss_dt;
typedef uint8_t input_image_dt;
typedef int8_t fc_weights_dt;
typedef int8_t fc_out_dt;

typedef float scales_dt;
// typedef scales_dt fused_scales_dt;
typedef float fused_scales_dt;
typedef float pooling_fused_scales_dt;
typedef uint8_t fused_scales_log_2_shifts_dt;
typedef int64_t relu_6_fused_scales_dt;
typedef int64_t layer_0_relu_6_fused_scales_dt;
// typedef scales_dt rec_scales_dt;
typedef float rec_scales_dt;

typedef int biases_dt;

const int8_t QUANTIZATION_MAX = 127;
const int8_t QUANTIZATION_MIN = -128;

struct fms_quantization_scheme
{
    fms_dt ofm_zero_point;
    scales_dt ifm_scale;
    rec_scales_dt ofm_scale_rec;
    scales_dt ofm_scale;
    biases_dt fused_zero_point;
    fused_scales_dt fused_scales;
};

#endif