#include "../../headers/conv_kernels.h"

#if (FUSION_MODE == ALL_MODES || FUSION_MODE == NOT_FUSED) && DATA_LAYOUT == HWC && DATA_TYPE == FLOAT_DTYPE

__global__ void pw_conv_h_w(fms_dt *ifms, fms_dt *ofms, weights_dt *pw_weights,
                            fused_scales_dt *fused_scales,
                            biases_dt *fused_zps,
                            const int compact_layer_depth,
                            const int pw_num_filters,
                            const int pw_ifm_width,
                            const int pw_ofm_width,
                            const int pw_layer_weights_offset,
                            const int pw_layer_fused_params_offset,
                            const fms_dt pw_ofms_zp,
                            const scales_dt pw_relu_threshold,
                            const int layer_activation,
                            const int parallel_h,
                            const int parallel_w,
                            const int parallel_hw)
{
    

}

__global__ void dw_conv3x3_h_w(fms_dt *ifms, fms_dt *ofms, weights_dt *weights,
                               fused_scales_dt *fused_scales,
                               biases_dt *fused_zps,
                               const int dw_ifm_depth,
                               const int dw_ifm_height,
                               const int dw_ifm_width,
                               const int dw_ofm_height,
                               const int dw_ofm_width,
                               const int strides,
                               const int padding_top,
                               const int padding_bottom,
                               const int padding_left,
                               const int padding_right,
                               const int padded_tile_width,
                               const int dw_layer_weights_offset,
                               const int dw_layer_fused_params_offset,
                               const fms_dt dw_ifms_zp,
                               const fms_dt dw_ofms_zp,
                               const scales_dt dw_relu_threshold,
                               const int parallel_h,
                               const int parallel_w)
{

}

void convolutionGPU_h_w(fms_dt *ifms, fms_dt *ofms,
                        weights_dt *pw_weights,
                        weights_dt *dw_weights,
                        fused_scales_dt *fused_scales,
                        biases_dt *fused_zps,
                        int *fused_params_offset,
                        layer_specs l_specs,
                        const int test_iteration,
                        float &exec_time)
{
    
}

#endif