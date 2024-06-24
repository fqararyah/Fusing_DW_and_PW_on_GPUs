#include "dtype_defs.h"
#include "common_funcs.h"
#include "../model_specs/layers_specs.h"
#include "dtype_defs.h"
#include "general_specs.h"
#include "parallalism_and_tiling.h"
#include "../utils/utils.h"
#include <cooperative_groups.h>
#include <iostream>
#include "sm_20_intrinsics.h"
#include <cassert>

void convolutionGPU(fms_dt *ifms, fms_dt *ofms, weights_dt *weights,
                    fused_scales_dt *fused_scales,
                    biases_dt *fused_zps,
                    layer_specs l_specs);

void convolutionGPU_fw_seq(fms_dt *ifms, fms_dt *d_ifms, fms_dt *d_ofms, weights_dt *weights,
                           fused_scales_dt *fused_scales,
                           biases_dt *fused_zps,
                           int *fused_params_offset,
                           layer_specs *ls_specs,
                           const int test_iterations,
                           const int num_layers,
                           float &exec_time);

void convolutionGPU_fw(fms_dt *ifms, fms_dt *ofms, weights_dt *weights,
                       fused_scales_dt *fused_scales,
                       biases_dt *fused_zps,
                       int *fused_params_offset,
                       layer_specs l_specs,
                       const int test_iteration,
                       float &exec_time);

void pw_convolutionGPU_f_w_v2_chw(fms_dt *ifms, fms_dt *ofms,
                                  weights_dt *pw_weights,
                                  fused_scales_dt *fused_scales,
                                  biases_dt *fused_zps,
                                  layer_specs pw_1_l_specs,
                                  int *fused_params_offsets,
                                  const int iteration,
                                  int *layers_parallelism_w,
                                  float &exec_time);

void fused_pw_pw_convolutionGPU_chw(fms_dt *ifms, fms_dt *ofms,
                                    weights_dt *pw_weights,
                                    fused_scales_dt *fused_scales,
                                    biases_dt *fused_zps,
                                    layer_specs pw_1_l_specs,
                                    layer_specs pw_2_l_specs,
                                    int *fused_params_offsets,
                                    const int iteration,
                                    int *layers_parallelism_w,
                                    float &exec_time);

void convolutionGPU_f_w_chw(fms_dt *ifms, fms_dt *ofms,
                            weights_dt *pw_weights,
                            weights_dt *dw_weights,
                            fused_scales_dt *fused_scales,
                            biases_dt *fused_zps,
                            int *fused_params_offset,
                            layer_specs l_specs,
                            const int test_iteration,
                            float &exec_time);

void convolutionGPU_h_w_chw(fms_dt *ifms, fms_dt *ofms,
                            weights_dt *pw_weights,
                            weights_dt *dw_weights,
                            fused_scales_dt *fused_scales,
                            biases_dt *fused_zps,
                            int *fused_params_offset,
                            layer_specs l_specs,
                            const int test_iteration,
                            float &exec_time);

void convolutionGPU_h_w_chw_wide(fms_dt *ifms, fms_dt *ofms,
                                 weights_dt *pw_weights,
                                 weights_dt *dw_weights,
                                 fused_scales_dt *fused_scales,
                                 biases_dt *fused_zps,
                                 int *fused_params_offset,
                                 layer_specs l_specs,
                                 const int test_iteration,
                                 float &exec_time);

void fused_dwpw_convolutionGPU_chw(fms_dt *ifms, fms_dt *ofms,
                                   weights_dt *pw_weghts,
                                   weights_dt *dw_weights,
                                   fused_scales_dt *fused_scales,
                                   biases_dt *fused_zps,
                                   layer_specs dw_l_specs,
                                   layer_specs pw_l_specs,
                                   int *fused_params_offsets,
                                   const int test_iterations,
                                   float &exec_time);

void fused_pw_dw_convolutionGPU_h_w_chw(fms_dt *ifms, fms_dt *ofms,
                                        weights_dt *pw_weights,
                                        weights_dt *dw_weights,
                                        fused_scales_dt *fused_scales,
                                        biases_dt *fused_zps,
                                        layer_specs pw_l_specs,
                                        layer_specs dw_l_specs,
                                        int *fused_params_offsets,
                                        const int test_iterations,
                                        float &exec_time,
                                        const int num_sms);

void fused_pw_dw_convolutionGPU_h_w_chw_wide(fms_dt *ifms, fms_dt *ofms,
                                             weights_dt *pw_weights,
                                             weights_dt *dw_weights,
                                             fused_scales_dt *fused_scales,
                                             biases_dt *fused_zps,
                                             layer_specs pw_l_specs,
                                             layer_specs dw_l_specs,
                                             int *fused_params_offsets,
                                             const int iteration,
                                             float &exec_time,
                                             const int num_sms);

void convolutionGPU_h_w_wide_hwc(fms_dt *ifms, fms_dt *ofms,
                                 weights_dt *pw_weights,
                                 weights_dt *dw_weights,
                                 fused_scales_dt *fused_scales,
                                 biases_dt *fused_zps,
                                 int *fused_params_offset,
                                 layer_specs l_specs,
                                 const int test_iteration,
                                 float &exec_time);

void convolutionCPU(fms_dt *ifms, fms_dt *ofms, weights_dt *weights,
                    fused_scales_dt *fused_scales,
                    biases_dt *fused_zps,
                    layer_specs l_specs);

void convolutionCPU_fw(fms_dt *ifms, fms_dt *ofms,
                       weights_dt *weights,
                       weights_dt *dw_weights,
                       fused_scales_dt *fused_scales,
                       biases_dt *fused_zps,
                       int *fused_params_offsets,
                       layer_specs l_specs);

void convolutionGPU_dot(fms_dt *ifms, fms_dt *ofms, weights_dt *weights,
                        fused_scales_dt *fused_scales,
                        biases_dt *fused_zps,
                        int *fused_params_offset,
                        layer_specs l_specs,
                        const int test_iteration,
                        float &exec_time);