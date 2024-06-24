#include "cuda.h"
#include "cuda_runtime.h"
#include "../../headers/dtype_defs.h"
#include "../../headers/general_specs.h"
#include "../../headers/parallalism_and_tiling.h"
#include "../../headers/simulation_constants.h"
#include <iostream>
#include "../../headers/other_layers.h"
#include "../../headers/common_funcs.h"

using namespace std;

#ifndef OTHER_LAYERS_GPU
#define OTHER_LAYERS_GPU

__global__ void gpu_add_kernel(fms_dt *src, fms_dt *dst,
                               const int layer_ofms_size,
                               scales_dt close_layer_scale,
                               fms_dt close_layer_zp,
                               scales_dt far_layer_scale,
                               fms_dt far_layer_zp,
                               scales_dt add_layer_scale_rec,
                               fms_dt add_layer_zp,
                               const int block_share,
                               const int work_per_thread)
{

    const int index_in_fms = blockIdx.x * block_share + threadIdx.x * work_per_thread;

    for (int pack_i = index_in_fms; pack_i < index_in_fms + work_per_thread; pack_i++)
    {
        if (pack_i < layer_ofms_size)
        {
            // if(i < 112)printf("%d, %d, \n", dst[i], src[i]);
            fms_dt src_fms_val = src[pack_i];
            fms_dt dst_fms_val = dst[pack_i];
            fms_dt result = 0;

            for (int i = 0; i < PACKED_ITEMS; i++)
            {
                float scaled_result = (close_layer_scale * (EXTRACT_8_32(dst_fms_val, i) - close_layer_zp) +
                                       far_layer_scale * (EXTRACT_8_32(src_fms_val, i) - far_layer_zp)) *
                                          add_layer_scale_rec +
                                      add_layer_zp;
                scaled_result = scaled_result + 0.5 - (scaled_result < 0);
                PACK_32_8(result, clamp((int16_t)scaled_result), i);
            }

            dst[pack_i] = result;
        }
    }
}

void gpu_add(fms_dt *src, fms_dt *dst, layer_specs conv_l_specs, Settings_struct settings)
{
    const int layer_ofms_size = conv_l_specs.layer_ofm_height * conv_l_specs.layer_ofm_width *
                                conv_l_specs.layer_num_fils / PACKED_ITEMS;

    scales_dt close_layer_scale = conv_l_specs.layer_ofms_scale;
    fms_dt close_layer_zp = conv_l_specs.layer_ofms_zero_point;

    scales_dt far_layer_scale = conv_l_specs.skip_connection_other_layer_scale;
    fms_dt far_layer_zp = conv_l_specs.skip_connection_other_layer_zero_point;

    scales_dt add_layer_scale_rec = conv_l_specs.add_layer_scale_reciprocal;
    fms_dt add_layer_zp = conv_l_specs.add_layer_zero_point;

    const int threads_per_block = 1024;
    int num_blocks = (layer_ofms_size + threads_per_block - 1) / threads_per_block;
    if (num_blocks > 2 * settings.num_sms)
    {
        num_blocks = 2 * settings.num_sms;
    }
    int block_share = (layer_ofms_size + num_blocks - 1) / num_blocks;
    int work_per_thread = (block_share + threads_per_block - 1) / threads_per_block;
    dim3 threads(threads_per_block, 1, 1);
    dim3 blocks(num_blocks, 1, 1);

    gpu_add_kernel<<<blocks, threads>>>(src, dst, layer_ofms_size, close_layer_scale, close_layer_zp,
                                        far_layer_scale, far_layer_zp, add_layer_scale_rec, add_layer_zp,
                                        block_share, work_per_thread);

    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess)
    {
        cout << "the error of code: " << kernel_error << " has happened\n";
    }
}

__global__ void gpu_avgpool_all_hw_kernel(fms_dt *fms,
                                          scales_dt scale,
                                          fms_dt ifm_zp,
                                          fms_dt ofm_zp,
                                          const int avgpool_input_depth,
                                          const int avgpool_input_height,
                                          const int avgpool_input_width,
                                          const int avgpool_input_hw,
                                          const int start_writing_index,
                                          const int threads_per_block)
{

    const int index_in_ofms = blockIdx.x * threads_per_block + threadIdx.x;
    const int base_index_in_ifms = index_in_ofms * avgpool_input_hw;
    pss_dt tmp = 0;
    for (int hw = 0; hw < avgpool_input_hw; hw++)
    {
        tmp += fms[base_index_in_ifms + hw];
    }

    pss_f_dt scaled_tmp = (tmp / avgpool_input_hw - ifm_zp) * scale + ofm_zp;

    fms[start_writing_index + index_in_ofms] = clamp(scaled_tmp);
}

void gpu_avgpool_all_hw(fms_dt *fms,
                        const pooling_layer_specs layer_specs_struct)
{

    const int avgpool_input_depth = layer_specs_struct.ifm_depth;
    const int avgpool_input_height = layer_specs_struct.ifm_height;
    const int avgpool_input_width = layer_specs_struct.ifm_width;
    const int avgpool_input_hw = avgpool_input_height * avgpool_input_width;

    const int threads_per_block = 128;
    const int num_blocks = (avgpool_input_depth + threads_per_block - 1) / threads_per_block;

    dim3 threads(threads_per_block, 1, 1);
    dim3 blocks(num_blocks, 1, 1);
    const int start_writing_index = MAX_FMS_SIZE_PACKED - avgpool_input_depth;

    gpu_avgpool_all_hw_kernel<<<blocks, threads>>>(fms,
                                                   layer_specs_struct.fused_scale,
                                                   layer_specs_struct.ifms_zero_point,
                                                   layer_specs_struct.ofms_zero_point,
                                                   avgpool_input_depth,
                                                   avgpool_input_height,
                                                   avgpool_input_width,
                                                   avgpool_input_hw,
                                                   start_writing_index,
                                                   threads_per_block);
}

#endif