#include "../../headers/conv_kernels.h"

#if (FUSION_MODE == ALL_MODES || FUSION_MODE == FUSED_F_W) && DATA_TYPE == INT8_DTYPE

using namespace std;
namespace cg = cooperative_groups;

void inline __device__ fill_dw_weights_tile(weights_dt *dw_weights,
                                            int8_t *weights_tile,
                                            const int dw_filter_area,
                                            const int layer_weights_offset,
                                            const int compact_layer_depth,
                                            const int start_filling_depth,
                                            const int offset_in_tile_depth)
{
    for (int i = 0; i < dw_filter_area; i++)
    {
        weights_dt current_weight = dw_weights[layer_weights_offset + i * compact_layer_depth + start_filling_depth];
        for (int d = 0; d < PACKED_ITEMS; d++)
        {
            weights_tile[(offset_in_tile_depth + d) * dw_filter_area + i] = EXTRACT_8_32(current_weight, d);
        }
    }
}

void inline __device__ fill_dw_scales_tile(scales_dt *fused_scales,
                                           biases_dt *fused_zps,
                                           scales_dt *fused_scales_tile,
                                           biases_dt *fused_zps_tile,
                                           const int start_filling_depth,
                                           const int fused_params_offset,
                                           const int offset_in_tile_depth)
{
    for (int f = 0; f < PACKED_ITEMS; f++)
    {
        fused_scales_tile[offset_in_tile_depth + f] = fused_scales[fused_params_offset + start_filling_depth + f];
        fused_zps_tile[offset_in_tile_depth + f] = fused_zps[fused_params_offset + start_filling_depth + f];
    }
}

__global__ void pw_conv_f_w_v2(fms_dt *ifms, fms_dt *ofms, weights_dt *pw_weights,
                               fused_scales_dt *fused_scales,
                               biases_dt *fused_zps,
                               const int pw_1_compact_layer_depth,
                               const int pw_1_compact_layer_num_filters,
                               const int pw_1_ifm_width,
                               const int pw_1_compact_ifm_width_depth,
                               const int pw_1_ofm_height,
                               const int pw_1_ofm_width,
                               const int pw_1_compact_ofm_width_depth,
                               const int pw_1_num_filters_to_parallelism_f_ratio,
                               const int pw_1_layer_weights_offset,
                               const int pw_1_layer_fused_params_offset,
                               const fms_dt pw_1_ofms_zp,
                               const scales_dt pw_1_relu_threshold,
                               const int layer_activation,
                               const int parallel_d,
                               const int parallel_w
                               )
{

    const int thread_f = threadIdx.z;

    const int block_w = blockIdx.y;
    const int block_f = blockIdx.x;
    const int abs_h_index = blockIdx.z;
    const int abs_w_index = block_w * parallel_w + threadIdx.x;

    //for (int i_f = 0; i_f < pw_1_num_filters_to_parallelism_f_ratio; i_f++)
    {
        const int pw_1_compact_filter_index = block_f * parallel_d + thread_f;
        const int pw_1_filter_index = pw_1_compact_filter_index * PACKED_ITEMS;

        scales_dt scale0 = fused_scales[pw_1_layer_fused_params_offset + pw_1_filter_index];
        scales_dt scale1 = fused_scales[pw_1_layer_fused_params_offset + pw_1_filter_index + 1];
        scales_dt scale2 = fused_scales[pw_1_layer_fused_params_offset + pw_1_filter_index + 2];
        scales_dt scale3 = fused_scales[pw_1_layer_fused_params_offset + pw_1_filter_index + 3];

        biases_dt fused_zp0 = fused_zps[pw_1_layer_fused_params_offset + pw_1_filter_index];
        biases_dt fused_zp1 = fused_zps[pw_1_layer_fused_params_offset + pw_1_filter_index + 1];
        biases_dt fused_zp2 = fused_zps[pw_1_layer_fused_params_offset + pw_1_filter_index + 2];
        biases_dt fused_zp3 = fused_zps[pw_1_layer_fused_params_offset + pw_1_filter_index + 3];

        const int base_index_in_ifms = abs_h_index * pw_1_ifm_width + abs_w_index;

        const int base_index_in_ofms = abs_h_index * pw_1_ofm_width +
                                       pw_1_compact_filter_index * pw_1_ofm_height * pw_1_ofm_width + abs_w_index;

        const int base_index_pw_1_weights = pw_1_layer_weights_offset + pw_1_filter_index * pw_1_compact_layer_depth;

        if (abs_w_index < pw_1_ofm_width)
        {
            pss_dt sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            int a = 0;
            for (int d = 0; d < pw_1_compact_layer_depth; d++)
            {
                fms_dt fms_val = ifms[base_index_in_ifms + d * pw_1_ifm_width * pw_1_ifm_width]; // TODO

                sum0 += __dp4a(fms_val, pw_weights[base_index_pw_1_weights + d], a);
                sum1 += __dp4a(fms_val, pw_weights[base_index_pw_1_weights + pw_1_compact_layer_depth + d], a);
                sum2 += __dp4a(fms_val, pw_weights[base_index_pw_1_weights + 2 * pw_1_compact_layer_depth + d], a);
                sum3 += __dp4a(fms_val, pw_weights[base_index_pw_1_weights + 3 * pw_1_compact_layer_depth + d], a);
            }
            if (layer_activation == RELU6)
            {
                ofms[base_index_in_ofms] = PACK_32_8s(quant_relu6(sum0, scale0, fused_zp0, pw_1_ofms_zp, pw_1_relu_threshold),
                                                      quant_relu6(sum1, scale1, fused_zp1, pw_1_ofms_zp, pw_1_relu_threshold),
                                                      quant_relu6(sum2, scale2, fused_zp2, pw_1_ofms_zp, pw_1_relu_threshold),
                                                      quant_relu6(sum3, scale3, fused_zp3, pw_1_ofms_zp, pw_1_relu_threshold));
            }
            else if (layer_activation == 0)
            {
                ofms[base_index_in_ofms] = PACK_32_8s(quant_no_activation(sum0, scale0, fused_zp0, pw_1_ofms_zp),
                                                      quant_no_activation(sum1, scale1, fused_zp1, pw_1_ofms_zp),
                                                      quant_no_activation(sum2, scale2, fused_zp2, pw_1_ofms_zp),
                                                      quant_no_activation(sum3, scale3, fused_zp3, pw_1_ofms_zp));
            }
        }
    }
}

void pw_convolutionGPU_f_w_v2_chw(fms_dt *ifms, fms_dt *ofms,
                                  weights_dt *pw_weights,
                                  fused_scales_dt *fused_scales,
                                  biases_dt *fused_zps,
                                  layer_specs pw_1_l_specs,
                                  int *fused_params_offsets,
                                  const int iteration,
                                  int *layers_parallelism_w,
                                  float &exec_time)
{

    const int pw_1_compact_layer_depth = (pw_1_l_specs.layer_depth / PACKED_ITEMS);
    const int pw_1_compact_layer_num_filters = (pw_1_l_specs.layer_num_fils / PACKED_ITEMS);

    int parallel_w = layers_parallelism_w[pw_1_l_specs.layer_index];

    int parallel_d = pw_1_compact_layer_depth < pw_1_compact_layer_num_filters ? pw_1_compact_layer_depth : pw_1_compact_layer_num_filters;

    int pw_1_num_filters_to_parallelism_f_ratio = pw_1_compact_layer_num_filters / parallel_d;
    if (pw_1_num_filters_to_parallelism_f_ratio < 1)
    {
        pw_1_num_filters_to_parallelism_f_ratio = 1;
    }

    dim3 threads(parallel_w, 1, parallel_d);
    dim3 blocks(pw_1_num_filters_to_parallelism_f_ratio, (pw_1_l_specs.layer_ifm_width + parallel_w - 1) / parallel_w,
                pw_1_l_specs.layer_ifm_height);

#if TIME_LAYER_BY_LAYER
    float elapsed_time;
    cudaEvent_t start_event, stop_event;
    cudaError_t err = cudaSuccess;

    err = (cudaEventCreate(&start_event));
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to cudaEventCreate start_event %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = (cudaEventCreate(&stop_event));
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to cudaEventCreate stop_event %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    if (iteration == 0)
    {
        printf("(PW_V2): layer %d\n", pw_1_l_specs.layer_index);
        //printf("parallel_w %d\n", pw_1_compact_layer_num_filters / parallel_d);
    }
    err = cudaEventRecord(start_event, 0);
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to cudaEventRecord start_event %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
#endif

    const int pw_1_layer_ifm_width = pw_1_l_specs.layer_ifm_width;
    const int pw_1_layer_ofm_width = pw_1_l_specs.layer_ofm_width;
    const int pw_1_compact_ofms_width_depth = pw_1_compact_layer_num_filters * pw_1_layer_ofm_width;

    pw_conv_f_w_v2<<<blocks, threads>>>(ifms, ofms, pw_weights,
                                        fused_scales,
                                        fused_zps,
                                        pw_1_compact_layer_depth,
                                        pw_1_compact_layer_num_filters,
                                        pw_1_layer_ifm_width,
                                        pw_1_compact_layer_depth * pw_1_layer_ifm_width,
                                        pw_1_l_specs.layer_ofm_height,
                                        pw_1_l_specs.layer_ofm_width,
                                        pw_1_compact_ofms_width_depth,
                                        pw_1_num_filters_to_parallelism_f_ratio,
                                        pw_1_l_specs.layer_weights_offset / PACKED_ITEMS,
                                        fused_params_offsets[pw_1_l_specs.layer_index],
                                        pw_1_l_specs.layer_ofms_zero_point,
                                        pw_1_l_specs.relu_threshold,
                                        pw_1_l_specs.layer_activation,
                                        parallel_d,
                                        parallel_w);

#if TIME_LAYER_BY_LAYER
    err = (cudaEventRecord(stop_event, 0));
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to cudaEventRecord stop_event %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = (cudaEventSynchronize(stop_event));
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to cudaEventSynchronize %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = (cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to cudaEventElapsedTime %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // printf("Measured time for sample = %.3fms\n", elapsed_time);
    if (iteration >= WARMUP_ITERATIONS)
    {
        exec_time += elapsed_time;
    }
#endif

    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess)
    {
        cout << "the error of code: " << kernel_error << " has happened\n";
    }
}

#endif