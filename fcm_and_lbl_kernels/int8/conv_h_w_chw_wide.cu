#include "../../headers/conv_kernels.h"

#if (FUSION_MODE == ALL_MODES || FUSION_MODE == NOT_FUSED) && DATA_LAYOUT == CHW && DATA_TYPE == INT8_DTYPE

__global__ void dw_conv3x3_h_w_wide(fms_dt *ifms, fms_dt *ofms, weights_dt *weights,
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
                                    const fms_dt packed_ifm_zp,
                                    const scales_dt dw_relu_threshold,
                                    const int parallel_h,
                                    const int parallel_w)
{

    const int thread_w = threadIdx.x;
    const int thread_h = threadIdx.y;
    const int thread_f = threadIdx.z;

    const int block_w = blockIdx.x;
    const int block_h = blockIdx.y;
    const int block_f = blockIdx.z;

    const int abs_dw_f_compact = block_f + thread_f;
    const int abs_dw_f = abs_dw_f_compact * PACKED_ITEMS;

    scales_dt scale0 = fused_scales[dw_layer_fused_params_offset + abs_dw_f];
    scales_dt scale1 = fused_scales[dw_layer_fused_params_offset + abs_dw_f + 1];
    scales_dt scale2 = fused_scales[dw_layer_fused_params_offset + abs_dw_f + 2];
    scales_dt scale3 = fused_scales[dw_layer_fused_params_offset + abs_dw_f + 3];

    biases_dt fused_zp0 = fused_zps[dw_layer_fused_params_offset + abs_dw_f];
    biases_dt fused_zp1 = fused_zps[dw_layer_fused_params_offset + abs_dw_f + 1];
    biases_dt fused_zp2 = fused_zps[dw_layer_fused_params_offset + abs_dw_f + 2];
    biases_dt fused_zp3 = fused_zps[dw_layer_fused_params_offset + abs_dw_f + 3];

    int8_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;

    weights_dt dw_filter_weights[FILTER_3x3_DIM * PACKED_ITEMS];
    for (int c_h = 0; c_h < FILTER_3x3_DIM; c_h++)
    {
        weights_dt weight_val0 =
            weights[dw_layer_weights_offset + (c_h * FILTER_3x3_DIM) + abs_dw_f_compact * FILTER_3x3_PADDED_AREA];
        weights_dt weight_val1 =
            weights[dw_layer_weights_offset + (c_h * FILTER_3x3_DIM + 1) + abs_dw_f_compact * FILTER_3x3_PADDED_AREA];
        weights_dt weight_val2 =
            weights[dw_layer_weights_offset + (c_h * FILTER_3x3_DIM + 2) + abs_dw_f_compact * FILTER_3x3_PADDED_AREA];
        for (int f = 0; f < PACKED_ITEMS; f++)
        {
            dw_filter_weights[f * FILTER_3x3_DIM + c_h] = PACK_32_8s(EXTRACT_8_32(weight_val0, f),
                                                                     EXTRACT_8_32(weight_val1, f),
                                                                     EXTRACT_8_32(weight_val2, f),
                                                                     0);
        }
    }

    // for (int h = 0; h < rows_per_thread; h++)
    {
        const int abs_row_index = (block_h * parallel_h + thread_h) * strides - padding_top;

        if ((block_h * parallel_h + thread_h) < dw_ofm_height)
        {

            // for (int i_w = 0; i_w < tile_w; i_w++)
            {
                pss_dt sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                const int abs_w_write = block_w * parallel_w + thread_w;
                const int abs_w_read = (block_w * parallel_w + thread_w) * strides - padding_left;

                int base_index_in_ofms = abs_dw_f_compact * dw_ofm_width * dw_ofm_height +
                                         (block_h * parallel_h + thread_h) * dw_ofm_width + abs_w_write;

                int base_index_in_ifms = abs_dw_f_compact * dw_ifm_height * dw_ifm_width +
                                         abs_row_index * dw_ifm_width + abs_w_read;

                if (abs_w_write < dw_ofm_width)
                {
                    for (int c_h = 0; c_h < FILTER_3x3_DIM; c_h++)
                    {
                        weights_dt weight_val0 = dw_filter_weights[c_h];
                        weights_dt weight_val1 = dw_filter_weights[FILTER_3x3_DIM + c_h];
                        weights_dt weight_val2 = dw_filter_weights[FILTER_3x3_DIM * 2 + c_h];
                        weights_dt weight_val3 = dw_filter_weights[FILTER_3x3_DIM * 3 + c_h];

                        fms_dt ifms_val0 = get_fms_val(ifms, abs_row_index + c_h, abs_w_read, dw_ifm_height, dw_ifm_width,
                                                       base_index_in_ifms + c_h * dw_ifm_width,
                                                       packed_ifm_zp);
                        fms_dt ifms_val1 = get_fms_val(ifms, abs_row_index + c_h, abs_w_read + 1, dw_ifm_height, dw_ifm_width,
                                                       base_index_in_ifms + c_h * dw_ifm_width + 1,
                                                       packed_ifm_zp);
                        fms_dt ifms_val2 = get_fms_val(ifms, abs_row_index + c_h, abs_w_read + 2, dw_ifm_height, dw_ifm_width,
                                                       base_index_in_ifms + c_h * dw_ifm_width + 2,
                                                       packed_ifm_zp);

                        sum0 +=
                            __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 0), EXTRACT_8_32(ifms_val1, 0), EXTRACT_8_32(ifms_val2, 0), 0), weight_val0, 0);
                        sum1 +=
                            __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 1), EXTRACT_8_32(ifms_val1, 1), EXTRACT_8_32(ifms_val2, 1), 0), weight_val1, 0);
                        sum2 +=
                            __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 2), EXTRACT_8_32(ifms_val1, 2), EXTRACT_8_32(ifms_val2, 2), 0), weight_val2, 0);
                        sum3 +=
                            __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 3), EXTRACT_8_32(ifms_val1, 3), EXTRACT_8_32(ifms_val2, 3), 0), weight_val3, 0);

                        // if (abs_row_index == 56 && abs_w_write == 5 && thread_f == 0)
                        // {
                        //     printf("%d % d %d\n", EXTRACT_8_32(ifms_val0, 0), EXTRACT_8_32(ifms_val1, 0), EXTRACT_8_32(ifms_val2, 0));
                        // }
                    }

                    q0 = quant_relu6(sum0, scale0, fused_zp0, dw_ofms_zp, dw_relu_threshold);
                    q1 = quant_relu6(sum1, scale1, fused_zp1, dw_ofms_zp, dw_relu_threshold);
                    q2 = quant_relu6(sum2, scale2, fused_zp2, dw_ofms_zp, dw_relu_threshold);
                    q3 = quant_relu6(sum3, scale3, fused_zp3, dw_ofms_zp, dw_relu_threshold);

                    // if (abs_row_index == 56 && abs_w_write == 5 && thread_f == 0)
                    // {
                    //     printf("\n%d\n", q0);
                    // }

                    // ofms_ifms_tile[row_offet_in_tile + thread_f * padded_tile_width + abs_w_write] = PACK_32_8s(q0, q1, q2, q3);
                    ofms[base_index_in_ofms] = PACK_32_8s(q0, q1, q2, q3);
                }
            }
        }
    }
}

__global__ void pw_conv_h_w_wide(fms_dt *ifms, fms_dt *ofms, weights_dt *pw_weights,
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
                                 const int parallel_w)
{

    const int thread_w = threadIdx.x;
    const int thread_h = threadIdx.y;

    const int thread_hw = thread_h * parallel_w + thread_w;

    const int block_f = blockIdx.x;
    const int block_h = blockIdx.y;
    const int block_w = blockIdx.z;

    scales_dt scale0, scale1, scale2, scale3;

    biases_dt fused_zp0, fused_zp1, fused_zp2, fused_zp3;

    __shared__ weights_dt weights_tile[MAX_COMPACT_DEPTH * PACKED_ITEMS];

    // for (int o_f = 0; o_f < packed_ofms_per_thread_f; o_f++)
    {
        const int abs_f_compact = block_f;
        const int base_index_pw_scales = abs_f_compact * PACKED_ITEMS;
        const int base_index_pw_weights = pw_layer_weights_offset + abs_f_compact * PACKED_ITEMS * compact_layer_depth;

        scale0 = fused_scales[pw_layer_fused_params_offset + base_index_pw_scales];
        scale1 = fused_scales[pw_layer_fused_params_offset + base_index_pw_scales + 1];
        scale2 = fused_scales[pw_layer_fused_params_offset + base_index_pw_scales + 2];
        scale3 = fused_scales[pw_layer_fused_params_offset + base_index_pw_scales + 3];

        fused_zp0 = fused_zps[pw_layer_fused_params_offset + base_index_pw_scales];
        fused_zp1 = fused_zps[pw_layer_fused_params_offset + base_index_pw_scales + 1];
        fused_zp2 = fused_zps[pw_layer_fused_params_offset + base_index_pw_scales + 2];
        fused_zp3 = fused_zps[pw_layer_fused_params_offset + base_index_pw_scales + 3];

        // if(thread_h == 0 && thread_w == 0)
        // for (int d = 0; d < compact_layer_depth; d++)
        if (thread_hw < compact_layer_depth)
        {
            weights_tile[thread_hw * PACKED_ITEMS] = pw_weights[base_index_pw_weights + thread_hw];
            weights_tile[thread_hw * PACKED_ITEMS + 1] = pw_weights[base_index_pw_weights + compact_layer_depth + thread_hw];
            weights_tile[thread_hw * PACKED_ITEMS + 2] = pw_weights[base_index_pw_weights + 2 * compact_layer_depth + thread_hw];
            weights_tile[thread_hw * PACKED_ITEMS + 3] = pw_weights[base_index_pw_weights + 3 * compact_layer_depth + thread_hw];
        }
        __syncthreads();
        // for (int h = 0; h < rows_per_thread; h++)
        {
            // for (int i_w = 0; i_w < tile_w; i_w++)
            if (block_h * parallel_h + thread_h < pw_ofm_width)
            {
                const int abs_w_write = block_w * parallel_w + thread_w;
                const int abs_w_read = block_w * parallel_w + thread_w; // TODO
                const int base_index_in_ofms = (block_h * parallel_h + thread_h) * pw_ofm_width +
                                               abs_f_compact * pw_ofm_width * pw_ofm_width + abs_w_write; // TODO

                if (abs_w_write < pw_ofm_width)
                {
                    pss_dt sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                    int a = 0;
                    for (int d = 0; d < compact_layer_depth; d++)
                    {
                        int base_index_in_ifms = (block_h * parallel_h + thread_h) * pw_ifm_width +
                                                 d * pw_ifm_width * pw_ifm_width + abs_w_read; // TODO

                        fms_dt ifms_val = ifms[base_index_in_ifms];

                        sum0 += __dp4a(ifms_val, weights_tile[d * PACKED_ITEMS], a);
                        sum1 += __dp4a(ifms_val, weights_tile[d * PACKED_ITEMS + 1], a);
                        sum2 += __dp4a(ifms_val, weights_tile[d * PACKED_ITEMS + 2], a);
                        sum3 += __dp4a(ifms_val, weights_tile[d * PACKED_ITEMS + 3], a);

                        // if (base_index_in_ofms + abs_w_write < 10)
                        // {
                        //     printf("%d * %d\n ", EXTRACT_8_32(pw_weights[base_index_pw_weights + d], 0), EXTRACT_8_32(fms_val, 0));
                        //     printf("%d * %d\n", EXTRACT_8_32(pw_weights[base_index_pw_weights + d], 1), EXTRACT_8_32(fms_val, 1));
                        //     printf("%d * %d\n", EXTRACT_8_32(pw_weights[base_index_pw_weights + d], 2), EXTRACT_8_32(fms_val, 2));
                        //     printf("%d * %d\n", EXTRACT_8_32(pw_weights[base_index_pw_weights + d], 3), EXTRACT_8_32(fms_val, 3));
                        // }
                    }

                    // if (base_index_in_ofms + abs_w_write < 10)
                    // {
                    //     printf("%d \n", quant_relu6(sum0, scale0, fused_zp0, pw_ofms_zp, pw_relu_threshold));
                    // }

                    if (layer_activation == RELU6)
                    {
                        ofms[base_index_in_ofms] = PACK_32_8s(quant_relu6(sum0, scale0, fused_zp0, pw_ofms_zp, pw_relu_threshold),
                                                              quant_relu6(sum1, scale1, fused_zp1, pw_ofms_zp, pw_relu_threshold),
                                                              quant_relu6(sum2, scale2, fused_zp2, pw_ofms_zp, pw_relu_threshold),
                                                              quant_relu6(sum3, scale3, fused_zp3, pw_ofms_zp, pw_relu_threshold));
                    }
                    else if (layer_activation == 0)
                    {
                        ofms[base_index_in_ofms] = PACK_32_8s(quant_no_activation(sum0, scale0, fused_zp0, pw_ofms_zp),
                                                              quant_no_activation(sum1, scale1, fused_zp1, pw_ofms_zp),
                                                              quant_no_activation(sum2, scale2, fused_zp2, pw_ofms_zp),
                                                              quant_no_activation(sum3, scale3, fused_zp3, pw_ofms_zp));
                    }
                }
            }
        }
    }
}

void convolutionGPU_h_w_chw_wide(fms_dt *ifms, fms_dt *ofms,
                                 weights_dt *pw_weights,
                                 weights_dt *dw_weights,
                                 fused_scales_dt *fused_scales,
                                 biases_dt *fused_zps,
                                 int *fused_params_offset,
                                 layer_specs l_specs,
                                 const int test_iteration,
                                 float &exec_time)
{

    const int num_filters = l_specs.layer_num_fils;

    const int ofms_width = l_specs.layer_ofm_width;
    const int ofms_height = l_specs.layer_ofm_height;
    const int compact_layer_depth = (l_specs.layer_depth >> 2);
    const int compact_layer_num_filters = (l_specs.layer_num_fils >> 2);

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

    err = cudaEventRecord(start_event, 0);
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to cudaEventRecord start_event %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
#endif

    if (test_iteration == 0)
    {
        printf("(conv_h_w_chw_wide_int8): layer %d\n", l_specs.layer_index);
    }

    const int padded_tile_w = least_pow_of_2_geq(l_specs.layer_ifm_width +
                                                 l_specs.padding_left + l_specs.padding_right);

    const int parallel_w = TILE_W_H_W > ofms_width ? least_pow_of_2_geq(ofms_width) : TILE_W_H_W;
    const int parallel_h = TILE_H_H_W > ofms_height ? least_pow_of_2_geq(ofms_height) : TILE_H_H_W;

    if (l_specs.conv_layer_type == DW_CONV)
    {
        uint8_t ifms_zp = (uint8_t)l_specs.layer_ifms_zero_point;
        uint8_t ifm_zps_to_pack[4] = {ifms_zp, ifms_zp, ifms_zp, ifms_zp};
        fms_dt packed_ifm_zp = PACK_32_8s(ifm_zps_to_pack);

        dim3 threads(parallel_w, parallel_h, 1);
        dim3 blocks((ofms_width + parallel_w) / parallel_w, (ofms_height + parallel_h) / parallel_h, compact_layer_num_filters);

        dw_conv3x3_h_w_wide<<<blocks, threads>>>(ifms, ofms, dw_weights, fused_scales, fused_zps,
                                                 l_specs.layer_depth,
                                                 l_specs.layer_ifm_height,
                                                 l_specs.layer_ifm_width,
                                                 l_specs.layer_ofm_height,
                                                 l_specs.layer_ofm_width,
                                                 l_specs.strides,
                                                 l_specs.padding_top,
                                                 l_specs.padding_bottom,
                                                 l_specs.padding_left,
                                                 l_specs.padding_right,
                                                 padded_tile_w,
                                                 l_specs.layer_weights_offset / PACKED_ITEMS,
                                                 fused_params_offset[l_specs.layer_index],
                                                 l_specs.layer_ifms_zero_point,
                                                 l_specs.layer_ofms_zero_point,
                                                 packed_ifm_zp,
                                                 l_specs.relu_threshold,
                                                 parallel_h,
                                                 parallel_w);
    }
    else if (l_specs.conv_layer_type == PW_CONV)
    {

        dim3 threads(parallel_w, parallel_h, 1);
        dim3 blocks(compact_layer_num_filters, (ofms_width + parallel_w) / parallel_w, (ofms_height + parallel_h) / parallel_h);
        // dim3 threads(16, 1, compact_layer_depth);
        // dim3 blocks(1, l_specs.layer_ifm_height / l_specs.strides, 1);

        pw_conv_h_w_wide<<<blocks, threads>>>(ifms, ofms, pw_weights, fused_scales, fused_zps,
                                              compact_layer_depth, num_filters,
                                              l_specs.layer_ifm_width,
                                              l_specs.layer_ofm_width,
                                              l_specs.layer_weights_offset / PACKED_ITEMS,
                                              fused_params_offset[l_specs.layer_index],
                                              l_specs.layer_ofms_zero_point,
                                              l_specs.relu_threshold,
                                              l_specs.layer_activation,
                                              parallel_h,
                                              parallel_w);
    }

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
    if (test_iteration >= WARMUP_ITERATIONS)
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