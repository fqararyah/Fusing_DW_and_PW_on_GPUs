#include "../../headers/conv_kernels.h"

#if (FUSION_MODE == ALL_MODES || FUSION_MODE == NOT_FUSED) && DATA_TYPE == FLOAT_DTYPE

__global__ void pw_conv_h_w_chw_wide(fms_dt *ifms, fms_dt *ofms, weights_dt *pw_weights,
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
    const int compact_layer_depth_to_parallel_hw_ratio = (compact_layer_depth + parallel_hw - 1) / parallel_hw;

    const int thread_w = threadIdx.x;
    const int thread_h = threadIdx.y;
    const int thread_f = threadIdx.z;

    const int thread_hw = thread_h * parallel_h + thread_w;

    // switching order to improve scheduling see the kernel call
    const int block_f = blockIdx.x;
    const int block_w = blockIdx.y;
    const int block_h = blockIdx.z;

    __shared__ weights_dt weights_tile[TILE_F_H_W_WIDE][256]; // TODO

    const int abs_f_compact = block_f * TILE_F_H_W_WIDE + thread_f;
    int base_index_pw_weights = pw_layer_weights_offset + abs_f_compact * PACKED_ITEMS * compact_layer_depth;

    // if(thread_h == 0 && thread_w == 0)
    // for (int d = 0; d < compact_layer_depth; d++)
    const int abs_w_write = block_w * parallel_w + thread_w;
    const int abs_w_read = block_w * parallel_w + thread_w; // TODO
    const int base_index_in_ofms = (block_h * parallel_h + thread_h) * pw_ofm_width +
                                   abs_f_compact * pw_ofm_width * pw_ofm_width + abs_w_write; // TODO

    // if (abs_w_write < pw_ofm_width)
    // {
    pss_dt sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
#if TILE_F_H_W_WIDE == 16
    pss_dt sum8 = 0, sum9 = 0, sum10 = 0, sum11 = 0, sum12 = 0, sum13 = 0, sum14 = 0, sum15 = 0;
#endif

    for (int i = 0; i < compact_layer_depth_to_parallel_hw_ratio; i++)
    {
        const int iter_ifms_offset = i * parallel_hw;
        for (int o_f = 0; o_f < TILE_F_H_W_WIDE; o_f++)
        {
            if (iter_ifms_offset + thread_hw < compact_layer_depth) // TODO
            {
                weights_tile[o_f][thread_hw] = pw_weights[base_index_pw_weights + iter_ifms_offset +
                                                          (o_f * compact_layer_depth) + thread_hw];
            }
        }

        __syncthreads();

        if ((block_h * parallel_h + thread_h) < pw_ofm_width && abs_w_write < pw_ofm_width)
        {

            for (int d = 0; d < parallel_hw && d + iter_ifms_offset < compact_layer_depth; d++)
            {
                int base_index_in_ifms = (block_h * parallel_h + thread_h) * pw_ifm_width +
                                         (d + iter_ifms_offset) * pw_ifm_width * pw_ifm_width + abs_w_read; // TODO

                fms_dt ifms_val = ifms[base_index_in_ifms];
                sum0 += ifms_val * weights_tile[0][d];
                sum1 += ifms_val * weights_tile[1][d];
                sum2 += ifms_val * weights_tile[2][d];
                sum3 += ifms_val * weights_tile[3][d];
                sum4 += ifms_val * weights_tile[4][d];
                sum5 += ifms_val * weights_tile[5][d];
                sum6 += ifms_val * weights_tile[6][d];
                sum7 += ifms_val * weights_tile[7][d];
#if TILE_F_H_W_WIDE == 16
                sum8 += ifms_val * weights_tile[8][d];
                sum9 += ifms_val * weights_tile[9][d];
                sum10 += ifms_val * weights_tile[10][d];
                sum11 += ifms_val * weights_tile[11][d];
                sum12 += ifms_val * weights_tile[12][d];
                sum13 += ifms_val * weights_tile[13][d];
                sum14 += ifms_val * weights_tile[14][d];
                sum15 += ifms_val * weights_tile[15][d];
#endif
            }
        }
        __syncthreads();
    }
    if ((block_h * parallel_h + thread_h) < pw_ofm_width && abs_w_write < pw_ofm_width)
    {
        if (sum0 < 0)
        {
            sum0 = 0;
        }
        else
        {
            sum0 = sum0 * DUMMY_SCALE + DUMMY_BIAS;
        }
        if (sum1 < 0)
        {
            sum1 = 0;
        }
        else
        {
            sum1 = sum1 * DUMMY_SCALE + DUMMY_BIAS;
        }
        if (sum2 < 0)
        {
            sum2 = 0;
        }
        else
        {
            sum2 = sum2 * DUMMY_SCALE + DUMMY_BIAS;
        }
        if (sum3 < 0)
        {
            sum3 = 0;
        }
        else
        {
            sum3 = sum3 * DUMMY_SCALE + DUMMY_BIAS;
        }
        if (sum4 < 0)
        {
            sum4 = 0;
        }
        else
        {
            sum4 = sum4 * DUMMY_SCALE + DUMMY_BIAS;
        }
        if (sum5 < 0)
        {
            sum5 = 0;
        }
        else
        {
            sum5 = sum5 * DUMMY_SCALE + DUMMY_BIAS;
        }
        if (sum6 < 0)
        {
            sum6 = 0;
        }
        else
        {
            sum6 = sum6 * DUMMY_SCALE + DUMMY_BIAS;
        }
        if (sum7 < 0)
        {
            sum7 = 0;
        }
        else
        {
            sum7 = sum7 * DUMMY_SCALE + DUMMY_BIAS;
        }
#if TILE_F_H_W_WIDE == 16
        if (sum8 < 0)
        {
            sum8 = 0;
        }
        else
        {
            sum8 = sum8 * DUMMY_SCALE + DUMMY_BIAS;
        }
        if (sum9 < 0)
        {
            sum9 = 0;
        }
        else
        {
            sum9 = sum9 * DUMMY_SCALE + DUMMY_BIAS;
        }
        if (sum10 < 0)
        {
            sum10 = 0;
        }
        else
        {
            sum10 = sum10 * DUMMY_SCALE + DUMMY_BIAS;
        }
        if (sum11 < 0)
        {
            sum11 = 0;
        }
        else
        {
            sum11 = sum11 * DUMMY_SCALE + DUMMY_BIAS;
        }
        if (sum12 < 0)
        {
            sum12 = 0;
        }
        else
        {
            sum12 = sum12 * DUMMY_SCALE + DUMMY_BIAS;
        }
        if (sum13 < 0)
        {
            sum13 = 0;
        }
        else
        {
            sum13 = sum13 * DUMMY_SCALE + DUMMY_BIAS;
        }
        if (sum14 < 0)
        {
            sum14 = 0;
        }
        else
        {
            sum14 = sum14 * DUMMY_SCALE + DUMMY_BIAS;
        }
        if (sum15 < 0)
        {
            sum15 = 0;
        }
        else
        {
            sum15 = sum15 * DUMMY_SCALE + DUMMY_BIAS;
        }
#endif
        ofms[base_index_in_ofms] = sum0;
        ofms[base_index_in_ofms + (pw_ofm_width * pw_ofm_width)] = sum1;
        ofms[base_index_in_ofms + 2 * (pw_ofm_width * pw_ofm_width)] = sum2;
        ofms[base_index_in_ofms + 3 * (pw_ofm_width * pw_ofm_width)] = sum3;
        ofms[base_index_in_ofms + 4 * (pw_ofm_width * pw_ofm_width)] = sum4;
        ofms[base_index_in_ofms + 5 * (pw_ofm_width * pw_ofm_width)] = sum5;
        ofms[base_index_in_ofms + 6 * (pw_ofm_width * pw_ofm_width)] = sum6;
        ofms[base_index_in_ofms + 7 * (pw_ofm_width * pw_ofm_width)] = sum7;
#if TILE_F_H_W_WIDE == 16
        ofms[base_index_in_ofms + 8 * (pw_ofm_width * pw_ofm_width)] = sum8;
        ofms[base_index_in_ofms + 9 * (pw_ofm_width * pw_ofm_width)] = sum9;
        ofms[base_index_in_ofms + 10 * (pw_ofm_width * pw_ofm_width)] = sum10;
        ofms[base_index_in_ofms + 11 * (pw_ofm_width * pw_ofm_width)] = sum11;
        ofms[base_index_in_ofms + 12 * (pw_ofm_width * pw_ofm_width)] = sum12;
        ofms[base_index_in_ofms + 13 * (pw_ofm_width * pw_ofm_width)] = sum13;
        ofms[base_index_in_ofms + 14 * (pw_ofm_width * pw_ofm_width)] = sum14;
        ofms[base_index_in_ofms + 15 * (pw_ofm_width * pw_ofm_width)] = sum15;
#endif
    }
}

__global__ void dw_conv3x3_h_w_chw_wide(fms_dt *ifms, fms_dt *ofms, weights_dt *weights,
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

    const int thread_w = threadIdx.x;
    const int thread_h = threadIdx.y;
    const int thread_f = threadIdx.z;

    const int block_w = blockIdx.x;
    const int block_h = blockIdx.y;
    const int block_f = blockIdx.z;

    const int abs_dw_f_compact = block_f + thread_f;
    const int abs_dw_f = abs_dw_f_compact * PACKED_ITEMS;

    __shared__ weights_dt dw_filter_weights[FILTER_3x3_AREA];

    if (thread_h == 0 && thread_w < FILTER_3x3_DIM)
    {
        for (int c_h = 0; c_h < FILTER_3x3_DIM; c_h++)
        {
            dw_filter_weights[c_h * FILTER_3x3_DIM + thread_w] =
                weights[dw_layer_weights_offset + (c_h * FILTER_3x3_DIM + thread_w) + abs_dw_f_compact * FILTER_3x3_PADDED_AREA];
        }
    }

    __syncthreads();

    // for (int h = 0; h < rows_per_thread; h++)
    {
        const int abs_row_index = (block_h * parallel_h + thread_h) * strides - padding_top;

        if ((block_h * parallel_h + thread_h) < dw_ofm_height)
        {

            // for (int i_w = 0; i_w < tile_w; i_w++)
            {
                pss_dt sum = 0;
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
                        for (int c_w = 0; c_w < FILTER_3x3_DIM; c_w++)
                        {

                            fms_dt ifms_val = get_fms_val(ifms, abs_row_index + c_h,
                                                          abs_w_read + c_w, dw_ifm_height, dw_ifm_width,
                                                          base_index_in_ifms + c_h * dw_ifm_width + c_w,
                                                          dw_ifms_zp);

                            sum += dw_filter_weights[c_h * FILTER_3x3_DIM + c_w] * ifms_val;
                        }
                    }
                    if (sum < 0)
                    {
                        sum = 0;
                    }
                    else
                    {
                        sum = sum * DUMMY_SCALE + DUMMY_BIAS;
                    }
                    ofms[base_index_in_ofms] = sum;
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
    const int compact_layer_depth = (l_specs.layer_depth / PACKED_ITEMS);
    const int compact_layer_num_filters = (l_specs.layer_num_fils / PACKED_ITEMS);

    const int parallel_w = TILE_W_H_W_WIDE > ofms_width ? least_pow_of_2_geq(ofms_width) : TILE_W_H_W_WIDE; 
    const int parallel_h = TILE_H_H_W_WIDE > ofms_height ? least_pow_of_2_geq(ofms_height) : TILE_H_H_W_WIDE;

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
        printf("(conv_h_w_wide): layer %d\n", l_specs.layer_index);
    }

    const int padded_tile_w = least_pow_of_2_geq(l_specs.layer_ifm_width +
                                                 l_specs.padding_left + l_specs.padding_right);

    if (l_specs.conv_layer_type == DW_CONV)
    {
        uint8_t ifms_zp = (uint8_t)l_specs.layer_ifms_zero_point;

        dim3 threads(parallel_w, parallel_h, 1);
        dim3 blocks((ofms_width + parallel_w) / parallel_w, (ofms_height + parallel_h) / parallel_h, compact_layer_num_filters);

        dw_conv3x3_h_w_chw_wide<<<blocks, threads>>>(ifms, ofms, dw_weights, fused_scales, fused_zps,
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
                                                     l_specs.relu_threshold,
                                                     parallel_h,
                                                     parallel_w);
    }
    else if (l_specs.conv_layer_type == PW_CONV)
    {

        dim3 threads(parallel_w, parallel_h, 1);
        // switching order to improve scheduling
        dim3 blocks(compact_layer_num_filters / TILE_F_H_W_WIDE,
                    (ofms_width + parallel_w) / parallel_w, (ofms_height + parallel_h) / parallel_h);

        pw_conv_h_w_chw_wide<<<blocks, threads>>>(ifms, ofms, pw_weights, fused_scales, fused_zps,
                                                  compact_layer_depth, num_filters,
                                                  l_specs.layer_ifm_width,
                                                  l_specs.layer_ofm_width,
                                                  l_specs.layer_weights_offset / PACKED_ITEMS,
                                                  fused_params_offset[l_specs.layer_index],
                                                  l_specs.layer_ofms_zero_point,
                                                  l_specs.relu_threshold,
                                                  l_specs.layer_activation,
                                                  parallel_h,
                                                  parallel_w,
                                                  parallel_h * parallel_w);
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