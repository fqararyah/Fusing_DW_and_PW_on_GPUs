#include "../../headers/conv_kernels.h"

#if COMPILE_FUSED && (FUSION_MODE == ALL_MODES || FUSION_MODE == NOT_FUSED) && DATA_TYPE == FLOAT_DTYPE

__global__ void pw_dw3x3_conv_h_w_chw(fms_dt *ifms, fms_dt *ofms, weights_dt *pw_weights,
                                  fused_scales_dt *fused_scales,
                                  biases_dt *fused_zps,
                                  const int compact_layer_depth,
                                  const int pw_num_filters,
                                  const int pw_ifm_width,
                                  const int pw_ofm_height,
                                  const int pw_ofm_width,
                                  const int pw_layer_weights_offset,
                                  const int pw_layer_fused_params_offset,
                                  const fms_dt pw_ofms_zp,
                                  const scales_dt pw_relu_threshold,
                                  weights_dt *dw_weights,
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
                                  const int parallel_w,
                                  const int compact_layer_depth_to_parallel_hw_ratio)
{

    const int thread_w = threadIdx.x;
    const int thread_h = threadIdx.y;
    const int thread_f = threadIdx.z;

    const int block_w = blockIdx.x;
    const int block_h = blockIdx.y;
    const int block_f = blockIdx.z;

    const int dw_ofms_hw = dw_ofm_height * dw_ofm_width;

    const int parallel_hw = parallel_h * parallel_w;

    __shared__ weights_dt dw_filter_weights[TILE_F_H_W][FILTER_3x3_PADDED_AREA];
    __shared__ weights_dt pw_weights_tile[TILE_F_H_W][256];
    __shared__ fms_dt ofms_ifms_tile[TILE_F_H_W * TILE_H_H_W * TILE_W_H_W];

    const int abs_f_compact = block_f * TILE_F_H_W + thread_f;
    int base_index_pw_weights = pw_layer_weights_offset + abs_f_compact * PACKED_ITEMS * compact_layer_depth;
    const int thread_hw = thread_h * parallel_h + thread_w;

    for (int o_f = 0; o_f < TILE_F_H_W; o_f++)
    {
        if (thread_h == 0 && thread_w < FILTER_3x3_DIM)
        {
            for (int c_h = 0; c_h < FILTER_3x3_DIM; c_h++)
            {
                dw_filter_weights[o_f][c_h * FILTER_3x3_DIM + thread_w] =
                    dw_weights[dw_layer_weights_offset + (c_h * FILTER_3x3_DIM + thread_w) +
                               (abs_f_compact + o_f) * FILTER_3x3_PADDED_AREA];
            }
        }
    }
    // ****************************************************************************************************
    pss_dt sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
#if TILE_F_H_W == 8
    pss_dt sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
#endif

    const int abs_w_write = block_w * parallel_w + thread_w;
    const int abs_w_read = block_w * parallel_w + thread_w; // TODO
    const int offet_in_tile_hw = thread_h * parallel_w + thread_w;

    for (int i = 0; i < compact_layer_depth_to_parallel_hw_ratio; i++)
    {
        const int iter_ifms_offset = i * parallel_hw;
        for (int o_f = 0; o_f < TILE_F_H_W; o_f++)
        {
            if (iter_ifms_offset + thread_hw < compact_layer_depth) // TODO
            {
                pw_weights_tile[o_f][thread_hw] = pw_weights[base_index_pw_weights + iter_ifms_offset +
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
                sum0 += ifms_val * pw_weights_tile[0][d];
                sum1 += ifms_val * pw_weights_tile[1][d];
                sum2 += ifms_val * pw_weights_tile[2][d];
                sum3 += ifms_val * pw_weights_tile[3][d];
#if TILE_F_H_W == 8
                sum4 += ifms_val * pw_weights_tile[4][d];
                sum5 += ifms_val * pw_weights_tile[5][d];
                sum6 += ifms_val * pw_weights_tile[6][d];
                sum7 += ifms_val * pw_weights_tile[7][d];
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
#if TILE_F_H_W == 8
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
#endif

        ofms_ifms_tile[offet_in_tile_hw] = sum0;
        ofms_ifms_tile[TILE_HW_H_W + offet_in_tile_hw] = sum1;
        ofms_ifms_tile[2 * TILE_HW_H_W + offet_in_tile_hw] = sum2;
        ofms_ifms_tile[3 * TILE_HW_H_W + offet_in_tile_hw] = sum3;
#if TILE_F_H_W == 8
        ofms_ifms_tile[4 * TILE_HW_H_W + offet_in_tile_hw] = sum4;
        ofms_ifms_tile[5 * TILE_HW_H_W + offet_in_tile_hw] = sum5;
        ofms_ifms_tile[6 * TILE_HW_H_W + offet_in_tile_hw] = sum6;
        ofms_ifms_tile[7 * TILE_HW_H_W + offet_in_tile_hw] = sum7;
#endif
    }

    __syncthreads();
    // ****************************************************************************************************
    {
        const int abs_row_index = (block_h * parallel_h + thread_h) * strides - padding_top;

        if ((block_h * parallel_h + thread_h) < dw_ofm_height)
        {
            const int abs_w_write = block_w * parallel_w + thread_w;
            const int w_read = thread_w * strides - padding_left;

            int base_index_in_ofms = abs_f_compact * dw_ofms_hw +
                                     (block_h * parallel_h + thread_h) * dw_ofm_width + abs_w_write;

            if (abs_w_write < dw_ofm_width)
            {
                const int row_index_in_tile = thread_h * strides - padding_top;
                int base_index_in_ifms_tile = row_index_in_tile * parallel_w + w_read;

                sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
#if TILE_F_H_W == 8
                sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
#endif
                for (int c_h = 0; c_h < FILTER_3x3_DIM; c_h++)
                {
                    for (int c_w = 0; c_w < FILTER_3x3_DIM; c_w++)
                    {

                        fms_dt ifms_val0, ifms_val1, ifms_val2, ifms_val3;
#if TILE_F_H_W == 8
                        fms_dt ifms_val4, ifms_val5, ifms_val6, ifms_val7;
#endif
                        get_fms_vals(ofms_ifms_tile, row_index_in_tile + c_h,
                                     w_read + c_w, dw_ifm_height, dw_ifm_width,
                                     base_index_in_ifms_tile + c_h * parallel_w + c_w,
                                     dw_ifms_zp,
                                     TILE_HW_H_W,
                                     ifms_val0,
                                     ifms_val1,
                                     ifms_val2,
                                     ifms_val3);
#if TILE_F_H_W == 8
                        get_fms_vals(ofms_ifms_tile, row_index_in_tile + c_h,
                                     w_read + c_w, dw_ifm_height, dw_ifm_width,
                                     base_index_in_ifms_tile + 4 * TILE_HW_H_W + c_h * parallel_w + c_w,
                                     dw_ifms_zp,
                                     TILE_HW_H_W,
                                     ifms_val4,
                                     ifms_val5,
                                     ifms_val6,
                                     ifms_val7);
#endif

                        sum0 += dw_filter_weights[0][c_h * FILTER_3x3_DIM + c_w] * ifms_val0;
                        sum1 += dw_filter_weights[1][c_h * FILTER_3x3_DIM + c_w] * ifms_val1;
                        sum2 += dw_filter_weights[2][c_h * FILTER_3x3_DIM + c_w] * ifms_val2;
                        sum3 += dw_filter_weights[3][c_h * FILTER_3x3_DIM + c_w] * ifms_val3;
#if TILE_F_H_W == 8
                        sum4 += dw_filter_weights[4][c_h * FILTER_3x3_DIM + c_w] * ifms_val4;
                        sum5 += dw_filter_weights[5][c_h * FILTER_3x3_DIM + c_w] * ifms_val5;
                        sum6 += dw_filter_weights[6][c_h * FILTER_3x3_DIM + c_w] * ifms_val6;
                        sum7 += dw_filter_weights[7][c_h * FILTER_3x3_DIM + c_w] * ifms_val7;
#endif
                    }
                }
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
#if TILE_F_H_W == 8
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
#endif
                ofms[base_index_in_ofms] = sum0;
                ofms[base_index_in_ofms + dw_ofms_hw] = sum1;
                ofms[base_index_in_ofms + 2 * dw_ofms_hw] = sum2;
                ofms[base_index_in_ofms + 3 * dw_ofms_hw] = sum3;
#if TILE_F_H_W == 8
                ofms[base_index_in_ofms + 4 * dw_ofms_hw] = sum4;
                ofms[base_index_in_ofms + 5 * dw_ofms_hw] = sum5;
                ofms[base_index_in_ofms + 6 * dw_ofms_hw] = sum6;
                ofms[base_index_in_ofms + 7 * dw_ofms_hw] = sum7;
#endif
                //}
            }
        }
    }
}

void fused_pw_dw_convolutionGPU_h_w_chw(fms_dt *ifms, fms_dt *ofms,
                                    weights_dt *pw_weights,
                                    weights_dt *dw_weights,
                                    fused_scales_dt *fused_scales,
                                    biases_dt *fused_zps,
                                    layer_specs pw_l_specs,
                                    layer_specs dw_l_specs,
                                    int *fused_params_offsets,
                                    const int iteration,
                                    float &exec_time,
                                    const int num_sms)
{

    const int num_filters = pw_l_specs.layer_num_fils;

    const int pw_ofms_width = pw_l_specs.layer_ofm_width;
    const int pw_ofms_height = pw_l_specs.layer_ofm_height;
    const int pw_compact_layer_depth = pw_l_specs.layer_depth / PACKED_ITEMS;

    const int dw_ofms_width = dw_l_specs.layer_ofm_width;
    const int dw_ofms_height = dw_l_specs.layer_ofm_height;
    const int dw_compact_layer_depth = dw_l_specs.layer_depth / PACKED_ITEMS;

    if (iteration == 0)
    {
        printf("%d, %d (FUSED_PWDW):\n", pw_l_specs.layer_index, dw_l_specs.layer_index);
    }

    const int compact_layer_depth = dw_l_specs.layer_depth / PACKED_ITEMS;

    const int parallel_w = TILE_W_H_W > pw_ofms_width ? least_pow_of_2_geq(pw_ofms_width) : TILE_W_H_W;
    const int parallel_h = TILE_H_H_W > pw_ofms_height  ? least_pow_of_2_geq(pw_ofms_height) : TILE_H_H_W;

    dim3 threads(parallel_w, parallel_h, 1);
    dim3 blocks((dw_ofms_width + parallel_w - 1) / parallel_w,
                (dw_ofms_height + parallel_h - 1) / parallel_h, dw_compact_layer_depth / TILE_F_H_W);

    uint8_t ifms_zp = (uint8_t)dw_l_specs.layer_ifms_zero_point;

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

    const int padded_tile_w = least_pow_of_2_geq(dw_l_specs.layer_ifm_width +
                                                 dw_l_specs.padding_left + dw_l_specs.padding_right);

    const int parallelism_hw = parallel_h * parallel_w;
    const int compact_layer_depth_to_parallel_hw_ratio = (pw_compact_layer_depth + parallelism_hw - 1) / parallelism_hw;

    pw_dw3x3_conv_h_w_chw<<<blocks, threads>>>(ifms, ofms, pw_weights, fused_scales, fused_zps,
                                           pw_compact_layer_depth, num_filters,
                                           pw_l_specs.layer_ifm_width,
                                           pw_l_specs.layer_ofm_height,
                                           pw_l_specs.layer_ofm_width,
                                           pw_l_specs.layer_weights_offset / PACKED_ITEMS,
                                           fused_params_offsets[pw_l_specs.layer_index],
                                           pw_l_specs.layer_ofms_zero_point,
                                           pw_l_specs.relu_threshold,
                                           //*******************
                                           dw_weights,
                                           dw_l_specs.layer_depth,
                                           dw_l_specs.layer_ifm_height,
                                           dw_l_specs.layer_ifm_width,
                                           dw_l_specs.layer_ofm_height,
                                           dw_l_specs.layer_ofm_width,
                                           dw_l_specs.strides,
                                           dw_l_specs.padding_top,
                                           dw_l_specs.padding_bottom,
                                           dw_l_specs.padding_left,
                                           dw_l_specs.padding_right,
                                           padded_tile_w,
                                           dw_l_specs.layer_weights_offset / PACKED_ITEMS,
                                           fused_params_offsets[dw_l_specs.layer_index],
                                           dw_l_specs.layer_ifms_zero_point,
                                           dw_l_specs.layer_ofms_zero_point,
                                           dw_l_specs.relu_threshold,
                                           parallel_h, parallel_w,
                                           compact_layer_depth_to_parallel_hw_ratio);

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