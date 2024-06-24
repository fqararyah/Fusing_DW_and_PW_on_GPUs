#include "../../headers/conv_kernels.h"

#if (FUSION_MODE == ALL_MODES || FUSION_MODE == FUSED_F_W) && DATA_TYPE == FLOAT_DTYPE

__global__ void dw_conv3x3_f_w_chw(fms_dt *ifms, fms_dt *ofms, weights_dt *weights,
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
                               const scales_dt dw_relu_threshold)
{

    const int compact_layer_num_filters_and_depth = (dw_ifm_depth / PACKED_ITEMS);
    const int dw_filter_area = FILTER_3x3_DIM * FILTER_3x3_DIM;

    const int rows_consumed_each_time = F_W_TILE_H;
    const int rows_produced_each_time = rows_consumed_each_time / strides;

    const int thread_w = threadIdx.x;
    const int thread_h = threadIdx.y;
    const int thread_f = threadIdx.z;

    const int block_w = blockIdx.x;
    const int block_h = blockIdx.y;

    const int abs_f_index = thread_f * PACKED_ITEMS * F_W_TILE_F;

    const int dw_ifms_width_depth = compact_layer_num_filters_and_depth * dw_ifm_width;

    const int rows_per_thread = rows_produced_each_time / F_W_PARALLEL_H;
    const int dw_ofms_hw = dw_ofm_height * dw_ofm_width;
    weights_dt dw_weight00, dw_weight01, dw_weight02,
        dw_weight10, dw_weight11, dw_weight12,
        dw_weight20, dw_weight21, dw_weight22;
    for (int o_f = 0; o_f < F_W_TILE_F; o_f++)
    {
        unsigned active_threads_mask = __activemask();

        weights_dt my_weight;
        if (thread_h < FILTER_3x3_DIM && thread_w < FILTER_3x3_DIM)
        {
            my_weight = weights[dw_layer_weights_offset +
                                (abs_f_index + o_f) * FILTER_3x3_PADDED_AREA +
                                (thread_h * FILTER_3x3_DIM + thread_w)];
        }

        dw_weight00 = __shfl_sync(active_threads_mask, my_weight, 0);
        dw_weight01 = __shfl_sync(active_threads_mask, my_weight, 1);
        dw_weight02 = __shfl_sync(active_threads_mask, my_weight, 2);
        dw_weight10 = __shfl_sync(active_threads_mask, my_weight, F_W_PARALLEL_W);
        dw_weight11 = __shfl_sync(active_threads_mask, my_weight, F_W_PARALLEL_W + 1);
        dw_weight12 = __shfl_sync(active_threads_mask, my_weight, F_W_PARALLEL_W + 2);
        dw_weight20 = __shfl_sync(active_threads_mask, my_weight, 2 * F_W_PARALLEL_W);
        dw_weight21 = __shfl_sync(active_threads_mask, my_weight, 2 * F_W_PARALLEL_W + 1);
        dw_weight22 = __shfl_sync(active_threads_mask, my_weight, 2 * F_W_PARALLEL_W + 2);

        for (int h = 0; h < rows_per_thread; h++)
        {
            const int abs_w_write = block_w * F_W_PARALLEL_W + thread_w;
            const int row_in_tile = (thread_h * rows_per_thread + h);
            const int abs_h_index_ofms = block_h * rows_produced_each_time + row_in_tile;
            const int base_index_in_ofms = abs_h_index_ofms *
                                               dw_ofm_width +
                                           (abs_f_index + o_f) * dw_ofm_height * dw_ofm_width;
            int abs_h_index_ifms = block_h * F_W_TILE_H + row_in_tile * strides - padding_top;
            if (abs_h_index_ofms < dw_ofm_height)
            {
                const int abs_w_read = (block_w * F_W_PARALLEL_W + thread_w) * strides - padding_left;
                int base_index_in_ifms = abs_h_index_ifms * dw_ifm_width +
                                         (abs_f_index + o_f) * dw_ifm_height * dw_ifm_width +
                                         abs_w_read;

                // for (int i_w = 0; i_w < tile_w; i_w++)
                {

                    pss_dt sum;

                    //*******************************************
                    fms_dt ifms_val00 = get_fms_val(ifms, abs_h_index_ifms, abs_w_read, dw_ifm_height, dw_ifm_width,
                                                    base_index_in_ifms,
                                                    dw_ifms_zp);
                    fms_dt ifms_val01 = get_fms_val(ifms, abs_h_index_ifms, abs_w_read + 1, dw_ifm_height, dw_ifm_width,
                                                    base_index_in_ifms + 1,
                                                    dw_ifms_zp);
                    fms_dt ifms_val02 = get_fms_val(ifms, abs_h_index_ifms, abs_w_read + 2, dw_ifm_height, dw_ifm_width,
                                                    base_index_in_ifms + 2,
                                                    dw_ifms_zp);
                    base_index_in_ifms += dw_ifm_width;
                    abs_h_index_ifms++;
                    fms_dt ifms_val10 = get_fms_val(ifms, abs_h_index_ifms, abs_w_read, dw_ifm_height, dw_ifm_width,
                                                    base_index_in_ifms,
                                                    dw_ifms_zp);
                    fms_dt ifms_val11 = get_fms_val(ifms, abs_h_index_ifms, abs_w_read + 1, dw_ifm_height, dw_ifm_width,
                                                    base_index_in_ifms + 1,
                                                    dw_ifms_zp);
                    fms_dt ifms_val12 = get_fms_val(ifms, abs_h_index_ifms, abs_w_read + 2, dw_ifm_height, dw_ifm_width,
                                                    base_index_in_ifms + 2,
                                                    dw_ifms_zp);
                    base_index_in_ifms += dw_ifm_width;
                    abs_h_index_ifms++;
                    fms_dt ifms_val20 = get_fms_val(ifms, abs_h_index_ifms, abs_w_read, dw_ifm_height, dw_ifm_width,
                                                    base_index_in_ifms,
                                                    dw_ifms_zp);
                    fms_dt ifms_val21 = get_fms_val(ifms, abs_h_index_ifms, abs_w_read + 1, dw_ifm_height, dw_ifm_width,
                                                    base_index_in_ifms + 1,
                                                    dw_ifms_zp);
                    fms_dt ifms_val22 = get_fms_val(ifms, abs_h_index_ifms, abs_w_read + 2, dw_ifm_height, dw_ifm_width,
                                                    base_index_in_ifms + 2,
                                                    dw_ifms_zp);

                    sum = dw_weight00 * ifms_val00 +
                          dw_weight01 * ifms_val01 +
                          dw_weight02 * ifms_val02 +
                          dw_weight10 * ifms_val10 +
                          dw_weight11 * ifms_val11 +
                          dw_weight12 * ifms_val12 +
                          dw_weight20 * ifms_val20 +
                          dw_weight21 * ifms_val21 +
                          dw_weight22 * ifms_val22;

                    // if (abs_f_index + o_f == 19 && abs_h_index == 30 && abs_w_write == 84)
                    // {
                    //     printf("%f * %f + ", dw_weight00, ifms_val00);
                    //     printf("%f * %f + ", dw_weight01, ifms_val01);
                    //     printf("%f * %f", dw_weight02, ifms_val02);
                    //     printf("\n");
                    //     printf("%f * %f + ", dw_weight10, ifms_val10);
                    //     printf("%f * %f + ", dw_weight11, ifms_val11);
                    //     printf("%f * %f", dw_weight12, ifms_val12);
                    //     printf("\n");
                    //     printf("%f * %f + ", dw_weight20, ifms_val20);
                    //     printf("%f * %f + ", dw_weight21, ifms_val21);
                    //     printf("%f * %f", dw_weight22, ifms_val22);
                    //     printf("\nsum: %f\n", sum);
                    //     printf("\n");
                    // }

                    if (sum < 0)
                    {
                        sum = 0;
                    }
                    else
                    {
                        sum = sum * DUMMY_SCALE + DUMMY_BIAS;
                    }

                    ofms[base_index_in_ofms + abs_w_write] = sum;
                }
            }
        }
    }
}

__global__ void pw_conv_f_w_chw(fms_dt *ifms, fms_dt *ofms, weights_dt *weights,
                            fused_scales_dt *fused_scales,
                            biases_dt *fused_zps,
                            const int compact_layer_depth,
                            const int num_filters,
                            const int ifm_height,
                            const int ofm_height,
                            const int ifm_width,
                            const int ofm_width,
                            const int layer_weights_offset,
                            const int layer_fused_params_offset,
                            const fms_dt ofms_zp,
                            const scales_dt relu_threshold,
                            const int layer_activation)
{

    const int compact_layer_num_filters = (num_filters / PACKED_ITEMS);
    const int strides = 1; // TODO

    __shared__ fms_dt ifms_tile[2 * F_W_TILE_F * 1024]; // TODO 1 is rows_produced_each_time and 4 is tile_w

    const int thread_w = threadIdx.x;
    const int thread_h = threadIdx.y;
    const int thread_f = threadIdx.z;

    const int block_w = blockIdx.x;
    const int block_h = blockIdx.y;

    const int dw_ifms_width_depth = compact_layer_depth * ifm_width;

    const int abs_f_index = thread_f * PACKED_ITEMS * F_W_TILE_F;
    const int rows_per_thread = F_W_TILE_H / F_W_PARALLEL_H;

    for (int h = 0; h < rows_per_thread; h++)
    {
        const int abs_w_read = block_w * F_W_PARALLEL_W + thread_w;
        const int abs_row_index = (block_h * F_W_TILE_H + (thread_h * rows_per_thread + h));
        for (int d = 0; d < compact_layer_depth; d++)
        {
            ifms_tile[d * F_W_TILE_H * F_W_TILE_W +
                      (thread_h * rows_per_thread + h) * F_W_TILE_W + thread_w] = ifms[abs_row_index * ifm_width +
                                                                                       d * ifm_height * ifm_width +
                                                                                       abs_w_read];
        }
    }

    {
        const int abs_w_write = block_w * F_W_PARALLEL_W + thread_w;

        const int base_index_weights = layer_weights_offset +
                                       abs_f_index * compact_layer_depth;

        for (int h = 0; h < rows_per_thread; h++)
        {
            const int abs_row_index = (block_h * F_W_TILE_H + (thread_h * rows_per_thread + h));
            const int row_offet_in_tile = (thread_h * rows_per_thread + h) * F_W_PARALLEL_W;
            const int base_index_in_ofms = abs_row_index * ofm_width +
                                           abs_f_index * ofm_width * ifm_height + abs_w_write;

            // for (int i_w = 0; i_w < tile_w; i_w++)
            if (abs_row_index < ofm_height)
            {
                const int abs_w_write = block_w * F_W_PARALLEL_W + thread_w;
                if (abs_w_write < ofm_width)
                {
                    pss_dt sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                    for (int d = 0; d < compact_layer_depth; d++)
                    {
                        fms_dt fms_val = ifms_tile[d * F_W_TILE_H * F_W_TILE_W +
                                                   (thread_h * rows_per_thread + h) * F_W_TILE_W + thread_w];
                        sum0 += fms_val * weights[base_index_weights + d];
                        sum1 += fms_val * weights[base_index_weights + compact_layer_depth + d];
                        sum2 += fms_val * weights[base_index_weights + 2 * compact_layer_depth + d];
                        sum3 += fms_val * weights[base_index_weights + 3 * compact_layer_depth + d];
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

                    const int ofms_hw = ofm_height * ofm_width;
                    ofms[base_index_in_ofms] = sum0;
                    ofms[base_index_in_ofms + ofms_hw] = sum1;
                    ofms[base_index_in_ofms + 2 * ofms_hw] = sum2;
                    ofms[base_index_in_ofms + 3 * ofms_hw] = sum3;
                }
            }
        }
    }
}

void convolutionGPU_f_w_chw(fms_dt *ifms, fms_dt *ofms,
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
        printf("(conv_f_w): : layer %d\n", l_specs.layer_index);
    }

    const int padded_tile_w = least_pow_of_2_geq(l_specs.layer_ifm_width +
                                                 l_specs.padding_left + l_specs.padding_right);

    dim3 threads(F_W_PARALLEL_W, F_W_PARALLEL_H, compact_layer_num_filters / F_W_TILE_F);
    dim3 blocks((l_specs.layer_ofm_width + F_W_TILE_W - 1) / F_W_TILE_W,
                (l_specs.layer_ifm_height + F_W_TILE_H - 1) / F_W_TILE_H, 1);

    if (l_specs.conv_layer_type == DW_CONV)
    {
        dw_conv3x3_f_w_chw<<<blocks, threads>>>(ifms, ofms, dw_weights, fused_scales, fused_zps,
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
                                            l_specs.relu_threshold);
    }
    else if (l_specs.conv_layer_type == PW_CONV)
    {
        // dim3 threads(16, 1, compact_layer_depth);
        // dim3 blocks(1, l_specs.layer_ifm_height / l_specs.strides, 1);
        pw_conv_f_w_chw<<<blocks, threads>>>(ifms, ofms, pw_weights, fused_scales, fused_zps,
                                         compact_layer_depth, num_filters,
                                         l_specs.layer_ifm_height,
                                         l_specs.layer_ofm_height,
                                         l_specs.layer_ifm_width,
                                         l_specs.layer_ofm_width,
                                         l_specs.layer_weights_offset / PACKED_ITEMS,
                                         fused_params_offset[l_specs.layer_index],
                                         l_specs.layer_ofms_zero_point,
                                         l_specs.relu_threshold,
                                         l_specs.layer_activation);
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