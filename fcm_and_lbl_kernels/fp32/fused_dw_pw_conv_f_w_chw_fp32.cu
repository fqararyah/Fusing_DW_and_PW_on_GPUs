#include "../../headers/conv_kernels.h"

#if COMPILE_FUSED && (FUSION_MODE == ALL_MODES || FUSION_MODE == FUSED_F_W) && DATA_TYPE == FLOAT_DTYPE

using namespace std;
namespace cg = cooperative_groups;


__global__ void dw_conv3x3_pw_conv_f_w_chw(fms_dt *ifms, fms_dt *ofms, weights_dt *pw_weights,
                                       fused_scales_dt *fused_scales,
                                       biases_dt *fused_zps,
                                       const int pw_compact_layer_depth,
                                       const int pw_compact_layer_num_filters,
                                       const int pw_ifm_width,
                                       const int pw_ofm_height,
                                       const int pw_ofm_width,
                                       const int pw_layer_weights_offset,
                                       const int pw_layer_fused_params_offset,
                                       const fms_dt pw_ofms_zp,
                                       const scales_dt pw_relu_threshold,
                                       weights_dt *dw_weights,
                                       const int compact_dw_depth,
                                       const int dw_ifm_height,
                                       const int dw_ifm_width,
                                       const int dw_ifms_width_depth,
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
                                       const int rows_produced_each_time,
                                       const int rows_per_thread,
                                       const int packed_ofms_per_thread_f)
{

    __shared__ fms_dt ofms_ifms_tile[2 * F_W_TILE_F * 1024]; // TODO 2 is rows_per_thread

    const int thread_w = threadIdx.x;
    const int thread_h = threadIdx.y;
    const int thread_f = threadIdx.z;

    const int block_w = blockIdx.x;
    const int block_h = blockIdx.y;

    const int abs_f_index = thread_f * PACKED_ITEMS * F_W_TILE_F;

    const int abs_w_write = block_w * F_W_PARALLEL_W + thread_w;

    if (abs_w_write < dw_ofm_width)
    {
        weights_dt dw_weight00, dw_weight01, dw_weight02,
            dw_weight10, dw_weight11, dw_weight12,
            dw_weight20, dw_weight21, dw_weight22;
        for (int o_f = 0; o_f < F_W_TILE_F; o_f++)
        {
            unsigned active_threads_mask = __activemask();

            const int offset_f_w = (abs_f_index + o_f) * F_W_PARALLEL_W * F_W_TILE_H + thread_w;
            weights_dt my_weight;
            if (thread_h < FILTER_3x3_DIM && thread_w < FILTER_3x3_DIM)
            {
                my_weight = dw_weights[dw_layer_weights_offset +
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
                const int row_in_tile = (thread_h * rows_per_thread + h);
                const int abs_h_index_ofms = block_h * rows_produced_each_time + row_in_tile;
                const int row_offet_in_tile = row_in_tile * F_W_PARALLEL_W;
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

                        ofms_ifms_tile[row_offet_in_tile + offset_f_w] = sum;
                        // ofms[base_index_in_ofms + abs_w_write] = PACK_32_8s(q0, q1, q2, q3);
                    }
                }
            }
        }
    }

    __syncthreads();

    const int pw_ofms_hw = pw_ofm_height * pw_ofm_width;

    for (int o_f = 0; o_f < packed_ofms_per_thread_f; o_f++)
    {
        const int current_abs_f_index = abs_f_index * packed_ofms_per_thread_f + o_f * F_W_TILE_F;
        if (current_abs_f_index < pw_compact_layer_num_filters)
        {
            const int base_index_pw_weights = pw_layer_weights_offset +
                                              current_abs_f_index *
                                                  pw_compact_layer_depth;

            for (int h = 0; h < rows_per_thread; h++)
            {
                const int abs_row_index_in_ofms = (block_h * rows_produced_each_time + (thread_h * rows_per_thread + h));

                // for (int i_w = 0; i_w < tile_w; i_w++)const int rows_per_thread = rows_produced_each_time / F_W_PARALLEL_H;
                if (abs_row_index_in_ofms < pw_ofm_height)
                {
                    const int row_offet_in_tile = (thread_h * rows_per_thread + h) * F_W_PARALLEL_W;
                    const int base_index_in_ofms = abs_row_index_in_ofms * pw_ofm_width +
                                                   current_abs_f_index * pw_ofms_hw +
                                                   abs_w_write;
                    const int abs_w_write = block_w * F_W_PARALLEL_W + thread_w;
                    if (abs_w_write < pw_ofm_width)
                    {
                        pss_dt sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                        for (int d = 0; d < pw_compact_layer_depth; d++)
                        {
                            fms_dt fms_val = ofms_ifms_tile[row_offet_in_tile +
                                                            d * F_W_TILE_H * F_W_PARALLEL_W +
                                                            thread_w];
                            sum0 += fms_val * pw_weights[base_index_pw_weights + d];
                            sum1 += fms_val * pw_weights[base_index_pw_weights + pw_compact_layer_depth + d];
                            sum2 += fms_val * pw_weights[base_index_pw_weights + 2 * pw_compact_layer_depth + d];
                            sum3 += fms_val * pw_weights[base_index_pw_weights + 3 * pw_compact_layer_depth + d];
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
                        ofms[base_index_in_ofms] = sum0;
                        ofms[base_index_in_ofms + pw_ofms_hw] = sum1;
                        ofms[base_index_in_ofms + 2 * pw_ofms_hw] = sum2;
                        ofms[base_index_in_ofms + 3 * pw_ofms_hw] = sum3;
                    }
                }
            }
        }
    }
}

void fused_dwpw_convolutionGPU_chw(fms_dt *ifms, fms_dt *ofms,
                               weights_dt *pw_weights,
                               weights_dt *dw_weights,
                               fused_scales_dt *fused_scales,
                               biases_dt *fused_zps,
                               layer_specs dw_l_specs,
                               layer_specs pw_l_specs,
                               int *fused_params_offsets,
                               const int iteration,
                               float &exec_time)
{

    const int pw_ofms_width = pw_l_specs.layer_ofm_width;
    const int pw_ofms_height = pw_l_specs.layer_ofm_height;
    const int pw_compact_layer_depth = (pw_l_specs.layer_depth / PACKED_ITEMS);

    const int dw_ofms_width = dw_l_specs.layer_ofm_width;
    const int dw_ofms_height = dw_l_specs.layer_ofm_height;
    const int dw_compact_layer_depth = (dw_l_specs.layer_depth / PACKED_ITEMS);

    const int rows_consumed_each_time = (TILE_H_FW - (dw_l_specs.filter_size - dw_l_specs.strides));
    const int compact_layer_depth = dw_l_specs.layer_depth / PACKED_ITEMS;

    dim3 threads(F_W_PARALLEL_W, F_W_PARALLEL_H, compact_layer_depth / F_W_TILE_F);
    dim3 blocks((dw_l_specs.layer_ofm_width + F_W_TILE_W - 1) / F_W_TILE_W,
                (dw_l_specs.layer_ifm_height + F_W_TILE_H - 1) / F_W_TILE_H, 1);

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
    if (iteration == 0)
    {
        printf("%d, %d (FUSED_DWPW):\n", pw_l_specs.layer_index, dw_l_specs.layer_index);
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

    const int rows_produced_each_time = F_W_TILE_H / dw_l_specs.strides;
    const int rows_per_thread = rows_produced_each_time / F_W_PARALLEL_H;
    const int dw_ifm_width = dw_l_specs.layer_ifm_width;

    const int compact_dw_depth = dw_l_specs.layer_depth / PACKED_ITEMS;
    const int dw_ifms_width_depth = compact_dw_depth * dw_ifm_width;

    const int pw_compact_layer_num_filters = (pw_l_specs.layer_num_fils / PACKED_ITEMS);

    const int packed_ofms_per_thread_f = pw_compact_layer_num_filters / pw_compact_layer_depth < 1
                                             ? 1
                                             : pw_compact_layer_num_filters / pw_compact_layer_depth;

    dw_conv3x3_pw_conv_f_w_chw<<<blocks, threads>>>(ifms, ofms, pw_weights, fused_scales, fused_zps,
                                                pw_compact_layer_depth,
                                                pw_compact_layer_num_filters,
                                                pw_l_specs.layer_ifm_width,
                                                pw_l_specs.layer_ofm_height,
                                                pw_l_specs.layer_ofm_width,
                                                pw_l_specs.layer_weights_offset / PACKED_ITEMS,
                                                fused_params_offsets[pw_l_specs.layer_index],
                                                pw_l_specs.layer_ofms_zero_point,
                                                pw_l_specs.relu_threshold,
                                                //*******************
                                                dw_weights,
                                                compact_dw_depth,
                                                dw_l_specs.layer_ifm_height,
                                                dw_ifm_width,
                                                dw_ifms_width_depth,
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
                                                rows_produced_each_time,
                                                rows_per_thread,
                                                packed_ofms_per_thread_f);

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