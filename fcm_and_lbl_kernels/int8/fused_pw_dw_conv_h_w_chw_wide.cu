#include "../../headers/conv_kernels.h"

#if COMPILE_FUSED && (FUSION_MODE == ALL_MODES || FUSION_MODE == FUSED_H_W) && DATA_LAYOUT == CHW && DATA_TYPE == INT8_DTYPE

using namespace std;
namespace cg = cooperative_groups;

__global__ void pw_dw3x3_conv_h_w_wide(fms_dt *ifms, fms_dt *ofms, weights_dt *pw_weights,
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
                                       const fms_dt packed_ifm_zp,
                                       const scales_dt dw_relu_threshold,
                                       const int parallel_h,
                                       const int parallel_w,
                                       const int compact_layer_depth_to_parallel_hw_ratio)
{

    const int thread_w = threadIdx.x;
    const int thread_h = threadIdx.y;
    const int thread_f = threadIdx.z;

    const int block_f = blockIdx.x;
    const int block_h = blockIdx.y;
    const int block_w = blockIdx.z;

    const int rows_consumed_each_time = parallel_h - (FILTER_3x3_DIM - 1);
    const int cols_consumed_each_time = parallel_w - (FILTER_3x3_DIM - 1);
    const int rows_produced_each_time = rows_consumed_each_time / strides;
    const int cols_produced_each_time = cols_consumed_each_time / strides;
    const int strided_thread_hw = (thread_h * strides) * parallel_w + (thread_w * strides);

    const int pw_abs_w_write = block_w * cols_consumed_each_time + thread_w;
    const int pw_abs_h_write = block_h * rows_consumed_each_time + thread_h;
    const int pw_abs_w_read = pw_abs_w_write - padding_left;
    const int pw_abs_h_read = pw_abs_h_write - padding_top;
    const int dw_abs_w = block_w * cols_produced_each_time + thread_w;
    const int dw_abs_h = block_h * rows_produced_each_time + thread_h;
    int pw_outpus_starting_h = block_h == 0 ? padding_top : 0;
    int pw_outpus_starting_w = block_w == 0 ? padding_left : 0;

    const int parallel_hw = parallel_h * parallel_w;
    const int thread_hw = thread_h * parallel_h + thread_w;

    const int abs_dw_f_compact = block_f + thread_f;
    const int abs_dw_f = abs_dw_f_compact * PACKED_ITEMS;

    const int base_index_pw_weights = pw_layer_weights_offset + block_f * PACKED_ITEMS * compact_layer_depth;

    __shared__ weights_dt weights_tile[MAX_PW_COMPACT_DEPTH_FUSED * PACKED_ITEMS];
    __shared__ fms_dt ofms_ifms_tile[TILE_H_H_W * TILE_W_H_W];

    __shared__ weights_dt dw_filter_weights[FILTER_3x3_AREA * PACKED_ITEMS];

    scales_dt scale0, scale1, scale2, scale3;
    biases_dt fused_zp0, fused_zp1, fused_zp2, fused_zp3;
    pss_dt sum0, sum1, sum2, sum3;

    for (int i = 0; i < compact_layer_depth_to_parallel_hw_ratio; i++)
    {
        const int iter_offset = i * parallel_hw + thread_hw;
        if (iter_offset < compact_layer_depth) // TODO
        {
            weights_tile[iter_offset * PACKED_ITEMS] = pw_weights[base_index_pw_weights + iter_offset];
            weights_tile[iter_offset * PACKED_ITEMS + 1] = pw_weights[base_index_pw_weights + compact_layer_depth + iter_offset];
            weights_tile[iter_offset * PACKED_ITEMS + 2] = pw_weights[base_index_pw_weights + 2 * compact_layer_depth + iter_offset];
            weights_tile[iter_offset * PACKED_ITEMS + 3] = pw_weights[base_index_pw_weights + 3 * compact_layer_depth + iter_offset];
        }
    }
    __syncthreads();

    if (pw_abs_h_write < pw_outpus_starting_h ||
        pw_abs_w_write < pw_outpus_starting_w ||
        pw_abs_h_write >= pw_ofm_height + padding_top || pw_abs_w_write >= pw_ofm_width + padding_left)
    {
        ofms_ifms_tile[thread_hw] = packed_ifm_zp;
    }
    else
    {
        const int abs_w_write = block_w * parallel_w + thread_w;
        if (abs_w_write < pw_ofm_width)
        {
            if (thread_h == 0 && thread_w == 0)
            {
                for (int c_h = 0; c_h < FILTER_3x3_DIM; c_h++)
                {
                    weights_dt weight_val0 =
                        dw_weights[dw_layer_weights_offset + (c_h * FILTER_3x3_DIM) + abs_dw_f_compact * FILTER_3x3_PADDED_AREA];
                    weights_dt weight_val1 =
                        dw_weights[dw_layer_weights_offset + (c_h * FILTER_3x3_DIM + 1) + abs_dw_f_compact * FILTER_3x3_PADDED_AREA];
                    weights_dt weight_val2 =
                        dw_weights[dw_layer_weights_offset + (c_h * FILTER_3x3_DIM + 2) + abs_dw_f_compact * FILTER_3x3_PADDED_AREA];
                    for (int f = 0; f < PACKED_ITEMS; f++)
                    {
                        dw_filter_weights[f * FILTER_3x3_DIM + c_h] = PACK_32_8s(EXTRACT_8_32(weight_val0, f),
                                                                                 EXTRACT_8_32(weight_val1, f),
                                                                                 EXTRACT_8_32(weight_val2, f),
                                                                                 0);
                    }
                }
            }

            const int offet_in_tile = thread_h * parallel_w + thread_w;
            const int base_index_pw_scales = block_f * PACKED_ITEMS;

            scale0 = fused_scales[pw_layer_fused_params_offset + base_index_pw_scales];
            scale1 = fused_scales[pw_layer_fused_params_offset + base_index_pw_scales + 1];
            scale2 = fused_scales[pw_layer_fused_params_offset + base_index_pw_scales + 2];
            scale3 = fused_scales[pw_layer_fused_params_offset + base_index_pw_scales + 3];

            fused_zp0 = fused_zps[pw_layer_fused_params_offset + base_index_pw_scales];
            fused_zp1 = fused_zps[pw_layer_fused_params_offset + base_index_pw_scales + 1];
            fused_zp2 = fused_zps[pw_layer_fused_params_offset + base_index_pw_scales + 2];
            fused_zp3 = fused_zps[pw_layer_fused_params_offset + base_index_pw_scales + 3];

            sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            int base_index_in_ifms = pw_abs_h_read * pw_ifm_width + pw_abs_w_read;
            const int pw_ifms_hw = pw_ifm_width * pw_ifm_width; // TODO
            for (int d = 0; d < compact_layer_depth; d++)
            {

                const int d_offset = d * PACKED_ITEMS;

                fms_dt ifms_val = ifms[base_index_in_ifms + d * pw_ifms_hw];

                sum0 += __dp4a(ifms_val, weights_tile[d_offset], 0);
                sum1 += __dp4a(ifms_val, weights_tile[d_offset + 1], 0);
                sum2 += __dp4a(ifms_val, weights_tile[d_offset + 2], 0);
                sum3 += __dp4a(ifms_val, weights_tile[d_offset + 3], 0);

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
            ofms_ifms_tile[offet_in_tile] = PACK_32_8s(quant_relu6(sum0, scale0, fused_zp0, pw_ofms_zp, pw_relu_threshold),
                                                       quant_relu6(sum1, scale1, fused_zp1, pw_ofms_zp, pw_relu_threshold),
                                                       quant_relu6(sum2, scale2, fused_zp2, pw_ofms_zp, pw_relu_threshold),
                                                       quant_relu6(sum3, scale3, fused_zp3, pw_ofms_zp, pw_relu_threshold));
        }
    }
    //**********************************************************
    __syncthreads();

    // for (int h = 0; h < rows_per_thread; h++)
    {

        int base_index_in_ofms = abs_dw_f_compact * dw_ofm_height * dw_ofm_width +
                                 dw_abs_h * dw_ofm_width + dw_abs_w;

        if (thread_h < parallel_h / strides && thread_w < parallel_w / strides)
        {
            scale0 = fused_scales[dw_layer_fused_params_offset + abs_dw_f];
            scale1 = fused_scales[dw_layer_fused_params_offset + abs_dw_f + 1];
            scale2 = fused_scales[dw_layer_fused_params_offset + abs_dw_f + 2];
            scale3 = fused_scales[dw_layer_fused_params_offset + abs_dw_f + 3];

            fused_zp0 = fused_zps[dw_layer_fused_params_offset + abs_dw_f];
            fused_zp1 = fused_zps[dw_layer_fused_params_offset + abs_dw_f + 1];
            fused_zp2 = fused_zps[dw_layer_fused_params_offset + abs_dw_f + 2];
            fused_zp3 = fused_zps[dw_layer_fused_params_offset + abs_dw_f + 3];
            sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

            for (int c_h = 0; c_h < FILTER_3x3_DIM; c_h++)
            {

                weights_dt weight_val0 = dw_filter_weights[c_h];
                weights_dt weight_val1 = dw_filter_weights[FILTER_3x3_DIM + c_h];
                weights_dt weight_val2 = dw_filter_weights[FILTER_3x3_DIM * 2 + c_h];
                weights_dt weight_val3 = dw_filter_weights[FILTER_3x3_DIM * 3 + c_h];

                int filter_offset_in_tile = c_h * TILE_W_H_W_WIDE;
                fms_dt ifms_val0 = ofms_ifms_tile[strided_thread_hw + filter_offset_in_tile];
                filter_offset_in_tile++;
                fms_dt ifms_val1 = ofms_ifms_tile[strided_thread_hw + filter_offset_in_tile];
                filter_offset_in_tile++;
                fms_dt ifms_val2 = ofms_ifms_tile[strided_thread_hw + filter_offset_in_tile];

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

            // q0 = quant_relu6(sum0, scale0, fused_zp0, dw_ofms_zp, dw_relu_threshold);
            // q1 = quant_relu6(sum1, scale1, fused_zp1, dw_ofms_zp, dw_relu_threshold);
            // q2 = quant_relu6(sum2, scale2, fused_zp2, dw_ofms_zp, dw_relu_threshold);
            // q3 = quant_relu6(sum3, scale3, fused_zp3, dw_ofms_zp, dw_relu_threshold);

            // if (abs_row_index == 56 && abs_w_write == 5 && thread_f == 0)
            // {
            //     printf("\n%d\n", q0);
            // }

            // ofms_ifms_tile[row_offet_in_tile + thread_f * padded_tile_width + abs_w_write] = PACK_32_8s(q0, q1, q2, q3);
            ofms[base_index_in_ofms] = PACK_32_8s(quant_relu6(sum0, scale0, fused_zp0, dw_ofms_zp, dw_relu_threshold),
                                                  quant_relu6(sum1, scale1, fused_zp1, dw_ofms_zp, dw_relu_threshold),
                                                  quant_relu6(sum2, scale2, fused_zp2, dw_ofms_zp, dw_relu_threshold),
                                                  quant_relu6(sum3, scale3, fused_zp3, dw_ofms_zp, dw_relu_threshold));
        }
    }
}

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
                                             const int num_sms)
{

    const int num_filters = pw_l_specs.layer_num_fils;

    const int pw_ofms_width = pw_l_specs.layer_ofm_width;
    const int pw_ofms_height = pw_l_specs.layer_ofm_height;
    const int pw_compact_layer_depth = (pw_l_specs.layer_depth / PACKED_ITEMS);

    const int dw_ofms_width = dw_l_specs.layer_ofm_width;
    const int dw_ofms_height = dw_l_specs.layer_ofm_height;
    const int dw_compact_layer_depth = (dw_l_specs.layer_depth / PACKED_ITEMS);

    if (iteration == 0)
    {
        printf("%d, %d (FUSED_PWDW_wide):\n", pw_l_specs.layer_index, dw_l_specs.layer_index);
    }

    const int parallel_w = TILE_W_H_W > pw_ofms_width ? least_pow_of_2_geq(pw_ofms_width) : TILE_W_H_W;
    const int parallel_h = TILE_H_H_W > pw_ofms_height ? least_pow_of_2_geq(pw_ofms_height) : TILE_H_H_W;

    const int strides = dw_l_specs.strides;

    const int rows_produced_each_time = (parallel_h - (FILTER_3x3_DIM - strides)) / strides;
    const int cols_produced_each_time = (parallel_w - (FILTER_3x3_DIM - strides)) / strides;

    dim3 threads(parallel_w, parallel_h, 1);
    dim3 blocks(dw_compact_layer_depth,
                (dw_ofms_width + cols_produced_each_time) / cols_produced_each_time,
                (dw_ofms_height + rows_produced_each_time) / rows_produced_each_time);

    uint8_t ifms_zp = (uint8_t)dw_l_specs.layer_ifms_zero_point;
    uint8_t ifm_zps_to_pack[4] = {ifms_zp, ifms_zp, ifms_zp, ifms_zp};
    fms_dt packed_ifm_zp = PACK_32_8s(ifm_zps_to_pack);

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

    const int hw_parallelism = parallel_h * parallel_w;
    const int compact_layer_depth_to_parallel_hw_ratio = (pw_compact_layer_depth + hw_parallelism - 1) / hw_parallelism;

    pw_dw3x3_conv_h_w_wide<<<blocks, threads>>>(ifms, ofms, pw_weights, fused_scales, fused_zps,
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
                                                packed_ifm_zp,
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