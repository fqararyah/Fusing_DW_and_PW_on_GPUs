#include "../../headers/conv_kernels.h"

#if COMPILE_FUSED && (FUSION_MODE == ALL_MODES || FUSION_MODE == FUSED_F_W) && DATA_LAYOUT == CHW && DATA_TYPE == INT8_DTYPE

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

__global__ void dw_conv3x3_pw_conv_f_w(fms_dt *ifms, fms_dt *ofms, weights_dt *pw_weights,
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
                                       const fms_dt packed_ifm_zp,
                                       const scales_dt dw_relu_threshold,
                                       const int rows_produced_each_time,
                                       const int rows_per_thread,
                                       const int packed_ofms_per_thread_f)
{

    __shared__ fms_dt ofms_ifms_tile[1 * MAX_LAYER_DW / PACKED_ITEMS]; // TODO 1 is rows_produced_each_time

    const int thread_w = threadIdx.x;
    const int thread_h = threadIdx.y;
    const int thread_f = threadIdx.z;

    const int block_w = blockIdx.x;
    const int block_h = blockIdx.y;

    const int abs_dw_f = thread_f * PACKED_ITEMS;

    scales_dt scale0 = fused_scales[dw_layer_fused_params_offset + abs_dw_f];
    scales_dt scale1 = fused_scales[dw_layer_fused_params_offset + abs_dw_f + 1];
    scales_dt scale2 = fused_scales[dw_layer_fused_params_offset + abs_dw_f + 2];
    scales_dt scale3 = fused_scales[dw_layer_fused_params_offset + abs_dw_f + 3];

    biases_dt fused_zp0 = fused_zps[dw_layer_fused_params_offset + abs_dw_f];
    biases_dt fused_zp1 = fused_zps[dw_layer_fused_params_offset + abs_dw_f + 1];
    biases_dt fused_zp2 = fused_zps[dw_layer_fused_params_offset + abs_dw_f + 2];
    biases_dt fused_zp3 = fused_zps[dw_layer_fused_params_offset + abs_dw_f + 3];

    int8_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;

    weights_dt my_weight;
    if (thread_h < FILTER_3x3_DIM && thread_w < FILTER_3x3_DIM)
    {
        my_weight = dw_weights[dw_layer_weights_offset + (thread_h * FILTER_3x3_DIM + thread_w) + thread_f * FILTER_3x3_PADDED_AREA];
        // if (thread_f == 0 && block_h == 0 && block_w == 0)
        // {
        //     printf("%d, %d >>> %d\n", thread_h, thread_w, EXTRACT_8_32(my_weight, 0));
        // }
    }

    const int offset_f_w = thread_f * F_W_PARALLEL_W * F_W_TILE_H + thread_w;
    const int abs_w_write = block_w * F_W_PARALLEL_W + thread_w;

    if (abs_w_write < dw_ofm_width)
    {
        unsigned active_threads_mask = __activemask();

        weights_dt weight_row00, weight_row01, weight_row02,
            weight_row10, weight_row11, weight_row12,
            weight_row20, weight_row21, weight_row22,
            weight_row30, weight_row31, weight_row32;

        weights_dt weight_val0, weight_val1, weight_val2;

        weight_val0 = __shfl_sync(active_threads_mask, my_weight, 0);
        weight_val1 = __shfl_sync(active_threads_mask, my_weight, 1);
        weight_val2 = __shfl_sync(active_threads_mask, my_weight, 2);

        weight_row00 = PACK_32_8s(EXTRACT_8_32(weight_val0, 0), EXTRACT_8_32(weight_val1, 0), EXTRACT_8_32(weight_val2, 0), 0);
        weight_row10 = PACK_32_8s(EXTRACT_8_32(weight_val0, 1), EXTRACT_8_32(weight_val1, 1), EXTRACT_8_32(weight_val2, 1), 0);
        weight_row20 = PACK_32_8s(EXTRACT_8_32(weight_val0, 2), EXTRACT_8_32(weight_val1, 2), EXTRACT_8_32(weight_val2, 2), 0);
        weight_row30 = PACK_32_8s(EXTRACT_8_32(weight_val0, 3), EXTRACT_8_32(weight_val1, 3), EXTRACT_8_32(weight_val2, 3), 0);

        weight_val0 = __shfl_sync(active_threads_mask, my_weight, F_W_PARALLEL_W);
        weight_val1 = __shfl_sync(active_threads_mask, my_weight, F_W_PARALLEL_W + 1);
        weight_val2 = __shfl_sync(active_threads_mask, my_weight, F_W_PARALLEL_W + 2);

        weight_row01 = PACK_32_8s(EXTRACT_8_32(weight_val0, 0), EXTRACT_8_32(weight_val1, 0), EXTRACT_8_32(weight_val2, 0), 0);
        weight_row11 = PACK_32_8s(EXTRACT_8_32(weight_val0, 1), EXTRACT_8_32(weight_val1, 1), EXTRACT_8_32(weight_val2, 1), 0);
        weight_row21 = PACK_32_8s(EXTRACT_8_32(weight_val0, 2), EXTRACT_8_32(weight_val1, 2), EXTRACT_8_32(weight_val2, 2), 0);
        weight_row31 = PACK_32_8s(EXTRACT_8_32(weight_val0, 3), EXTRACT_8_32(weight_val1, 3), EXTRACT_8_32(weight_val2, 3), 0);

        weight_val0 = __shfl_sync(active_threads_mask, my_weight, 2 * F_W_PARALLEL_W);
        weight_val1 = __shfl_sync(active_threads_mask, my_weight, 2 * F_W_PARALLEL_W + 1);
        weight_val2 = __shfl_sync(active_threads_mask, my_weight, 2 * F_W_PARALLEL_W + 2);

        weight_row02 = PACK_32_8s(EXTRACT_8_32(weight_val0, 0), EXTRACT_8_32(weight_val1, 0), EXTRACT_8_32(weight_val2, 0), 0);
        weight_row12 = PACK_32_8s(EXTRACT_8_32(weight_val0, 1), EXTRACT_8_32(weight_val1, 1), EXTRACT_8_32(weight_val2, 1), 0);
        weight_row22 = PACK_32_8s(EXTRACT_8_32(weight_val0, 2), EXTRACT_8_32(weight_val1, 2), EXTRACT_8_32(weight_val2, 2), 0);
        weight_row32 = PACK_32_8s(EXTRACT_8_32(weight_val0, 3), EXTRACT_8_32(weight_val1, 3), EXTRACT_8_32(weight_val2, 3), 0);

        for (int h = 0; h < rows_per_thread; h++)
        {
            const int row_in_tile = (thread_h * rows_per_thread + h);
            const int row_offet_in_tile = row_in_tile * F_W_PARALLEL_W;
            const int abs_row_index = block_h * F_W_TILE_H + row_in_tile * strides - padding_top;
            if (block_h * rows_produced_each_time + row_in_tile < dw_ofm_height)
            {
                const int abs_w_read = (block_w * F_W_PARALLEL_W + thread_w) * strides - padding_left;
                const int base_index_in_ifms = abs_row_index * dw_ifm_width +
                                               thread_f * dw_ifm_height * dw_ifm_width +
                                               abs_w_read;

                // for (int i_w = 0; i_w < tile_w; i_w++)
                {

                    pss_dt sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

                    //*******************************************
                    fms_dt ifms_val0 = get_fms_val(ifms, abs_row_index, abs_w_read, dw_ifm_height, dw_ifm_width,
                                                   base_index_in_ifms,
                                                   packed_ifm_zp);
                    fms_dt ifms_val1 = get_fms_val(ifms, abs_row_index, abs_w_read + 1, dw_ifm_height, dw_ifm_width,
                                                   base_index_in_ifms + 1,
                                                   packed_ifm_zp);
                    fms_dt ifms_val2 = get_fms_val(ifms, abs_row_index, abs_w_read + 2, dw_ifm_height, dw_ifm_width,
                                                   base_index_in_ifms + 2,
                                                   packed_ifm_zp);

                    sum0 +=
                        __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 0), EXTRACT_8_32(ifms_val1, 0), EXTRACT_8_32(ifms_val2, 0), 0),
                               weight_row00, 0);
                    sum1 +=
                        __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 1), EXTRACT_8_32(ifms_val1, 1), EXTRACT_8_32(ifms_val2, 1), 0),
                               weight_row10, 0);
                    sum2 +=
                        __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 2), EXTRACT_8_32(ifms_val1, 2), EXTRACT_8_32(ifms_val2, 2), 0),
                               weight_row20, 0);
                    sum3 +=
                        __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 3), EXTRACT_8_32(ifms_val1, 3), EXTRACT_8_32(ifms_val2, 3), 0),
                               weight_row30, 0);
                    //*******************************************
                    ifms_val0 = get_fms_val(ifms, abs_row_index + 1, abs_w_read, dw_ifm_height, dw_ifm_width,
                                            base_index_in_ifms + dw_ifm_width,
                                            packed_ifm_zp);
                    ifms_val1 = get_fms_val(ifms, abs_row_index + 1, abs_w_read + 1, dw_ifm_height, dw_ifm_width,
                                            base_index_in_ifms + dw_ifm_width + 1,
                                            packed_ifm_zp);
                    ifms_val2 = get_fms_val(ifms, abs_row_index + 1, abs_w_read + 2, dw_ifm_height, dw_ifm_width,
                                            base_index_in_ifms + dw_ifm_width + 2,
                                            packed_ifm_zp);

                    sum0 +=
                        __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 0), EXTRACT_8_32(ifms_val1, 0), EXTRACT_8_32(ifms_val2, 0), 0),
                               weight_row01, 0);
                    sum1 +=
                        __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 1), EXTRACT_8_32(ifms_val1, 1), EXTRACT_8_32(ifms_val2, 1), 0),
                               weight_row11, 0);
                    sum2 +=
                        __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 2), EXTRACT_8_32(ifms_val1, 2), EXTRACT_8_32(ifms_val2, 2), 0),
                               weight_row21, 0);
                    sum3 +=
                        __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 3), EXTRACT_8_32(ifms_val1, 3), EXTRACT_8_32(ifms_val2, 3), 0),
                               weight_row31, 0);
                    //*******************************************
                    ifms_val0 = get_fms_val(ifms, abs_row_index + 2, abs_w_read, dw_ifm_height, dw_ifm_width,
                                            base_index_in_ifms + 2 * dw_ifm_width,
                                            packed_ifm_zp);
                    ifms_val1 = get_fms_val(ifms, abs_row_index + 2, abs_w_read + 1, dw_ifm_height, dw_ifm_width,
                                            base_index_in_ifms + 2 * dw_ifm_width + 1,
                                            packed_ifm_zp);
                    ifms_val2 = get_fms_val(ifms, abs_row_index + 2, abs_w_read + 2, dw_ifm_height, dw_ifm_width,
                                            base_index_in_ifms + 2 * dw_ifm_width + 2,
                                            packed_ifm_zp);

                    sum0 +=
                        __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 0), EXTRACT_8_32(ifms_val1, 0), EXTRACT_8_32(ifms_val2, 0), 0),
                               weight_row02, 0);
                    sum1 +=
                        __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 1), EXTRACT_8_32(ifms_val1, 1), EXTRACT_8_32(ifms_val2, 1), 0),
                               weight_row12, 0);
                    sum2 +=
                        __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 2), EXTRACT_8_32(ifms_val1, 2), EXTRACT_8_32(ifms_val2, 2), 0),
                               weight_row22, 0);
                    sum3 +=
                        __dp4a(PACK_32_8s(EXTRACT_8_32(ifms_val0, 3), EXTRACT_8_32(ifms_val1, 3), EXTRACT_8_32(ifms_val2, 3), 0),
                               weight_row32, 0);

                    //*******************************************

                    q0 = quant_relu6(sum0, scale0, fused_zp0, dw_ofms_zp, dw_relu_threshold);
                    q1 = quant_relu6(sum1, scale1, fused_zp1, dw_ofms_zp, dw_relu_threshold);
                    q2 = quant_relu6(sum2, scale2, fused_zp2, dw_ofms_zp, dw_relu_threshold);
                    q3 = quant_relu6(sum3, scale3, fused_zp3, dw_ofms_zp, dw_relu_threshold);

                    // if (abs_row_index == 56 && abs_w_write == 5 && thread_f == 0)
                    // {
                    //     printf("\n%d\n", q0);
                    // }

                    ofms_ifms_tile[row_offet_in_tile + offset_f_w] = PACK_32_8s(q0, q1, q2, q3);
                    // ofms[base_index_in_ofms + abs_w_write] = PACK_32_8s(q0, q1, q2, q3);
                }
            }
        }
    }

    __syncthreads();

    for (int o_f = 0; o_f < packed_ofms_per_thread_f; o_f++)
    {
        if (thread_f < pw_compact_layer_num_filters)
        {
            const int abs_f_compact = thread_f * packed_ofms_per_thread_f + o_f;
            const int base_index_pw_scales = abs_f_compact * PACKED_ITEMS;
            const int base_index_pw_weights = pw_layer_weights_offset + abs_f_compact * PACKED_ITEMS * pw_compact_layer_depth;

            scale0 = fused_scales[pw_layer_fused_params_offset + base_index_pw_scales];
            scale1 = fused_scales[pw_layer_fused_params_offset + base_index_pw_scales + 1];
            scale2 = fused_scales[pw_layer_fused_params_offset + base_index_pw_scales + 2];
            scale3 = fused_scales[pw_layer_fused_params_offset + base_index_pw_scales + 3];

            fused_zp0 = fused_zps[pw_layer_fused_params_offset + base_index_pw_scales];
            fused_zp1 = fused_zps[pw_layer_fused_params_offset + base_index_pw_scales + 1];
            fused_zp2 = fused_zps[pw_layer_fused_params_offset + base_index_pw_scales + 2];
            fused_zp3 = fused_zps[pw_layer_fused_params_offset + base_index_pw_scales + 3];

            for (int h = 0; h < rows_per_thread; h++)
            {
                const int abs_row_index_in_ofms = (block_h * rows_produced_each_time + (thread_h * rows_per_thread + h));
                const int row_offet_in_tile = (thread_h * rows_per_thread + h) * F_W_PARALLEL_W;
                const int base_index_in_ofms = abs_row_index_in_ofms * pw_ofm_width +
                                               abs_f_compact * pw_ofm_width * pw_ofm_height + abs_w_write;

                // for (int i_w = 0; i_w < tile_w; i_w++)
                if (abs_row_index_in_ofms < pw_ofm_height)
                {
                    const int abs_w_write = block_w * F_W_PARALLEL_W + thread_w;
                    if (abs_w_write < pw_ofm_width)
                    {
                        pss_dt sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                        int a = 0;
                        for (int d = 0; d < pw_compact_layer_depth; d++)
                        {
                            fms_dt fms_val = ofms_ifms_tile[row_offet_in_tile + d * F_W_TILE_H * F_W_PARALLEL_W + thread_w];

                            // if (thread_f == 0 && abs_w_write == 0 && block_h * rows_produced_each_time + h == 0)
                            // {
                            //     printf("%d * %d\n ", EXTRACT_8_32(pw_weights[base_index_pw_weights + d], 0), EXTRACT_8_32(fms_val, 0));
                            //     printf("%d * %d\n", EXTRACT_8_32(pw_weights[base_index_pw_weights + d], 1), EXTRACT_8_32(fms_val, 1));
                            //     printf("%d * %d\n", EXTRACT_8_32(pw_weights[base_index_pw_weights + d], 2), EXTRACT_8_32(fms_val, 2));
                            //     printf("%d * %d\n", EXTRACT_8_32(pw_weights[base_index_pw_weights + d], 3), EXTRACT_8_32(fms_val, 3));
                            // }
                            sum0 += __dp4a(fms_val, pw_weights[base_index_pw_weights + d], a);
                            sum1 += __dp4a(fms_val, pw_weights[base_index_pw_weights + pw_compact_layer_depth + d], a);
                            sum2 += __dp4a(fms_val, pw_weights[base_index_pw_weights + 2 * pw_compact_layer_depth + d], a);
                            sum3 += __dp4a(fms_val, pw_weights[base_index_pw_weights + 3 * pw_compact_layer_depth + d], a);
                        }
                        ofms[base_index_in_ofms] = PACK_32_8s(quant_relu6(sum0, scale0, fused_zp0, pw_ofms_zp, pw_relu_threshold),
                                                              quant_relu6(sum1, scale1, fused_zp1, pw_ofms_zp, pw_relu_threshold),
                                                              quant_relu6(sum2, scale2, fused_zp2, pw_ofms_zp, pw_relu_threshold),
                                                              quant_relu6(sum3, scale3, fused_zp3, pw_ofms_zp, pw_relu_threshold));
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

    const int pw_compact_layer_depth = (pw_l_specs.layer_depth >> 2);

    const int compact_layer_depth = dw_l_specs.layer_depth / PACKED_ITEMS;

    dim3 threads(F_W_PARALLEL_W, F_W_PARALLEL_H, compact_layer_depth);
    dim3 blocks((dw_l_specs.layer_ifm_width + F_W_TILE_W - 1) / F_W_TILE_W,
                (dw_l_specs.layer_ifm_height + F_W_TILE_H - 1) / F_W_TILE_H, 1);

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

    const int pw_compact_layer_num_filters = (pw_l_specs.layer_num_fils >> 2);

    const int packed_ofms_per_thread_f = pw_compact_layer_num_filters / pw_compact_layer_depth < 1
                                             ? 1
                                             : pw_compact_layer_num_filters / pw_compact_layer_depth;

    dw_conv3x3_pw_conv_f_w<<<blocks, threads>>>(ifms, ofms, pw_weights, fused_scales, fused_zps,
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
                                                packed_ifm_zp,
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