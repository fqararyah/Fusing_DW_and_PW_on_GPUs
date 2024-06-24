#include "../../headers/conv_kernels.h"

#if COMPILE_FUSED && (FUSION_MODE == ALL_MODES || FUSION_MODE == FUSED_F_W) && DATA_TYPE == FLOAT_DTYPE

using namespace std;
namespace cg = cooperative_groups;

__global__ void pw_pw_conv_f_w_chw(fms_dt *ifms, fms_dt *ofms, weights_dt *pw_weights,
                               fused_scales_dt *fused_scales,
                               biases_dt *fused_zps,
                               const int pw_1_compact_layer_depth,
                               const int pw_1_compact_layer_num_filters,
                               const int pw_1_ifm_width,
                               const int pw_1_compact_ifm_width_depth,
                               const int pw_1_ofm_height,
                               const int pw_1_ofm_width,
                               const int pw_1_compact_ofm_width_depth,
                               const int pw_1_layer_weights_offset,
                               const int pw_1_layer_fused_params_offset,
                               const fms_dt pw_1_ofms_zp,
                               const scales_dt pw_1_relu_threshold,
                               const int pw_1_layer_activation,
                               const int pw_2_compact_layer_depth,
                               const int pw_2_compact_layer_num_filters,
                               const int pw_2_ifm_width,
                               const int pw_2_compact_ifm_width_depth,
                               const int pw_2_ofm_height,
                               const int pw_2_ofm_width,
                               const int pw_2_compact_ofm_width_depth,
                               const int pw_2_filters_to_parallelism_f_ratio,
                               const int pw_2_layer_weights_offset,
                               const int pw_2_layer_fused_params_offset,
                               const fms_dt pw_2_ofms_zp,
                               const scales_dt pw_2_relu_threshold,
                               const int parallel_w)
{

    const int thread_w = threadIdx.x;
    const int thread_f = threadIdx.z;

    const int block_w = blockIdx.x;
    const int abs_h_index = blockIdx.y;

    const int abs_f = thread_f * F_W_V2_TILE_F;

    const int pw_1_filter_index = abs_f * PACKED_ITEMS;

    const int pw_1_ifm_hw = pw_1_ifm_width * pw_1_ifm_width; // TODO
    const int pw_2_ofm_hw = pw_2_ofm_width * pw_2_ofm_width; // TODO

    __shared__ fms_dt ofms_ifms_tile[PW_PW_MAX_FMS_BUFFER_SZ];
    //__shared__ fms_dt pw_2_weights[PW_PW_MAX_WEIGHTS_BUFFER_SZ];

    scales_dt scale0, scale1, scale2, scale3;

    biases_dt fused_zp0, fused_zp1, fused_zp2, fused_zp3;

    const int abs_w_index = block_w * parallel_w + thread_w;

    const int base_index_pw_1_weights = pw_1_layer_weights_offset + pw_1_filter_index * pw_1_compact_layer_depth;
    const int base_index_ifms = abs_h_index * pw_1_ifm_width + abs_w_index;

    if (abs_w_index < pw_1_ofm_width)
    {
        pss_dt sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        for (int d = 0; d < pw_1_compact_layer_depth; d++)
        {
            fms_dt fms_val = ifms[base_index_ifms + d * pw_1_ifm_hw];
            // ifms_tile[d * parallel_w + thread_w];

            sum0 += fms_val * pw_weights[base_index_pw_1_weights + d];
            sum1 += fms_val * pw_weights[base_index_pw_1_weights + pw_1_compact_layer_depth + d];
            sum2 += fms_val * pw_weights[base_index_pw_1_weights + 2 * pw_1_compact_layer_depth + d];
            sum3 += fms_val * pw_weights[base_index_pw_1_weights + 3 * pw_1_compact_layer_depth + d];
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
        ofms_ifms_tile[abs_f * parallel_w + thread_w] = sum0;
        ofms_ifms_tile[(abs_f + 1) * parallel_w + thread_w] = sum1;
        ofms_ifms_tile[(abs_f + 2) * parallel_w + thread_w] = sum2;
        ofms_ifms_tile[(abs_f + 3) * parallel_w + thread_w] = sum3;
    }
    //}

    __syncthreads();
    // if (thread_w >= parallel_w)
    {
        // const int abs_w_index = block_w * parallel_w + (thread_w - parallel_w);

        for (int o_f = 0; o_f < pw_2_filters_to_parallelism_f_ratio; o_f++)
        {
            const int pw_2_compact_filter_index = abs_f * pw_2_filters_to_parallelism_f_ratio + o_f * F_W_V2_TILE_F;
            const int pw_2_filter_index = pw_2_compact_filter_index * PACKED_ITEMS;

            if (pw_2_compact_filter_index < pw_2_compact_layer_num_filters)
            {
                const int f_offset = pw_2_filter_index * pw_2_compact_layer_depth;
                const int base_index_pw_2_weights = pw_2_layer_weights_offset + f_offset;

                if (abs_w_index < pw_2_ofm_width)
                {
                    const int base_index_in_ofms = abs_h_index * pw_2_ofm_width +
                                                   pw_2_compact_filter_index * pw_2_ofm_hw + abs_w_index;

                    pss_dt sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                    int a = 0;
                    for (int d = 0; d < pw_2_compact_layer_depth; d++)
                    {
                        fms_dt fms_val = ofms_ifms_tile[d * parallel_w + thread_w];

                        sum0 += fms_val * pw_weights[base_index_pw_2_weights + d];
                        sum1 += fms_val * pw_weights[base_index_pw_2_weights + pw_2_compact_layer_depth + d];
                        sum2 += fms_val * pw_weights[base_index_pw_2_weights + 2 * pw_2_compact_layer_depth + d];
                        sum3 += fms_val * pw_weights[base_index_pw_2_weights + 3 * pw_2_compact_layer_depth + d];
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
                    ofms[base_index_in_ofms + pw_2_ofm_hw] = sum1;
                    ofms[base_index_in_ofms + 2 * pw_2_ofm_hw] = sum2;
                    ofms[base_index_in_ofms + 3 * pw_2_ofm_hw] = sum3;
                }
            }
        }
    }
}

void fused_pw_pw_convolutionGPU_chw(fms_dt *ifms, fms_dt *ofms,
                                weights_dt *pw_weights,
                                fused_scales_dt *fused_scales,
                                biases_dt *fused_zps,
                                layer_specs pw_1_l_specs,
                                layer_specs pw_2_l_specs,
                                int *fused_params_offsets,
                                const int iteration,
                                int *layers_parallelism_w,
                                float &exec_time)
{

    const int pw_1_ofms_width = pw_1_l_specs.layer_ofm_width;
    const int pw_1_ofms_height = pw_1_l_specs.layer_ofm_height;
    const int pw_1_compact_layer_depth = (pw_1_l_specs.layer_depth / PACKED_ITEMS);
    const int pw_1_compact_layer_num_filters = (pw_1_l_specs.layer_num_fils / PACKED_ITEMS);

    int parallel_w = layers_parallelism_w[pw_1_l_specs.layer_index];

    dim3 threads(parallel_w, 1, pw_1_compact_layer_num_filters / F_W_V2_TILE_F);
    dim3 blocks((pw_1_l_specs.layer_ifm_width + parallel_w - 1) / parallel_w,
                pw_1_l_specs.layer_ifm_height, 1);

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
        printf("%d, %d (FUSED_PWPW):\n", pw_1_l_specs.layer_index, pw_2_l_specs.layer_index);
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

    const int pw_1_layer_depth = pw_1_l_specs.layer_depth;
    const int pw_1_layer_ifm_width = pw_1_l_specs.layer_ifm_width;
    const int pw_1_layer_ofm_width = pw_1_l_specs.layer_ofm_width;
    const int pw_1_compact_ofms_width_depth = pw_1_compact_layer_num_filters * pw_1_layer_ofm_width;
    const int pw_2_layer_depth = pw_2_l_specs.layer_depth;
    const int pw_2_layer_ifm_width = pw_2_l_specs.layer_ifm_width;
    const int pw_2_layer_ofm_width = pw_2_l_specs.layer_ofm_width;
    const int pw_2_compact_layer_num_filters = pw_2_l_specs.layer_num_fils / PACKED_ITEMS;
    const int pw_2_compact_ofms_width_depth = pw_2_compact_layer_num_filters * pw_2_layer_ofm_width;
    const int pw_2_compact_layer_depth = pw_2_layer_depth / PACKED_ITEMS;

    int pw_2_filters_to_parallelism_f_ratio = pw_2_compact_layer_num_filters / (pw_1_compact_layer_num_filters / F_W_V2_TILE_F);
    if (pw_2_filters_to_parallelism_f_ratio < 1)
    {
        pw_2_filters_to_parallelism_f_ratio = 1;
    }

    pw_pw_conv_f_w_chw<<<blocks, threads>>>(ifms, ofms, pw_weights,
                                        fused_scales,
                                        fused_zps,
                                        pw_1_compact_layer_depth,
                                        pw_1_compact_layer_num_filters,
                                        pw_1_layer_ifm_width,
                                        pw_1_compact_layer_depth * pw_1_layer_ifm_width,
                                        pw_1_l_specs.layer_ofm_height,
                                        pw_1_l_specs.layer_ofm_width,
                                        pw_1_compact_ofms_width_depth,
                                        pw_1_l_specs.layer_weights_offset / PACKED_ITEMS,
                                        fused_params_offsets[pw_1_l_specs.layer_index],
                                        pw_1_l_specs.layer_ofms_zero_point,
                                        pw_1_l_specs.relu_threshold,
                                        pw_1_l_specs.layer_activation,
                                        pw_2_compact_layer_depth,
                                        pw_2_compact_layer_num_filters,
                                        pw_2_layer_ifm_width,
                                        pw_2_compact_layer_depth * pw_2_layer_ifm_width,
                                        pw_2_l_specs.layer_ofm_height,
                                        pw_2_l_specs.layer_ofm_width,
                                        pw_2_compact_ofms_width_depth,
                                        pw_2_filters_to_parallelism_f_ratio,
                                        pw_2_l_specs.layer_weights_offset / PACKED_ITEMS,
                                        fused_params_offsets[pw_2_l_specs.layer_index],
                                        pw_2_l_specs.layer_ofms_zero_point,
                                        pw_2_l_specs.relu_threshold,
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