#include "../../headers/conv_kernels.h"

#if (FUSION_MODE == ALL_MODES || FUSION_MODE == FUSED_F_W) && DATA_TYPE == FLOAT_DTYPE

using namespace std;
namespace cg = cooperative_groups;

__global__ void pw_conv_f_w_v2_chw(fms_dt *ifms, fms_dt *ofms, weights_dt *pw_weights,
                               fused_scales_dt *fused_scales,
                               biases_dt *fused_zps,
                               const int pw_1_compact_layer_depth,
                               const int pw_1_compact_layer_num_filters,
                               const int pw_1_ifm_width,
                               const int pw_1_compact_ifm_width_depth,
                               const int pw_1_ofm_height,
                               const int pw_1_ofm_width,
                               const int pw_1_compact_ofm_width_depth,
                               const int pw_1_depth_to_parallelism_f_ratio,
                               const int pw_1_layer_weights_offset,
                               const int pw_1_layer_fused_params_offset,
                               const fms_dt pw_1_ofms_zp,
                               const scales_dt pw_1_relu_threshold,
                               const int layer_activation,
                               const int parallel_w)
{

    const int thread_w = threadIdx.x;
    const int thread_f = threadIdx.z;

    const int pw_1_filter_index = thread_f * PACKED_ITEMS * F_W_V2_TILE_F;

    const int block_w = blockIdx.x;
    const int abs_h_index = blockIdx.y;
    const int abs_w_index = blockIdx.x * parallel_w + threadIdx.x;

    const int base_index_in_ofms = abs_h_index * pw_1_ofm_width +
                                   pw_1_filter_index * pw_1_ofm_width * pw_1_ofm_height + abs_w_index;

    const int base_index_in_ifms = abs_h_index * pw_1_ifm_width + abs_w_index; // TODO

    const int base_index_pw_1_weights = pw_1_layer_weights_offset + pw_1_filter_index * pw_1_compact_layer_depth;

    if (abs_w_index < pw_1_ofm_width)
    {
        pss_dt sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        for (int d = 0; d < pw_1_compact_layer_depth; d++)
        {
            fms_dt fms_val = ifms[base_index_in_ifms + d * pw_1_ifm_width * pw_1_ifm_width]; // TODO

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
        ofms[base_index_in_ofms] = sum0;
        ofms[base_index_in_ofms + pw_1_ofm_width * pw_1_ofm_width] = sum1;
        ofms[base_index_in_ofms + 2 * pw_1_ofm_width * pw_1_ofm_width] = sum2;
        ofms[base_index_in_ofms + 3 * pw_1_ofm_width * pw_1_ofm_width] = sum3;
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
        printf("(PW_V2): layer %d\n", pw_1_l_specs.layer_index);
        printf("%d, %d\n", parallel_w, pw_1_compact_layer_num_filters);
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

    int pw_1_depth_to_parallelism_f_ratio = pw_1_compact_layer_depth / pw_1_compact_layer_num_filters;
    if (pw_1_depth_to_parallelism_f_ratio < 1)
    {
        pw_1_depth_to_parallelism_f_ratio = 1;
    }

    pw_conv_f_w_v2_chw<<<blocks, threads>>>(ifms, ofms, pw_weights,
                                        fused_scales,
                                        fused_zps,
                                        pw_1_compact_layer_depth,
                                        pw_1_compact_layer_num_filters,
                                        pw_1_layer_ifm_width,
                                        pw_1_compact_layer_depth * pw_1_layer_ifm_width,
                                        pw_1_l_specs.layer_ofm_height,
                                        pw_1_l_specs.layer_ofm_width,
                                        pw_1_compact_ofms_width_depth,
                                        pw_1_depth_to_parallelism_f_ratio,
                                        pw_1_l_specs.layer_weights_offset / PACKED_ITEMS,
                                        fused_params_offsets[pw_1_l_specs.layer_index],
                                        pw_1_l_specs.layer_ofms_zero_point,
                                        pw_1_l_specs.relu_threshold,
                                        pw_1_l_specs.layer_activation,
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