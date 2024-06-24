#include "dtype_defs.h"
#include "simulation_constants.h"

#ifndef GENERAL_SPECS
#define GENERAL_SPECS

#define S_CONV 0
#define PW_CONV 1
#define DW_CONV 2
#define ADD_LAYER_ID 3
#define AVG_POOL_LAYER_ID 4

#define RELU 6
#define RELU6 6 // TODO

#if DATA_TYPE == INT8_DTYPE
const int PACKED_ITEMS = 4;
#elif DATA_TYPE == FLOAT_DTYPE
const int PACKED_ITEMS = 1;
#endif

struct fc_layer_specs
{
    const fms_dt ifm_zero_point;
};

struct layer_specs
{
    int layer_index;
    int conv_layer_type;
    int layer_num_fils;
    int strides;
    int filter_size;
    int padding_left;
    int padding_right;
    int padding_top;
    int padding_bottom;
    int layer_depth;
    int layer_ifm_height;
    int layer_ifm_width;
    int layer_ofm_height;
    int layer_ofm_width;
    int layer_activation;
    int layer_num_of_ifm_tiles_h;
    int layer_num_of_ifm_tiles_w;
    int layer_num_of_ofm_tiles_h;
    int layer_num_of_ofm_tiles_w;
    int layer_weights_offset;
    bool write_to_result_or_channels;
    bool write_to_tmp;
    int followed_by;
    fms_dt layer_ifms_zero_point;
    scales_dt layer_ofms_scale;
    int relu_threshold;
    fms_dt layer_ofms_zero_point;
    rec_scales_dt add_layer_scale_reciprocal;
    biases_dt add_layer_zero_point;
    scales_dt skip_connection_other_layer_scale;
    biases_dt skip_connection_other_layer_zero_point;
    int data_layout;
};

struct pooling_layer_specs
{
    int ifm_depth;
    int ifm_height;
    int ifm_width;
    int ofm_depth;
    int ofm_height;
    int ofm_width;
    bool full_hw;
    pooling_fused_scales_dt fused_scale;
    biases_dt ifms_zero_point;
    biases_dt ofms_zero_point;
};

// max of all possible to make it generic
#if COMPARE_WITH_CUDNN
const int MAX_PADDING = 1;
const int MAX_COMPACT_DEPTH = 2048 / PACKED_ITEMS;
const int MAX_PW_COMPACT_DEPTH_FUSED = 728 / PACKED_ITEMS; // among the fused
const int MAX_FMS_SIZE = 96 * 112 * 112;
const int MAX_FMS_SIZE_PACKED = MAX_FMS_SIZE / PACKED_ITEMS;
const int MAX_DW_FILTER_DIM = 3;

#elif MODEL_ID == XCE_R
const int MAX_PADDING = 1; // apart from the first layer
const int MAX_COMPACT_DEPTH = 2048 / PACKED_ITEMS;
const int MAX_PW_COMPACT_DEPTH_FUSED = 728 / PACKED_ITEMS; // among the fused
const int MAX_FMS_SIZE = 128 * 109 * 109;
const int MAX_FMS_SIZE_PACKED = MAX_FMS_SIZE / PACKED_ITEMS;
const int MAX_TMP_FMS_SIZE = 24 * 56 * 56;
const int MAX_TMP_FMS_SIZE_PACKED = MAX_TMP_FMS_SIZE / PACKED_ITEMS;
const int MAX_DW_FILTER_DIM = 3; // apart from the first layer

#elif MODEL_ID == RESNET50
const int MAX_PADDING = 1; // apart from the first layer
const int MAX_COMPACT_DEPTH = 2048 / PACKED_ITEMS;
const int MAX_PW_COMPACT_DEPTH_FUSED = 2048 / PACKED_ITEMS; // among the fused
const int MAX_FMS_SIZE = 256 * 56 * 56;
const int MAX_FMS_SIZE_PACKED = MAX_FMS_SIZE / PACKED_ITEMS;
const int MAX_TMP_FMS_SIZE = 24 * 56 * 56;
const int MAX_TMP_FMS_SIZE_PACKED = MAX_TMP_FMS_SIZE / PACKED_ITEMS;
const int MAX_DW_FILTER_DIM = 3; // apart from the first layer

#elif MODEL_ID == MOB_V1
const int MAX_PADDING = 1;
const int MAX_COMPACT_DEPTH = 1024 / PACKED_ITEMS;
const int MAX_PW_COMPACT_DEPTH_FUSED = 512 / PACKED_ITEMS; // among the fused
const int MAX_FMS_SIZE = 64 * 112 * 112;
const int MAX_FMS_SIZE_PACKED = MAX_FMS_SIZE / PACKED_ITEMS;
const int MAX_TMP_FMS_SIZE = 24 * 56 * 56;
const int MAX_TMP_FMS_SIZE_PACKED = MAX_TMP_FMS_SIZE / PACKED_ITEMS;
const int MAX_DW_FILTER_DIM = 3;

#elif MODEL_ID == MOB_V2
const int MAX_PADDING = 1;
const int MAX_COMPACT_DEPTH = 960 / PACKED_ITEMS;
const int MAX_PW_COMPACT_DEPTH_FUSED = 160 / PACKED_ITEMS; // among the fused
const int MAX_FMS_SIZE = 96 * 112 * 112;
const int MAX_FMS_SIZE_PACKED = MAX_FMS_SIZE / PACKED_ITEMS;
const int MAX_TMP_FMS_SIZE = 24 * 56 * 56;
const int MAX_TMP_FMS_SIZE_PACKED = MAX_TMP_FMS_SIZE / PACKED_ITEMS;
const int MAX_DW_FILTER_DIM = 3;

#elif MODEL_ID == GPROX_3
const int MAX_PADDING = 1;
const int MAX_COMPACT_DEPTH = 1536 / PACKED_ITEMS;
const int MAX_PW_COMPACT_DEPTH_FUSED = 256 / PACKED_ITEMS; // among the fused
const int MAX_FMS_SIZE = 72 * 112 * 112;
const int MAX_FMS_SIZE_PACKED = MAX_FMS_SIZE / PACKED_ITEMS;
const int MAX_TMP_FMS_SIZE = 24 * 56 * 56;
const int MAX_TMP_FMS_SIZE_PACKED = MAX_TMP_FMS_SIZE / PACKED_ITEMS;
const int MAX_DW_FILTER_DIM = 3;
#endif

enum fusion_types
{
    pwdw,
    dwpw,
    pwpw,
    pwdw_wide
};

// the following are weird definitions, but for some reason replacing a kernel param by a constant can reduce running time considerably
const int FILTER_3x3_DIM = 3;
const int FILTER_3x3_AREA = FILTER_3x3_DIM * FILTER_3x3_DIM;
const int FILTER_3x3_PADDED_AREA = 16;
const int STRIDES_2 = 2;

#endif