#ifndef LAYERS_SPECS
#define LAYERS_SPECS

const int MODEL_NUM_LAYERS = 57;
const int MAX_LAYER_DW = 8 * 2048;
const int conv_layers_indices[57] = {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0};
//****************************
 const int first_conv_layer_num_fils = 32;
const int first_conv_layer_depth = 3;
const int first_conv_layer_filter_dim = 3;
 const int first_conv_layer_strides = 2;
 const int first_conv_layer_padding_left = 0;
 const int first_conv_layer_padding_right = 1;
 const int first_conv_layer_ifm_width = 224;
 //****************************
//****************************
 const int layer_4_dw_num_fils = 32;
const int layer_4_dw_depth = 32;
const int layer_4_dw_filter_dim = 3;
 const int layer_4_dw_ifm_width = 112;
 //****************************
//****************************
 const int layer_5_pw_num_fils = 24;
const int layer_5_pw_depth = 32;
const int layer_5_pw_filter_dim = 1;
 const int layer_5_pw_ifm_width = 112;
 //****************************
//****************************
 const int layer_6_pw_num_fils = 72;
const int layer_6_pw_depth = 24;
const int layer_6_pw_filter_dim = 1;
 const int layer_6_pw_ifm_width = 112;
 //****************************
//****************************
 const int layer_7_dw_num_fils = 72;
const int layer_7_dw_depth = 72;
const int layer_7_dw_filter_dim = 3;
 const int layer_7_dw_ifm_width = 112;
 //****************************
//****************************
 const int layer_8_pw_num_fils = 32;
const int layer_8_pw_depth = 72;
const int layer_8_pw_filter_dim = 1;
 const int layer_8_pw_ifm_width = 56;
 //****************************
//****************************
 const int layer_9_pw_num_fils = 96;
const int layer_9_pw_depth = 32;
const int layer_9_pw_filter_dim = 1;
 const int layer_9_pw_ifm_width = 56;
 //****************************
//****************************
 const int layer_10_dw_num_fils = 96;
const int layer_10_dw_depth = 96;
const int layer_10_dw_filter_dim = 3;
 const int layer_10_dw_ifm_width = 56;
 //****************************
//****************************
 const int layer_11_pw_num_fils = 56;
const int layer_11_pw_depth = 96;
const int layer_11_pw_filter_dim = 1;
 const int layer_11_pw_ifm_width = 28;
 //****************************
//****************************
 const int layer_12_pw_num_fils = 168;
const int layer_12_pw_depth = 56;
const int layer_12_pw_filter_dim = 1;
 const int layer_12_pw_ifm_width = 28;
 //****************************
//****************************
 const int layer_13_dw_num_fils = 168;
const int layer_13_dw_depth = 168;
const int layer_13_dw_filter_dim = 3;
 const int layer_13_dw_ifm_width = 28;
 //****************************
//****************************
 const int layer_14_pw_num_fils = 56;
const int layer_14_pw_depth = 168;
const int layer_14_pw_filter_dim = 1;
 const int layer_14_pw_ifm_width = 28;
 //****************************
//****************************
 const int layer_16_pw_num_fils = 336;
const int layer_16_pw_depth = 56;
const int layer_16_pw_filter_dim = 1;
 const int layer_16_pw_ifm_width = 28;
 //****************************
//****************************
 const int layer_17_dw_num_fils = 336;
const int layer_17_dw_depth = 336;
const int layer_17_dw_filter_dim = 3;
 const int layer_17_dw_ifm_width = 28;
 //****************************
//****************************
 const int layer_18_pw_num_fils = 112;
const int layer_18_pw_depth = 336;
const int layer_18_pw_filter_dim = 1;
 const int layer_18_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_19_pw_num_fils = 336;
const int layer_19_pw_depth = 112;
const int layer_19_pw_filter_dim = 1;
 const int layer_19_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_20_dw_num_fils = 336;
const int layer_20_dw_depth = 336;
const int layer_20_dw_filter_dim = 3;
 const int layer_20_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_21_pw_num_fils = 112;
const int layer_21_pw_depth = 336;
const int layer_21_pw_filter_dim = 1;
 const int layer_21_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_23_pw_num_fils = 672;
const int layer_23_pw_depth = 112;
const int layer_23_pw_filter_dim = 1;
 const int layer_23_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_24_dw_num_fils = 672;
const int layer_24_dw_depth = 672;
const int layer_24_dw_filter_dim = 3;
 const int layer_24_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_25_pw_num_fils = 128;
const int layer_25_pw_depth = 672;
const int layer_25_pw_filter_dim = 1;
 const int layer_25_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_26_pw_num_fils = 384;
const int layer_26_pw_depth = 128;
const int layer_26_pw_filter_dim = 1;
 const int layer_26_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_27_dw_num_fils = 384;
const int layer_27_dw_depth = 384;
const int layer_27_dw_filter_dim = 3;
 const int layer_27_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_28_pw_num_fils = 128;
const int layer_28_pw_depth = 384;
const int layer_28_pw_filter_dim = 1;
 const int layer_28_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_30_pw_num_fils = 384;
const int layer_30_pw_depth = 128;
const int layer_30_pw_filter_dim = 1;
 const int layer_30_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_31_dw_num_fils = 384;
const int layer_31_dw_depth = 384;
const int layer_31_dw_filter_dim = 3;
 const int layer_31_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_32_pw_num_fils = 128;
const int layer_32_pw_depth = 384;
const int layer_32_pw_filter_dim = 1;
 const int layer_32_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_34_pw_num_fils = 768;
const int layer_34_pw_depth = 128;
const int layer_34_pw_filter_dim = 1;
 const int layer_34_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_35_dw_num_fils = 768;
const int layer_35_dw_depth = 768;
const int layer_35_dw_filter_dim = 3;
 const int layer_35_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_36_pw_num_fils = 256;
const int layer_36_pw_depth = 768;
const int layer_36_pw_filter_dim = 1;
 const int layer_36_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_37_pw_num_fils = 1536;
const int layer_37_pw_depth = 256;
const int layer_37_pw_filter_dim = 1;
 const int layer_37_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_38_dw_num_fils = 1536;
const int layer_38_dw_depth = 1536;
const int layer_38_dw_filter_dim = 3;
 const int layer_38_dw_ifm_width = 7;
 //****************************
//****************************
 const int layer_39_pw_num_fils = 256;
const int layer_39_pw_depth = 1536;
const int layer_39_pw_filter_dim = 1;
 const int layer_39_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_41_pw_num_fils = 1536;
const int layer_41_pw_depth = 256;
const int layer_41_pw_filter_dim = 1;
 const int layer_41_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_42_dw_num_fils = 1536;
const int layer_42_dw_depth = 1536;
const int layer_42_dw_filter_dim = 3;
 const int layer_42_dw_ifm_width = 7;
 //****************************
//****************************
 const int layer_43_pw_num_fils = 256;
const int layer_43_pw_depth = 1536;
const int layer_43_pw_filter_dim = 1;
 const int layer_43_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_45_pw_num_fils = 1536;
const int layer_45_pw_depth = 256;
const int layer_45_pw_filter_dim = 1;
 const int layer_45_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_46_dw_num_fils = 1536;
const int layer_46_dw_depth = 1536;
const int layer_46_dw_filter_dim = 3;
 const int layer_46_dw_ifm_width = 7;
 //****************************
//****************************
 const int layer_47_pw_num_fils = 256;
const int layer_47_pw_depth = 1536;
const int layer_47_pw_filter_dim = 1;
 const int layer_47_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_49_pw_num_fils = 1536;
const int layer_49_pw_depth = 256;
const int layer_49_pw_filter_dim = 1;
 const int layer_49_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_50_dw_num_fils = 1536;
const int layer_50_dw_depth = 1536;
const int layer_50_dw_filter_dim = 3;
 const int layer_50_dw_ifm_width = 7;
 //****************************
//****************************
 const int layer_51_pw_num_fils = 432;
const int layer_51_pw_depth = 1536;
const int layer_51_pw_filter_dim = 1;
 const int layer_51_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_52_pw_num_fils = 1280;
const int layer_52_pw_depth = 432;
const int layer_52_pw_filter_dim = 1;
 const int layer_52_pw_ifm_width = 7;
 //****************************
const pooling_layer_specs layer_53_avgpool_specs = {
                1.8097904154646356,//const pooling_fused_scales_dt fused_scale; 
                -128,//const biases_dt ifms_zero_point;
                -128,//const biases_dt ofms_zero_point;
                };
#endif
