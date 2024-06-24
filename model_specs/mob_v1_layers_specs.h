#ifndef LAYERS_SPECS
#define LAYERS_SPECS

const int MODEL_NUM_LAYERS = 40;
const int MAX_LAYER_DW = 8 * 1024;
const int conv_layers_indices[40] = {0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0};
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
 const int layer_2_dw_num_fils = 32;
const int layer_2_dw_depth = 32;
const int layer_2_dw_filter_dim = 3;
 const int layer_2_dw_ifm_width = 112;
 //****************************
//****************************
 const int layer_3_pw_num_fils = 64;
const int layer_3_pw_depth = 32;
const int layer_3_pw_filter_dim = 1;
 const int layer_3_pw_ifm_width = 112;
 //****************************
//****************************
 const int layer_5_dw_num_fils = 64;
const int layer_5_dw_depth = 64;
const int layer_5_dw_filter_dim = 3;
 const int layer_5_dw_ifm_width = 112;
 //****************************
//****************************
 const int layer_6_pw_num_fils = 128;
const int layer_6_pw_depth = 64;
const int layer_6_pw_filter_dim = 1;
 const int layer_6_pw_ifm_width = 56;
 //****************************
//****************************
 const int layer_7_dw_num_fils = 128;
const int layer_7_dw_depth = 128;
const int layer_7_dw_filter_dim = 3;
 const int layer_7_dw_ifm_width = 56;
 //****************************
//****************************
 const int layer_8_pw_num_fils = 128;
const int layer_8_pw_depth = 128;
const int layer_8_pw_filter_dim = 1;
 const int layer_8_pw_ifm_width = 56;
 //****************************
//****************************
 const int layer_10_dw_num_fils = 128;
const int layer_10_dw_depth = 128;
const int layer_10_dw_filter_dim = 3;
 const int layer_10_dw_ifm_width = 56;
 //****************************
//****************************
 const int layer_11_pw_num_fils = 256;
const int layer_11_pw_depth = 128;
const int layer_11_pw_filter_dim = 1;
 const int layer_11_pw_ifm_width = 28;
 //****************************
//****************************
 const int layer_12_dw_num_fils = 256;
const int layer_12_dw_depth = 256;
const int layer_12_dw_filter_dim = 3;
 const int layer_12_dw_ifm_width = 28;
 //****************************
//****************************
 const int layer_13_pw_num_fils = 256;
const int layer_13_pw_depth = 256;
const int layer_13_pw_filter_dim = 1;
 const int layer_13_pw_ifm_width = 28;
 //****************************
//****************************
 const int layer_15_dw_num_fils = 256;
const int layer_15_dw_depth = 256;
const int layer_15_dw_filter_dim = 3;
 const int layer_15_dw_ifm_width = 28;
 //****************************
//****************************
 const int layer_16_pw_num_fils = 512;
const int layer_16_pw_depth = 256;
const int layer_16_pw_filter_dim = 1;
 const int layer_16_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_17_dw_num_fils = 512;
const int layer_17_dw_depth = 512;
const int layer_17_dw_filter_dim = 3;
 const int layer_17_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_18_pw_num_fils = 512;
const int layer_18_pw_depth = 512;
const int layer_18_pw_filter_dim = 1;
 const int layer_18_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_19_dw_num_fils = 512;
const int layer_19_dw_depth = 512;
const int layer_19_dw_filter_dim = 3;
 const int layer_19_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_20_pw_num_fils = 512;
const int layer_20_pw_depth = 512;
const int layer_20_pw_filter_dim = 1;
 const int layer_20_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_21_dw_num_fils = 512;
const int layer_21_dw_depth = 512;
const int layer_21_dw_filter_dim = 3;
 const int layer_21_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_22_pw_num_fils = 512;
const int layer_22_pw_depth = 512;
const int layer_22_pw_filter_dim = 1;
 const int layer_22_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_23_dw_num_fils = 512;
const int layer_23_dw_depth = 512;
const int layer_23_dw_filter_dim = 3;
 const int layer_23_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_24_pw_num_fils = 512;
const int layer_24_pw_depth = 512;
const int layer_24_pw_filter_dim = 1;
 const int layer_24_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_25_dw_num_fils = 512;
const int layer_25_dw_depth = 512;
const int layer_25_dw_filter_dim = 3;
 const int layer_25_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_26_pw_num_fils = 512;
const int layer_26_pw_depth = 512;
const int layer_26_pw_filter_dim = 1;
 const int layer_26_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_28_dw_num_fils = 512;
const int layer_28_dw_depth = 512;
const int layer_28_dw_filter_dim = 3;
 const int layer_28_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_29_pw_num_fils = 1024;
const int layer_29_pw_depth = 512;
const int layer_29_pw_filter_dim = 1;
 const int layer_29_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_30_dw_num_fils = 1024;
const int layer_30_dw_depth = 1024;
const int layer_30_dw_filter_dim = 3;
 const int layer_30_dw_ifm_width = 7;
 //****************************
//****************************
 const int layer_31_pw_num_fils = 1024;
const int layer_31_pw_depth = 1024;
const int layer_31_pw_filter_dim = 1;
 const int layer_31_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_33_pw_num_fils = 1000;
const int layer_33_pw_depth = 1024;
const int layer_33_pw_filter_dim = 1;
 const int layer_33_pw_ifm_width = 1;
 //****************************
#endif
