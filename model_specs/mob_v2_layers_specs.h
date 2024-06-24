#ifndef LAYERS_SPECS
#define LAYERS_SPECS

const int MODEL_NUM_LAYERS = 71;
const int MAX_LAYER_DW = 128 * 128;
const int conv_layers_indices[71] = {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0};
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
 const int layer_3_pw_num_fils = 16;
const int layer_3_pw_depth = 32;
const int layer_3_pw_filter_dim = 1;
 const int layer_3_pw_ifm_width = 112;
 //****************************
//****************************
 const int layer_4_pw_num_fils = 96;
const int layer_4_pw_depth = 16;
const int layer_4_pw_filter_dim = 1;
 const int layer_4_pw_ifm_width = 112;
 //****************************
//****************************
 const int layer_6_dw_num_fils = 96;
const int layer_6_dw_depth = 96;
const int layer_6_dw_filter_dim = 3;
 const int layer_6_dw_ifm_width = 112;
 //****************************
//****************************
 const int layer_7_pw_num_fils = 24;
const int layer_7_pw_depth = 96;
const int layer_7_pw_filter_dim = 1;
 const int layer_7_pw_ifm_width = 56;
 //****************************
//****************************
 const int layer_8_pw_num_fils = 144;
const int layer_8_pw_depth = 24;
const int layer_8_pw_filter_dim = 1;
 const int layer_8_pw_ifm_width = 56;
 //****************************
//****************************
 const int layer_9_dw_num_fils = 144;
const int layer_9_dw_depth = 144;
const int layer_9_dw_filter_dim = 3;
 const int layer_9_dw_ifm_width = 56;
 //****************************
//****************************
 const int layer_10_pw_num_fils = 24;
const int layer_10_pw_depth = 144;
const int layer_10_pw_filter_dim = 1;
 const int layer_10_pw_ifm_width = 56;
 //****************************
//****************************
 const int layer_12_pw_num_fils = 144;
const int layer_12_pw_depth = 24;
const int layer_12_pw_filter_dim = 1;
 const int layer_12_pw_ifm_width = 56;
 //****************************
//****************************
 const int layer_14_dw_num_fils = 144;
const int layer_14_dw_depth = 144;
const int layer_14_dw_filter_dim = 3;
 const int layer_14_dw_ifm_width = 56;
 //****************************
//****************************
 const int layer_15_pw_num_fils = 32;
const int layer_15_pw_depth = 144;
const int layer_15_pw_filter_dim = 1;
 const int layer_15_pw_ifm_width = 28;
 //****************************
//****************************
 const int layer_16_pw_num_fils = 192;
const int layer_16_pw_depth = 32;
const int layer_16_pw_filter_dim = 1;
 const int layer_16_pw_ifm_width = 28;
 //****************************
//****************************
 const int layer_17_dw_num_fils = 192;
const int layer_17_dw_depth = 192;
const int layer_17_dw_filter_dim = 3;
 const int layer_17_dw_ifm_width = 28;
 //****************************
//****************************
 const int layer_18_pw_num_fils = 32;
const int layer_18_pw_depth = 192;
const int layer_18_pw_filter_dim = 1;
 const int layer_18_pw_ifm_width = 28;
 //****************************
//****************************
 const int layer_20_pw_num_fils = 192;
const int layer_20_pw_depth = 32;
const int layer_20_pw_filter_dim = 1;
 const int layer_20_pw_ifm_width = 28;
 //****************************
//****************************
 const int layer_21_dw_num_fils = 192;
const int layer_21_dw_depth = 192;
const int layer_21_dw_filter_dim = 3;
 const int layer_21_dw_ifm_width = 28;
 //****************************
//****************************
 const int layer_22_pw_num_fils = 32;
const int layer_22_pw_depth = 192;
const int layer_22_pw_filter_dim = 1;
 const int layer_22_pw_ifm_width = 28;
 //****************************
//****************************
 const int layer_24_pw_num_fils = 192;
const int layer_24_pw_depth = 32;
const int layer_24_pw_filter_dim = 1;
 const int layer_24_pw_ifm_width = 28;
 //****************************
//****************************
 const int layer_26_dw_num_fils = 192;
const int layer_26_dw_depth = 192;
const int layer_26_dw_filter_dim = 3;
 const int layer_26_dw_ifm_width = 28;
 //****************************
//****************************
 const int layer_27_pw_num_fils = 64;
const int layer_27_pw_depth = 192;
const int layer_27_pw_filter_dim = 1;
 const int layer_27_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_28_pw_num_fils = 384;
const int layer_28_pw_depth = 64;
const int layer_28_pw_filter_dim = 1;
 const int layer_28_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_29_dw_num_fils = 384;
const int layer_29_dw_depth = 384;
const int layer_29_dw_filter_dim = 3;
 const int layer_29_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_30_pw_num_fils = 64;
const int layer_30_pw_depth = 384;
const int layer_30_pw_filter_dim = 1;
 const int layer_30_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_32_pw_num_fils = 384;
const int layer_32_pw_depth = 64;
const int layer_32_pw_filter_dim = 1;
 const int layer_32_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_33_dw_num_fils = 384;
const int layer_33_dw_depth = 384;
const int layer_33_dw_filter_dim = 3;
 const int layer_33_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_34_pw_num_fils = 64;
const int layer_34_pw_depth = 384;
const int layer_34_pw_filter_dim = 1;
 const int layer_34_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_36_pw_num_fils = 384;
const int layer_36_pw_depth = 64;
const int layer_36_pw_filter_dim = 1;
 const int layer_36_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_37_dw_num_fils = 384;
const int layer_37_dw_depth = 384;
const int layer_37_dw_filter_dim = 3;
 const int layer_37_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_38_pw_num_fils = 64;
const int layer_38_pw_depth = 384;
const int layer_38_pw_filter_dim = 1;
 const int layer_38_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_40_pw_num_fils = 384;
const int layer_40_pw_depth = 64;
const int layer_40_pw_filter_dim = 1;
 const int layer_40_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_41_dw_num_fils = 384;
const int layer_41_dw_depth = 384;
const int layer_41_dw_filter_dim = 3;
 const int layer_41_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_42_pw_num_fils = 96;
const int layer_42_pw_depth = 384;
const int layer_42_pw_filter_dim = 1;
 const int layer_42_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_43_pw_num_fils = 576;
const int layer_43_pw_depth = 96;
const int layer_43_pw_filter_dim = 1;
 const int layer_43_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_44_dw_num_fils = 576;
const int layer_44_dw_depth = 576;
const int layer_44_dw_filter_dim = 3;
 const int layer_44_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_45_pw_num_fils = 96;
const int layer_45_pw_depth = 576;
const int layer_45_pw_filter_dim = 1;
 const int layer_45_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_47_pw_num_fils = 576;
const int layer_47_pw_depth = 96;
const int layer_47_pw_filter_dim = 1;
 const int layer_47_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_48_dw_num_fils = 576;
const int layer_48_dw_depth = 576;
const int layer_48_dw_filter_dim = 3;
 const int layer_48_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_49_pw_num_fils = 96;
const int layer_49_pw_depth = 576;
const int layer_49_pw_filter_dim = 1;
 const int layer_49_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_51_pw_num_fils = 576;
const int layer_51_pw_depth = 96;
const int layer_51_pw_filter_dim = 1;
 const int layer_51_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_53_dw_num_fils = 576;
const int layer_53_dw_depth = 576;
const int layer_53_dw_filter_dim = 3;
 const int layer_53_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_54_pw_num_fils = 160;
const int layer_54_pw_depth = 576;
const int layer_54_pw_filter_dim = 1;
 const int layer_54_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_55_pw_num_fils = 960;
const int layer_55_pw_depth = 160;
const int layer_55_pw_filter_dim = 1;
 const int layer_55_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_56_dw_num_fils = 960;
const int layer_56_dw_depth = 960;
const int layer_56_dw_filter_dim = 3;
 const int layer_56_dw_ifm_width = 7;
 //****************************
//****************************
 const int layer_57_pw_num_fils = 160;
const int layer_57_pw_depth = 960;
const int layer_57_pw_filter_dim = 1;
 const int layer_57_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_59_pw_num_fils = 960;
const int layer_59_pw_depth = 160;
const int layer_59_pw_filter_dim = 1;
 const int layer_59_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_60_dw_num_fils = 960;
const int layer_60_dw_depth = 960;
const int layer_60_dw_filter_dim = 3;
 const int layer_60_dw_ifm_width = 7;
 //****************************
//****************************
 const int layer_61_pw_num_fils = 160;
const int layer_61_pw_depth = 960;
const int layer_61_pw_filter_dim = 1;
 const int layer_61_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_63_pw_num_fils = 960;
const int layer_63_pw_depth = 160;
const int layer_63_pw_filter_dim = 1;
 const int layer_63_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_64_dw_num_fils = 960;
const int layer_64_dw_depth = 960;
const int layer_64_dw_filter_dim = 3;
 const int layer_64_dw_ifm_width = 7;
 //****************************
//****************************
 const int layer_65_pw_num_fils = 320;
const int layer_65_pw_depth = 960;
const int layer_65_pw_filter_dim = 1;
 const int layer_65_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_66_pw_num_fils = 1280;
const int layer_66_pw_depth = 320;
const int layer_66_pw_filter_dim = 1;
 const int layer_66_pw_ifm_width = 7;
 //****************************
const fc_layer_specs layer_68_fc_specs = {
                -128,//const fms_dt ifm_zero_point
                };
#endif
