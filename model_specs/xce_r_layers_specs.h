#ifndef LAYERS_SPECS
#define LAYERS_SPECS

const int MODEL_NUM_LAYERS = 106;
const int MAX_LAYER_DW = 32 * 1024;
const int conv_layers_indices[106] = {0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0};
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
 const int layer_2_s_num_fils = 64;
const int layer_2_s_depth = 32;
const int layer_2_s_filter_dim = 3;
 const int layer_2_s_ifm_width = 111;
 //****************************
//****************************
 const int layer_3_dw_num_fils = 64;
const int layer_3_dw_depth = 64;
const int layer_3_dw_filter_dim = 3;
 const int layer_3_dw_ifm_width = 109;
 //****************************
//****************************
 const int layer_4_pw_num_fils = 128;
const int layer_4_pw_depth = 64;
const int layer_4_pw_filter_dim = 1;
 const int layer_4_pw_ifm_width = 109;
 //****************************
//****************************
 const int layer_5_dw_num_fils = 128;
const int layer_5_dw_depth = 128;
const int layer_5_dw_filter_dim = 3;
 const int layer_5_dw_ifm_width = 109;
 //****************************
//****************************
 const int layer_6_pw_num_fils = 128;
const int layer_6_pw_depth = 128;
const int layer_6_pw_filter_dim = 1;
 const int layer_6_pw_ifm_width = 109;
 //****************************
//****************************
 const int layer_8_pw_num_fils = 128;
const int layer_8_pw_depth = 64;
const int layer_8_pw_filter_dim = 1;
 const int layer_8_pw_ifm_width = 109;
 //****************************
//****************************
 const int layer_11_dw_num_fils = 128;
const int layer_11_dw_depth = 128;
const int layer_11_dw_filter_dim = 3;
 const int layer_11_dw_ifm_width = 55;
 //****************************
//****************************
 const int layer_12_pw_num_fils = 256;
const int layer_12_pw_depth = 128;
const int layer_12_pw_filter_dim = 1;
 const int layer_12_pw_ifm_width = 55;
 //****************************
//****************************
 const int layer_13_dw_num_fils = 256;
const int layer_13_dw_depth = 256;
const int layer_13_dw_filter_dim = 3;
 const int layer_13_dw_ifm_width = 55;
 //****************************
//****************************
 const int layer_14_pw_num_fils = 256;
const int layer_14_pw_depth = 256;
const int layer_14_pw_filter_dim = 1;
 const int layer_14_pw_ifm_width = 55;
 //****************************
//****************************
 const int layer_16_pw_num_fils = 256;
const int layer_16_pw_depth = 128;
const int layer_16_pw_filter_dim = 1;
 const int layer_16_pw_ifm_width = 55;
 //****************************
//****************************
 const int layer_19_dw_num_fils = 256;
const int layer_19_dw_depth = 256;
const int layer_19_dw_filter_dim = 3;
 const int layer_19_dw_ifm_width = 28;
 //****************************
//****************************
 const int layer_20_pw_num_fils = 728;
const int layer_20_pw_depth = 256;
const int layer_20_pw_filter_dim = 1;
 const int layer_20_pw_ifm_width = 28;
 //****************************
//****************************
 const int layer_21_dw_num_fils = 728;
const int layer_21_dw_depth = 728;
const int layer_21_dw_filter_dim = 3;
 const int layer_21_dw_ifm_width = 28;
 //****************************
//****************************
 const int layer_22_pw_num_fils = 728;
const int layer_22_pw_depth = 728;
const int layer_22_pw_filter_dim = 1;
 const int layer_22_pw_ifm_width = 28;
 //****************************
//****************************
 const int layer_24_pw_num_fils = 728;
const int layer_24_pw_depth = 256;
const int layer_24_pw_filter_dim = 1;
 const int layer_24_pw_ifm_width = 28;
 //****************************
//****************************
 const int layer_27_dw_num_fils = 728;
const int layer_27_dw_depth = 728;
const int layer_27_dw_filter_dim = 3;
 const int layer_27_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_28_pw_num_fils = 728;
const int layer_28_pw_depth = 728;
const int layer_28_pw_filter_dim = 1;
 const int layer_28_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_29_dw_num_fils = 728;
const int layer_29_dw_depth = 728;
const int layer_29_dw_filter_dim = 3;
 const int layer_29_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_30_pw_num_fils = 728;
const int layer_30_pw_depth = 728;
const int layer_30_pw_filter_dim = 1;
 const int layer_30_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_31_dw_num_fils = 728;
const int layer_31_dw_depth = 728;
const int layer_31_dw_filter_dim = 3;
 const int layer_31_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_32_pw_num_fils = 728;
const int layer_32_pw_depth = 728;
const int layer_32_pw_filter_dim = 1;
 const int layer_32_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_35_dw_num_fils = 728;
const int layer_35_dw_depth = 728;
const int layer_35_dw_filter_dim = 3;
 const int layer_35_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_36_pw_num_fils = 728;
const int layer_36_pw_depth = 728;
const int layer_36_pw_filter_dim = 1;
 const int layer_36_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_37_dw_num_fils = 728;
const int layer_37_dw_depth = 728;
const int layer_37_dw_filter_dim = 3;
 const int layer_37_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_38_pw_num_fils = 728;
const int layer_38_pw_depth = 728;
const int layer_38_pw_filter_dim = 1;
 const int layer_38_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_39_dw_num_fils = 728;
const int layer_39_dw_depth = 728;
const int layer_39_dw_filter_dim = 3;
 const int layer_39_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_40_pw_num_fils = 728;
const int layer_40_pw_depth = 728;
const int layer_40_pw_filter_dim = 1;
 const int layer_40_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_43_dw_num_fils = 728;
const int layer_43_dw_depth = 728;
const int layer_43_dw_filter_dim = 3;
 const int layer_43_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_44_pw_num_fils = 728;
const int layer_44_pw_depth = 728;
const int layer_44_pw_filter_dim = 1;
 const int layer_44_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_45_dw_num_fils = 728;
const int layer_45_dw_depth = 728;
const int layer_45_dw_filter_dim = 3;
 const int layer_45_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_46_pw_num_fils = 728;
const int layer_46_pw_depth = 728;
const int layer_46_pw_filter_dim = 1;
 const int layer_46_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_47_dw_num_fils = 728;
const int layer_47_dw_depth = 728;
const int layer_47_dw_filter_dim = 3;
 const int layer_47_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_48_pw_num_fils = 728;
const int layer_48_pw_depth = 728;
const int layer_48_pw_filter_dim = 1;
 const int layer_48_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_51_dw_num_fils = 728;
const int layer_51_dw_depth = 728;
const int layer_51_dw_filter_dim = 3;
 const int layer_51_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_52_pw_num_fils = 728;
const int layer_52_pw_depth = 728;
const int layer_52_pw_filter_dim = 1;
 const int layer_52_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_53_dw_num_fils = 728;
const int layer_53_dw_depth = 728;
const int layer_53_dw_filter_dim = 3;
 const int layer_53_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_54_pw_num_fils = 728;
const int layer_54_pw_depth = 728;
const int layer_54_pw_filter_dim = 1;
 const int layer_54_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_55_dw_num_fils = 728;
const int layer_55_dw_depth = 728;
const int layer_55_dw_filter_dim = 3;
 const int layer_55_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_56_pw_num_fils = 728;
const int layer_56_pw_depth = 728;
const int layer_56_pw_filter_dim = 1;
 const int layer_56_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_59_dw_num_fils = 728;
const int layer_59_dw_depth = 728;
const int layer_59_dw_filter_dim = 3;
 const int layer_59_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_60_pw_num_fils = 728;
const int layer_60_pw_depth = 728;
const int layer_60_pw_filter_dim = 1;
 const int layer_60_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_61_dw_num_fils = 728;
const int layer_61_dw_depth = 728;
const int layer_61_dw_filter_dim = 3;
 const int layer_61_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_62_pw_num_fils = 728;
const int layer_62_pw_depth = 728;
const int layer_62_pw_filter_dim = 1;
 const int layer_62_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_63_dw_num_fils = 728;
const int layer_63_dw_depth = 728;
const int layer_63_dw_filter_dim = 3;
 const int layer_63_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_64_pw_num_fils = 728;
const int layer_64_pw_depth = 728;
const int layer_64_pw_filter_dim = 1;
 const int layer_64_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_67_dw_num_fils = 728;
const int layer_67_dw_depth = 728;
const int layer_67_dw_filter_dim = 3;
 const int layer_67_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_68_pw_num_fils = 728;
const int layer_68_pw_depth = 728;
const int layer_68_pw_filter_dim = 1;
 const int layer_68_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_69_dw_num_fils = 728;
const int layer_69_dw_depth = 728;
const int layer_69_dw_filter_dim = 3;
 const int layer_69_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_70_pw_num_fils = 728;
const int layer_70_pw_depth = 728;
const int layer_70_pw_filter_dim = 1;
 const int layer_70_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_71_dw_num_fils = 728;
const int layer_71_dw_depth = 728;
const int layer_71_dw_filter_dim = 3;
 const int layer_71_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_72_pw_num_fils = 728;
const int layer_72_pw_depth = 728;
const int layer_72_pw_filter_dim = 1;
 const int layer_72_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_75_dw_num_fils = 728;
const int layer_75_dw_depth = 728;
const int layer_75_dw_filter_dim = 3;
 const int layer_75_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_76_pw_num_fils = 728;
const int layer_76_pw_depth = 728;
const int layer_76_pw_filter_dim = 1;
 const int layer_76_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_77_dw_num_fils = 728;
const int layer_77_dw_depth = 728;
const int layer_77_dw_filter_dim = 3;
 const int layer_77_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_78_pw_num_fils = 728;
const int layer_78_pw_depth = 728;
const int layer_78_pw_filter_dim = 1;
 const int layer_78_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_79_dw_num_fils = 728;
const int layer_79_dw_depth = 728;
const int layer_79_dw_filter_dim = 3;
 const int layer_79_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_80_pw_num_fils = 728;
const int layer_80_pw_depth = 728;
const int layer_80_pw_filter_dim = 1;
 const int layer_80_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_83_dw_num_fils = 728;
const int layer_83_dw_depth = 728;
const int layer_83_dw_filter_dim = 3;
 const int layer_83_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_84_pw_num_fils = 728;
const int layer_84_pw_depth = 728;
const int layer_84_pw_filter_dim = 1;
 const int layer_84_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_85_dw_num_fils = 728;
const int layer_85_dw_depth = 728;
const int layer_85_dw_filter_dim = 3;
 const int layer_85_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_86_pw_num_fils = 728;
const int layer_86_pw_depth = 728;
const int layer_86_pw_filter_dim = 1;
 const int layer_86_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_87_dw_num_fils = 728;
const int layer_87_dw_depth = 728;
const int layer_87_dw_filter_dim = 3;
 const int layer_87_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_88_pw_num_fils = 728;
const int layer_88_pw_depth = 728;
const int layer_88_pw_filter_dim = 1;
 const int layer_88_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_91_dw_num_fils = 728;
const int layer_91_dw_depth = 728;
const int layer_91_dw_filter_dim = 3;
 const int layer_91_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_92_pw_num_fils = 728;
const int layer_92_pw_depth = 728;
const int layer_92_pw_filter_dim = 1;
 const int layer_92_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_93_dw_num_fils = 728;
const int layer_93_dw_depth = 728;
const int layer_93_dw_filter_dim = 3;
 const int layer_93_dw_ifm_width = 14;
 //****************************
//****************************
 const int layer_94_pw_num_fils = 1024;
const int layer_94_pw_depth = 728;
const int layer_94_pw_filter_dim = 1;
 const int layer_94_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_96_pw_num_fils = 1024;
const int layer_96_pw_depth = 728;
const int layer_96_pw_filter_dim = 1;
 const int layer_96_pw_ifm_width = 14;
 //****************************
//****************************
 const int layer_98_dw_num_fils = 1024;
const int layer_98_dw_depth = 1024;
const int layer_98_dw_filter_dim = 3;
 const int layer_98_dw_ifm_width = 7;
 //****************************
//****************************
 const int layer_99_pw_num_fils = 1536;
const int layer_99_pw_depth = 1024;
const int layer_99_pw_filter_dim = 1;
 const int layer_99_pw_ifm_width = 7;
 //****************************
//****************************
 const int layer_100_dw_num_fils = 1536;
const int layer_100_dw_depth = 1536;
const int layer_100_dw_filter_dim = 3;
 const int layer_100_dw_ifm_width = 7;
 //****************************
//****************************
 const int layer_101_pw_num_fils = 2048;
const int layer_101_pw_depth = 1536;
const int layer_101_pw_filter_dim = 1;
 const int layer_101_pw_ifm_width = 7;
 //****************************
const pooling_layer_specs layer_102_avgpool_specs = {
                1.8902983942238927,//const pooling_fused_scales_dt fused_scale; 
                -128,//const biases_dt ifms_zero_point;
                -128,//const biases_dt ofms_zero_point;
                };
#endif
