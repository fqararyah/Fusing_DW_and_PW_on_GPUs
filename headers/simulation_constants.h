
#ifndef SIMULATION_CONSTANTS
#define SIMULATION_CONSTANTS

#define COMPILE_FUSED 1
#define MOB_V1 1
#define MOB_V2 2
#define MOB_V2_0_5 25
#define MOB_V2_0_25 225
#define MOB_V2_0_75 275
#define MNAS 3
#define PROX 4
#define RESNET50 5
#define XCE_R 6
#define GPROX_3 7
#define MODEL_ID MOB_V1

#define HWC 0
#define HCW 1
#define CHW 2
#define COMPARE_WITH_CUDNN 0
#define DATA_LAYOUT CHW

#define WEIGHTS_SHARED_WITH_GPU 1
#define PADDED_DW_WEIGHTS 1

#define DSC_MODEL (MODEL_ID == MOB_V1 || MODEL_ID == MOB_V2 || MODEL_ID == XCE_R || GPROX_3) && !COMPARE_WITH_CUDNN

#define NOT_FUSED 0
#define FUSED_F_W 1
#define FUSED_H_W 2
#define ALL_MODES (NOT_FUSED + FUSED_F_W + FUSED_H_W)
#define FUSION_MODE ALL_MODES

#define FW 1

#define WARMUP_ITERATIONS 400

#define WARP_SIZE 32

#define TIME_LAYER_BY_LAYER 0

#define MAX_THREADS_PER_BLOCK 1024

#define INT8_DTYPE 1
#define FLOAT_DTYPE 0
#define MIXED_LAYOUT 0

#if COMPARE_WITH_CUDNN
#define DUMMY_SCALE 1.0
#define DWDUMMY_SCALE 1.0
#define DUMMY_BIAS 0
#else
#define DUMMY_SCALE 0.001
#define DWDUMMY_SCALE 0.001
#define DUMMY_BIAS 1.0
#endif

#define DATA_TYPE INT8_DTYPE

#define GTX1660 0
#define ORIN 1
#define NANO 2

#define USED_GPU GTX1660

#endif