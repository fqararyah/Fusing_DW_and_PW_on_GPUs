#ifndef PARALLELISM_AND_TILING
#define PARALLELISM_AND_TILING

const int TILE_H = 2;
const int TILE_H_FW = 3;
const int F_W_TILE_H = 8;
const int F_W_TILE_F = 4;
const int F_W_PARALLEL_H = 4;
const int F_W_TILE_W = 8;
const int F_W_PARALLEL_W = 8;
const int PW_PW_MAX_FMS_BUFFER_SZ = 1024;
const int PW_PW_MAX_WEIGHTS_BUFFER_SZ = 6144 / PACKED_ITEMS;//384*64/4

#if USED_GPU == GTX1660
const int TILE_H_H_W = 16;
const int TILE_W_H_W = 16;
const int TILE_H_H_W_WIDE = 16;
const int TILE_W_H_W_WIDE = 16;
#elif USED_GPU == ORIN
const int TILE_H_H_W = 16;
const int TILE_W_H_W = 16;
const int TILE_H_H_W_WIDE = 16;
const int TILE_W_H_W_WIDE = 32;
#endif

const int TILE_HW_H_W = TILE_H_H_W * TILE_W_H_W;
const int TILE_HW_H_W_WIDE = TILE_H_H_W_WIDE * TILE_W_H_W_WIDE;
#if ((MODEL_ID == MOB_V2 || MODEL_ID == GPROX_3) && !COMPARE_WITH_CUDNN ) || DATA_TYPE == INT8_DTYPE
#define TILE_F_H_W 4
#define TILE_F_H_W_WIDE TILE_F_H_W
#else
#define TILE_F_H_W 8
#define TILE_F_H_W_WIDE 8
#endif

const int F_W_V2_TILE_F = 4;
const int MAX_STRIDES = 2;
const int MAX_PADDED_TILE_H = TILE_H_FW + 2 * MAX_PADDING;
const int TILE_W = 56;
const int MAX_PADDED_TILE_W = TILE_W + 2 * MAX_PADDING;
const int FW_MAX_PADDED_TILE_W = 128;
const int PARALLELISM_OFMS = 1;
const int PARALLELISM_W = 112;
const int PARALLELISM_IFMS = 1;
const int TILE_HW = TILE_H * TILE_W;

#endif