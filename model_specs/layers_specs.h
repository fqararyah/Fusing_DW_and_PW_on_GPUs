#include "../headers/simulation_constants.h"
#include "../headers/general_specs.h"
#include "../headers/parallalism_and_tiling.h"

#if MODEL_ID == MOB_V1
#include "mob_v1_layers_specs.h"
#elif MODEL_ID == MOB_V2
#include "mob_v2_layers_specs.h"
#elif MODEL_ID == RESNET50
#include "resnet50_layers_specs.h"
#elif MODEL_ID == XCE_R
#include "xce_r_layers_specs.h"
#elif MODEL_ID == GPROX_3
#include "gprox_3_layers_specs.h"
#endif

void layer_specs_init(layer_specs *layer_specs_seq,  pooling_layer_specs * pooling_layer_specs_seq);