#include "general_specs.h"
#include "utils/utils.h"

void cpu_add(fms_dt *src, fms_dt *dst, layer_specs conv_l_specs);
void cpu_avgpool_all_hw(fms_dt *fms, const pooling_layer_specs layer_specs_struct);

void gpu_add(fms_dt *src, fms_dt *dst, layer_specs conv_l_specs, Settings_struct settings);
void gpu_avgpool_all_hw(fms_dt *fms, const pooling_layer_specs layer_specs_struct);