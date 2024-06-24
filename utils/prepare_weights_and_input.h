#include "../headers/dtype_defs.h"
#include "../headers/general_specs.h"
#include "../headers/parallalism_and_tiling.h"
#include "../headers/common_funcs.h"
#include "utils.h"

#include <fstream>
#include <iostream>
#include <cassert>

using namespace std;

void fill_layer_input(string file_name, fms_dt layer_input[MAX_FMS_SIZE],
                      const layer_specs layer_specs_struct);

void fill_layer_input_fw_hzw(string file_name, fms_dt layer_input[MAX_FMS_SIZE],
                             const layer_specs layer_specs_struct);

void fill_layer_input_fw_hwz(string file_name, fms_dt layer_input[MAX_FMS_SIZE],
                             const layer_specs layer_specs_struct);

void fill_layer_input_fwh_zhw(string file_name, fms_dt layer_input[MAX_FMS_SIZE_PACKED],
                              const layer_specs layer_specs_struct); // fhw mewans full height width

void verify_fill_layer_input(string file_name, fms_dt ifms[MAX_FMS_SIZE],
                             const layer_specs layer_specs_struct);

void verify_fill_layer_input_fw(string file_name, fms_dt ifms[MAX_FMS_SIZE],
                                const layer_specs layer_specs_struct);

int count_weights(string file_name);

void load_weights(string file_name,
                  weights_dt weights[], layer_specs layer_specs_seq[]);

void transform_pw_weights_fc_to_cf(weights_dt *weights, layer_specs l_specs);
void transform_dw_weights_fc_to_cf(weights_dt *weights, layer_specs l_specs);
void transform_pw_layers_weights(weights_dt *weights, layer_specs layer_specs_seq[]);
void transform_dw_layers_weights(weights_dt *weights, layer_specs layer_specs_seq[]);

void load_dw_weights(string file_name,
                     weights_dt weights[]);

void load_dw_weights_padded_filters(string file_name,
                                    weights_dt weights[],
                                    layer_specs layer_specs_seq[]);

void load_dw_weights_cpu(string file_name,
                         weights_dt weights[]);

void load_dw_weights_cpu_padded(string file_name,
                                weights_dt weights[], layer_specs layer_specs_seq[]);

void verify_load_weights(string file_name, weights_dt weights[], const int num_of_weights);

void fill_layer_input_cpu(string file_name, fms_dt layer_input[MAX_FMS_SIZE],
                          const layer_specs layer_specs_struct);

void load_weights_cpu(string file_name,
                      weights_dt weights[]);

void load_scales(string file_name, fused_scales_dt fused_scales[]);

void load_zps(string file_name,
              biases_dt fused_zps[]);

int read_count_from_a_file(string file_name);
