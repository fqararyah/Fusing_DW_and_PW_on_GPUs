#include <cinttypes>
#include "cuda.h"
#include "cuda_runtime.h"
#include "../headers/dtype_defs.h"
#include "../headers/general_specs.h"
#include "../headers/parallalism_and_tiling.h"
#include "../headers/simulation_constants.h"
#include "../headers/common_funcs.h"
#include "../model_specs/layers_specs.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <bits/stdc++.h>

using namespace std;

#ifndef UTILS
#define UTILS

struct Settings_struct
{
    string ifms_file_name;
    string fusion_file;
    string dump_file_name;
    int test_iterations;
    int first_layer;
    int num_layers;
    int num_sms;
    bool run_fused;
    bool run_unfused;
    int bench;
};

struct Fusion_struct{
    int first_layer_index;
    int second_layer_index;
    fusion_types fusion_type;
};

fusion_types get_fusion_type(string s);

void dump_gpu_output_chw(string file_name, fms_dt *ofms_gpu, layer_specs layer_specs_struct);

void read_settings(string file_name, Settings_struct &settings_struct);

void read_fusions_list(string file_name, Fusion_struct *layers_fusions);

void read_ints_from_file(string file_name,
                         int ints[]);

bool compare_cpu_and_gpu_outputs(fms_dt *ofms_cpu, fms_dt *ofms_gpu, layer_specs layer_specs_struct);
bool compare_cpu_and_gpu_outputs_fw_hzw(fms_dt *ofms_cpu, fms_dt *ofms_gpu, layer_specs layer_specs_struct);
bool compare_cpu_and_gpu_outputs_fw_hwz(fms_dt *ofms_cpu, fms_dt *ofms_gpu, layer_specs layer_specs_struct);
bool compare_cpu_and_gpu_outputs_fhw_zhw(fms_dt *ofms_cpu, fms_dt *ofms_gpu, layer_specs layer_specs_struct);

void dump_outputs_hwz(string file_name, fms_dt *ofms, layer_specs layer_specs_struct);

string get_model_prefix();
void dump_cpu_output(string file_name, fms_dt ifms[MAX_FMS_SIZE],
                     const layer_specs layer_specs_struct);

void dump_gpu_output(string file_name, fms_dt *ofms_gpu, layer_specs layer_specs_struct);
void dump_gpu_output_fw(string file_name, fms_dt *ofms_gpu, layer_specs layer_specs_struct);

int get_conv_layer_index(const int starting_layer_index, const int offset);

int least_pow_of_2_geq(int inp);
int largest_pow_of_2_leq(int inp);

void get_pw_f_w_v2_parallelism_w( layer_specs *layer_specs_seq, int *parallelism_w, const int num_layers, bool fused);
void get_pw_f_w_v2_parallelism_w_v2(layer_specs *layer_specs_seq, int *layers_parallelism_w, const int num_layers, bool fused);

#endif