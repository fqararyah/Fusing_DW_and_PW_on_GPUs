import _constants
import json
import os
import math

def get_my_path():
    return str(os.path.dirname(__file__))

settings = {}

def is_pow_of_2(n):
    return (n & (n-1) == 0) and n != 0

def least_pow_of_2_geq(num):
    return int( math.pow( 2, math.ceil(math.log(num, 2)) ) )

def pow_of_2_leq(num):
    return int( math.pow( 2, math.floor(math.log(num, 2)) ) )

def get_last_conv_layer(model_dag):

    for i in range(len(model_dag) - 1, 0, -1):
        layer_specs = model_dag[i]
        if is_conv_layer(layer_specs):
            return layer_specs

def get_num_conv_layers(model_dag):

    num_conv_layers = 0

    for i in range(len(model_dag)):
        layer_specs = model_dag[i]
        if is_conv_layer(layer_specs):
            num_conv_layers += 1

    return num_conv_layers

def get_num_conv_layers_in_range(model_dag, start_layer, end_layer):

    num_conv_layers = 0
    for layer_index in range(start_layer, end_layer):
        layer_specs = model_dag[layer_index]
        if is_conv_layer(layer_specs):
            num_conv_layers += 1

    return num_conv_layers

def get_conv_layer_index_from_offset(model_dag, anchor_layer_index, layer_offset):

    num_conv_layers = 0
    layer_index = anchor_layer_index + 1
    while num_conv_layers < layer_offset and layer_index < len(model_dag):
        layer_specs = model_dag[layer_index]
        if is_conv_layer(layer_specs):
            num_conv_layers += 1

        layer_index += 1

    if num_conv_layers != layer_offset:
        return -1
    
    return layer_index - 1

def is_conv_layer(layer_specs):
    return 'type' in layer_specs and layer_specs['type'] in ['s', 'pw', 'dw']

def is_dw_conv_layer(layer_specs):
    return 'type' in layer_specs and layer_specs['type'] in ['dw']

def is_s_conv_layer(layer_specs):
    return 'type' in layer_specs and layer_specs['type'] in ['s']

def is_pw_conv_layer(layer_specs):
    return 'type' in layer_specs and layer_specs['type'] in ['pw']

def is_fc_layer(layer_specs):
    return 'type' in layer_specs and layer_specs['type'] in ['fc']

def is_add_layer(layer_specs):
    return 'add' in layer_specs['name']


def calc_fms_size(fms_shape):
    size = 1
    for i in fms_shape:
        size *= i

    return size


def get_max_ifms_and_ofms_after_pipe(model_dag, starting_layer, pipeline_len):

    end_layer = starting_layer + pipeline_len
    current_layer = starting_layer
    layer_index = starting_layer
    max_ifms = 0
    max_ofms = 0
    while current_layer < end_layer:
        layer_specs = model_dag[layer_index]
        if is_conv_layer(layer_specs):
            current_layer += 1
        layer_index += 1
    
    for i in range(layer_index, len(model_dag)):
        layer_specs = model_dag[i]
        if not is_conv_layer(layer_specs):
            continue

        ifms_shape = layer_specs['ifms_shape']
        ofms_shape = layer_specs['ofms_shape']
        ifms_size = ifms_shape[0] * ifms_shape[1] * ifms_shape[2]
        ofms_size = ofms_shape[0] * ofms_shape[1] * ofms_shape[2]

        max_ifms = max(max_ifms, ifms_size)
        max_ofms = max(max_ofms, ofms_size)

    return max_ifms, max_ofms


def parse_line(line, seperator='::'):
    key_val = line.replace(' ', '').replace('\n', '').split(seperator)
    if len(key_val) < 2:
        return '', ''
    
    if '*M*' in key_val[1]:
        key_val[1] = key_val[1].replace('*M*', settings['PYTHON_MODEL_NAME'])
    return key_val[0], key_val[1]


def read_settings():
    if(len(settings) == 0):
        with open(_constants.SETTINGS_FILE_PATH, 'r') as f:
            for line in f:
                key, val = parse_line(line)
                if key != '':
                    settings[key.upper()] = val


def read_model_dag():
    read_settings()
    f = open(settings['MODEL_DAG_FILE'])
    return json.load(f)


def read_hw_config():
    read_settings()
    hw_configs = {}
    hw_configs_file_name = ''
    if os.path.exists(settings['HW_CONFIG_FILE']):
        hw_configs_file_name = settings['HW_CONFIG_FILE']
    elif os.path.exists(get_my_path() + '/' + settings['HW_CONFIG_FILE']):
        hw_configs_file_name = get_my_path() + '/' + settings['HW_CONFIG_FILE']
    else:
        print(get_my_path() + '/' + settings['HW_CONFIG_FILE'], 'HW CONFIG DOES NOT EXITS!!')
        exit()

    with open(hw_configs_file_name, 'r') as f:
        for line in f:
            key, val = parse_line(line)
            if key != '':
                hw_configs[key.lower()] = val

    return hw_configs

def get_layers_weights_sizes_start_to_end_layers(model_dag, start_layer, end_layer):

    layers_weight_sizes = {}
    for layer_index in range(start_layer, end_layer):
        layer_specs = model_dag[layer_index]
        if is_conv_layer(layer_specs):
            weights_size = 1
            weights_shape = layer_specs['weights_shape']
            for i in range(len(weights_shape)):
                weights_size *= weights_shape[i]

            layers_weight_sizes[layer_index] = weights_size

    return layers_weight_sizes

def get_layer_ifms_depth(layer_specs):
    return layer_specs['ifms_shape'][0]

def get_layer_ifms_height(layer_specs):
    return layer_specs['ifms_shape'][1]

def get_layer_ifms_width(layer_specs):
    return layer_specs['ifms_shape'][2]

def get_layer_ofms_depth(layer_specs):
    return layer_specs['ofms_shape'][0]

def get_layer_ofms_height(layer_specs):
    return layer_specs['ofms_shape'][1]

def get_layer_ofms_width(layer_specs):
    return layer_specs['ofms_shape'][2]

def get_layer_ifms_size(layer_specs):
    ifms_shape = layer_specs['ifms_shape']
    return ifms_shape[0] * ifms_shape[1] * ifms_shape[2]

def get_layer_ofms_size(layer_specs):
    ifms_shape = layer_specs['ofms_shape']
    return ifms_shape[0] * ifms_shape[1] * ifms_shape[2]

def get_layer_num_filters(layer_specs):
    if is_dw_conv_layer(layer_specs):
        return 1
    
    return layer_specs['weights_shape'][0]

def get_layer_filter_size(layer_specs):
    weights_shape = layer_specs['weights_shape']
    filter_size = 1
    if is_pw_conv_layer(layer_specs):
        filter_size = weights_shape[-1]
    elif is_dw_conv_layer(layer_specs):
        filter_size *= weights_shape[-2]
    elif is_s_conv_layer(layer_specs):
        filter_size *= weights_shape[-3]
    
    return filter_size

def get_layer_weights_size(layer_specs):
    weights_shape = layer_specs['weights_shape']
    weights_size = 1
    for i in weights_shape:
        weights_size *= i
    return weights_size
        
def get_layer_ifms_size_index(model_dag, layer_index):
    ifms_shape = model_dag[layer_index]['ifms_shape']
    return ifms_shape[0] * ifms_shape[1] * ifms_shape[2]

def get_layer_ofms_size_index(model_dag, layer_index):
    ofms_shape = model_dag[layer_index]['ofms_shape']
    return ofms_shape[0] * ofms_shape[1] * ofms_shape[2]

def get_layer_weights_size_index(model_dag, layer_index):
    weights_shape = model_dag[layer_index]['weights_shape']
    weights_size = 1
    for i in weights_shape:
        weights_size *= i

    return weights_size

def get_layer_ifms_ofms_and_weights_sizes(model_dag, layer_index):
    return [get_layer_ifms_size(model_dag, layer_index), \
        get_layer_ofms_size(model_dag, layer_index), \
        get_layer_weights_size(model_dag, layer_index)]

def get_conv_siblings(model_dag, layer_specs):
    conv_siblings = []
    assert(len(layer_specs['parents']) == 1)
    subling_layers_indices = model_dag[layer_specs['parents'][0]]['children']
    for sibling_layer_indix in subling_layers_indices:
        if is_conv_layer(model_dag[sibling_layer_indix]):
            conv_siblings.append(sibling_layer_indix)
    
    return conv_siblings
        
def is_last_child(model_dag, layer_specs):
    layer_children = layer_specs['children']
    for child_index in layer_children:
        if max(model_dag[child_index]['parents']) != layer_specs['id']:
            return False
    return True

def has_a_distant_child(model_dag, layer_specs):
    layer_children = layer_specs['children']
    for child_index in layer_children:
        if child_index > layer_specs['id'] + len(layer_children):
            return True
    return False

def has_a_distant_parent(model_dag, layer_specs):
    layer_parent = layer_specs['parents']
    for parent_index in layer_parent:
        if parent_index < layer_specs['id'] - len(layer_parent):
            return True
    return False

def find_non_conv_child(model_dag, layer_specs):
    layer_children = layer_specs['children']
    for child_index in layer_children:
        child = model_dag[child_index]
        if not is_conv_layer(child):
            return child_index
    return -1

def fused_add_layer_index(model_dag, layer_specs):
    layer_children = layer_specs['children']
    for child_index in layer_children:
        child = model_dag[child_index]
        if is_add_layer(child):
            return child_index
    return -1

def is_there_a_cycly_in_range_execlude_last(model_dag, first_layer_index, last_layer_index):
    for layer_index in range(first_layer_index, last_layer_index):
        layer_specs = model_dag[layer_index]
        layer_children = layer_specs['children']
        if len(layer_children) > 1:
            for child_index in layer_children:
                if child_index < last_layer_index:
                    return True
    return False

def get_fc_weights(model_dag):
    for layer_index in range(len(model_dag)):
        layer_specs = model_dag[layer_index]
        if is_fc_layer(layer_specs):
            ifms_rows = layer_specs['ifms_shape'][0]
            ofms_cols = layer_specs['ofms_shape'][-1]
            return ifms_rows * ofms_cols
    return 0