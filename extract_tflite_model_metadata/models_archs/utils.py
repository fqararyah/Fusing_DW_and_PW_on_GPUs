from multiprocessing.dummy import active_children
import sys
import pathlib
import json

current_dir = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(current_dir)
print(current_dir)
import classes

DELIMITER = '::'

NET_PREFIX = 'mob_v2'
NET_FULL_NAME = 'mobilenet_v2'
input_folder = current_dir + '/models_archs/models/'\
    + NET_FULL_NAME + '/'
IFMS_FILE = input_folder + 'layers_inputs.txt'
OFMS_FILE = input_folder + 'layers_outputs.txt'
LAYERS_TYPES_FILE = input_folder + 'layers_types.txt'
SECONDARY_LAYERS_TYPES_FILE = input_folder + 'secondary_layers_types.txt'
LAYERS_WEIGHTS_FILE = input_folder + 'layers_weights.txt'
LAYERS_STRIDES_FILE = input_folder + 'layers_strides.txt'
EXPANSION_PROJECTION_FILE = input_folder + 'expansion_projection.txt'
LAYERS_RELUS_FILE = input_folder + 'layers_relus.txt'
LAYERS_SKIP_CONNECTIONS_FILE = input_folder + 'skip_connections_indices.txt'
LAYERS_ACTIVATIONS_FILE = input_folder + 'layers_activations.txt'
LAYERS_EXECUTION_SEQUENCE = input_folder + 'layers_execution_sequence.txt'
MODEL_DAG_FILE = input_folder + 'model_dag.json'

def set_globals(prefix, full_name):
    global NET_PREFIX, NET_FULL_NAME, input_folder, IFMS_FILE, OFMS_FILE, LAYERS_TYPES_FILE, LAYERS_WEIGHTS_FILE, LAYERS_STRIDES_FILE, EXPANSION_PROJECTION_FILE, LAYERS_RELUS_FILE, LAYERS_SKIP_CONNECTIONS_FILE, SECONDARY_LAYERS_TYPES_FILE, LAYERS_ACTIVATIONS_FILE,\
        LAYERS_EXECUTION_SEQUENCE, MODEL_DAG_FILE
    NET_PREFIX = prefix
    NET_FULL_NAME = full_name
    input_folder = current_dir + '/models/'\
        + NET_FULL_NAME + '/'
    IFMS_FILE = input_folder + 'layers_inputs.txt'
    OFMS_FILE = input_folder + 'layers_outputs.txt'
    LAYERS_TYPES_FILE = input_folder + 'layers_types.txt'
    SECONDARY_LAYERS_TYPES_FILE = input_folder + 'secondary_layers_types.txt'
    LAYERS_WEIGHTS_FILE = input_folder + 'layers_weights.txt'
    LAYERS_STRIDES_FILE = input_folder + 'layers_strides.txt'
    EXPANSION_PROJECTION_FILE = input_folder + 'expansion_projection.txt'
    LAYERS_RELUS_FILE = input_folder + 'layers_relus.txt'
    LAYERS_ACTIVATIONS_FILE = input_folder + 'layers_activations.txt'
    LAYERS_SKIP_CONNECTIONS_FILE = input_folder + 'skip_connections_indices.txt'
    LAYERS_EXECUTION_SEQUENCE = input_folder + 'layers_execution_sequence.txt'
    MODEL_DAG_FILE = input_folder + 'model_dag.json'


def clean_line(line):
    return line.replace(' ', '').replace('\n', '')


def read_layers_input_shapes():
    layers_inputs = []
    with open(IFMS_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            splits = line.split('x')
            if len(splits) > 0:
                if len(splits) == 3:
                    layers_inputs.append(classes.feature_map(
                        int(splits[0]), int(splits[1]), int(splits[2])))
                elif len(splits) == 1:
                    layers_inputs.append(classes.feature_map(
                        int(splits[0]), 1, 1))

    return layers_inputs


def read_layers_output_shapes():
    layers_outputs = []
    with open(OFMS_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            splits = line.split('x')
            if len(splits) > 0:
                splits = line.split('x')
                if len(splits) == 3:
                    layers_outputs.append(classes.feature_map(
                        int(splits[0]), int(splits[1]), int(splits[2])))
                elif len(splits) == 1:
                    layers_outputs.append(classes.feature_map(
                        int(splits[0]), 1, 1))

    return layers_outputs


def read_layers_weight_shapes(layers_types):
    layers_weights = []
    count = 0
    with open(LAYERS_WEIGHTS_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            splits = line.split('x')
            if len(splits) > 0:
                if layers_types[count] == 'dw':
                    layers_weights.append(classes.weights(int(splits[0]), 1,
                                                          int(splits[1]) if len(splits) > 1 else 1, int(splits[2]) if len(splits) > 2 else 1))
                else:
                    layers_weights.append(classes.weights(int(splits[0]), int(splits[1]),
                                                          int(splits[2]) if len(splits) > 2 else 1, int(splits[3]) if len(splits) > 3 else 1))
                count += 1

    return layers_weights


def read_layers_strides():
    layers_strides = []
    with open(LAYERS_STRIDES_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            layers_strides.append(int(line))

    return layers_strides


def read_layers_types():
    layers_types = []
    with open(LAYERS_TYPES_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            layers_types.append(line)

    return layers_types


def read_secondary_layers_types():
    layers_types = []
    with open(SECONDARY_LAYERS_TYPES_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            layers_types.append(line)

    return layers_types


def read_expansion_projection():
    expansion_projection = []
    with open(EXPANSION_PROJECTION_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            expansion_projection.append(int(line))

    return expansion_projection


def read_layers_relus():
    layers_relus = []
    with open(LAYERS_RELUS_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            layers_relus.append(int(line))

    return layers_relus


def read_layers_activations():
    layers_activations = []
    with open(LAYERS_ACTIVATIONS_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            layers_activations.append(line.replace('\n', '').replace(' ', ''))

    return layers_activations


def read_skip_connections_indices():
    skip_connections_indices = {}
    with open(LAYERS_SKIP_CONNECTIONS_FILE, 'r') as f:
        for line in f:
            line = clean_line(line)
            line = clean_line(line)
            skip_connections_indices[int(line)] = 1

    return skip_connections_indices


def read_layers_execution_sequence():
    layers_execution_sequence = []
    with open(LAYERS_EXECUTION_SEQUENCE, 'r') as f:
        for line in f:
            layers_execution_sequence.append(
                line.replace('\n', '').replace(' ', ''))

    return layers_execution_sequence

def read_model_dag():
    f = open(MODEL_DAG_FILE)
    return json.load(f)

def read_model_dag_file(dag_file):
    f = open(dag_file)
    return json.load(f)

def get_layers_op_counts(model_dag):

    layers_num_of_ops = []
    for layer_specs in model_dag:
        if 'type' in layer_specs and layer_specs['type'] in ['s', 'dw', 'pw']:
            layers_ofms_shape = layer_specs['ofms_shape']
            layers_weights_shape = layer_specs['weights_shape']

            layer_num_of_ops = 1
            for i in layers_weights_shape:
                layer_num_of_ops *= i

            layer_num_of_ops *= layers_ofms_shape[1] * layers_ofms_shape[2]

            layers_num_of_ops.append(2 * layer_num_of_ops)

    return layers_num_of_ops

def get_dw_laye_op_counts(model_dag):

    layers_num_of_ops = []
    for layer_specs in model_dag:
        if 'type' in layer_specs and layer_specs['type'] in ['dw']:
            layers_ofms_shape = layer_specs['ofms_shape']
            layers_weights_shape = layer_specs['weights_shape']

            layer_num_of_ops = 1
            for i in layers_weights_shape:
                layer_num_of_ops *= i

            layer_num_of_ops *= layers_ofms_shape[1] * layers_ofms_shape[2]

            layers_num_of_ops.append(2 * layer_num_of_ops)

    return layers_num_of_ops