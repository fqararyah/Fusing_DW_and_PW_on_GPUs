import json
from operator import mod
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras.applications.mobilenet_v2 as mob_v2
import tensorflow.keras.applications.mobilenet as mob_v1
import tensorflow.keras.applications as models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import pathlib
from models_archs import utils
import tflite_ops_names
import extraction_constants as ec


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#################################################################################################################
ACTIVATION_FUNCTIONS = ['relu6', 'relu']
np.random.seed(0)

weights_fms_dir = ec.MODEL_NAME
model_arch_dir = './models_archs/models/' + ec.MODEL_NAME + '/'
tflite_models_dir = pathlib.Path("./")
tflite_model_quant_file = tflite_models_dir / \
    (ec.MODEL_NAME + '_' + str(ec.PRECISION) + ".tflite")

if not os.path.exists(tflite_model_quant_file):
    import run_and_save_model

#################################################################################################################
interpreter = tf.lite.Interpreter(model_path=str(
    tflite_model_quant_file), experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
ops_details_list = interpreter._get_ops_details()
tensors_details_list = interpreter.get_tensor_details()
#################################################################################################################
# prepare image
test_image = './ILSVRC2012_val_00000932.JPEG'
#'/media/SSD2TB/shared/vedliot_evaluation/D3.3_Accuracy_Evaluation/imagenet/imagenet_val2012/ILSVRC2012_val_00018455.JPEG'
a_test_image = load_img(test_image, target_size=(224, 224))

if input_details['dtype'] == np.uint8:
    numpy_image = img_to_array(a_test_image, dtype=np.uint8)
elif input_details['dtype'] == np.int8:
    numpy_image = img_to_array(a_test_image, dtype=np.int8)
image_batch = np.expand_dims(numpy_image, axis=0)
#################################################################################################################
# invoke mode
interpreter.set_tensor(input_details["index"], image_batch)
interpreter.invoke()

tensor_details = interpreter.get_tensor_details()
#################################################################################################################

model_dag = []
tmp_ofms_to_layer_indeices_map = {}

op_index_comp = 0
for op_details in ops_details_list:
    model_dag_entry = {}
    op_name = op_details['op_name'].lower()
    model_dag_entry['name'] = op_name
    op_index = op_details['index']
    print('processing layer:', op_index)
    assert op_index == op_index_comp
    model_dag_entry['id'] = op_index
    op_index_comp += 1
    op_inputs = op_details['inputs']
    op_outputs = op_details['outputs']
    op_inputs = sorted(op_inputs)

    op_ofms_tensor_details = tensors_details_list[op_outputs[0]]
    op_ofms_tensor = interpreter.get_tensor(op_outputs[0])
    op_ofms_tensor = np.squeeze(op_ofms_tensor)

    for activation in ACTIVATION_FUNCTIONS:
        if activation in op_ofms_tensor_details['name'].lower():
            model_dag_entry['activation'] = activation.upper()
            break
        else:
            model_dag_entry['activation'] = '0'

    if op_ofms_tensor.ndim == 3:
        op_ofms_tensor = np.transpose(op_ofms_tensor, (2, 0, 1))
    op_ofms_tensor = np.reshape(op_ofms_tensor, (op_ofms_tensor.size))
    file_name = 'ofms_' + str(op_index) + '.txt'
    directory = './'+weights_fms_dir+'/fms/' 
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        os.makedirs(directory)
    np.savetxt(directory +
               file_name, op_ofms_tensor, fmt='%i')

    tmp_ofms_to_layer_indeices_map[op_outputs[0]] = op_index
    model_dag_entry['parents'] = []
    model_dag_entry['children'] = []
    for op_input in op_inputs:
        if op_input in tmp_ofms_to_layer_indeices_map:
            parent_index = tmp_ofms_to_layer_indeices_map[op_input]
            model_dag_entry['parents'].append(parent_index)
            model_dag[parent_index]['children'].append(op_index)

    if op_name == 'add':
        model_dag_entry['ifms_scales'] = []
        model_dag_entry['ifms_zero_points'] = []
        for op_input in op_inputs:
            op_ifms_tensor_details = tensors_details_list[op_input]
            assert(
                len(op_ifms_tensor_details['quantization_parameters']['scales']) == 1)
            model_dag_entry['ifms_scales'].append(
                float(op_ifms_tensor_details['quantization_parameters']['scales'][0]))
            assert(
                len(op_ifms_tensor_details['quantization_parameters']['zero_points']) == 1)
            model_dag_entry['ifms_zero_points'].append(
                int(op_ifms_tensor_details['quantization_parameters']['zero_points'][0]))

    if op_name in tflite_ops_names.TFLITE_CONV_OP_NAMES or op_name in tflite_ops_names.TFLITE_FULLY_CONNECTED_OP_NAMES:
        # assuming the op_inputs are of the weights, then the biases, theen the IFMs (based on my observation)
        if op_inputs[0] < 0 or op_inputs[1] < 0 or op_inputs[-1] < 0:
              print(op_name, 'has missing tensors')
        else:
            op_ifms_tensor = interpreter.get_tensor(op_inputs[-1]) #inputs tensor
            op_ifms_tensor_details = tensors_details_list[op_inputs[-1]]
            op_weights_tensor = interpreter.get_tensor(op_inputs[0]) #weights tensor
            op_weights_tensor_details = tensors_details_list[op_inputs[0]]
            op_biases_tensor = interpreter.get_tensor(op_inputs[1])#biases tensor
            op_biases_tensor_details = tensors_details_list[op_inputs[1]]

            if len(op_biases_tensor.shape) > 1 and len(op_weights_tensor.shape) == 1:
                op_weights_tensor = op_biases_tensor
                op_weights_tensor_details = op_biases_tensor_details
                op_biases_tensor = interpreter.get_tensor(op_inputs[0])
                op_biases_tensor_details = tensors_details_list[op_inputs[0]]
            #print(op_ifms_tensor_details['quantization_parameters']['scales'])
            assert(
                len(op_ifms_tensor_details['quantization_parameters']['scales']) == 1)
            model_dag_entry['ifms_scales'] = float(
                op_ifms_tensor_details['quantization_parameters']['scales'][0])
            assert(
                len(op_ifms_tensor_details['quantization_parameters']['zero_points']) == 1)
            model_dag_entry['ifms_zero_points'] = int(
                op_ifms_tensor_details['quantization_parameters']['zero_points'][0])

            op_ifms_tensor = np.squeeze(op_ifms_tensor)
            if op_name in tflite_ops_names.TFLITE_CONV_OP_NAMES and op_ifms_tensor.ndim == 3:
                op_ifms_tensor = np.transpose(op_ifms_tensor, (2, 0, 1))
            op_ifms_tensor = np.reshape(op_ifms_tensor, (op_ifms_tensor.size))
            file_name = 'ifms_' + str(op_index) + '.txt'
            np.savetxt('./'+weights_fms_dir+'/fms/' +
                    file_name, op_ifms_tensor, fmt='%i')

            op_weights_tensor = np.squeeze(op_weights_tensor)
            if op_weights_tensor.ndim == 4:
                op_weights_tensor = np.transpose(op_weights_tensor, (0, 3, 1, 2))
            elif op_weights_tensor.ndim == 3:
                op_weights_tensor = np.transpose(op_weights_tensor, (2, 0, 1))
            op_weights_tensor = np.reshape(
                op_weights_tensor, (op_weights_tensor.size))
            file_name = 'weights_' + str(op_index)

            directory = './'+weights_fms_dir+'/weights/' 
            if not os.path.exists(directory):
                # If it doesn't exist, create it
                os.makedirs(directory)
            np.savetxt(directory + file_name +
                    '.txt', op_weights_tensor, fmt='%i')

            op_weights_scales_tensor = op_weights_tensor_details['quantization_parameters']['scales']
            np.savetxt(directory + file_name +
                    '_scales.txt', op_weights_scales_tensor)

            op_weights_zero_pooints_tensor = op_weights_tensor_details[
                'quantization_parameters']['zero_points']
            np.savetxt('./'+weights_fms_dir+'/weights/' + file_name +
                    '_zps.txt', op_weights_zero_pooints_tensor, fmt='%i')

            directory = './'+weights_fms_dir+'/biases/' 
            if not os.path.exists(directory):
                # If it doesn't exist, create it
                os.makedirs(directory)
            file_name = 'biases_' + str(op_index) + '.txt'
            np.savetxt(directory +
                    file_name, op_biases_tensor, fmt='%i')

            op_biases_scales_tensor = op_biases_tensor_details['quantization_parameters']['scales']
            np.savetxt('./'+weights_fms_dir+'/biases/' + file_name +
                    '_scales.txt', op_biases_scales_tensor)

            op_biases_zero_points_tensor = op_biases_tensor_details[
                'quantization_parameters']['zero_points']
            np.savetxt('./'+weights_fms_dir+'/biases/' + file_name +
                    '_zps.txt', op_biases_zero_points_tensor, fmt='%i')

            op_weights_shape = [int(i) for i in op_weights_tensor_details['shape']]

            if 'depthwise' in op_name:
                model_dag_entry['type'] = 'dw'
                model_dag_entry['weights_shape'] = [
                    op_weights_shape[3], op_weights_shape[1], op_weights_shape[2]]
            elif  op_name in tflite_ops_names.TFLITE_FULLY_CONNECTED_OP_NAMES:
                model_dag_entry['type'] = 'fc'
            elif op_weights_shape[1] == 1 and op_weights_shape[2] == 1:
                model_dag_entry['type'] = 'pw'
                model_dag_entry['weights_shape'] = [
                    op_weights_shape[0], op_weights_shape[3]]
            elif len(op_weights_shape) == 4:
                model_dag_entry['type'] = 's'
                model_dag_entry['weights_shape'] = [
                    op_weights_shape[0], op_weights_shape[3], op_weights_shape[1], op_weights_shape[2]]
            else:
                model_dag_entry['weights_shape'] = [i for i in op_weights_shape]

            if len(op_ifms_tensor_details['shape']) == 4:
                model_dag_entry['ifms_shape'] = [int(op_ifms_tensor_details['shape'][3]), int(op_ifms_tensor_details['shape'][1]),
                                                int(op_ifms_tensor_details['shape'][2])]
                model_dag_entry['ofms_shape'] = [int(op_ofms_tensor_details['shape'][3]), int(op_ofms_tensor_details['shape'][1]),
                                                int(op_ofms_tensor_details['shape'][2])]
            else:
                model_dag_entry['ifms_shape'] = [
                    int(i) for i in op_ifms_tensor_details['shape']]
                model_dag_entry['ofms_shape'] = [
                    int(i) for i in op_ofms_tensor_details['shape']]

            model_dag_entry['strides'] = int(
                model_dag_entry['ifms_shape'][-1] / model_dag_entry['ofms_shape'][-1])

    else:
        if op_name == 'pad':
            for input_index in op_inputs:
                input_details = tensors_details_list[input_index]
                if 'paddings' in input_details['name']:
                    paddings = interpreter.get_tensor(input_index)
                    padding_t_b = paddings[1]
                    padding_l_r = paddings[2]
                    #print(padding_t_b[0], padding_t_b[1], padding_l_r[0], padding_l_r[1])
                    model_dag_entry['padding_top'] = int(padding_t_b[0])
                    model_dag_entry['padding_bottom'] = int(padding_t_b[1])
                    model_dag_entry['padding_left'] = int(padding_l_r[0])
                    model_dag_entry['padding_right'] = int(padding_l_r[1])

        op_ifms_tensor_details = tensors_details_list[op_input]
        # assert(
        if len(op_ifms_tensor_details['quantization_parameters']['scales']) == 1:
            model_dag_entry['ifms_scales'] = float(
                op_ifms_tensor_details['quantization_parameters']['scales'][0])
        else:
            model_dag_entry['ifms_scales'] = 1.0

        if len(op_ifms_tensor_details['quantization_parameters']['zero_points']) == 1:
            model_dag_entry['ifms_zero_points'] = int(
                op_ifms_tensor_details['quantization_parameters']['zero_points'][0])
        else:
            model_dag_entry['ifms_zero_points'] = 0
            
        op_ifms_tensor_details = tensors_details_list[op_inputs[-1]]
        model_dag_entry['ifms_shape'] = [
            int(i) for i in op_ifms_tensor_details['shape']]
        model_dag_entry['ofms_shape'] = [
            int(i) for i in op_ofms_tensor_details['shape']]
        
        op_ifms_tensor = interpreter.get_tensor(op_inputs[-1])
        op_ifms_tensor = np.squeeze(op_ifms_tensor)
        if op_name in tflite_ops_names.TFLITE_CONV_OP_NAMES and op_ifms_tensor.ndim == 3:
            op_ifms_tensor = np.transpose(op_ifms_tensor, (2, 0, 1))
        op_ifms_tensor = np.reshape(op_ifms_tensor, (op_ifms_tensor.size))
        file_name = 'ifms_' + str(op_index) + '.txt'
        np.savetxt('./'+weights_fms_dir+'/fms/' +
                file_name, op_ifms_tensor, fmt='%i')

    assert(
        len(op_ofms_tensor_details['quantization_parameters']['scales']) <= 1)
    if len(op_ofms_tensor_details['quantization_parameters']['scales']) == 1:
        model_dag_entry['ofms_scales'] = float(
            op_ofms_tensor_details['quantization_parameters']['scales'][0])
    assert(
        len(op_ofms_tensor_details['quantization_parameters']['zero_points']) <= 1)
    if len(op_ofms_tensor_details['quantization_parameters']['zero_points']) == 1:
        model_dag_entry['ofms_zero_points'] = int(
            op_ofms_tensor_details['quantization_parameters']['zero_points'][0])

    if op_name in tflite_ops_names.TFLITE_AVG_POOL_OP_NAMES:
        model_dag_entry['type'] = 'avgpool'

    model_dag.append(model_dag_entry)

json_object = json.dumps(model_dag)

if not os.path.exists(model_arch_dir):
    # If it doesn't exist, create it
    os.makedirs(model_arch_dir)

with open(model_arch_dir + "model_dag.json", "w") as outfile:
    outfile.write(json_object)
