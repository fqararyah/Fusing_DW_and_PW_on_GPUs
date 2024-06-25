#import estimator as est
import estimator_v2 as est
import sys

sys.path.append('/home/fareed/wd/hybrid_strategy/')
import utils

COMP_TO_DM_ACCEPTED_RATIO = 10  # TODO


def layer_by_layer_dm_estimation(model_dag, hw_configs, layer_by_layer_dms):
    i = 0
    for layer_specs in model_dag:
        if utils.is_conv_layer(layer_specs):
            layer_by_layer_dms[i] = est.estimate_dm([layer_specs], hw_configs)
            # if layer_by_layer_dms[i] != -1:
            #     print(layer_specs['type'], layer_by_layer_dms[i].total_dm, layer_by_layer_dms[i].l1_tile_filters,
            #           layer_by_layer_dms[i].l1_tile_h, layer_by_layer_dms[i].l1_tile_w)

        i += 1


def pw_dw_dm_estimation(model_dag, hw_configs, fused_dw_pw_layer_dms):
    i = 0
    for layer_specs in model_dag:
        if utils.is_pw_conv_layer(layer_specs):
            next_conv_layer_index = utils.get_conv_layer_index_from_offset(
                model_dag, i, 1)
            if next_conv_layer_index != -1:
                next_layer_specs = model_dag[next_conv_layer_index]
                if utils.is_dw_conv_layer(next_layer_specs):
                    fused_dw_pw_layer_dms[i] = est.estimate_dm(
                        [layer_specs, next_layer_specs], hw_configs)
                    
                    # if fused_dw_pw_layer_dms[i] != -1:
                    #     print(layer_specs['type'], fused_dw_pw_layer_dms[i].total_dm / 1000000, fused_dw_pw_layer_dms[i].l1_tile_filters,
                    #     fused_dw_pw_layer_dms[i].l1_tile_h, fused_dw_pw_layer_dms[i].l1_tile_w)

        i += 1

def dw_pw_dm_estimation(model_dag, hw_configs, fused_dw_pw_layer_dms):
    i = 0
    for layer_specs in model_dag:
        if utils.is_dw_conv_layer(layer_specs):
            next_conv_layer_index = utils.get_conv_layer_index_from_offset(
                model_dag, i, 1)
            if next_conv_layer_index != -1:
                next_layer_specs = model_dag[next_conv_layer_index]
                if utils.is_pw_conv_layer(next_layer_specs):
                    fused_dw_pw_layer_dms[i] = est.estimate_dm(
                        [layer_specs, next_layer_specs], hw_configs)
                    
                    # if fused_dw_pw_layer_dms[i] != -1:
                    #     print(layer_specs['type'], fused_dw_pw_layer_dms[i].total_dm / 1000000, fused_dw_pw_layer_dms[i].l1_tile_filters,
                    #     fused_dw_pw_layer_dms[i].l1_tile_h, fused_dw_pw_layer_dms[i].l1_tile_w)

        i += 1

def pw_pw_dm_estimation(model_dag, hw_configs, fused_pw_pw_layer_dms):
    i = 0
    for layer_specs in model_dag:
        if utils.is_pw_conv_layer(layer_specs):
            next_conv_layer_index = utils.get_conv_layer_index_from_offset(
                model_dag, i, 1)
            if next_conv_layer_index != -1:
                next_layer_specs = model_dag[next_conv_layer_index]
                if utils.is_pw_conv_layer(next_layer_specs):
                    fused_pw_pw_layer_dms[i] = est.estimate_dm(
                        [layer_specs, next_layer_specs], hw_configs)
                    
                    # if fused_dw_pw_layer_dms[i] != -1:
                    #     print(layer_specs['type'], fused_dw_pw_layer_dms[i].total_dm / 1000000, fused_dw_pw_layer_dms[i].l1_tile_filters,
                    #     fused_dw_pw_layer_dms[i].l1_tile_h, fused_dw_pw_layer_dms[i].l1_tile_w)

        i += 1
        
def to_fuse_or_not_to(layer_by_layer_dms, fused_layer_dms, fusion_dict):

    for layer_index in range(len(layer_by_layer_dms)):
        fusion_dict[layer_index] = None

        if layer_index in fused_layer_dms:
            layer_0 = layer_index
            fusion_and_splitting_info = fused_layer_dms[layer_0]
            layer_1 = fusion_and_splitting_info.fused_with
            if layer_1 == -1:
                continue

            avoided_dm = ((layer_by_layer_dms[layer_0].total_dm +
                           layer_by_layer_dms[layer_1].total_dm) - fusion_and_splitting_info.total_dm)

            redundant_comp_to_saved_dm_ratio = fusion_and_splitting_info.redundant_comp / avoided_dm

            if avoided_dm > 0 and redundant_comp_to_saved_dm_ratio < COMP_TO_DM_ACCEPTED_RATIO:
                fusion_dict[layer_0] = layer_1
                fused_layer_dms[layer_0].saved_dm = avoided_dm
                #print(avoided_dm)


def build_cnn(model_dag, hw_configs):
    layer_by_layer_dms = [None] * len(model_dag)
    fused_pw_dw_layer_dms = {}
    fused_pw_dw_layer_dms_rev = {}
    pwdw_fusion_dict = {}
    pwdw_fusion_dict_rev = {}

    layer_by_layer_dm_estimation(model_dag, hw_configs, layer_by_layer_dms)
    pw_dw_dm_estimation(model_dag, hw_configs, fused_pw_dw_layer_dms)
    to_fuse_or_not_to(layer_by_layer_dms, fused_pw_dw_layer_dms, pwdw_fusion_dict)

    print('**********pwdw***********')
    fused_layers = {}
    for layer_0, layer_1 in pwdw_fusion_dict.items():
        if layer_1 != None:
            print('layers:', layer_0, layer_1)
            pwdw_fusion_dict_rev[layer_1] = layer_0
            fused_layers[layer_0] = 1
            fused_layers[layer_1] = 1
            fused_pw_dw_layer_dms_rev[layer_1] = fused_pw_dw_layer_dms[layer_0]
            print('tile dims:', fused_pw_dw_layer_dms[layer_0].l1_tile_filters,
                   fused_pw_dw_layer_dms[layer_0].l1_tile_h, fused_pw_dw_layer_dms[layer_0].l1_tile_w)
    ###################################dwpw##########################################
    fused_dw_pw_layer_dms = {}
    fused_dw_pw_layer_dms_rev = {}
    dwpw_fusion_dict = {}  
    dwpw_fusion_dict_rev = {}  
    dw_pw_dm_estimation(model_dag, hw_configs, fused_dw_pw_layer_dms)
    to_fuse_or_not_to(layer_by_layer_dms, fused_dw_pw_layer_dms, dwpw_fusion_dict)
    
    print('**********dwpw***********')
    for layer_0, layer_1 in dwpw_fusion_dict.items():
        if layer_1 != None:
            print('layers:', layer_0, layer_1)
            dwpw_fusion_dict_rev[layer_1] = layer_0
            fused_layers[layer_0] = 1
            fused_layers[layer_1] = 1
            fused_dw_pw_layer_dms_rev[layer_1] = fused_dw_pw_layer_dms[layer_0]
            print('tile dims:', fused_dw_pw_layer_dms[layer_0].l1_tile_filters, 
                  fused_dw_pw_layer_dms[layer_0].l1_tile_h, fused_dw_pw_layer_dms[layer_0].l1_tile_w)
    ##################################pwpw###########################################
    fused_pw_pw_layer_dms = {}
    fused_pw_pw_layer_dms_rev = {}
    pwpw_fusion_dict = {}  
    pwpw_fusion_dict_rev = {}  
    pw_pw_dm_estimation(model_dag, hw_configs, fused_pw_pw_layer_dms)
    to_fuse_or_not_to(layer_by_layer_dms, fused_pw_pw_layer_dms, pwpw_fusion_dict)
    
    print('**********pwpw***********')
    for layer_0, layer_1 in pwpw_fusion_dict.items():
        if layer_1 != None:
            print('layers:', layer_0, layer_1)
            pwpw_fusion_dict_rev[layer_1] = layer_0
            fused_layers[layer_0] = 1
            fused_layers[layer_1] = 1
            fused_pw_pw_layer_dms_rev[layer_1] = fused_pw_pw_layer_dms[layer_0]
            print('tile dims:', fused_pw_pw_layer_dms[layer_0].l1_tile_filters, 
                  fused_pw_pw_layer_dms[layer_0].l1_tile_h, fused_pw_pw_layer_dms[layer_0].l1_tile_w)
    print('********not fused********')
    for layer_index in range(len(layer_by_layer_dms)):
        if layer_index not in fused_layers and layer_by_layer_dms[layer_index] is not None and layer_by_layer_dms[layer_index] != -1:
            print('Layer:', layer_index)
            print('tile dims:', layer_by_layer_dms[layer_index].l1_tile_filters, 
                  layer_by_layer_dms[layer_index].l1_tile_h, layer_by_layer_dms[layer_index].l1_tile_w)
    print('**************************')


