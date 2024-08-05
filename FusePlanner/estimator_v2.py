import utils
import builder_utils
from builder_utils import Dimension
import sys
import tiling_and_dm_info
import math

def calculate_overlap (ifms_h, ifms_w, filter_dim, ifms_tile_h, ifms_tile_w, strides):
    if filter_dim <= strides or ifms_tile_h > ifms_h:
        return 0
    return (math.ceil(ifms_h / ifms_tile_h) - 1) * (filter_dim - strides) * ifms_w + \
            (math.ceil(ifms_w / ifms_tile_w) - 1) * (filter_dim - strides) * ifms_h

def adjust_tile_and_calculate_overlap(ifm_h, ifm_w, filter_dim, ifms_tile_h, ifms_tile_w, strides, dimension_to_adjust = None):

    if dimension_to_adjust == Dimension.HEIGHT:
        ifms_tile_h *= 2
    elif dimension_to_adjust == Dimension.WIDTH:
        ifms_tile_w *= 2

    num_of_ifm_tiles_h =  1 + (ifm_h // ifms_tile_h)
    num_of_ifm_tiles_w = 1 + (ifm_w // ifms_tile_w)
    
    new_overlap = 0
    if filter_dim > 1:
        new_overlap = calculate_overlap(ifm_h, ifm_w, filter_dim, ifms_tile_h, ifms_tile_w, strides)

    return ifms_tile_h, ifms_tile_w, num_of_ifm_tiles_h, num_of_ifm_tiles_w, new_overlap

def dw_gma(ifms_size, ofms_size, weights_size, num_of_weights_tile_hw, overlap, num_of_ifm_tiles_h, num_of_ifm_tiles_w):
    
    return num_of_weights_tile_hw * ifms_size + \
                ofms_size + \
                2 * overlap + \
                weights_size * (num_of_ifm_tiles_h * num_of_ifm_tiles_w)

def pw_gma(ifms_size, ofms_size, weights_size, num_of_weights_tiles, num_of_ifm_tiles_h, num_of_ifm_tiles_w):
    
    return num_of_weights_tiles * ifms_size + \
                ofms_size + \
                weights_size * (num_of_ifm_tiles_h * num_of_ifm_tiles_w)
        
def pw_dw_gma(ifms_size, ofms_size, pw_weights_size, dw_weights_size, 
              num_of_pw_weights_tiles, num_of_dw_weights_tiles,
              overlap,
              num_of_ifm_tiles_h, num_of_ifm_tiles_w,
              pw_ifms_d):
    
    return max(num_of_pw_weights_tiles, num_of_dw_weights_tiles) * ifms_size + \
                ofms_size + \
                2 * overlap * pw_ifms_d+ \
                (pw_weights_size + dw_weights_size) * (num_of_ifm_tiles_h * num_of_ifm_tiles_w)

def dw_pw_gma(ifms_size, ofms_size, pw_weights_size, dw_weights_size, 
              num_of_pw_weights_tiles, num_of_dw_weights_tiles,
              overlap,
              num_of_ifm_tiles_h, num_of_ifm_tiles_w,
              dw_ifms_d):
    
    return max(num_of_pw_weights_tiles, num_of_dw_weights_tiles) * ifms_size + \
                ofms_size + \
                2 * overlap * dw_ifms_d + \
                (pw_weights_size + dw_weights_size) * (num_of_ifm_tiles_h * num_of_ifm_tiles_w)

def pw_pw_gma(ifms_size, ofms_size, pw_1_weights_size, pw_2_weights_size, 
              num_of_pw_1_weights_tiles, num_of_pw_2_weights_tiles,
              num_of_ifm_tiles_h, num_of_ifm_tiles_w):
    
    return max(num_of_pw_1_weights_tiles, num_of_pw_2_weights_tiles) * ifms_size + \
                ofms_size + \
                (pw_1_weights_size * num_of_pw_2_weights_tiles + pw_2_weights_size) * (num_of_ifm_tiles_h * num_of_ifm_tiles_w)

def is_viable_tile(ifms_h, ifms_w, ifms_tile_h, ifms_tile_w, num_of_tiles,\
            weights_tile_size, ifms_tile_size, ofms_tile_size, num_of_sms, l1_sz, shmem_sz):
    
    return ifms_tile_h > 0 and ifms_tile_w > 0 and \
          ifms_tile_h <= builder_utils.least_pow_of_2_geq(ifms_h) and ifms_tile_w <= builder_utils.least_pow_of_2_geq(ifms_w) and \
            num_of_tiles >= num_of_sms and \
            weights_tile_size + ifms_tile_size <= shmem_sz and \
                weights_tile_size + ifms_tile_size + ofms_tile_size <= l1_sz
            
def dw_estimate_min_dm_v2(layer_specs, hw_configs):
    splitting_and_fusion_info = tiling_and_dm_info.Splitting_and_fusion_info()

    [ifms_d, ifms_h, ifms_w] = layer_specs['ifms_shape']
    [ofms_d, ofms_h, ofms_w] = layer_specs['ofms_shape']
    filter_dim = layer_specs['weights_shape'][-1]
    weights_size = layer_specs['weights_shape'][0] * layer_specs['weights_shape'][1] * layer_specs['weights_shape'][2]
    strides = layer_specs['strides']

    ifms_hw = ifms_h * ifms_w
    ifms_size = ifms_d * ifms_hw
    ofms_size = ofms_d * ofms_h * ofms_w
    num_of_sms = int(hw_configs['sms'])
    l1_sz = int(hw_configs['l1']) * 1024
    shmem_sz = int(hw_configs['shmem']) * 1024

    num_of_weights_tile_hw = 1 #OS dataflow requirements, may add other options in the future
    weights_tile_size = filter_dim * filter_dim
    
    ifms_tile_h= builder_utils.least_pow_of_2_geq(filter_dim)
    num_of_ifm_tiles_h = ifms_h / ifms_tile_h
    ifms_tile_w= builder_utils.least_pow_of_2_geq(filter_dim)
    num_of_ifm_tiles_w = ifms_w / ifms_tile_w
    ifms_tile_d = 1
    
    splitting_and_fusion_info.l1_tile_d = ifms_tile_d
    ifms_tile_hw = ifms_tile_h * ifms_tile_w
    ifms_tile_size = ifms_tile_hw * ifms_tile_d
    ofms_tile_size = ifms_tile_size / (strides * strides)
    
    num_of_ifms_tile_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w
    
    overlap = calculate_overlap(ifms_h, ifms_w, filter_dim, ifms_tile_h, ifms_tile_w, strides)
    
    gma = dw_gma(ifms_size, ofms_size, weights_size, num_of_weights_tile_hw,
                 overlap, num_of_ifm_tiles_h, num_of_ifm_tiles_w)
        
    iter = 0
    min_gma = gma
    splitting_and_fusion_info.l1_tile_h = ifms_tile_h / strides
    splitting_and_fusion_info.l1_tile_w = ifms_tile_w / strides
    splitting_and_fusion_info.l1_tile_filters = 1
    splitting_and_fusion_info.total_dm =gma
    
    while is_viable_tile(ifms_h, ifms_w, ifms_tile_h, ifms_tile_w, num_of_ifms_tile_hw * ifms_d,
                         weights_tile_size, ifms_tile_size, ofms_tile_size, num_of_sms, l1_sz, shmem_sz):
        
        if iter % 2 == 0:
            ifms_tile_h, ifms_tile_w, num_of_ifm_tiles_h, num_of_ifm_tiles_w, overlap = \
            adjust_tile_and_calculate_overlap(ifms_h, ifms_w, filter_dim, ifms_tile_h, ifms_tile_w, strides, Dimension.WIDTH)
        else:
            ifms_tile_h, ifms_tile_w, num_of_ifm_tiles_h, num_of_ifm_tiles_w, overlap = \
            adjust_tile_and_calculate_overlap(ifms_h, ifms_w, filter_dim, ifms_tile_h, ifms_tile_w, strides, Dimension.HEIGHT)
        
        #print(ifms_tile_h, ifms_tile_w, num_of_ifm_tiles_h, num_of_ifm_tiles_w, overlap)
        current_gma = dw_gma(ifms_size, ofms_size, weights_size, num_of_weights_tile_hw,
                 overlap, num_of_ifm_tiles_h, num_of_ifm_tiles_w)
        
        num_of_ifms_tile_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w
        ifms_tile_size = ifms_tile_hw * ifms_tile_d
        ofms_tile_size = ifms_tile_size / (strides * strides)
        
        if current_gma < min_gma:
            min_gma = current_gma
            splitting_and_fusion_info.l1_tile_h = int(ifms_tile_h / strides)
            splitting_and_fusion_info.l1_tile_w = int(ifms_tile_w / strides)
            splitting_and_fusion_info.l1_tile_filters = 1
            splitting_and_fusion_info.total_dm = current_gma
            
        iter += 1
        
    return splitting_and_fusion_info

def pw_estimate_min_dm_v2(layer_specs, hw_configs):
    splitting_and_fusion_info = tiling_and_dm_info.Splitting_and_fusion_info()

    [ifms_d, ifms_h, ifms_w] = layer_specs['ifms_shape']
    [ofms_d, ofms_h, ofms_w] = layer_specs['ofms_shape']
    [num_of_filters, filter_d] = layer_specs['weights_shape']
    weights_size = num_of_filters * filter_d
    strides = layer_specs['strides']

    ifms_hw = ifms_h * ifms_w
    ifms_size = ifms_d * ifms_hw
    ofms_size = ofms_d * ofms_h * ofms_w
    
    num_of_sms = int(hw_configs['sms'])
    l1_sz = int(hw_configs['l1']) * 1024
    shmem_sz = int(hw_configs['shmem']) * 1024
    
    min_gma = 1000000000000
    splitting_and_fusion_info.total_dm = min_gma
    num_of_filters_in_a_tile = 1
    if not utils.is_pow_of_2(num_of_filters):
        num_of_filters_in_a_tile = int(num_of_filters / utils.pow_of_2_leq(num_of_filters))
    while num_of_filters_in_a_tile < num_of_filters:
        weights_tile_size = num_of_filters_in_a_tile * filter_d
        num_of_weights_tiles = num_of_filters / num_of_filters_in_a_tile
    
        ifms_tile_h=1
        num_of_ifm_tiles_h = ifms_h / ifms_tile_h
        ifms_tile_w= 1
        num_of_ifm_tiles_w = ifms_w / ifms_tile_w
        ifms_tile_d = filter_d
        
        splitting_and_fusion_info.l1_tile_d = ifms_tile_d
        ifms_tile_hw = ifms_tile_h * ifms_tile_w
        ifms_tile_size = ifms_tile_hw * ifms_tile_d
        ofms_tile_size = ifms_tile_hw * num_of_filters_in_a_tile / (strides * strides)
        
        num_of_ifms_tile_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w
        num_of_all_tiles = num_of_ifms_tile_hw * num_of_weights_tiles
        
        if is_viable_tile(ifms_h, ifms_w, ifms_tile_h, ifms_tile_w, num_of_all_tiles,
                            weights_tile_size, ifms_tile_size, ofms_tile_size, num_of_sms, l1_sz, shmem_sz):   
            current_gma = pw_gma(ifms_size, ofms_size, weights_size, num_of_weights_tiles,
                        num_of_ifm_tiles_h, num_of_ifm_tiles_w)
                
            iter = 0
            if current_gma < min_gma:
                min_gma = current_gma
                splitting_and_fusion_info.l1_tile_h = ifms_tile_h / strides
                splitting_and_fusion_info.l1_tile_w = ifms_tile_w / strides
                splitting_and_fusion_info.l1_tile_filters = num_of_filters_in_a_tile
                splitting_and_fusion_info.total_dm = current_gma
        
        while is_viable_tile(ifms_h, ifms_w, ifms_tile_h, ifms_tile_w, num_of_all_tiles,
                            weights_tile_size, ifms_tile_size, ofms_tile_size, num_of_sms, l1_sz, shmem_sz):
            
            if iter % 2 == 0:
                ifms_tile_h, ifms_tile_w, num_of_ifm_tiles_h, num_of_ifm_tiles_w, _ = \
                adjust_tile_and_calculate_overlap(ifms_h, ifms_w, 1, ifms_tile_h, ifms_tile_w, strides, Dimension.WIDTH)
            else:
                ifms_tile_h, ifms_tile_w, num_of_ifm_tiles_h, num_of_ifm_tiles_w, _ = \
                adjust_tile_and_calculate_overlap(ifms_h, ifms_w, 1, ifms_tile_h, ifms_tile_w, strides, Dimension.HEIGHT)
            
            current_gma = pw_gma(ifms_size, ofms_size, weights_size, num_of_weights_tiles,
                                num_of_ifm_tiles_h, num_of_ifm_tiles_w)
            
            num_of_ifms_tile_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w
            num_of_all_tiles = num_of_ifms_tile_hw * num_of_weights_tiles
            ifms_tile_hw = ifms_tile_h * ifms_tile_w
            ifms_tile_size = ifms_tile_hw * ifms_tile_d
            ofms_tile_size = ifms_tile_hw * num_of_filters_in_a_tile / (strides * strides)
            
            #print(ifms_h, ifms_tile_h, ifms_tile_w, weights_tile_size, ifms_tile_size, ofms_tile_size, current_gma)
            if current_gma < min_gma:
                min_gma = current_gma
                splitting_and_fusion_info.l1_tile_h = int(ifms_tile_h / strides)
                splitting_and_fusion_info.l1_tile_w = int(ifms_tile_w / strides)
                splitting_and_fusion_info.l1_tile_filters = num_of_filters_in_a_tile
                splitting_and_fusion_info.total_dm = current_gma
                
            iter += 1
        
        num_of_filters_in_a_tile *= 2
        
    return splitting_and_fusion_info

def pw_dw_estimate_min_dm_v2(pw_layer_specs, dw_layer_specs, hw_configs):
    splitting_and_fusion_info = tiling_and_dm_info.Splitting_and_fusion_info()

    [ifms_d, ifms_h, ifms_w] = pw_layer_specs['ifms_shape']
    [dw_ifms_d, dw_ifms_h, dw_ifms_w] = dw_layer_specs['ifms_shape']
    [ofms_d, ofms_h, ofms_w] = dw_layer_specs['ofms_shape']
    [pw_num_of_filters, pw_filter_d] = pw_layer_specs['weights_shape']
    pw_weights_size = pw_num_of_filters * pw_filter_d
    dw_weights_shape = dw_layer_specs['weights_shape']
    dw_weights_size = dw_weights_shape[0] * dw_weights_shape[1] * dw_weights_shape[2]
    pw_strides = pw_layer_specs['strides']
    dw_filter_dim = dw_weights_shape[-1]
    dw_strides = pw_layer_specs['strides']
    
    ifms_hw = ifms_h * ifms_w
    ifms_size = ifms_d * ifms_hw
    ofms_size = ofms_d * ofms_h * ofms_w
    
    num_of_sms = int(hw_configs['sms'])
    l1_sz = int(hw_configs['l1']) * 1024
    shmem_sz = int(hw_configs['shmem']) * 1024
    
    min_gma = 1000000000000
    splitting_and_fusion_info.fused_with = dw_layer_specs['id']
    splitting_and_fusion_info.total_dm = min_gma
    num_of_filters_in_a_tile = 1
    if not utils.is_pow_of_2(pw_num_of_filters):
        num_of_filters_in_a_tile = int(pw_num_of_filters / utils.pow_of_2_leq(pw_num_of_filters))

    while num_of_filters_in_a_tile < pw_num_of_filters:
        pw_weights_tile_size = num_of_filters_in_a_tile * pw_filter_d
        dw_weights_tile_size = num_of_filters_in_a_tile * dw_filter_dim
        num_of_weights_tiles = pw_num_of_filters / num_of_filters_in_a_tile
    
        comm_tile_h= builder_utils.least_pow_of_2_geq(dw_filter_dim)
        ifms_tile_h = comm_tile_h * pw_strides
        num_of_ifm_tiles_h = ifms_h / ifms_tile_h
        comm_tile_w= builder_utils.least_pow_of_2_geq(dw_filter_dim)
        ifms_tile_w = comm_tile_w * pw_strides
        num_of_ifm_tiles_w = ifms_w / ifms_tile_w
        
        dw_ifms_tile_d = num_of_filters_in_a_tile
        pw_ifms_tile_d = pw_filter_d
        
        splitting_and_fusion_info.l1_tile_d = pw_ifms_tile_d
        splitting_and_fusion_info.l2_tile_d = dw_ifms_tile_d
        
        ifms_tile_hw = ifms_tile_h * ifms_tile_w
        ifms_tile_size = min(ifms_tile_hw * pw_ifms_tile_d, ifms_tile_h * ifms_w * pw_ifms_tile_d)
        comm_tile_size = min(comm_tile_h * comm_tile_w * dw_ifms_tile_d, comm_tile_h * (ifms_w / pw_strides) * dw_ifms_tile_d)
        ofms_tile_size = comm_tile_size / (dw_strides * dw_strides)
        num_of_ifms_tile_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w
        num_of_all_tiles = num_of_ifms_tile_hw * num_of_weights_tiles
        
        _, _, _, _, overlap = adjust_tile_and_calculate_overlap(dw_ifms_h, dw_ifms_w, dw_filter_dim, comm_tile_w,
                                                  comm_tile_w, dw_strides)
        
        if is_viable_tile(ifms_h, ifms_w, ifms_tile_h, ifms_tile_w, num_of_all_tiles,
                            pw_weights_tile_size + dw_weights_tile_size, ifms_tile_size + comm_tile_size,
                            ofms_tile_size, num_of_sms, l1_sz, shmem_sz):
            
            current_gma = pw_dw_gma(ifms_size, ofms_size, pw_weights_size, dw_weights_size, num_of_weights_tiles,
                            num_of_weights_tiles, overlap, num_of_ifm_tiles_h, num_of_ifm_tiles_w, pw_filter_d)
            
            if current_gma <= min_gma:
                min_gma = current_gma
                splitting_and_fusion_info.l1_tile_h = ifms_tile_h / pw_strides
                splitting_and_fusion_info.l1_tile_w = ifms_tile_w / pw_strides
                splitting_and_fusion_info.l1_tile_filters = num_of_filters_in_a_tile
                splitting_and_fusion_info.l2_tile_h = ifms_tile_h / (dw_strides * pw_strides)
                splitting_and_fusion_info.l2_tile_w = ifms_tile_w / (dw_strides * pw_strides)
                splitting_and_fusion_info.l2_tile_filters = num_of_filters_in_a_tile
                splitting_and_fusion_info.total_dm = current_gma
                splitting_and_fusion_info.redundant_comp = overlap * ifms_d * pw_num_of_filters / \
                      ((ifms_size * pw_num_of_filters) + (dw_ifms_w * dw_ifms_h * dw_ifms_w) * dw_filter_dim * dw_filter_dim)

        iter = 0 
        while is_viable_tile(ifms_h, ifms_w, ifms_tile_h, ifms_tile_w, num_of_all_tiles,
                            pw_weights_tile_size + dw_weights_tile_size, ifms_tile_size + comm_tile_size,
                            ofms_tile_size, num_of_sms, l1_sz, shmem_sz):
            
            if iter % 2 == 0:
                comm_tile_h, comm_tile_w, num_of_ifm_tiles_h, num_of_ifm_tiles_w, overlap = \
                adjust_tile_and_calculate_overlap(dw_ifms_h, dw_ifms_w, dw_filter_dim, comm_tile_h,
                                                  comm_tile_w, dw_strides, Dimension.WIDTH)
            else:
                comm_tile_h, comm_tile_w, num_of_ifm_tiles_h, num_of_ifm_tiles_w, overlap = \
                adjust_tile_and_calculate_overlap(dw_ifms_h, dw_ifms_w, dw_filter_dim, comm_tile_h,
                                                  comm_tile_w, dw_strides, Dimension.HEIGHT)
            
            current_gma = pw_dw_gma(ifms_size, ofms_size, pw_weights_size, dw_weights_size, num_of_weights_tiles,
                        num_of_weights_tiles, overlap, num_of_ifm_tiles_h, num_of_ifm_tiles_w, pw_filter_d)
            
            ifms_tile_h = comm_tile_h * pw_strides
            ifms_tile_w = comm_tile_w * pw_strides
            ifms_tile_hw = ifms_tile_h * ifms_tile_w
            ifms_tile_size = min(ifms_tile_hw * pw_ifms_tile_d, ifms_tile_h * ifms_w * pw_ifms_tile_d)
            comm_tile_size = min(comm_tile_h * comm_tile_w * dw_ifms_tile_d, comm_tile_h * (ifms_w / pw_strides) * dw_ifms_tile_d)
            ofms_tile_size = comm_tile_size / (dw_strides * dw_strides)
            num_of_ifms_tile_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w
            num_of_all_tiles = num_of_ifms_tile_hw * num_of_weights_tiles

            # print(ifms_h, ifms_tile_h, ifms_tile_w, dw_weights_tile_size, pw_weights_tile_size, ifms_tile_size, comm_tile_size,
            #                ofms_tile_size, current_gma/1000000)
            
            if current_gma <= min_gma:
                min_gma = current_gma
                splitting_and_fusion_info.l1_tile_h = ifms_tile_h
                splitting_and_fusion_info.l1_tile_w = ifms_tile_w
                splitting_and_fusion_info.l1_tile_filters = num_of_filters_in_a_tile
                splitting_and_fusion_info.l2_tile_h = ifms_tile_h / (dw_strides * pw_strides)
                splitting_and_fusion_info.l2_tile_w = ifms_tile_w / (dw_strides * pw_strides)
                splitting_and_fusion_info.l2_tile_filters = num_of_weights_tiles
                splitting_and_fusion_info.total_dm = current_gma
                splitting_and_fusion_info.redundant_comp = overlap * ifms_d * pw_num_of_filters / \
                    ((ifms_size * pw_num_of_filters) + (dw_ifms_w * dw_ifms_h * dw_ifms_w) * dw_filter_dim * dw_filter_dim)
                
            iter += 1
        
        num_of_filters_in_a_tile *= 2
        
    return splitting_and_fusion_info

def dw_pw_estimate_min_dm_v2(dw_layer_specs, pw_layer_specs, hw_configs):
    splitting_and_fusion_info = tiling_and_dm_info.Splitting_and_fusion_info()

    [ifms_d, ifms_h, ifms_w] = pw_layer_specs['ifms_shape']
    [dw_ifms_d, dw_ifms_h, dw_ifms_w] = dw_layer_specs['ifms_shape']
    [ofms_d, ofms_h, ofms_w] = dw_layer_specs['ofms_shape']
    [pw_num_of_filters, pw_filter_d] = pw_layer_specs['weights_shape']
    pw_weights_size = pw_num_of_filters * pw_filter_d
    dw_weights_shape = dw_layer_specs['weights_shape']
    dw_weights_size = dw_weights_shape[0] * dw_weights_shape[1] * dw_weights_shape[2]
    pw_strides = pw_layer_specs['strides']
    dw_filter_dim = dw_weights_shape[-1]
    dw_strides = pw_layer_specs['strides']
    
    ifms_hw = ifms_h * ifms_w
    ifms_size = ifms_d * ifms_hw
    ofms_size = ofms_d * ofms_h * ofms_w
    
    num_of_sms = int(hw_configs['sms'])
    l1_sz = int(hw_configs['l1']) * 1024
    shmem_sz = int(hw_configs['shmem']) * 1024
    
    min_gma = 1000000000000
    splitting_and_fusion_info.total_dm = min_gma
    splitting_and_fusion_info.fused_with = pw_layer_specs['id']
    num_of_filters_in_a_tile = 1
    if not utils.is_pow_of_2(pw_num_of_filters):
        num_of_filters_in_a_tile = int(pw_num_of_filters / utils.pow_of_2_leq(pw_num_of_filters))
    while num_of_filters_in_a_tile < pw_num_of_filters:
        pw_weights_tile_size = num_of_filters_in_a_tile * pw_filter_d
        dw_weights_tile_size = num_of_filters_in_a_tile * dw_filter_dim
        num_of_weights_tiles = pw_num_of_filters / num_of_filters_in_a_tile
    
        comm_tile_h= 1 # pw_filter_dim
        ifms_tile_h = comm_tile_h * dw_strides + dw_filter_dim - dw_strides
        num_of_ifm_tiles_h = ifms_h / ifms_tile_h
        comm_tile_w= 1 # pw_filter_dim
        ifms_tile_w = comm_tile_w * dw_strides + dw_filter_dim - dw_strides
        num_of_ifm_tiles_w = ifms_w / ifms_tile_w
        
        dw_ifms_tile_d = dw_ifms_d
        pw_ifms_tile_d = pw_filter_d
        
        splitting_and_fusion_info.l1_tile_d = dw_ifms_tile_d
        splitting_and_fusion_info.l2_tile_d = pw_ifms_tile_d
        
        ifms_tile_hw = ifms_tile_h * ifms_tile_w
        ifms_tile_size = min(ifms_tile_hw * dw_ifms_tile_d, ifms_tile_h * ifms_w * dw_ifms_tile_d)
        comm_tile_size = min(comm_tile_h * comm_tile_w * pw_ifms_tile_d, comm_tile_h * (ifms_w / pw_strides) * pw_ifms_tile_d)
        ofms_tile_size = comm_tile_size / (pw_strides * pw_strides)
        num_of_ifms_tile_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w
        num_of_all_tiles = num_of_ifms_tile_hw * num_of_weights_tiles
        
        _, _, _, _, overlap = adjust_tile_and_calculate_overlap(dw_ifms_h, dw_ifms_w, 1, ifms_tile_h,
                                                  ifms_tile_w, dw_strides)
        
        if is_viable_tile(ifms_h, ifms_w, ifms_tile_h, ifms_tile_w, num_of_all_tiles,
                            pw_weights_tile_size + dw_weights_tile_size, ifms_tile_size + comm_tile_size,
                            ofms_tile_size, num_of_sms, l1_sz, shmem_sz):
            
            current_gma = dw_pw_gma(ifms_size, ofms_size, pw_weights_size, dw_weights_size, num_of_weights_tiles,
                            1, overlap, num_of_ifm_tiles_h, num_of_ifm_tiles_w, dw_ifms_d)
            
            if current_gma <= min_gma:
                min_gma = current_gma
                splitting_and_fusion_info.l1_tile_h = ifms_tile_h
                splitting_and_fusion_info.l1_tile_w = ifms_tile_w
                splitting_and_fusion_info.l1_tile_filters = num_of_filters_in_a_tile
                splitting_and_fusion_info.l2_tile_h = ifms_tile_h / (dw_strides * pw_strides)
                splitting_and_fusion_info.l2_tile_w = ifms_tile_w / (dw_strides * pw_strides)
                splitting_and_fusion_info.l2_tile_filters = num_of_filters_in_a_tile
                splitting_and_fusion_info.total_dm = current_gma

        iter = 0 
        while is_viable_tile(ifms_h, ifms_w, ifms_tile_h, ifms_tile_w, num_of_all_tiles,
                            pw_weights_tile_size + dw_weights_tile_size, ifms_tile_size + comm_tile_size,
                            ofms_tile_size, num_of_sms, l1_sz, shmem_sz):
            
            if iter % 2 == 0:
                comm_tile_h, comm_tile_w, num_of_ifm_tiles_h, num_of_ifm_tiles_w, overlap = \
                adjust_tile_and_calculate_overlap(dw_ifms_h, dw_ifms_w, 1, ifms_tile_h,
                                                  ifms_tile_w, dw_strides, Dimension.WIDTH)
            else:
                comm_tile_h, comm_tile_w, num_of_ifm_tiles_h, num_of_ifm_tiles_w, overlap = \
                adjust_tile_and_calculate_overlap(dw_ifms_h, dw_ifms_w, 1, ifms_tile_h,
                                                  ifms_tile_w, dw_strides, Dimension.HEIGHT)
            
            current_gma = dw_pw_gma(ifms_size, ofms_size, pw_weights_size, dw_weights_size, num_of_weights_tiles,
                        1, overlap, num_of_ifm_tiles_h, num_of_ifm_tiles_w, dw_ifms_d)
            
            ifms_tile_h = comm_tile_h * dw_strides
            ifms_tile_w = comm_tile_w * dw_strides
            ifms_tile_hw = ifms_tile_h * ifms_tile_w
            ifms_tile_size = min(ifms_tile_hw * dw_ifms_tile_d, ifms_tile_h * ifms_w * dw_ifms_tile_d)
            comm_tile_size = min(comm_tile_h * comm_tile_w * dw_ifms_tile_d, comm_tile_h * (ifms_w / dw_strides) * dw_ifms_tile_d)
            ofms_tile_size = comm_tile_size / (dw_strides * dw_strides)
            num_of_ifms_tile_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w
            num_of_all_tiles = num_of_ifms_tile_hw * num_of_weights_tiles

            # print(ifms_h, ifms_tile_h, ifms_tile_w, dw_weights_tile_size, pw_weights_tile_size, ifms_tile_size, comm_tile_size,
            #                ofms_tile_size, current_gma/1000000)
            
            if current_gma <= min_gma:
                min_gma = current_gma
                splitting_and_fusion_info.l1_tile_h = ifms_tile_h
                splitting_and_fusion_info.l1_tile_w = ifms_tile_w
                splitting_and_fusion_info.l1_tile_filters = num_of_filters_in_a_tile
                splitting_and_fusion_info.l2_tile_h = ifms_tile_h / (dw_strides * pw_strides)
                splitting_and_fusion_info.l2_tile_w = ifms_tile_w / (dw_strides * pw_strides)
                splitting_and_fusion_info.l2_tile_filters = num_of_filters_in_a_tile
                splitting_and_fusion_info.total_dm = current_gma
                
            iter += 1
        
        num_of_filters_in_a_tile *= 2
        
    return splitting_and_fusion_info

def pw_pw_estimate_min_dm_v2(pw_layer_1_specs, pw_layer_2_specs, hw_configs):
    splitting_and_fusion_info = tiling_and_dm_info.Splitting_and_fusion_info()

    [ifms_d_1, ifms_h_1, ifms_w_1] = pw_layer_1_specs['ifms_shape']
    [ifms_d_2, ifms_h_2, ifms_w_2] = pw_layer_2_specs['ifms_shape']
    [ofms_d, ofms_h, ofms_w] = pw_layer_2_specs['ofms_shape']
    [pw_1_num_of_filters, pw_1_filter_d] = pw_layer_1_specs['weights_shape']
    [pw_2_num_of_filters, pw_2_filter_d] = pw_layer_2_specs['weights_shape']
    pw_1_weights_size = pw_1_num_of_filters * pw_1_filter_d
    pw_2_weights_size = pw_2_num_of_filters * pw_2_filter_d
    pw_1_weights_tile_size = pw_1_weights_size
    pw_1_strides = pw_layer_1_specs['strides']
    pw_2_strides = pw_layer_2_specs['strides']
    
    ifms_hw = ifms_h_1 * ifms_w_1
    ifms_size = ifms_d_1 * ifms_hw
    ofms_size = ofms_d * ofms_h * ofms_w
    
    num_of_sms = int(hw_configs['sms'])
    l1_sz = int(hw_configs['l1']) * 1024
    shmem_sz = int(hw_configs['shmem']) * 1024
    
    min_gma = 1000000000000
    splitting_and_fusion_info.fused_with = pw_layer_2_specs['id']
    splitting_and_fusion_info.l1_tile_filters = pw_1_num_of_filters
    splitting_and_fusion_info.total_dm = min_gma
    num_of_filters_in_a_tile = 1
    if not utils.is_pow_of_2(pw_2_num_of_filters):
        num_of_filters_in_a_tile = int(pw_2_num_of_filters / utils.pow_of_2_leq(pw_2_num_of_filters))

    while num_of_filters_in_a_tile < pw_2_num_of_filters:
        pw_2_weights_tile_size = num_of_filters_in_a_tile * pw_2_filter_d
        num_of_weights_tiles = pw_2_num_of_filters / num_of_filters_in_a_tile
    
        comm_tile_h= 1 # pw_filter_dim
        ifms_tile_h = comm_tile_h * pw_1_strides
        num_of_ifm_tiles_h = ifms_h_1 / ifms_tile_h
        comm_tile_w= 1 # pw_filter_dim
        ifms_tile_w = comm_tile_w * pw_1_strides
        num_of_ifm_tiles_w = ifms_w_1 / ifms_tile_w
        
        pw_2_ifms_tile_d = pw_2_filter_d
        pw_1_ifms_tile_d = pw_1_filter_d
        
        splitting_and_fusion_info.l1_tile_d = pw_1_ifms_tile_d
        splitting_and_fusion_info.l2_tile_d = pw_2_ifms_tile_d
        
        ifms_tile_hw = ifms_tile_h * ifms_tile_w
        ifms_tile_size = min(ifms_tile_hw * pw_1_ifms_tile_d, ifms_tile_h * ifms_w_1 * pw_1_ifms_tile_d)
        comm_tile_size = min(comm_tile_h * comm_tile_w * pw_2_ifms_tile_d, comm_tile_h * (ifms_w_1 / pw_1_strides) * pw_2_ifms_tile_d)
        ofms_tile_size = comm_tile_size / (pw_2_strides * pw_2_strides)
        num_of_ifms_tile_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w
        num_of_all_tiles = num_of_ifms_tile_hw * num_of_weights_tiles
        
        if is_viable_tile(ifms_h_1, ifms_w_1, ifms_tile_h, ifms_tile_w, num_of_all_tiles,
                            pw_2_weights_tile_size + pw_1_weights_tile_size, ifms_tile_size + comm_tile_size,
                            ofms_tile_size, num_of_sms, l1_sz, shmem_sz):
            
            current_gma = pw_pw_gma(ifms_size, ofms_size, pw_1_weights_size, pw_2_weights_size, 1, 
                                    num_of_weights_tiles, num_of_ifm_tiles_h, num_of_ifm_tiles_w)
            
            if current_gma <= min_gma:
                min_gma = current_gma
                splitting_and_fusion_info.l1_tile_h = ifms_tile_h
                splitting_and_fusion_info.l1_tile_w = ifms_tile_w
                splitting_and_fusion_info.l2_tile_h = ifms_tile_h / (pw_1_strides * pw_2_strides)
                splitting_and_fusion_info.l2_tile_w = ifms_tile_w / (pw_1_strides * pw_2_strides)
                splitting_and_fusion_info.l2_tile_filters = num_of_filters_in_a_tile
                splitting_and_fusion_info.total_dm = current_gma

        iter = 0 
        while is_viable_tile(ifms_h_1, ifms_w_1, ifms_tile_h, ifms_tile_w, num_of_all_tiles,
                            pw_2_weights_tile_size + pw_1_weights_tile_size, ifms_tile_size + comm_tile_size,
                            ofms_tile_size, num_of_sms, l1_sz, shmem_sz):
            if iter % 2 == 0:
                comm_tile_h, comm_tile_w, num_of_ifm_tiles_h, num_of_ifm_tiles_w, _ = \
                adjust_tile_and_calculate_overlap(ifms_h_1, ifms_w_1, 1, ifms_tile_h,
                                                  ifms_tile_w, pw_1_strides, Dimension.WIDTH)
            else:
                comm_tile_h, comm_tile_w, num_of_ifm_tiles_h, num_of_ifm_tiles_w, _ = \
                adjust_tile_and_calculate_overlap(ifms_h_1, ifms_w_1, 1, ifms_tile_h,
                                                  ifms_tile_w, pw_1_strides, Dimension.HEIGHT)
            
            current_gma = pw_pw_gma(ifms_size, ofms_size, pw_1_weights_size, pw_2_weights_size, 1,
                        num_of_weights_tiles, num_of_ifm_tiles_h, num_of_ifm_tiles_w)
            
            ifms_tile_h = comm_tile_h * pw_1_strides
            ifms_tile_w = comm_tile_w * pw_1_strides
            ifms_tile_hw = ifms_tile_h * ifms_tile_w
            ifms_tile_size = min(ifms_tile_hw * pw_1_ifms_tile_d, ifms_tile_h * ifms_w_1 * pw_1_ifms_tile_d)
            comm_tile_size = min(comm_tile_h * comm_tile_w * pw_2_ifms_tile_d, comm_tile_h * (ifms_w_1 / pw_1_strides) * pw_2_ifms_tile_d)
            ofms_tile_size = comm_tile_size / (pw_1_strides * pw_1_strides)
            num_of_ifms_tile_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w
            num_of_all_tiles = num_of_ifms_tile_hw * num_of_weights_tiles

            # print(ifms_h, ifms_tile_h, ifms_tile_w, dw_weights_tile_size, pw_weights_tile_size, ifms_tile_size, comm_tile_size,
            #                ofms_tile_size, current_gma/1000000)
            
            if current_gma <= min_gma:
                min_gma = current_gma
                splitting_and_fusion_info.l1_tile_h = ifms_tile_h
                splitting_and_fusion_info.l1_tile_w = ifms_tile_w
                splitting_and_fusion_info.l2_tile_h = ifms_tile_h / (pw_1_strides * pw_2_strides)
                splitting_and_fusion_info.l2_tile_w = ifms_tile_w / (pw_1_strides * pw_2_strides)
                splitting_and_fusion_info.l2_tile_filters = num_of_filters_in_a_tile
                splitting_and_fusion_info.total_dm = current_gma
                
            iter += 1
        
        num_of_filters_in_a_tile *= 2
        
    return splitting_and_fusion_info

def estimate_dm(layer_specs, hw_configs):
    if len(layer_specs) == 1:
        layer_specs = layer_specs[0]
        if utils.is_conv_layer(layer_specs):
            layer_type = layer_specs['type']
            if layer_type == 'pw':
                return pw_estimate_min_dm_v2(layer_specs, hw_configs)
            elif layer_type == 'dw':
                return dw_estimate_min_dm_v2(layer_specs, hw_configs)
            else:
                return -1

    elif len(layer_specs) == 2:
        layer_0_type = layer_specs[0]['type']
        layer_1_type = layer_specs[1]['type']
        if layer_0_type == 'pw' and layer_1_type == 'dw':
            return pw_dw_estimate_min_dm_v2(layer_specs[0], layer_specs[1], hw_configs)
        elif layer_0_type == 'dw' and layer_1_type == 'pw':
            return dw_pw_estimate_min_dm_v2(layer_specs[0], layer_specs[1], hw_configs)
        if layer_0_type == 'pw' and layer_1_type == 'pw':
            return pw_pw_estimate_min_dm_v2(layer_specs[0], layer_specs[1], hw_configs)