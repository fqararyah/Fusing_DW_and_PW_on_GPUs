import utils
import math
import sys
import tiling_and_dm_info
import builder_utils

def dw_estimate_min_dm(layer_specs, hw_configs):
    # print('dw')
    splitting_and_fusion_info = tiling_and_dm_info.Splitting_and_fusion_info()

    [ifms_d, ifms_h, ifms_w] = layer_specs['ifms_shape']
    [ofms_d, ofms_h, ofms_w] = layer_specs['ofms_shape']
    filter_dim = layer_specs['weights_shape'][-1]
    strides = layer_specs['strides']

    ifms_hw = ifms_h * ifms_w
    ifms_size = ifms_d * ifms_hw
    ofms_size = ofms_d * ofms_h * ofms_w
    num_of_sms = int(hw_configs['sms'])
    l1_sz = int(hw_configs['l1']) * 1024

    dm = ifms_size + ofms_size
    redundant_dm = 0
    # best situation is to allocate a whole FM to an SM, this way there is no redundant data movement
    # note that in this scenario, only one dw filter is required per SM, and since a dw filter is 2d with
    # small number of elements (9, 25 or 49), it is negligible compared to the fms dm
    num_of_tiles_in_ifm = 1

    if ifms_hw > l1_sz:
        num_of_tiles_in_ifm = math.ceil(ifms_hw / l1_sz)

    if ifms_d * num_of_tiles_in_ifm < num_of_sms:
        num_of_tiles_in_ifm = math.ceil(num_of_sms / ifms_d)

    if num_of_tiles_in_ifm > 1:
        tile_hw = math.ceil(ifms_hw / num_of_tiles_in_ifm)
        # size that reduces DM (square is the rectangle of minimum perimeter given the area)
        tile_hw, tile_h = builder_utils.size_to_sqr_dim(
            tile_hw)  # size_to_rect_hw(tile_size)
        tile_w = tile_h
        redundant_dm = ifms_d * (filter_dim - strides) * ((ifms_w * (math.ceil(ifms_h / tile_h) - 1)) +
                                                          (ifms_h * (math.ceil(ifms_w / tile_w) - 1)))

    splitting_and_fusion_info.total_dm = dm + redundant_dm
    return splitting_and_fusion_info


def pw_estimate_min_dm(layer_specs, hw_configs):
    # print('pw')
    splitting_and_fusion_info = tiling_and_dm_info.Splitting_and_fusion_info()

    [ifms_d, ifms_h, ifms_w] = layer_specs['ifms_shape']
    [ofms_d, ofms_h, ofms_w] = layer_specs['ofms_shape']
    [num_filters, weights_d] = layer_specs['weights_shape']
    strides = layer_specs['strides']

    # since this is pw, and there is no data sharing, no need to bring data betweem strides
    ifms_size = ifms_d * ifms_h * ifms_w / (strides * strides)
    ofms_size = ofms_d * ofms_h * ofms_w
    weights_size = num_filters * weights_d

    num_of_sms = int(hw_configs['sms'])
    warp_size = int(hw_configs['warp_size'])

    # in the case there is not enough parallelism to keep the GPU busy of ofms are not divisible on h or w anyway
    # this case needs further handling in case it appears
    assert((ofms_w == 1 and ofms_h == 1) or num_of_sms *
           warp_size * 2 <= num_filters * ofms_h * ofms_w)

    weights_tile_sz = int(hw_configs['l1']) * 1024
    ifms_tile_sz = weights_tile_sz  # shared memory

    dm = ofms_size

    if weights_size >= ifms_size and ifms_size <= ifms_tile_sz:
        num_of_pw_Weight_tiles = max(
            math.ceil(weights_size / weights_tile_sz), num_of_sms)
        dm += weights_size + num_of_pw_Weight_tiles * ifms_size

    elif weights_size < ifms_size and weights_size <= weights_tile_sz:
        num_of_ifms_tiles = max(
            math.ceil(ifms_size / ifms_tile_sz), num_of_sms)
        dm += ifms_size + num_of_ifms_tiles * weights_size

    elif weights_size >= ifms_size:
        num_of_pw_Weight_tiles = max(
            math.ceil(weights_size / weights_tile_sz), num_of_sms)
        dm += weights_size + num_of_pw_Weight_tiles * \
            math.ceil(ifms_size / ifms_tile_sz) * ifms_tile_sz

    else:
        num_of_ifms_tiles = max(
            math.ceil(ifms_size / ifms_tile_sz), num_of_sms)
        dm += ifms_size + num_of_ifms_tiles * \
            math.ceil(weights_size / weights_tile_sz) * weights_tile_sz

    splitting_and_fusion_info.total_dm = int(dm)
    return splitting_and_fusion_info


def dw_pw_estimate_min_dm(dw_layer_specs, pw_layer_specs, hw_configs):
    splitting_and_fusion_info = tiling_and_dm_info.Splitting_and_fusion_info()

    [dw_ifms_d, dw_ifms_h, dw_ifms_w] = dw_layer_specs['ifms_shape']
    [dw_ofms_d, dw_ofms_h, dw_ofms_w] = dw_layer_specs['ofms_shape']
    dw_filter_dim = dw_layer_specs['weights_shape'][-1]
    dw_weights_size = dw_layer_specs['weights_shape'][0] * dw_layer_specs['weights_shape'][1] * \
        dw_layer_specs['weights_shape'][2]

    dw_ifms_hw = dw_ifms_h * dw_ifms_w
    dw_ifms_size = dw_ifms_d * dw_ifms_hw
    dw_ofms_size = dw_ofms_d * dw_ofms_h * dw_ofms_w
    dw_strides = dw_layer_specs['strides']
    dw_comps = dw_weights_size * dw_ofms_size / dw_ifms_d

    [pw_ifms_d, pw_ifms_h, pw_ifms_w] = pw_layer_specs['ifms_shape']
    [pw_ofms_d, pw_ofms_h, pw_ofms_w] = pw_layer_specs['ofms_shape']
    [pw_num_filters, pw_weights_d] = pw_layer_specs['weights_shape']
    pw_strides = pw_layer_specs['strides']

    # since this is pw, and there is no data sharing, no need to bring data betweem strides
    pw_ifms_size = pw_ifms_d * pw_ifms_h * \
        pw_ifms_w / (pw_strides * pw_strides)
    pw_ofms_size = pw_ofms_d * pw_ofms_h * pw_ofms_w
    pw_weights_size = pw_num_filters * pw_weights_d

    num_of_sms = int(hw_configs['sms'])
    l1_size = int(hw_configs['l1']) * 1024

    weights_tile_sz = l1_size

    # initially assumae max ifms tile size to reduce redundant load of the haloes
    ifms_tile_sz = l1_size / 2  # equally shared with the IRs between the DW and PW

    dm = pw_ofms_size  # + num_of_tiles_in_ifm * dw_weights_size
    redundant_comp = 0

    tile_h, tile_w = dw_ifms_h, dw_ifms_w
    #

    if pw_weights_size >= dw_ifms_size and dw_ifms_size <= ifms_tile_sz:
        # each SM will do the DW for all
        num_of_pw_Weight_tiles = max(
            math.ceil(pw_weights_size / weights_tile_sz), num_of_sms)
        dm += pw_weights_size + \
            num_of_pw_Weight_tiles * (dw_ifms_size + dw_weights_size)
        redundant_comp += (num_of_pw_Weight_tiles - 1) * dw_comps

    elif pw_weights_size < pw_ifms_size and pw_weights_size <= weights_tile_sz:
        num_of_tiles_in_ifm = max(dw_ifms_size / ifms_tile_sz, num_of_sms)
        ifms_tile_sz = math.ceil(dw_ifms_size / num_of_tiles_in_ifm)
        ifms_tile_hw = math.ceil(ifms_tile_sz / dw_ifms_d)

        ifms_tile_hw, tile_h = builder_utils.size_to_sqr_dim(
            ifms_tile_hw)  # size_to_rect_hw(ifms_tile_hw)
        tile_w = tile_h

        ifms_tile_sz = ifms_tile_hw * dw_ifms_d
        num_of_tiles_in_ifm = math.ceil(dw_ifms_size / ifms_tile_sz)

        # pw weights replicated
        dm += num_of_tiles_in_ifm * pw_weights_size

        # dw ifms haloes
        redundant_dm = dw_ifms_d * (dw_filter_dim - dw_strides) * ((dw_ifms_w * (math.ceil(dw_ifms_h / tile_h) - 1)) +
                                                                   (dw_ifms_h * (math.ceil(dw_ifms_w / tile_w) - 1)))
        dm += redundant_dm

    elif pw_weights_size >= pw_ifms_size:
        num_of_pw_weight_tiles = max(
            math.ceil(pw_weights_size / weights_tile_sz), num_of_sms)
        dm += pw_weights_size + num_of_pw_weight_tiles * \
            (math.ceil(dw_ifms_size / ifms_tile_sz)
             * ifms_tile_sz + dw_weights_size)

        redundant_comp += (num_of_pw_weight_tiles - 1) * dw_comps

    else:
        dm += dw_ifms_size + math.ceil(dw_ifms_size / ifms_tile_sz) * \
            math.ceil(pw_weights_size / weights_tile_sz) * weights_tile_sz

        num_of_tiles_in_ifm = max(dw_ifms_size / ifms_tile_sz, num_of_sms)
        ifms_tile_sz = math.ceil(dw_ifms_size / num_of_tiles_in_ifm)
        ifms_tile_hw = math.ceil(ifms_tile_sz / dw_ifms_d)

        ifms_tile_hw, tile_h, tile_w = builder_utils.size_to_sqr_dim(
            ifms_tile_hw)  # size_to_rect_hw(ifms_tile_hw)
        tile_w = tile_h

        ifms_tile_sz = ifms_tile_hw * dw_ifms_d
        num_of_tiles_in_ifm = math.ceil(dw_ifms_size / ifms_tile_sz)

        # dw ifms haloes
        redundant_dm = dw_ifms_d * (dw_filter_dim - dw_strides) * ((dw_ifms_w * (math.ceil(dw_ifms_h / tile_h) - 1)) +
                                                                   (dw_ifms_h * (math.ceil(dw_ifms_w / tile_w) - 1)))
        # pw weights replicated
        redundant_dm += num_of_tiles_in_ifm * pw_weights_size

        dm += redundant_dm

    splitting_and_fusion_info.fused_with = pw_layer_specs['id']
    splitting_and_fusion_info.redundant_comp = redundant_comp
    splitting_and_fusion_info.redundant_comp_ratio = redundant_comp / \
        (dw_comps + pw_ofms_size * pw_num_filters)
    splitting_and_fusion_info.total_dm = dm
    splitting_and_fusion_info.tile_h = tile_h
    splitting_and_fusion_info.tile_w = tile_w

    return splitting_and_fusion_info


def pw_dw_estimate_min_dm(pw_layer_specs, dw_layer_specs, hw_configs):
    splitting_and_fusion_info = tiling_and_dm_info.Splitting_and_fusion_info()

    [dw_ifms_d, dw_ifms_h, dw_ifms_w] = dw_layer_specs['ifms_shape']
    [dw_ofms_d, dw_ofms_h, dw_ofms_w] = dw_layer_specs['ofms_shape']
    dw_filter_dim = dw_layer_specs['weights_shape'][-1]
    dw_weights_size = dw_layer_specs['weights_shape'][0] * dw_layer_specs['weights_shape'][1] * \
        dw_layer_specs['weights_shape'][2]

    dw_ifms_hw = dw_ifms_h * dw_ifms_w
    dw_ifms_size = dw_ifms_d * dw_ifms_hw
    dw_ofms_size = dw_ofms_d * dw_ofms_h * dw_ofms_w
    dw_strides = dw_layer_specs['strides']
    dw_comps = dw_weights_size * dw_ofms_size / dw_ifms_d

    [pw_ifms_d, pw_ifms_h, pw_ifms_w] = pw_layer_specs['ifms_shape']
    [pw_ofms_d, pw_ofms_h, pw_ofms_w] = pw_layer_specs['ofms_shape']
    [pw_num_filters, pw_weights_d] = pw_layer_specs['weights_shape']
    pw_strides = pw_layer_specs['strides']

    # since this is pw, and there is no data sharing, no need to bring data betweem strides
    pw_ifms_size = pw_ifms_d * pw_ifms_h * \
        pw_ifms_w / (pw_strides * pw_strides)
    pw_ofms_size = pw_ofms_d * pw_ofms_h * pw_ofms_w
    pw_weights_size = pw_num_filters * pw_weights_d

    num_of_sms = int(hw_configs['sms'])
    l1_size = int(hw_configs['l1']) * 1024
    warp_size = int(hw_configs['warp_size'])

    # in the case there is not enough parallelism to keep the GPU busy of ofms are not divisible on h or w anyway
    # this case needs further handling in case it appears
    assert((pw_ofms_w == 1 and pw_ofms_h == 1) or num_of_sms *
           warp_size * 2 <= pw_num_filters * pw_ofms_h * pw_ofms_w)

    weights_tile_sz = l1_size
    ifms_tile_sz = l1_size / 2  # equally shared with the IRs between the DW and PW

    dm = pw_ofms_size

    redundant_comp = 0

    tile_h, tile_w = dw_ifms_h, dw_ifms_w

    #
    if pw_weights_size >= pw_ifms_size and pw_ifms_size <= ifms_tile_sz:
        num_of_pw_Weight_tiles = max(
            math.ceil(pw_weights_size / weights_tile_sz), num_of_sms)
        dm += pw_weights_size + num_of_pw_Weight_tiles * pw_ifms_size
        dm += dw_weights_size

    elif pw_weights_size < pw_ifms_size and pw_weights_size <= weights_tile_sz:
        num_of_ifms_tiles = max(
            math.ceil(pw_ifms_size / ifms_tile_sz), num_of_sms)
        dm += pw_ifms_size + num_of_ifms_tiles * pw_weights_size

        tile_hw = math.ceil(dw_ifms_hw / num_of_ifms_tiles)
        # size that reduces DM (square is the rectangle of minimum perimeter given the area)
        tile_hw, tile_h = builder_utils.size_to_sqr_dim(
            tile_hw)  # size_to_rect_hw(tile_size)
        tile_w = tile_h
        dm += dw_ifms_d * (dw_filter_dim - dw_strides) * ((dw_ifms_w * (math.ceil(dw_ifms_h / tile_h) - 1)) +
                                                    (dw_ifms_h * (math.ceil(dw_ifms_w / tile_w) - 1)))

    elif pw_weights_size >= pw_ifms_size:
        num_of_pw_Weight_tiles = max(
            math.ceil(pw_weights_size / weights_tile_sz), num_of_sms)
        dm += pw_weights_size + num_of_pw_Weight_tiles * \
            math.ceil(pw_ifms_size / ifms_tile_sz) * ifms_tile_sz

    else:
        num_of_ifms_tiles = max(
            math.ceil(pw_ifms_size / ifms_tile_sz), num_of_sms)
        dm += pw_ifms_size + num_of_ifms_tiles * \
            math.ceil(pw_weights_size / weights_tile_sz) * weights_tile_sz

    splitting_and_fusion_info.fused_with = dw_layer_specs['id']
    splitting_and_fusion_info.redundant_comp = redundant_comp
    splitting_and_fusion_info.redundant_comp_ratio = redundant_comp / \
        (dw_comps + pw_ofms_size * pw_num_filters)
    splitting_and_fusion_info.total_dm = dm
    splitting_and_fusion_info.tile_h = tile_h
    splitting_and_fusion_info.tile_w = tile_w

    return splitting_and_fusion_info


def estimate_dm(layer_specs, hw_configs):
    if len(layer_specs) == 1:
        layer_specs = layer_specs[0]
        if utils.is_conv_layer(layer_specs):
            layer_type = layer_specs['type']
            if layer_type == 'pw':
                return pw_estimate_min_dm(layer_specs, hw_configs)
            elif layer_type == 'dw':
                return dw_estimate_min_dm(layer_specs, hw_configs)
            else:
                return -1

    elif len(layer_specs) == 2:
        layer_0_type = layer_specs[0]['type']
        layer_1_type = layer_specs[1]['type']
        if layer_0_type == 'dw' and layer_1_type == 'pw':
            return dw_pw_estimate_min_dm(layer_specs[0], layer_specs[1], hw_configs)
