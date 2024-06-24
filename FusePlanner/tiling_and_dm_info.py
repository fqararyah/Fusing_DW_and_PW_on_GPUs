from dataclasses import dataclass

@dataclass
class Splitting_and_fusion_info:
    fused_with: int = -1
    
    l1_tile_h: int = -1
    l1_tile_w: int = -1
    l1_tile_d: int = -1
    l1_tile_filters: int = -1
    
    l2_tile_h: int = -1
    l2_tile_w: int = -1
    l2_tile_d: int = -1
    l2_tile_filters: int = -1

    total_dm: int = -1
    saved_dm: int = -1
    redundant_comp: int = -1
    redundant_comp_ratio: float = 0