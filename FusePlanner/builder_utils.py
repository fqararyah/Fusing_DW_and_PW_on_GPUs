import math
from enum import Enum

class Dimension(Enum):
    DEPTH = 0
    HEIGHT = 1
    WIDTH = 2

def size_to_sqr_dim(size):
    hw = 1
    steps = 0
    while hw <= size / 4:
        hw = hw << 2
        steps += 2

    return hw, hw >> int(steps / 2)


def size_to_rect_hw(size):
    h = int(math.floor(math.sqrt(size)))
    w = int(math.ceil(math.sqrt(size)))
    size = h * w
    return size, h, w

def least_pow_of_2_geq(num):
    return int( math.pow( 2, math.ceil(math.log(num, 2)) ) )