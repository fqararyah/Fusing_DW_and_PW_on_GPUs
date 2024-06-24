#path to a dataset
DATA_PATH = '/media/SSD2TB/shared/imagenet_val2012'
#in 
MODEL_NAME = 'gprox_3'
PRECISION = 8 #32
DO_INFERENCE = False
if DO_INFERENCE:
    NUM_IMAGES = 100
    PREDICTIONS_DIR = 'predictions'
