import sys
import builder

import utils

hw_configs = utils.read_hw_config()
model_dag = model_dag = utils.read_model_dag()

builder.build_cnn(model_dag, hw_configs)

