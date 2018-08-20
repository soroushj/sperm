#!/usr/bin/env python3

import os
import sys
import warnings
import importlib.util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

models_dir_path = 'models'
model_input_shape = (64, 64, 1)

assert len(sys.argv) == 2
model_name = sys.argv[1]
modeldef_spec = importlib.util.spec_from_file_location('modeldef', os.path.join(models_dir_path, model_name + '.py'))
modeldef = importlib.util.module_from_spec(modeldef_spec)
modeldef_spec.loader.exec_module(modeldef)
model = modeldef.get_model(model_input_shape)
model.summary()
