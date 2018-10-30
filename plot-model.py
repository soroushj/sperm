#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model
from keras.utils import plot_model

models_dir = 'saved-results/models'
model_names = os.listdir(models_dir)
model_names.sort()

for model_name in model_names:
    model_filename = os.path.join(models_dir, model_name)
    plot_filename = model_filename[:-2] + 'png'
    model = load_model(model_filename)
    plot_model(model, to_file=plot_filename)
