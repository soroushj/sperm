#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')
import os
from tqdm import tqdm
from scipy import misc
import numpy as np
from keras.models import load_model
from keras import activations
from vis.utils import utils
from vis.visualization import visualize_cam, overlay

def make_dirs(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

images_dir = '../sperm-data-preparation/images/01-gray-crop-64'
leveled_images_dir = '../sperm-data-preparation/images/05-2-autolevels'
image_names = os.listdir(images_dir)
image_names.sort()
models_dir = 'saved/final-models'
model_names = os.listdir(models_dir)
model_names.sort()
out_dir = 'saved/attention'

for model_name in model_names:
    print('Model:', model_name)
    model = load_model(os.path.join(models_dir, model_name))
    model.layers[-1].activation = activations.linear
    model = utils.apply_modifications(model)
    model_basename = os.path.splitext(model_name)[0]
    cam_dir = os.path.join(out_dir, model_basename, 'cam')
    cam_image_dir = os.path.join(out_dir, model_basename, 'cam_image')
    cam_leveled_image_dir = os.path.join(out_dir, model_basename, 'cam_leveled_image')
    make_dirs(cam_dir, cam_image_dir, cam_leveled_image_dir)
    for image_name in tqdm(image_names):
        image = misc.imread(os.path.join(images_dir, image_name))
        leveled_image = misc.imread(os.path.join(leveled_images_dir, image_name))
        x = image.reshape((64, 64, 1)).astype(np.float32) / 255
        x -= x.mean()
        image = np.stack((image,)*3, -1)
        leveled_image = np.stack((leveled_image,)*3, -1)
        cam = visualize_cam(model=model, layer_idx=-1, filter_indices=None, seed_input=x)
        cam_image = overlay(cam, image, alpha=0.33)
        cam_leveled_image = overlay(cam, leveled_image, alpha=0.33)
        misc.imsave(os.path.join(cam_dir, image_name), cam)
        misc.imsave(os.path.join(cam_image_dir, image_name), cam_image)
        misc.imsave(os.path.join(cam_leveled_image_dir, image_name), cam_leveled_image)
