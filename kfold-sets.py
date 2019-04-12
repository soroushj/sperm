#!/usr/bin/env python3

import os
import numpy as np
import imageio

images_dir = 'data/images/gray'
folds_dir = 'data-folds'
num_folds = 5
class_i_dict = {'h': 9, 'v': 10, 't': 11, 'a': 13}
class_c_dict = {'h': '0', 'v': '0', 't': '0', 'a': 'n'}

labels = class_i_dict.keys()

def get_image(image_name):
    image_path = os.path.join(images_dir, image_name)
    image = imageio.imread(image_path)
    cropped = image[32:-32,32:-32]
    cropped = cropped.astype(np.float32) * (1.0 / 255.0)
    cropped -= cropped.mean()
    return cropped

def get_set(k, set_name):
    set_path = os.path.join(folds_dir, '{}.{}.txt'.format(k, set_name))
    with open(set_path) as set_file:
        image_names = set_file.read().split()
    x = np.ndarray(shape=(len(image_names), 64, 64), dtype=np.float32)
    y = {}
    for label in labels:
        y[label] = np.ndarray(shape=(len(image_names), ), dtype=np.float32)
    for i, image_name in enumerate(image_names):
        x[i,:,:] = get_image(image_name)
        for label in labels:
            y[label][i] = float(image_name[class_i_dict[label]] != class_c_dict[label])
    return x, y

set_names = ['valid', 'test']
for k in range(num_folds):
    for set_name in set_names:
        x, y = get_set(k, set_name)
        print('x:', x)
        print('y:', y)
        print()
