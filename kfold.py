#!/usr/bin/env python3

import os

images_dir = 'data/images/gray'
num_folds = 5

images = os.listdir(images_dir)
images.sort()
n_rotate = len(images) // num_folds

def get_sets(k):
    train = []
    valid = []
    test = []
    return train, valid, test
