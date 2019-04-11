#!/usr/bin/env python3

import os
from random import shuffle

images_dir = 'data/images/gray'
num_folds = 5
train_size = 1000

images = os.listdir(images_dir)
images.sort()
fold_size = len(images) // num_folds

def get_sets(k):
    assert 0 <= k < num_folds
    a = k * fold_size
    b = (k + 1) * fold_size
    test = images[a:b]
    non_test = images[:a] + images[b:]
    shuffle(non_test)
    train = non_test[:train_size]
    valid = non_test[train_size:]
    return train, valid, test

for k in range(num_folds):
    print(get_sets(k))