#!/usr/bin/env python3

import os
from random import shuffle

images_dir = 'data/images/gray'
folds_dir = 'data-folds'
num_folds = 5
train_size = 1000

images = os.listdir(images_dir)
images.sort()
fold_size = len(images) // num_folds
test_size = fold_size
valid_size = len(images) - test_size - train_size

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

set_names = ['train', 'valid', 'test']
for k in range(num_folds):
    for set_data, set_name in zip(get_sets(k), set_names):
        set_path = os.path.join(folds_dir, '{}.{}.txt'.format(k, set_name))
        with open(set_path, 'w') as set_file:
            set_file.write('{}\n'.format('\n'.join(set_data)))
