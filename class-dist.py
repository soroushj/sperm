#!/usr/bin/env python3

import os

images_dir = 'data/images/gray'

class_i_dict = {'h': 9, 'v': 10, 't': 11, 'a': 13}
class_c_dict = {'h': '0', 'v': '0', 't': '0', 'a': 'n'}
set_i_dict = {
    'whole': (0, 1540),
    'train': (0, 1000),
    'valid': (1000, 1240),
    'test': (1240, 1540)
}

names = os.listdir(images_dir)
names.sort()
num_pos_dict = {
    'whole': {'h': 0, 'v': 0, 't': 0, 'a': 0},
    'train': {'h': 0, 'v': 0, 't': 0, 'a': 0},
    'valid': {'h': 0, 'v': 0, 't': 0, 'a': 0},
    'test': {'h': 0, 'v': 0, 't': 0, 'a': 0}
}
labels = ('a', 'h', 'v', 't')
set_names = ('whole', 'train', 'valid', 'test')

for i, name in enumerate(names):
    for label in labels:
        is_pos = name[class_i_dict[label]] == class_c_dict[label]
        if is_pos:
            for set_name, (lower, upper) in set_i_dict.items():
                if lower <= i < upper:
                    num_pos_dict[set_name][label] += 1

for set_name in set_names:
    print(set_name)
    print('label\t#pos\t#neg\t%pos')
    for label in labels:
        lower, upper = set_i_dict[set_name]
        total = upper - lower
        num_pos = num_pos_dict[set_name][label]
        num_neg = total - num_pos
        ratio_pos = num_pos / total
        print('{}\t{}\t{}\t{:.4f}'.format(label,num_pos, num_neg, ratio_pos))
    print()
