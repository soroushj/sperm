#!/usr/bin/env python3

import os
import math
import pandas as pd

results_dir = 'saved-results/kfold'
model = 'm1'
labels = ['a', 'h', 'v']
config = 0
num_folds = 5
# cols = ['epoch', 'test-acc', 'test-precision', 'test-recall', 'test-f0.5-score', 'test-tp', 'test-fp', 'test-fn', 'test-tn']
cols = ['test-acc', 'test-f0.5-score']

def get_result(csv_path):
    # return pd.read_csv(csv_path).sort_values('valid-loss')[cols].iloc[0]
    return pd.read_csv(csv_path).sort_values('valid-loss').iloc[0]

s = 0.0
for label in labels:
    for fold in range(num_folds):
        csv_filename = '{}.{}.{}.{}.csv'.format(model, label, config, fold)
        g_csv_path = os.path.join(results_dir, 'results', csv_filename)
        b_csv_path = os.path.join(results_dir, 'results-bad', csv_filename)
        g_result = get_result(g_csv_path)
        b_result = get_result(b_csv_path)

        metric = 'test-f0.5-score'
        diff = g_result[metric] - b_result[metric]
        s += diff
        print('{}{} {:+.2f}'.format(label, fold, 100 * diff))

print()
print('{:+.2f}'.format(s * 100))
