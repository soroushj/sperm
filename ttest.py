#!/usr/bin/env python3

import os
import csv
from scipy.stats import ttest_rel

prediction_files = {
    'h': ('saved-results/predictions/m1.h.0.9385.csv', 'saved-results/ghasemian/predictions/ghasemian.h.csv'),
    'v': ('saved-results/predictions/m1.v.2.9781.csv', 'saved-results/ghasemian/predictions/ghasemian.v.csv')
}

def get_test_col_list(csv_path, col_index):
    col_list = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if int(row[0]) > 1240:
                col_list.append(float(row[col_index]))
    assert len(col_list) == 300
    return col_list

for label, files in sorted(prediction_files.items()):
    a = get_test_col_list(files[0], 1)
    b = get_test_col_list(files[1], 2)
    r = ttest_rel(a, b)
    print('{}: {}'.format(label, r))
