#!/usr/bin/env python3

import os
import csv
from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar

prediction_files = {
    'h': ('saved-results/predictions/m1.h.0.9385.csv', 'saved-results/ghasemian/predictions/ghasemian.h.csv'),
    'v': ('saved-results/predictions/m1.v.2.9781.csv', 'saved-results/ghasemian/predictions/ghasemian.v.csv')
}

def get_test_col_list(csv_path):
    col_list = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if int(row[0]) > 1240:
                col_list.append(float(row[-1] == 'True'))
    assert len(col_list) == 300
    return col_list

for label, files in sorted(prediction_files.items()):
    a = get_test_col_list(files[0])
    b = get_test_col_list(files[1])
    # contingency table
    # 00=YY 01=YN
    # 10=NY 11=NN
    ct = [
        [0, 0],
        [0, 0]
    ]
    for x, y in zip(a, b):
        if x and y:
            ct[0][0] += 1
        elif x and not y:
            ct[0][1] += 1
        elif not x and y:
            ct[1][0] += 1
        elif not x and not y:
            ct[1][1] += 1
    assert sum(ct[0]) + sum(ct[1]) == 300
    tt = ttest_rel(a, b)
    mc = mcnemar(ct, exact=True)
    print(' t-test {}\t\tstat={:+.2f}\tpval={:.6f}\t\tstat={}\tpval={}'.format(label, tt.statistic, tt.pvalue, tt.statistic, tt.pvalue))
    print('mcnemar {}\t\tstat={:+.2f}\tpval={:.6f}\t\tstat={}\t\tpval={}'.format(label, mc.statistic, mc.pvalue, mc.statistic, mc.pvalue))
    print()
