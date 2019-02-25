#!/usr/bin/env python3

import csv

for path in ('ghasemian.h.csv', 'ghasemian.v.csv', 'ghasemian.t.csv'):
    tp, tn, fp, fn = 0, 0, 0, 0
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            sample, actual, pred = tuple(map(int, row[:3]))
            if sample > 1240:
                if actual == pred:
                    if actual == 0:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if actual == 0:
                        fp += 1
                    else:
                        fn += 1
    assert tp + tn + fp + fn == 300
    print(path)
    print('tp = {}\ntn = {}\nfp = {}\nfn = {}'.format(tp, tn, fp, fn))
    print()