#!/usr/bin/env python3

import os
import csv
from sklearn.metrics import roc_auc_score

predictions_dirs = [
    'saved-results/predictions',
    'saved-results/ghasemian/predictions',
]

for predictions_dir in predictions_dirs:
    for filename in os.listdir(predictions_dir):
        with open(os.path.join(predictions_dir, filename)) as f:
            reader = csv.reader(f)
            next(reader)
            y_true = []
            y_score = []
            for row in reader:
                if int(row[0]) > 1240:
                    y_true.append(float(row[2]))
                    y_score.append(float(row[1]))
            assert len(y_true) == 300
            auc = roc_auc_score(y_true, y_score)
            print('{}: {:.4f}'.format(filename, auc))
