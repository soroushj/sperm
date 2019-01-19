#!/usr/bin/env python3

import os
import csv
from sklearn.metrics import roc_auc_score

predictions_dir = 'saved-results/predictions'

for filename in os.listdir(predictions_dir):
    label = filename[3]
    in_file = open(os.path.join(predictions_dir, filename))
    reader = csv.reader(in_file)
    next(reader)
    y_true = []
    y_score = []
    for row in reader:
        y_true.append(int(row[2]))
        y_score.append(float(row[1]))
    auc = roc_auc_score(y_true, y_score)
    print('{}: {:.4f}'.format(label, auc))
    in_file.close()
