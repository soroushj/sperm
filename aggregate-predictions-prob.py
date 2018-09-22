#!/usr/bin/env python3

import os
import csv
from functools import reduce

predictions_dir = 'saved-results/predictions'

in_files = [open(os.path.join(predictions_dir, fn)) for fn in os.listdir(predictions_dir)]
readers = [csv.reader(f) for f in in_files]
out_file = open('saved-results/aggregated-predictions-prob.csv', 'w')
writer = csv.writer(out_file)

writer.writerow(['Sample', 'Prediction', 'Actual Class', 'Predicted Class', 'Truth'])
for reader in readers:
    next(reader)

for rows in zip(*readers):
    sample = rows[0][0]
    prediction = sum(map(lambda r: float(r[1]), rows)) / len(rows)
    actual_class = reduce(lambda c1, c2: c1 or c2, map(lambda r: int(r[2]), rows))
    predicted_class = 0 if prediction < 0.5 else 1
    truth = actual_class == predicted_class
    writer.writerow([sample, prediction, actual_class, predicted_class, truth])

out_file.close()
for in_file in in_files:
    in_file.close()
