#!/usr/bin/env python3

import os
import csv
from functools import reduce

predictions_dir = 'saved-results/predictions'

in_files = [open(os.path.join(predictions_dir, fn)) for fn in os.listdir(predictions_dir)]
readers = [csv.reader(f) for f in in_files]
out_file = open('saved-results/aggregated-predictions-bin.csv', 'w')
writer = csv.writer(out_file)

writer.writerow(['Sample', 'Actual Class', 'Predicted Class', 'Truth'])
for reader in readers:
    next(reader)

for rows in zip(*readers):
    sample = rows[0][0]
    actual_class = reduce(lambda c1, c2: c1 or c2, map(lambda r: int(r[2]), rows))
    predicted_class = reduce(lambda c1, c2: c1 or c2, map(lambda r: int(r[3]), rows))
    truth = actual_class == predicted_class
    writer.writerow([sample, actual_class, predicted_class, truth])

out_file.close()
for in_file in in_files:
    in_file.close()
