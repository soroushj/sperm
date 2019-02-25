#!/usr/bin/env python3

import csv

# labels/preds: h, v, t

with open('final-map.csv') as map_file:
    map_reader = csv.reader(map_file)
    id_to_name = {int(row[1][:4]): row[0] for row in map_reader}

with open('final-map.csv') as map_file:
    map_reader = csv.reader(map_file)
    id_to_labels = {int(row[1][:4]): tuple(map(int, row[1][9:12])) for row in map_reader}

with open('final-tags.csv') as tag_file:
    with open('Result-Feature-New-corrected-pno-synched-with-mhsma.csv') as res_file:
        tag_reader = csv.reader(tag_file)
        res_reader = csv.reader(res_file)
        next(tag_reader)
        next(res_reader)
        name_to_preds = {}
        for tag_row, res_row in zip(tag_reader, res_reader):
            h = int(res_row[6])
            v = int(res_row[7])
            t = int(res_row[9])
            assert h in (0, 1)
            assert v in (0, 1)
            assert t in (0, 1)
            h = 1 - h
            name_to_preds[tag_row[9]] = (h, v, t)

header = ['Sample', 'Actual Class', 'Predicted Class', 'Truth']
metrics = []
for l, c in enumerate(('h', 'v', 't')):
    total_truth = 0
    test_truth = 0
    with open('predictions/ghasemian.{}.csv'.format(c), 'w') as eval_file:
        evl_writer = csv.writer(eval_file)
        evl_writer.writerow(header)
        for i in range(1, 1541):
            actual = id_to_labels[i][l]
            pred = name_to_preds[id_to_name[i]][l]
            truth = actual == pred
            evl_writer.writerow([i, actual, pred, truth])
            if truth:
                total_truth += 1
                if i > 1240:
                    test_truth += 1
    total_acc = total_truth / 1540
    test_acc = test_truth / 300
    metrics.append('total accuracy {}: {:.4f}'.format(c, total_acc))
    metrics.append('test accuracy {}: {:.4f}'.format(c, test_acc))
print('\n'.join(metrics))
