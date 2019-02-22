#!/usr/bin/env python3

import csv

with open('final-map.csv') as f:
    reader = csv.raeder(f)
    new_index_to_common_name = {row[1][:4]: row[0] for row in reader}

print(new_index_to_common_name)
