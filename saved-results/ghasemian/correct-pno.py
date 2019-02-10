#!/usr/bin/env python3

import csv

with open('Result-Feature-New-corrected-pno.csv', 'w') as out_file:
    writer = csv.writer(out_file)
    with open('Result-Feature-New.csv') as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            if not row[1]:
                row[1] = last_patient_no
            else:
                last_patient_no = row[1]
            writer.writerow(row)
