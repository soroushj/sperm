#!/usr/bin/env python3

import math

def mcc(tp, tn, fp, fn):
    return (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

acrosome_tp = 171
acrosome_tn = 59
acrosome_fp = 28
acrosome_fn = 42
print('acrosome: {:.4f}'.format(mcc(acrosome_tp, acrosome_tn, acrosome_fp, acrosome_fn)))

head_tp = 187
head_tn = 44
head_fp = 37
head_fn = 32
print('head: {:.4f}'.format(mcc(head_tp, head_tn, head_fp, head_fn)))

vacuole_tp = 251
vacuole_tn = 23
vacuole_fp = 15
vacuole_fn = 11
print('vacuole: {:.4f}'.format(mcc(vacuole_tp, vacuole_tn, vacuole_fp, vacuole_fn)))
