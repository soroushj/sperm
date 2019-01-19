#!/usr/bin/env python3

import math

def mcc(tp, tn, fp, fn):
    return (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

def g_mean(tp, tn, fp, fn):
    return tp / math.sqrt((tp + fp) * (tp + fn))

acrosome_tp = 171
acrosome_tn = 59
acrosome_fp = 28
acrosome_fn = 42

head_tp = 187
head_tn = 44
head_fp = 37
head_fn = 32

vacuole_tp = 251
vacuole_tn = 23
vacuole_fp = 15
vacuole_fn = 11

print('mcc, acrosome: {:.4f}'.format(mcc(acrosome_tp, acrosome_tn, acrosome_fp, acrosome_fn)))
print('mcc, head: {:.4f}'.format(mcc(head_tp, head_tn, head_fp, head_fn)))
print('mcc, vacuole: {:.4f}'.format(mcc(vacuole_tp, vacuole_tn, vacuole_fp, vacuole_fn)))

print('g_mean, acrosome: {:.4f}'.format(g_mean(acrosome_tp, acrosome_tn, acrosome_fp, acrosome_fn)))
print('g_mean, head: {:.4f}'.format(g_mean(head_tp, head_tn, head_fp, head_fn)))
print('g_mean, vacuole: {:.4f}'.format(g_mean(vacuole_tp, vacuole_tn, vacuole_fp, vacuole_fn)))
