#!/usr/bin/env python3

import math

def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def precision(tp, tn, fp, fn):
    return tp / (tp + fp)

def recall(tp, tn, fp, fn):
    return tp / (tp + fn)

def f_beta(tp, tn, fp, fn, beta):
    p = precision(tp, tn, fp, fn)
    r = recall(tp, tn, fp, fn)
    return (1 + beta**2) * p * r / (beta**2 * p + r)

def g_mean(tp, tn, fp, fn):
    return tp / math.sqrt((tp + fp) * (tp + fn))

def mcc(tp, tn, fp, fn):
    return (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

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

gh_tail_tp = 274
gh_tail_tn = 3
gh_tail_fp = 10
gh_tail_fn = 13

gh_head_tp = 168
gh_head_tn = 15
gh_head_fp = 51
gh_head_fn = 66

gh_vacuole_tp = 218
gh_vacuole_tn = 23
gh_vacuole_fp = 44
gh_vacuole_fn = 15

print('ours, accuracy, acrosome: {:.4f}'.format(accuracy(acrosome_tp, acrosome_tn, acrosome_fp, acrosome_fn)))
print('ours, accuracy, head: {:.4f}'.format(accuracy(head_tp, head_tn, head_fp, head_fn)))
print('ours, accuracy, vacuole: {:.4f}'.format(accuracy(vacuole_tp, vacuole_tn, vacuole_fp, vacuole_fn)))
print()
print('ours, precision, acrosome: {:.4f}'.format(precision(acrosome_tp, acrosome_tn, acrosome_fp, acrosome_fn)))
print('ours, precision, head: {:.4f}'.format(precision(head_tp, head_tn, head_fp, head_fn)))
print('ours, precision, vacuole: {:.4f}'.format(precision(vacuole_tp, vacuole_tn, vacuole_fp, vacuole_fn)))
print()
print('ours, recall, acrosome: {:.4f}'.format(recall(acrosome_tp, acrosome_tn, acrosome_fp, acrosome_fn)))
print('ours, recall, head: {:.4f}'.format(recall(head_tp, head_tn, head_fp, head_fn)))
print('ours, recall, vacuole: {:.4f}'.format(recall(vacuole_tp, vacuole_tn, vacuole_fp, vacuole_fn)))
print()
print('ours, f_0.5, acrosome: {:.4f}'.format(f_beta(acrosome_tp, acrosome_tn, acrosome_fp, acrosome_fn, 0.5)))
print('ours, f_0.5, head: {:.4f}'.format(f_beta(head_tp, head_tn, head_fp, head_fn, 0.5)))
print('ours, f_0.5, vacuole: {:.4f}'.format(f_beta(vacuole_tp, vacuole_tn, vacuole_fp, vacuole_fn, 0.5)))
print()
print('ours, g_mean, acrosome: {:.4f}'.format(g_mean(acrosome_tp, acrosome_tn, acrosome_fp, acrosome_fn)))
print('ours, g_mean, head: {:.4f}'.format(g_mean(head_tp, head_tn, head_fp, head_fn)))
print('ours, g_mean, vacuole: {:.4f}'.format(g_mean(vacuole_tp, vacuole_tn, vacuole_fp, vacuole_fn)))
print()
print('ours, mcc, acrosome: {:.4f}'.format(mcc(acrosome_tp, acrosome_tn, acrosome_fp, acrosome_fn)))
print('ours, mcc, head: {:.4f}'.format(mcc(head_tp, head_tn, head_fp, head_fn)))
print('ours, mcc, vacuole: {:.4f}'.format(mcc(vacuole_tp, vacuole_tn, vacuole_fp, vacuole_fn)))
print()
print()
print()
print('ghasemian, accuracy, tail: {:.4f}'.format(accuracy(gh_tail_tp, gh_tail_tn, gh_tail_fp, gh_tail_fn)))
print('ghasemian, accuracy, head: {:.4f}'.format(accuracy(gh_head_tp, gh_head_tn, gh_head_fp, gh_head_fn)))
print('ghasemian, accuracy, vacuole: {:.4f}'.format(accuracy(gh_vacuole_tp, gh_vacuole_tn, gh_vacuole_fp, gh_vacuole_fn)))
print()
print('ghasemian, precision, tail: {:.4f}'.format(precision(gh_tail_tp, gh_tail_tn, gh_tail_fp, gh_tail_fn)))
print('ghasemian, precision, head: {:.4f}'.format(precision(gh_head_tp, gh_head_tn, gh_head_fp, gh_head_fn)))
print('ghasemian, precision, vacuole: {:.4f}'.format(precision(gh_vacuole_tp, gh_vacuole_tn, gh_vacuole_fp, gh_vacuole_fn)))
print()
print('ghasemian, recall, tail: {:.4f}'.format(recall(gh_tail_tp, gh_tail_tn, gh_tail_fp, gh_tail_fn)))
print('ghasemian, recall, head: {:.4f}'.format(recall(gh_head_tp, gh_head_tn, gh_head_fp, gh_head_fn)))
print('ghasemian, recall, vacuole: {:.4f}'.format(recall(gh_vacuole_tp, gh_vacuole_tn, gh_vacuole_fp, gh_vacuole_fn)))
print()
print('ghasemian, f_0.5, tail: {:.4f}'.format(f_beta(gh_tail_tp, gh_tail_tn, gh_tail_fp, gh_tail_fn, 0.5)))
print('ghasemian, f_0.5, head: {:.4f}'.format(f_beta(gh_head_tp, gh_head_tn, gh_head_fp, gh_head_fn, 0.5)))
print('ghasemian, f_0.5, vacuole: {:.4f}'.format(f_beta(gh_vacuole_tp, gh_vacuole_tn, gh_vacuole_fp, gh_vacuole_fn, 0.5)))
print()
print('ghasemian, g_mean, tail: {:.4f}'.format(g_mean(gh_tail_tp, gh_tail_tn, gh_tail_fp, gh_tail_fn)))
print('ghasemian, g_mean, head: {:.4f}'.format(g_mean(gh_head_tp, gh_head_tn, gh_head_fp, gh_head_fn)))
print('ghasemian, g_mean, vacuole: {:.4f}'.format(g_mean(gh_vacuole_tp, gh_vacuole_tn, gh_vacuole_fp, gh_vacuole_fn)))
print()
print('ghasemian, mcc, tail: {:.4f}'.format(mcc(gh_tail_tp, gh_tail_tn, gh_tail_fp, gh_tail_fn)))
print('ghasemian, mcc, head: {:.4f}'.format(mcc(gh_head_tp, gh_head_tn, gh_head_fp, gh_head_fn)))
print('ghasemian, mcc, vacuole: {:.4f}'.format(mcc(gh_vacuole_tp, gh_vacuole_tn, gh_vacuole_fp, gh_vacuole_fn)))
