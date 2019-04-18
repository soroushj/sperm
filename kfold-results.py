#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import math
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score

results_dir = 'saved-results/kfold/results1'
checkpoints_dir = 'saved-results/kfold/checkpoints1'
folds_dir = 'data-folds'
model = 'm1'
labels = ['a', 'h', 'v']
config = 0
num_folds = 5
cols = ['epoch', 'test-acc', 'test-precision', 'test-recall', 'test-f0.5-score', 'test-tp', 'test-fp', 'test-fn', 'test-tn']
cols_new = ['epoch', 'accuracy', 'precision', 'recall', 'f05', 'tp', 'fp', 'fn', 'tn']
metrics = ['accuracy', 'precision', 'recall', 'f05', 'gmean', 'auc', 'mcc']

def get_result(csv_path):
    result = pd.read_csv(csv_path).sort_values('valid-loss')[cols].iloc[[0]]
    result.columns = cols_new
    return result

def g_mean(tp, tn, fp, fn):
    return tp / math.sqrt((tp + fp) * (tp + fn))

def mcc(tp, tn, fp, fn):
    return (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

result_dfs = []
for label in labels:
    for fold in range(num_folds):
        result_path = os.path.join(results_dir, '{}.{}.{}.{}.csv'.format(model, label, config, fold))
        result = get_result(result_path)
        epoch = result[['epoch']].iloc[0].values[0]
        model_path = os.path.join(checkpoints_dir, '{}.{}.{}.{}.{}.h5'.format(model, label, config, fold, epoch))
        model = load_model(model_path)
        x_path = os.path.join(folds_dir, '{}.test.x.npy'.format(fold))
        y_path = os.path.join(folds_dir, '{}.test.y.{}.npy'.format(fold, label))
        x = np.expand_dims(np.load(x_path), axis=-1)
        y = np.load(y_path)
        predictions = model.predict(x, batch_size=y.shape[0], verbose=0).reshape(y.shape)
        auc = roc_auc_score(y, predictions)
        result.insert(0, 'label', [label])
        result.insert(1, 'fold', [fold])
        tp, fp, fn, tn = result[['tp', 'fp', 'fn', 'tn']].iloc[0].values
        result.insert(7, 'gmean', [g_mean(tp, tn, fp, fn)])
        result.insert(8, 'auc', [auc])
        result.insert(9, 'mcc', [mcc(tp, tn, fp, fn)])
        result_dfs.append(result)

results = pd.concat(result_dfs)
results_path = results_dir + '.csv'
results.to_csv(results_path, index=False)
print('Saved', results_path)

agg_cols = ['label']
for metric in metrics:
    agg_cols.append(metric + '-mean')
    agg_cols.append(metric + '-std')

results_agg = pd.DataFrame(columns=agg_cols)
for i, label in enumerate(labels):
    label_results = results.loc[results['label'] == label]
    agg_data = {'label': label}
    for metric in metrics:
        agg_data[metric + '-mean'] = label_results[metric].mean()
        agg_data[metric + '-std'] = label_results[metric].std()
    results_agg.loc[i] = agg_data

results_agg_path = results_dir + '.agg.csv'
results_agg.to_csv(results_agg_path, index=False)
print('Saved', results_agg_path)
