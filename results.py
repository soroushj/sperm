#!/usr/bin/env python3

import pandas as pd

csv_file_path = 'saved-results/results/m1.a.0.csv'

df = pd.read_csv(csv_file_path)
df = df.sort_values('valid-loss')

print('head:')
print(df.head())
print()
print('first row - dataframe:')
print(df.iloc[[0]])
print()
print('first row - series:')
print(df.iloc[0])
