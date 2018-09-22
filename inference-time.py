#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
from keras.models import load_model

model_filename = 'saved/final-models/m1.a.0.8963.h5'
x_filename = 'data/dataset/gray/x_test.npy'
n_samples = 32
n_iters = 100

model = load_model(model_filename)
x = np.load(x_filename)[:n_samples, :, :]
x = x.reshape((*x.shape, 1))
n_samples = x.shape[0]

start_time = time.time()
for _ in range(n_iters):
    model.predict(x, batch_size=n_samples)
end_time = time.time()

elapsed_time = end_time - start_time
avg_inference_time = elapsed_time / (n_iters * n_samples)
print('Average inference time:', avg_inference_time, 'seconds')
