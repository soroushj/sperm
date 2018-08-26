#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import csv
from scipy import misc
import numpy as np
from keras.models import load_model

class_i_dict = {'h': 9, 'v': 10, 't': 11, 'a': 13}
class_c_dict = {'h': '0', 'v': '0', 't': '0', 'a': 'n'}
def get_class(image_name, label):
    return 0 if image_name[class_i_dict[label]] == class_c_dict[label] else 1

images_dir = '../sperm-data-preparation/images/01-gray-crop-64'
image_names = os.listdir(images_dir)
image_names.sort()
models_dir = 'saved/final-models'
model_names = os.listdir(models_dir)
model_names.sort()
predictions_dir = 'saved/predictions'
header = ['Sample', 'Prediction', 'Actual Class', 'Predicted Class', 'Truth']

print('Loading images...', end=' ')
x = np.ndarray(shape=(len(image_names), 64, 64, 1), dtype=np.float32)
for i, image_name in enumerate(image_names):
    image = misc.imread(os.path.join(images_dir, image_name)).reshape((64, 64, 1)).astype(np.float32) / 255
    image -= image.mean()
    x[i, :, :, :] = image
print('done')

for model_name in model_names:
    print('Model:', model_name)
    model = load_model(os.path.join(models_dir, model_name))
    predictions = model.predict(x, batch_size=64, verbose=2)
    predictions = predictions.reshape(predictions.shape[:1])
    label = model_name[model_name.find('.') + 1]
    model_basename = os.path.splitext(model_name)[0]
    predictions_filename = os.path.join(predictions_dir, model_basename + '.csv')
    predictions_file = open(predictions_filename, 'w', newline='')
    predictions_writer = csv.writer(predictions_file)
    predictions_writer.writerow(header)
    for i, prediction in enumerate(predictions):
        actual_class = get_class(image_names[i], label)
        predicted_class = 0 if prediction < 0.5 else 1
        predictions_writer.writerow([i + 1, prediction, actual_class, predicted_class, actual_class == predicted_class])
    predictions_file.close()
