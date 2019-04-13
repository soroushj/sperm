#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import csv
import time
import math
import random
import warnings
import traceback
import importlib.util
from scipy import misc
import numpy as np
import cv2 as cv
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
try:
    from tensorflow.python import keras
except ImportError:
    from tensorflow.contrib.keras.python import keras

models_dir_path = 'models'
results_dir_path = 'results'
checkpoints_dir_path = 'checkpoints'
dataset_name_default = 'gray'
save_summary = False
print_summary = False
flush_epochs = 10
train_size = 1000
image_size = 128
patch_size = 64
folds_dir = 'data-folds'
num_folds = 5


assert os.path.isdir(results_dir_path)
assert os.path.isdir(checkpoints_dir_path)
assert image_size % 2 == 0
assert patch_size % 2 == 0
assert 4 <= len(sys.argv) <= 6
model_name = sys.argv[1]
modeldef_spec = importlib.util.spec_from_file_location('modeldef', os.path.join(models_dir_path, model_name + '.py'))
modeldef = importlib.util.module_from_spec(modeldef_spec)
modeldef_spec.loader.exec_module(modeldef)
label_type = sys.argv[2]
assert label_type in ('h', 'v', 't', 'a')
num_epochs = int(sys.argv[3])
assert num_epochs > 0
if len(sys.argv) > 4:
    settings = int(sys.argv[4])
else:
    settings = 0
assert 0 <= settings <= 3
no_augmentation = settings & 1
no_oversampling = settings & 2
if len(sys.argv) > 5:
    fold_k = int(sys.argv[5])
else:
    fold_k = 0
assert 0 <= fold_k < num_folds
training_name = model_name + '.' + label_type + '.' + str(settings) + '.' + str(fold_k)
class_i_dict = {'h': 9, 'v': 10, 't': 11, 'a': 13}
class_c_dict = {'h': '0', 'v': '0', 't': '0', 'a': 'n'}
class_i = class_i_dict[label_type]
class_c = class_c_dict[label_type]
if 'dataset_name' in modeldef.preprocess_config:
    dataset_name = modeldef.preprocess_config['dataset_name']
    assert type(dataset_name) is str
else:
    dataset_name = dataset_name_default
shift_range = modeldef.preprocess_config['shift_range']
assert shift_range >= 0
rotate_range = modeldef.preprocess_config['rotate_range']
assert rotate_range >= 0
flip_ud = modeldef.preprocess_config['flip_ud']
assert type(flip_ud) is bool
flip_lr = modeldef.preprocess_config['flip_lr']
assert type(flip_lr) is bool
scale_range = modeldef.preprocess_config['scale_range']
assert scale_range >= 1
pca_whitening = modeldef.preprocess_config['pca_whitening']
assert type(pca_whitening) is bool
pca_epsilon = modeldef.preprocess_config['pca_epsilon']
assert pca_epsilon > 0
batch_size = modeldef.train_config['batch_size']
assert batch_size > 0
log_header = [
#    0        1            2
    'epoch', 'train-acc', 'train-loss',
#    3            4             5                  6               7                   8                   9           10          11          12
    'valid-acc', 'valid-loss', 'valid-precision', 'valid-recall', 'valid-f1.0-score', 'valid-f0.5-score', 'valid-tp', 'valid-fp', 'valid-fn', 'valid-tn',
#    13          14           15                16             17                 18                 19         20         21         22
    'test-acc', 'test-loss', 'test-precision', 'test-recall', 'test-f1.0-score', 'test-f0.5-score', 'test-tp', 'test-fp', 'test-fn', 'test-tn'
]
images_dir_path = 'data/images/' + dataset_name
x_valid_file_path = os.path.join(folds_dir, '{}.valid.x.npy'.format(fold_k))
y_valid_file_path = os.path.join(folds_dir, '{}.valid.y.{}.npy'.format(fold_k, label_type))
x_test_file_path = os.path.join(folds_dir, '{}.test.x.npy'.format(fold_k))
y_test_file_path = os.path.join(folds_dir, '{}.test.y.{}.npy'.format(fold_k, label_type))
image_center = image_size // 2
patch_center = patch_size // 2
crop_margin = (image_size - patch_size) // 2
canvas_size = math.ceil(math.sqrt(2 * image_size ** 2))
canvas_size += canvas_size % 2
canvas_center = canvas_size // 2
canvas_margin = (canvas_size - image_size) // 2
scale_log = math.log(scale_range)
x_batch_shape = (batch_size, patch_size, patch_size, 1)
y_batch_shape = (batch_size,)
model_input_shape = (patch_size, patch_size, 1)


def rotate(img, cx, cy , angle):
    return cv.warpAffine(img, cv.getRotationMatrix2D((cx, cy), angle, 1.0), (img.shape[1], img.shape[0]), flags=cv.INTER_LINEAR)

def pca_whiten(X):
    X_cov = np.dot(X.T, X)
    d, V = np.linalg.eigh(X_cov)
    D = np.diag(1 / np.sqrt(d + pca_epsilon))
    W = np.dot(np.dot(V, D), V.T)
    return np.dot(X, W)

def augmented_crop(img_name):
    canvas = loaded_canvases[img_name]
    rotated = rotate(canvas, canvas_center, canvas_center, random.random() * rotate_range)
    crop_x = canvas_center - patch_center + random.randint(-shift_range, shift_range)
    crop_y = canvas_center - patch_center + random.randint(-shift_range, shift_range)
    cropped = rotated[crop_y:crop_y+patch_size, crop_x:crop_x+patch_size]
    if flip_ud and random.randrange(2):
        cropped = np.flipud(cropped)
    if flip_lr and random.randrange(2):
        cropped = np.fliplr(cropped)
    cropped = cropped.astype(np.float32) * (math.exp(random.uniform(-scale_log, scale_log)) / 255)
    cropped -= cropped.mean()
    if pca_whitening:
        cropped = pca_whiten(cropped)
    return cropped

def real_crop(img_name):
    return loaded_crops[img_name]

final_crop = real_crop if no_augmentation else augmented_crop

def get_batch_oversample():
    x = np.ndarray(shape=x_batch_shape, dtype=np.float32)
    y = np.ndarray(shape=y_batch_shape, dtype=np.float32)
    for i in range(batch_size):
        random_class = random.randrange(2)
        sample_index = sample_indices[random_class]
        if sample_index == 0:
            random.shuffle(classes_samples[random_class])
        sample = classes_samples[random_class][sample_index]
        sample_indices[random_class] = (sample_index + 1) % len(classes_samples[random_class])
        x[i, :, :, :] = final_crop(sample).reshape(model_input_shape)
        y[i] = 0 if sample[class_i] == class_c else 1
    return (x, y)

def get_batch_real():
    global sample_index_real
    x = np.ndarray(shape=x_batch_shape, dtype=np.float32)
    y = np.ndarray(shape=y_batch_shape, dtype=np.float32)
    for i in range(batch_size):
        if sample_index_real == 0:
            random.shuffle(train_samples)
        sample = train_samples[sample_index_real]
        sample_index_real = (sample_index_real + 1) % train_size
        x[i, :, :, :] = final_crop(sample).reshape(model_input_shape)
        y[i] = 0 if sample[class_i] == class_c else 1
    return (x, y)

get_batch = get_batch_real if no_oversampling else get_batch_oversample

def get_checkpoint_path(epoch):
    return os.path.join(checkpoints_dir_path, training_name + '.' + str(epoch) + '.h5')

def remove_checkpoint(epoch):
    checkpoint_path = get_checkpoint_path(epoch)
    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)

M = 60
H = 60 * M
D = 24 * H
U = ['d', 'h', 'm', 's']
def eta():
    num_done = session_epoch_counter + 1
    if num_done == 0:
        return 'unknown'
    if num_done >= num_epochs:
        return 'compeleted'
    s = round((num_epochs - num_done) * (time.time() - start_time) / num_done)
    d = s // D
    s = s % D
    h = s // H
    s = s % H
    m = s // M
    s = s % M
    V = [d, h, m, s]
    return ' '.join([str(v) + u for v, u in zip(V, U) if v != 0])

finalized = False
def finalize():
    global finalized
    if finalized:
        return
    print('Finalizing')
    if not last_checkpoint_saved:
        model.save(get_checkpoint_path(model_epoch_counter))
    model_log_file.close()
    finalized = True


best_checkpoints = {
    'valid_loss': {'epoch': 0, 'value': math.inf},
    'valid_acc': {'epoch': 0, 'value': 0.0},
    'valid_f10': {'epoch': 0, 'value': 0.0},
    'valid_f05': {'epoch': 0, 'value': 0.0},
    'test_loss': {'epoch': 0, 'value': math.inf},
    'test_acc': {'epoch': 0, 'value': 0.0},
    'test_f10': {'epoch': 0, 'value': 0.0},
    'test_f05': {'epoch': 0, 'value': 0.0}
}

model_log_file_path = os.path.join(results_dir_path, training_name + '.csv')
model_log_file_exists = os.path.isfile(model_log_file_path) and os.path.getsize(model_log_file_path)
if model_log_file_exists:
    with open(model_log_file_path, 'r', newline='') as model_log_file:
        model_log_reader = csv.reader(model_log_file)
        print('Reading old checkpoints')
        for row_counter, row in enumerate(model_log_reader):
            if row_counter == 0:
                for r, h in zip(row, log_header):
                    assert r == h
            else:
                assert int(row[0]) == row_counter

                checkpoint_valid_loss = float(row[4])
                checkpoint_valid_acc = float(row[3])
                checkpoint_valid_f10 = float(row[7])
                checkpoint_valid_f05 = float(row[8])
                if checkpoint_valid_loss < best_checkpoints['valid_loss']['value']:
                    best_checkpoints['valid_loss']['value'] = checkpoint_valid_loss
                    best_checkpoints['valid_loss']['epoch'] = row_counter
                if checkpoint_valid_acc > best_checkpoints['valid_acc']['value']:
                    best_checkpoints['valid_acc']['value'] = checkpoint_valid_acc
                    best_checkpoints['valid_acc']['epoch'] = row_counter
                if checkpoint_valid_f10 > best_checkpoints['valid_f10']['value']:
                    best_checkpoints['valid_f10']['value'] = checkpoint_valid_f10
                    best_checkpoints['valid_f10']['epoch'] = row_counter
                if checkpoint_valid_f05 > best_checkpoints['valid_f05']['value']:
                    best_checkpoints['valid_f05']['value'] = checkpoint_valid_f05
                    best_checkpoints['valid_f05']['epoch'] = row_counter
                
                checkpoint_test_loss = float(row[14])
                checkpoint_test_acc = float(row[13])
                checkpoint_test_f10 = float(row[17])
                checkpoint_test_f05 = float(row[18])
                if checkpoint_test_loss < best_checkpoints['test_loss']['value']:
                    best_checkpoints['test_loss']['value'] = checkpoint_test_loss
                    best_checkpoints['test_loss']['epoch'] = row_counter
                if checkpoint_test_acc > best_checkpoints['test_acc']['value']:
                    best_checkpoints['test_acc']['value'] = checkpoint_test_acc
                    best_checkpoints['test_acc']['epoch'] = row_counter
                if checkpoint_test_f10 > best_checkpoints['test_f10']['value']:
                    best_checkpoints['test_f10']['value'] = checkpoint_test_f10
                    best_checkpoints['test_f10']['epoch'] = row_counter
                if checkpoint_test_f05 > best_checkpoints['test_f05']['value']:
                    best_checkpoints['test_f05']['value'] = checkpoint_test_f05
                    best_checkpoints['test_f05']['epoch'] = row_counter
        model_epoch_counter = row_counter
else:
    model_epoch_counter = 0

if model_epoch_counter == 0:
    model = modeldef.get_model(model_input_shape)
    if save_summary:
        with open(os.path.join(models_dir_path, model_name + '.txt'), 'w') as model_summary_file:
            orig_stdout = sys.stdout
            sys.stdout = model_summary_file
            model.summary()
            sys.stdout = orig_stdout
else:
    model = keras.models.load_model(get_checkpoint_path(model_epoch_counter))

if print_summary:
    model.summary()

model_log_file = open(model_log_file_path, 'a', newline='')
model_log_writer = csv.writer(model_log_file)
if not model_log_file_exists:
    model_log_writer.writerow(log_header)


train_samples = os.listdir(images_dir_path)
train_samples.sort()
train_samples = train_samples[:train_size]
labels = [0, 1]
if no_oversampling:
    sample_index_real = 0
else:
    classes_samples = [[], []]
    sample_indices = [0, 0]
    for s in train_samples:
        classes_samples[0 if s[class_i] == class_c else 1].append(s)

x_valid = np.load(x_valid_file_path)
if pca_whitening:
    print('Whitening valid samples')
    for i in range(x_valid.shape[0]):
        x_valid[i, :, :] = pca_whiten(x_valid[i])
x_valid = x_valid.reshape(*x_valid.shape, 1)
y_valid = np.load(y_valid_file_path).astype(np.float32)

x_test = np.load(x_test_file_path)
if pca_whitening:
    print('Whitening test samples')
    for i in range(x_test.shape[0]):
        x_test[i, :, :] = pca_whiten(x_test[i])
x_test = x_test.reshape(*x_test.shape, 1)
y_test = np.load(y_test_file_path).astype(np.float32)

print('Loading images')
if no_augmentation:
    loaded_crops = {}
    for img_name in train_samples:
        image = misc.imread(os.path.join(images_dir_path, img_name))
        image = image[crop_margin:crop_margin+patch_size, crop_margin:crop_margin+patch_size]
        image = image.astype(np.float32) / 255
        image -= image.mean()
        loaded_crops[img_name] = image
else:
    loaded_canvases = {}
    for img_name in train_samples:
        image = misc.imread(os.path.join(images_dir_path, img_name))
        canvas = np.ndarray(shape=(canvas_size, canvas_size), dtype=np.uint8)
        canvas[:, :] = round(image.mean())
        canvas[canvas_margin:canvas_margin+image_size, canvas_margin:canvas_margin+image_size] = image
        loaded_canvases[img_name] = canvas

print('Starting')
last_checkpoint_saved = False
start_time = time.time()

try:
    for session_epoch_counter in range(num_epochs):

        train_loss, train_acc = model.train_on_batch(*get_batch())
        model_epoch_counter += 1
        last_checkpoint_saved = False

        valid_loss, valid_acc = model.evaluate(x_valid, y_valid, batch_size=x_valid.shape[0], verbose=0)
        y_valid_predicted = model.predict_classes(x_valid, batch_size=x_valid.shape[0], verbose=0).reshape(y_valid.shape).astype(y_valid.dtype)
        valid_tp, valid_fn, valid_fp, valid_tn = confusion_matrix(y_valid, y_valid_predicted, labels=labels).ravel()
        valid_precision, valid_recall, valid_f10, _ = precision_recall_fscore_support(y_valid, y_valid_predicted, beta=1.0, labels=labels)
        _, _, valid_f05, _ = precision_recall_fscore_support(y_valid, y_valid_predicted, beta=0.5, labels=labels)
        valid_precision = valid_precision[0]
        valid_recall = valid_recall[0]
        valid_f10 = valid_f10[0]
        valid_f05 = valid_f05[0]

        test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=x_test.shape[0], verbose=0)
        y_test_predicted = model.predict_classes(x_test, batch_size=x_test.shape[0], verbose=0).reshape(y_test.shape).astype(y_test.dtype)
        test_tp, test_fn, test_fp, test_tn = confusion_matrix(y_test, y_test_predicted, labels=labels).ravel()
        test_precision, test_recall, test_f10, _ = precision_recall_fscore_support(y_test, y_test_predicted, beta=1.0, labels=labels)
        _, _, test_f05, _ = precision_recall_fscore_support(y_test, y_test_predicted, beta=0.5, labels=labels)
        test_precision = test_precision[0]
        test_recall = test_recall[0]
        test_f10 = test_f10[0]
        test_f05 = test_f05[0]

        log_row = [
            model_epoch_counter, train_acc, train_loss,
            valid_acc, valid_loss, valid_precision, valid_recall, valid_f10, valid_f05, valid_tp, valid_fp, valid_fn, valid_tn,
            test_acc, test_loss, test_precision, test_recall, test_f10, test_f05, test_tp, test_fp, test_fn, test_tn
        ]
        model_log_writer.writerow(log_row)
        if model_epoch_counter % flush_epochs == 0:
            model_log_file.flush()

        epochs_to_remove = set()
        
        if valid_loss < best_checkpoints['valid_loss']['value']:
            print('New best valid loss:', valid_loss)
            best_checkpoints['valid_loss']['value'] = valid_loss
            epochs_to_remove.add(best_checkpoints['valid_loss']['epoch'])
            best_checkpoints['valid_loss']['epoch'] = model_epoch_counter
        if valid_acc > best_checkpoints['valid_acc']['value']:
            print('New best valid acc:', valid_acc)
            best_checkpoints['valid_acc']['value'] = valid_acc
            epochs_to_remove.add(best_checkpoints['valid_acc']['epoch'])
            best_checkpoints['valid_acc']['epoch'] = model_epoch_counter
        if valid_f10 > best_checkpoints['valid_f10']['value']:
            print('New best valid f1.0:', valid_f10)
            best_checkpoints['valid_f10']['value'] = valid_f10
            epochs_to_remove.add(best_checkpoints['valid_f10']['epoch'])
            best_checkpoints['valid_f10']['epoch'] = model_epoch_counter
        if valid_f05 > best_checkpoints['valid_f05']['value']:
            print('New best valid f0.5:', valid_f05)
            best_checkpoints['valid_f05']['value'] = valid_f05
            epochs_to_remove.add(best_checkpoints['valid_f05']['epoch'])
            best_checkpoints['valid_f05']['epoch'] = model_epoch_counter
        
        if test_loss < best_checkpoints['test_loss']['value']:
            print('New best test loss:', test_loss)
            best_checkpoints['test_loss']['value'] = test_loss
            epochs_to_remove.add(best_checkpoints['test_loss']['epoch'])
            best_checkpoints['test_loss']['epoch'] = model_epoch_counter
        if test_acc > best_checkpoints['test_acc']['value']:
            print('New best test acc:', test_acc)
            best_checkpoints['test_acc']['value'] = test_acc
            epochs_to_remove.add(best_checkpoints['test_acc']['epoch'])
            best_checkpoints['test_acc']['epoch'] = model_epoch_counter
        if test_f10 > best_checkpoints['test_f10']['value']:
            print('New best test f1.0:', test_f10)
            best_checkpoints['test_f10']['value'] = test_f10
            epochs_to_remove.add(best_checkpoints['test_f10']['epoch'])
            best_checkpoints['test_f10']['epoch'] = model_epoch_counter
        if test_f05 > best_checkpoints['test_f05']['value']:
            print('New best test f0.5:', test_f05)
            best_checkpoints['test_f05']['value'] = test_f05
            epochs_to_remove.add(best_checkpoints['test_f05']['epoch'])
            best_checkpoints['test_f05']['epoch'] = model_epoch_counter

        if len(epochs_to_remove) != 0:
            model.save(get_checkpoint_path(model_epoch_counter))
            last_checkpoint_saved = True
            for metric in best_checkpoints:
                if best_checkpoints[metric]['epoch'] in epochs_to_remove:
                    epochs_to_remove.remove(best_checkpoints[metric]['epoch'])
            for epoch_to_remove in epochs_to_remove:
                if epoch_to_remove != 0:
                    remove_checkpoint(epoch_to_remove)
        
        print(training_name + ' ', session_epoch_counter + 1, '/', num_epochs, ' (', model_epoch_counter, ') ETA ', eta(), sep='')

except KeyboardInterrupt:
    print()
    finalize()
except Exception:
    traceback.print_exc()
    finalize()

finalize()
