from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import matplotlib
import copy
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as plot
import seaborn as sns
import matplotlib.lines as mlines
import semantic_drift
import tensorflow as tf
import utils
import two_models_exp as tm
import pickle

MULTI_NUM = 20

num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test_orig) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

init_model = tm.custom_model(input_shape, num_classes)

params = {'model': init_model,
          'epochs': 10,
          'x_train': x_train,
          'y_train': y_train,
          'x_test': x_test,
          'y_test': y_test_orig,
          'size': 600,
          'diff': 0.05}

multi_res, additionals = tm.multiple_experiments(tm.get_losses_for_overlapping_labels_w_noise, MULTI_NUM, params)

with open('logs/' + tm.get_result_filename(params) + '_multi.pickle', 'wb') as f:
    pickle.dump(multi_res, f, pickle.HIGHEST_PROTOCOL)

with open('logs/' + tm.get_result_filename(params) + '_additionals.pickle', 'wb') as f:
    pickle.dump(additionals, f, pickle.HIGHEST_PROTOCOL)
