# -*- coding: utf-8 -*-

import os
import random as rn
import sys
from datetime import datetime

import joblib
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from skimage import io, img_as_ubyte
import keras_ft

if len(sys.argv) != 7:
    print('Usage: python 04_ft.py <base_dir> <dataset_name> <model_name> <batch_size> <img_multiplier> <epochs>')
    exit(1)

base_dir = sys.argv[1]
dataset_name = sys.argv[2]
model_name = sys.argv[3]
batch_size = int(sys.argv[4])
img_mult = int(sys.argv[5])
epochs = int(sys.argv[6])

ts = datetime.now()
ts_str = datetime.strftime(ts, '%Y%m%d_%H%M%S')

#begin set seed
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(42)

tf.set_random_seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
#end set seed

X = np.load(os.path.join(base_dir, 'X_%s.npy' % dataset_name))
y = np.load(os.path.join(base_dir, 'y_%s.npy' % dataset_name))
labels = joblib.load(os.path.join(base_dir, 'label_names_%s.pkl' % dataset_name))

shape = X.shape[1:]
n_classes = len(np.unique(y))

lbin = LabelBinarizer()
y = lbin.fit_transform(y)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

X = None

print('timestamp: %s' % ts_str)
print('model: %s, # classes: %d' % (model_name, n_classes))
print('shape: ', shape)
print('batch_size: %d, img_mult: %d, epochs: %d' % (batch_size, img_mult, epochs))

datagen = ImageDataGenerator(
    zca_whitening=False,
    rotation_range=270,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect')

ft = keras_ft.KerasFineTuner(model_name, shape, n_classes)

ft.fit_top_model(   X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    datagen=datagen,
                    img_mult=img_mult,
                    batch_size=batch_size,
                    epochs=epochs)

ft.fine_tune_model( X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    datagen=datagen,
                    img_mult=img_mult,
                    batch_size=batch_size,
                    epochs=epochs)

pred_dir = os.path.join(base_dir, '%s_predictions_%s_%s' % (ts_str, dataset_name, model_name))

if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

y_pred = ft.predict(X_valid)
y_pred_seq = lbin.inverse_transform(y_pred)

for i, c in enumerate(y_pred_seq):
    class_name = labels[c]
    class_dir = os.path.join(pred_dir, class_name)

    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
        print(class_dir)
    
    io.imsave(os.path.join(class_dir, '%d.jpg' % i), img_as_ubyte(X_valid[i]))
