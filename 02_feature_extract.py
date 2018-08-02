# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from keras import applications, utils

if len(sys.argv) != 4:
    print('Usage: python 02_feature_extract.py <base_dir> <dataset_name> <batch_size>')
    exit(1)

base_dir = sys.argv[1]
dataset_name = sys.argv[2]
batch_size = int(sys.argv[3])

X = np.load(os.path.join(base_dir, 'X_%s.npy' % dataset_name))
shape = X.shape[1:]

model_names = []
models = []

if shape[0] >= 32:
    model_names.append('nasnet')
    models.append(applications.nasnet.NASNetLarge(include_top=False, weights='imagenet', input_shape=shape, pooling='max'))

if shape[0] >= 48:
    model_names.append('vgg16')
    model_names.append('vgg19')
    models.append(applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=shape, pooling='max'))
    models.append(applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=shape, pooling='max'))

if shape[0] >= 71:
    model_names.append('xception')
    models.append(applications.xception.Xception(include_top=False, weights='imagenet', input_shape=shape, pooling='max'))

if shape[0] >= 139:
    model_names.append('inceptionv3')
    model_names.append('inception_resnetv2')
    models.append(applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=shape, pooling='max'))
    models.append(applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=shape, pooling='max'))

if shape[0] >= 197:
    model_names.append('resnet50')
    models.append(applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=shape, pooling='max'))

if shape[0] >= 221:
    model_names.append('densenet121')
    model_names.append('densenet169')
    model_names.append('densenet201')
    models.append(applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=shape, pooling='max'))
    models.append(applications.densenet.DenseNet169(include_top=False, weights='imagenet', input_shape=shape, pooling='max'))
    models.append(applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_shape=shape, pooling='max'))

for model_name, model in zip(model_names, models):
    print(model_name)

    X_features = model.predict(X, batch_size=batch_size)
    np.save(os.path.join(base_dir, 'X_%s_features_%s.npy' % (dataset_name, model_name)), X_features)
