# -*- coding: utf-8 -*-

import os
import sys
from glob import glob
import numpy as np
from skimage import io, transform
import joblib
import pandas as pd

def load_square_image(img_name, target_shape):
    img = io.imread(img_name)
    
    channels = 3
    greater_side = max(img.shape[:2])
    smaller_side = min(img.shape[:2])

    new_img = np.zeros((greater_side, greater_side, channels)).astype('uint8')
    new_img[:,:,:] = 255
    
    start = int((greater_side/2) - (smaller_side//2))
    end = start + smaller_side

    max_side = np.argmax(img.shape)
    
    if max_side == 0: #vertical
        new_img[:,start:end,0] = img
    elif max_side == 1: #horizontal
        new_img[start:end,:,0] = img

    for c in range(2):
        new_img[:,:,c+1] = new_img[:,:,0]

    new_img = transform.resize(new_img, target_shape, preserve_range=True) / 255.0

    return new_img

if len(sys.argv) != 3:
    print('Usage: python 01_create_dataset.py <data_dir> <image_side>')
    exit(1)

base_dir = sys.argv[1]
img_side = int(sys.argv[2])

shape = (img_side, img_side, 3)

#plankton datasets
datasets = dict()

datasets['furg'] = os.path.join(base_dir, 'furg')
datasets['laps'] = os.path.join(base_dir, 'laps_nobg_100')
datasets['ndsb'] = os.path.join(base_dir, 'ndsb')
datasets['japan'] = os.path.join(base_dir, 'LRoot_japan_dataset_review_jul2018')

for dataset, path in datasets.items():
    classes = sorted(glob(path + '/*'))
    
    label_names = []
    y_tmp = []
    X_tmp = []
    
    for i, c in enumerate(classes):
        if os.path.isdir(c):
            print(i, c)
            label_names.append(os.path.basename(c))

            files_in_class = sorted(glob(c + '/*.*'))
            
            for f in files_in_class:
                img = load_square_image(f, shape)
                X_tmp.append(img)
                y_tmp.append(i)

    np.save(os.path.join(base_dir, 'X_%s.npy' % dataset), np.array(X_tmp))
    np.save(os.path.join(base_dir, 'y_%s.npy' % dataset), np.array(y_tmp).astype('uint8'))
    joblib.dump(label_names, os.path.join(base_dir, 'label_names_%s.pkl' % dataset))
