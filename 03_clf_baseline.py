# -*- coding: utf-8 -*-

import os
import sys
from glob import glob

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

if len(sys.argv) != 3:
    print('Usage: python 03_clf_baseline.py <base_dir> <dataset_name>')
    exit(1)

base_dir = sys.argv[1]
dataset_name = sys.argv[2]

feature_files = glob(os.path.join(base_dir, 'X_%s_features_*.npy' % dataset_name))

for feature_file in feature_files:
    model = os.path.basename(feature_file).replace('X_%s_features_' % dataset_name, '').replace('.npy', '')
    X = np.load(feature_file)
    y = np.load(os.path.join(base_dir, 'y_%s.npy' % dataset_name))
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)

    print('%s (%s): %.4f' % (model, clf.__class__.__name__, accuracy_score(y_valid, y_pred)))

    print('Classification report:')
    print(classification_report(y_valid, y_pred))
