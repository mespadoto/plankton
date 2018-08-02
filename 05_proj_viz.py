import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from MulticoreTSNE import MulticoreTSNE

import keras_ft


def create_plot(X, y, labels, fig_name):
    _, ax = plt.subplots(figsize=(20, 18))
    for label_id, label in enumerate(labels):
        indices = np.where(y == label_id)
        ax.scatter(X[indices,0], X[indices,1], label=label, s=60, alpha=0.4, edgecolors='none')
        ax.axis('off')

    ax.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))
    plt.savefig(fig_name)


if len(sys.argv) != 5:
    print('Usage: python 05_proj_viz.py <base_dir> <model_name> <dataset_name> <model_file>')
    exit(1)


base_dir = sys.argv[1]
model_name = sys.argv[2]
dataset_name = sys.argv[3]
model_file = sys.argv[4]

labels = joblib.load(os.path.join(base_dir, 'label_names_%s.pkl' % dataset_name))
y = np.load(os.path.join(base_dir, 'y_%s.npy' % dataset_name))

if os.path.exists(model_file):
    X = np.load(os.path.join(base_dir, 'X_%s.npy' % dataset_name))
    shape = X.shape[1:]
    n_classes = len(np.unique(y))

    ft = keras_ft.KerasFineTuner(model_name, shape, n_classes)
    ft.load_weights(model_file)
    X = ft.get_embeddings(X)
else:
    X = np.load(os.path.join(base_dir, 'X_%s_features_%s.npy' % (dataset_name, model_name)))

dr = MulticoreTSNE(n_jobs=4, n_iter=3000, perplexity=30, random_state=42)
X_proj = dr.fit_transform(X)

create_plot(X_proj, y, labels, '%s_%s.png' % (model_name, model_file))
