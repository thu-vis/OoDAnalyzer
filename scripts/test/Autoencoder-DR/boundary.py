from sklearn.datasets import fetch_mldata
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

from scripts.utils.config_utils import config

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def encoder(x, e1, e2, e3, e4):
    test_data_num = x.shape[0]
    x = np.concatenate((x, np.ones((test_data_num, 1))), axis=1)
    x = np.dot(x, e1)
    x = sigmoid(x)
    x = np.concatenate((x, np.ones((test_data_num, 1))), axis=1)
    x = np.dot(x, e2)
    x = sigmoid(x)
    x = np.concatenate((x, np.ones((test_data_num, 1))), axis=1)
    x = np.dot(x, e3)
    x = sigmoid(x)
    x = np.concatenate((x, np.ones((test_data_num, 1))), axis=1)
    x = np.dot(x, e4)
    return x

def decoder(x, d1, d2, d3, d4):
    test_data_num = x.shape[0]
    x = np.concatenate((x, np.ones((test_data_num, 1))), axis=1)
    x = np.dot(x, d1)
    x = sigmoid(x)
    x = np.concatenate((x, np.ones((test_data_num, 1))), axis=1)
    x = np.dot(x, d2)
    x = sigmoid(x)
    x = np.concatenate((x, np.ones((test_data_num, 1))), axis=1)
    x = np.dot(x, d3)
    x = sigmoid(x)
    x = np.concatenate((x, np.ones((test_data_num, 1))), axis=1)
    x = np.dot(x, d4)
    x = sigmoid(x)
    return x

def load_weights():
    weights_path = os.path.join(config.raw_data_root,
                               "MNIST-1000-200-40-2",
                               "mnist_weights.mat")
    weights = sio.loadmat(weights_path)

    return weights["w1"], weights["w2"], weights["w3"], weights["w4"], \
           weights["w5"], weights["w6"], weights["w7"], weights["w8"]


def load_mnist():
    mnist = fetch_mldata("MNIST original")
    target = mnist["target"]
    data = mnist["data"]
    index = np.array(range(len(target)))
    np.random.shuffle(index)
    target = target[index]
    data = data[index,:]
    X_train = data[:60000,:]
    X_test = data[60000:,:]
    y_train = target[:60000]
    y_test = target[60000:]
    return X_train, y_train, X_test, y_test

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def decision_boundary():
    X_train, y_train, X_test, y_test = load_mnist()
    e1, e2, e3, e4, d1, d2, d3, d4 = load_weights()
    x = X_train / 255.0
    embed_x = encoder(x, e1, e2, e3, e4)
    idx = np.array(range(X_train.shape[0]))
    np.random.shuffle(idx)

    norm = plt.cm.colors.Normalize(vmax=9, vmin=0)

    fig, sub = plt.subplots(1,1)
    ax = sub
    xx, yy = make_meshgrid(embed_x[:,0], embed_x[:,1], h=0.2)
    sample_x = np.c_[xx.ravel(), yy.ravel()]
    sample_x = decoder(sample_x, d1, d2, d3, d4)
    sample_x = (sample_x * 255)
    clf = RandomForestClassifier(n_estimators=100, random_state=123)
    clf.fit(X_train, y_train)
    print(clf.score(X_test,y_test))
    sample_y = clf.predict(sample_x)
    sample_y = sample_y.reshape(xx.shape)
    color_map = plt.get_cmap("tab10")(y_train[idx[:1000]]/9.0)
    # ax.contourf(xx, yy, sample_y, cmap=plt.cm.coolwarm)
    ax.scatter(xx.reshape(-1), yy.reshape(-1), c=sample_y.reshape(-1), cmap=plt.get_cmap("tab10"), norm=norm, s=1)
    # ax.scatter(embed_x[idx[:1000],0], embed_x[idx[:1000],1], c=y_train[idx[:1000]], cmap=plt.cm.coolwarm,s = 2, edgecolors="k")
    # ax.scatter(embed_x[idx[:1000],0], embed_x[idx[:1000],1], c=color_map, s = 2)
    plt.show()

if __name__ == '__main__':
    decision_boundary()