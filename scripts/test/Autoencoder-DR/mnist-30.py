import numpy as np
import os
import warnings
import scipy.io as sio
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.datasets import fetch_mldata
from sklearn.manifold import TSNE

from scripts.utils.config_utils import config

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

raw_data_name = "MNIST-1000-200-40-2"

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
                                raw_data_name,
                               "mnist_weights.mat")
    weights = sio.loadmat(weights_path)

    return weights["w1"], weights["w2"], weights["w3"], weights["w4"], \
           weights["w5"], weights["w6"], weights["w7"], weights["w8"]

def reconstruction():
    test_data_num = 30
    mnist = fetch_mldata("MNIST original")
    target = mnist["target"]
    data = mnist["data"]
    idx = np.array(range(data.shape[0]))
    np.random.shuffle(idx)
    test_data = data[idx[:test_data_num],:]
    e1, e2, e3, e4, d1, d2, d3, d4 = load_weights()
    x = test_data.copy() / 255.0
    x = encoder(x, e1, e2, e3, e4)
    x = decoder(x, d1, d2, d3, d4)


    for i in range(test_data_num):
        origin_data = test_data[i,:]
        origin_data = origin_data.reshape(28, 28, -1).repeat(axis=2, repeats=3)
        recons_data = (x[i,:] * 255).astype(np.uint8)
        recons_data = recons_data.reshape(28, 28, -1).repeat(axis=2, repeats=3)
        origin_img = Image.fromarray(origin_data)
        recons_img = Image.fromarray(recons_data)
        origin_filepath = os.path.join(config.raw_data_root,
                                       raw_data_name,
                                       "origin_data",
                                       str(i) + ".jpg")
        recons_filepath = os.path.join(config.raw_data_root,
                                       raw_data_name,
                                       "recons_data",
                                       str(i) + ".jpg")
        origin_img.save(origin_filepath)
        recons_img.save(recons_filepath)

def two_d_plot():
    data_num = 6000
    mnist = fetch_mldata("MNIST original")
    target = mnist["target"]
    data = mnist["data"]
    idx = np.array(range(data.shape[0]))
    np.random.shuffle(idx)
    X = data[idx[:data_num],:]
    y = target[idx[:data_num]]
    e1, e2, e3, e4, d1, d2, d3, d4 = load_weights()
    x = X / 255.0
    x = encoder(x, e1, e2, e3, e4)
    color_map = plt.get_cmap("tab10")(y.astype(int))
    plt.scatter(x[:,0], x[:,1], s=3, c=color_map)
    plt.show()

def reconstructed_two_d_plot():
    mnist = fetch_mldata("MNIST original")
    target = mnist["target"]
    data = mnist["data"]
    x = data / 255.0
    e1, e2, e3, e4, d1, d2, d3, d4 = load_weights()
    x = encoder(x, e1, e2, e3, e4)
    x = decoder(x, d1, d2, d3, d4)
    recons_data = (x * 255).astype(np.uint8)
    recons_data = recons_data.astype(float)
    tsne = TSNE()
    tsne_x = tsne.fit_transform(recons_data)
    color_map = plt.get_cmap("tab10")(target.astype(int))
    plt.scatter(tsne_x[:,0], tsne_x[:,1], c=color_map, s=2)
    plt.show()


if __name__ == '__main__':
    reconstructed_two_d_plot()