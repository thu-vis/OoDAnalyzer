import numpy as np
import os
import sys
import random
import ctypes
import math
import tensorflow as tf
from time import time

from scipy.sparse import csr_matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from scipy.spatial.distance import cdist
from PIL import Image
from lapjv import lapjv
from lapjvo import lapjvo
from lap import lapjv as lap_jv
from lap import lapmod as lap_mod
import multiprocessing
from keras.datasets import cifar10

from sklearn.datasets import fetch_mldata

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.utils.data_utils import Data
from scripts.utils.log_utils import logger
from scripts.DensityBasedSampler import DensityBasedSampler
from scripts.utils.config_utils import config
from contextlib import redirect_stdout

from mosek.fusion import *


def time_cost_grid_layout(X, k=50, c_util_name="c_utils.dll"):
    t = time()
    if X.shape[0] < 10:
        print("the shape of X is less than 50, skip normalization")
    else:
        X -= X.min(axis=0)
        X /= X.max(axis=0)
    num = X.shape[0]
    square_len = math.ceil(np.sqrt(num))
    N = square_len * square_len
    grids = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / square_len, square_len),
                                  np.linspace(0, 1 - 1.0 / square_len, square_len))) \
        .reshape(-1, 2) + 0.5 / square_len

    original_cost_matrix = cdist(grids, X, "euclidean")
    # knn process
    dummy_points = np.ones((N - original_cost_matrix.shape[1], 2)) * 0.5
    # dummy at [0.5, 0.5]
    dummy_vertices = (1 - cdist(grids, dummy_points, "euclidean")) * 100
    cost_matrix = np.concatenate((original_cost_matrix, dummy_vertices), axis=1)

    libc = ctypes.cdll.LoadLibrary(os.path.join(config.scripts_root, c_util_name))
    cost_matrix = cost_matrix.astype(np.dtype('d'))
    rows, cols = cost_matrix.shape
    cost_matrix_1 = np.asarray(cost_matrix.copy())
    cost_matrix_2 = np.asarray(cost_matrix.T.copy())
    ptr1 = cost_matrix_1.ctypes.data_as(ctypes.c_char_p)
    ptr2 = cost_matrix_2.ctypes.data_as(ctypes.c_char_p)
    k = min(cost_matrix.shape[0], k)
    print("k: ", k)
    libc.knn_sparse(ptr1, rows, cols, k, False, 0)
    print("cost_matrix_1_diff:", ((cost_matrix_1 - cost_matrix)**2).sum())
    libc.knn_sparse(ptr2, cols, rows, k, False, 0)
    print("cost_matrix_2_diff:", ((cost_matrix_2 - cost_matrix.T)**2).sum())
    cost_matrix_2 = cost_matrix_2.T
    # logger.info("end knn preprocessing")

    # merge two sub-graph
    _cost_matrix = cost_matrix.copy()
    cost_matrix = np.maximum(cost_matrix_1, cost_matrix_2)
    print("cost_matrix_all_diff:", ((np.maximum(cost_matrix_1, cost_matrix_2) - _cost_matrix)**2).sum())

    # binning
    # cost_matrix = cost_matrix / original_cost_matrix.max() * 100
    # cost_matrix = np.ceil(cost_matrix).astype(np.int32)
    cost_matrix[cost_matrix == 0] = 10000000

    # begin LAP-JV
    # logger.info("begin LAP JV")
    #
    print("pypypy", cost_matrix.sum())

    row_asses, col_asses, info = lapjv(cost_matrix)
    cost = original_cost_matrix[col_asses[:num],
                                np.array(range(N))[:num]].sum()
    col_asses = col_asses[:num]
    grid_X = grids[col_asses]


    return time() - t, info[0], [row_asses, col_asses, info, original_cost_matrix]

class Experiment(object):
    def __init__(self, dataname):
        self.dataname = dataname
        self.data_dir = os.path.join(config.data_root, dataname, "knn")
        check_dir(self.data_dir)
        print("present data:", dataname)
        self._get_data()

    def _get_data(self):

        if self.dataname == "simple-case":
            square_len = 5
            grids = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / square_len, square_len),
                                          np.linspace(0, 1 - 1.0 / square_len, square_len))) \
                .reshape(-1, 2) + 0.001
            self.X = grids

        if self.dataname == config.dog_cat:
            data = Data(self.dataname)
            self.X = data.embed_X[np.array(data.train_idx)]

        if self.dataname == "simple-dogcat":
            data = Data(config.dog_cat)
            self.X = data.embed_X[np.array(data.train_idx)]
            # idx = np.array(range(self.X.shape[0]))
            # np.random.seed(123)
            # np.random.shuffle(idx)
            # self.X = self.X[idx[:2500]]


        if self.dataname == config.animals or self.dataname == config.rea:
            data = Data(self.dataname)
            valided_idx = data.train_idx + data.test_idx
            self.X = data.embed_X[np.array(valided_idx)]
            # print("**************test experiment*****************")
            # self.X = self.X[:150,:]
        elif self.dataname == config.svhn:
            data_dir = self.data_dir
            X_train = np.load(os.path.join(data_dir, "train_activations.npy"))
            y_train = np.load(os.path.join(data_dir, "train_probs.npy")).reshape(-1).astype(int)
            X_test = np.load(os.path.join(data_dir, "test_activations.npy"))
            y_test = np.load(os.path.join(data_dir, "test_probs.npy")).reshape(-1).astype(int)
            # X_train = X_train.reshape(X_train.shape[0], -1)
            # X_test = X_test.reshape(X_test.shape[0], -1)
            # X = np.concatenate((X_train, X_test), axis=0)
            X = X_train.reshape(X_train.shape[0], -1)
            print("svhn data shape:", X.shape)
            if os.path.exists(os.path.join(data_dir, "embed.pkl")):
                X = pickle_load_data(os.path.join(data_dir, "embed.pkl"))
            else:
                tsne = TSNE(n_components=2, random_state=123)
                X = tsne.fit_transform(X)
                pickle_save_data(os.path.join(data_dir, "embed.pkl"), X)
            self.X = X
            idx = np.array(range(self.X.shape[0]))
            np.random.shuffle(idx)
            self.X = self.X[idx[:4900]]

        elif self.dataname == config.mnist:
            data_dir = self.data_dir
            mnist = fetch_mldata("MNIST original")
            target = mnist["target"]
            X = mnist["data"][:60000]
            pca = PCA(n_components=32)
            X = pca.fit_transform(X)
            print("mnist: ",X.shape)
            print("get mnist data")
            if os.path.exists(os.path.join(data_dir, "embed.pkl")):
                X = pickle_load_data(os.path.join(data_dir, "embed.pkl"))
            else:
                tsne = TSNE(n_components=2, random_state=123)
                X = tsne.fit_transform(X)
                pickle_save_data(os.path.join(data_dir, "embed.pkl"), X)
            self.X = X
            idx = np.array(range(self.X.shape[0]))
            np.random.shuffle(idx)
            self.X = self.X[idx[:30000]]
            # idx = np.array(range(60000))
            # np.random.shuffle(idx)
            # self.X = self.X[idx[:1000]]
            # self.X = X[:60000]

        elif self.dataname[:3] == "gmm":
            nn = int(self.dataname[4:])
            kernels = [[random.random(), random.random()] for _ in range(5)]
            distrib_size = nn * nn / 5
            cov = [[0.2, 0], [0, 0.2]]
            random_data = []
            for ker in kernels:
                cnt = 0
                while cnt < distrib_size:
                    point = np.random.multivariate_normal(ker, cov)
                    if 0 < point[0] < 1 and 0 < point[1] < 1:
                        random_data.append(point)
                        cnt += 1
            self.X = np.array(random_data)

        elif self.dataname[:6] == "random":
            nn = int(self.dataname[7:])
            random.seed(123)
            random_data = [[random.random(), random.random()] for _ in range(nn * nn)]
            self.X = np.array(random_data)

        elif self.dataname == config.cifar10:
            data_dir = self.data_dir
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()

            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
            X = np.concatenate((X_train, X_test), axis=0)
            print("get cifa10 data")
            if os.path.exists(os.path.join(data_dir, "embed.pkl")):
                X = pickle_load_data(os.path.join(data_dir, "embed.pkl"))
            else:
                tsne = TSNE(n_components=2, random_state=123)
                X = tsne.fit_transform(X)
                pickle_save_data(os.path.join(data_dir, "embed.pkl"), X)
            self.X = X

    def experiment(self):
        experiment_file = open(os.path.join(r"D:\Project\Project2019\DataBias2019\Project\experiments\KNN-greedy",
                                            "experiment.txt"), "w")

        print("processing data shape:", self.X.shape)
        instance_num = e.X.shape[0]
        # for k in [5, 10, 20, 50]:
        for k in [5, 10, 20, 50, 100, 200, 500, 1000, (math.ceil(np.sqrt(instance_num)))**2]:
        # for k in [(math.ceil(np.sqrt(instance_num)))**2]:
        # for k in [5,10,20,50,100]:
        # for k in [52*52]:
            t, cost, res = time_cost_grid_layout(self.X.copy(), k, "c_utils.dll")
            t_r = cost_r = 0
            for n in range(10):
                _t_r, _cost_r, res_r = time_cost_grid_layout(self.X.copy(), k, "c_utils_random.dll")
                print("n: {}, t_r: {}".format(n, _t_r))
                cost_r = cost_r + _cost_r
            cost_r = cost_r / 10
            print("time: ", t, "cost: ", cost)
            print("time: ", t_r, "cost: ", cost_r)

            neighbor_range = 0
            instance_num = self.X.shape[0]
            square_len = int(np.sqrt(len(res[0])))
            count = 0
            # for id in range(square_len * square_len):
            for id in range(instance_num):
                a = res_r[1][id]
                b = res[1][id]
                xa = a % square_len;
                ya = a // square_len
                xb = b % square_len;
                yb = b // square_len
                if abs(xa - xb) <= neighbor_range and abs(ya - yb) <= neighbor_range:
                    count = count + 1
                else:
                    None

            IoU = count / (2 * instance_num - count)
            print("IoU", IoU)
            s = "{}\t{}\t{}\t{}\t{}".format(
                k, cost_r, cost, abs(cost_r-cost) / cost, IoU
            )
            experiment_file.writelines(s + "\n")

    def IoU(self, neighbor_range=1, k=10):
        # k = 1000
        res_star = pickle_load_data(os.path.join(config.data_root,
                                                 self.dataname, "knn",
                                                 "lapjvo.pkl"))
        res = pickle_load_data(os.path.join(config.data_root,
                                            self.dataname, "knn",
                                            str(k) + ".pkl"))
        instance_num = self.X.shape[0]
        square_len = int(np.sqrt(len(res[0])))
        count = 0
        all_num = 0
        cost_matrix = res[3]
        cost_sum = 0
        res_cost_sum = 0
        all_cost_a = 0
        all_cost_b = 0
        # for id in range(square_len * square_len):
        for id in range(instance_num):
            a = res_star[1][id]
            b = res[1][id]
            xa = a % square_len; ya = a // square_len
            xb = b % square_len; yb = b // square_len
            # cost_a = cost_matrix[a, id]
            # cost_b = cost_matrix[b, id]
            if abs(xa-xb) <= neighbor_range and abs(ya-yb) <= neighbor_range:
                count = count + 1
                # res_cost_sum = res_cost_sum + cost_b - cost_a
            else:
                # cost_sum = cost_sum +  cost_b - cost_a
                None
            # all_cost_a = all_cost_a + cost_matrix[a, id]
            # all_cost_b = all_cost_b + cost_matrix[b, id]
        col_star = res_star[1][:instance_num].tolist()
        col = res[1][:instance_num].tolist()
        sum  = 0


        IoU = count / (2*instance_num - count)
        print(IoU)
        # print(IoU, cost_sum, res_cost_sum, count)
        # print("cost a:", all_cost_a, "cost b:", all_cost_b)

def metric(a,b, square_len, neighboor_range):
    xa = a % square_len
    ya = a // square_len
    xb = b % square_len
    yb = b // square_len
    if abs(xa - xb) <= neighboor_range and abs(ya - yb) <= neighboor_range:
        return 1
    else:
        return 0



if __name__ == '__main__':
    e = Experiment(config.svhn)
    # e = Experiment(config.rea)
    print(e.dataname)
    instance_num = e.X.shape[0]
    e.experiment()
    #
    # for i in [2]:
    #     for k in [5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, (math.ceil(np.sqrt(instance_num)))**2]:
    #         e.IoU(i, k=k)

    # e.knn_check()
    # e.view()
    # for prefix in ["random_", "gmm_"]:
    #     for n in [30, 40, 50, 60, 70]:
    #         e = Experiment(prefix + str(n))
    #         e.experiment()