import numpy as np
import os
import math
import tensorflow as tf
from time import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from scipy.spatial.distance import cdist
from PIL import Image
from lapjv import lapjv

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.utils.data_utils import Data

class GridLayout(object):
    def __init__(self, dataname):
        self.dataname = dataname
        self.data = Data(self.dataname)
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = self.data.get_data("all")
        self.train_idx = self.data.train_idx
        self.valid_idx = self.data.valid_idx
        self.test_idx = self.data.test_idx
        self.embed_X_train, self.embed_X_valid, self.embed_X_test = self.data.get_embed_X("all")
        self.train_num, self.feature_num = self.X_train.shape
        self.num_class = self.y_train.max() + 1
        self.clf = None
        self.kernel = None


    def training(self, kernel="linear", C=1, gamma="auto"):
        self.kernel = kernel
        self.clf = SVC(kernel=kernel,
                       C=C, gamma=gamma,
                       verbose=1, max_iter=5000)
        print("parameter:", self.clf.get_params())
        print("training data shape:{}, test data shape: {}".format(self.X_train.shape, self.X_test.shape))
        self.clf.fit(self.X_train, self.y_train)
        train_score = self.clf.score(self.X_train, self.y_train)
        test_score = self.clf.score(self.X_test, self.y_test)
        if kernel == "linear":
            weights = self.clf.coef_
            margin = 1.0 / ((weights**2).sum())**0.5
        else:
            margin = "not defined"
        print("\n training acc: {}, test acc: {}, margin value: {}."
              .format(train_score, test_score, margin))

    def get_grid_layout(self):
        if self.clf is None:
            raise ValueError("you need train the svm if you want to get the support vector.")
        X = np.concatenate((self.X_train, self.X_test), axis=0)
        if hasattr(self.data, "embed_X_train"):
            print("using pre-computated embeddings")
            X_train_embeddings = self.data.embed_X_train
            X_test_embeddings = self.data.embed_X_test
            print("X_train_embeddings shape:{}, X_test_embeddings shape:{}"
                  .format(X_train_embeddings.shape, X_test_embeddings.shape))
        else:
            print("calculating tsne now...")
            tsne = TSNE(n_components=2, random_state=123)
            X_embeddings = tsne.fit_transform(X)
            X_train_embeddings = X_embeddings[:self.X_train.shape[0], :]
            X_test_embeddings = X_embeddings[self.X_train.shape[0]:, :]

        # X_train_embeddings = np.concatenate((X_train_embeddings,X_test_embeddings),axis=0)

        # Normalization
        X_train_embeddings -= X_train_embeddings.min(axis=0)
        X_train_embeddings /= X_train_embeddings.max(axis=0)
        X_test_embeddings -= X_test_embeddings.min(axis=0)
        X_test_embeddings /= X_test_embeddings.max(axis=0)

        train_num = X_train_embeddings.shape[0]
        N_sqrt_train = int(np.sqrt(X_train_embeddings.shape[0])) + 1
        for_embed_X_train = X_train_embeddings
        N_train = N_sqrt_train * N_sqrt_train
        N_sqrt_test = int(np.sqrt(X_test_embeddings.shape[0]))
        N_test = N_sqrt_test * N_sqrt_test
        for_embed_X_test = X_test_embeddings[:N_test]

        grid_train = np.dstack(np.meshgrid(np.linspace(0, 1, N_sqrt_train), np.linspace(0, 1, N_sqrt_train))).reshape(-1, 2)
        grid_test = np.dstack(np.meshgrid(np.linspace(0, 1, N_sqrt_test), np.linspace(0, 1, N_sqrt_test))).reshape(-1, 2)

        print("toal instance: {}".format(N_train))
        original_cost_matrix = cdist(grid_train, for_embed_X_train, "euclidean")
        max_val = 25
        cost_matrix = original_cost_matrix.copy()
        cost_matrix = cost_matrix/ (cost_matrix.max() * 1.01) * max_val
        cost_matrix = (cost_matrix + 0.5).astype(np.int32)
        # cost_matrix = np.random.rand(cost_matrix.shape[0], cost_matrix.shape[1])
        print("in python:\n", cost_matrix[:4,:4])
        # dummy_vertices = np.ones((N_train, N_train - cost_matrix.shape[1])) * 10000000
        # cost_matrix = np.concatenate((cost_matrix, dummy_vertices), axis=1)
        cost_matrix = cost_matrix[:train_num, :train_num]

        # cost_matrix = 5000 * np.random.random((5000, 5000))
        t = time()
        train_row_asses, train_col_asses, info = lapjv(cost_matrix)
        cost = info[0] / float(max_val)
        cost = original_cost_matrix[train_col_asses[:train_num],
                                                  np.array(range(N_train))[:train_num]].sum()
        print("lapjv time: {}, cost: {}".format(time() - t, cost))
        # exit()

        # cost_matrix = cdist(grid_test, for_embed_X_test, "euclidean")
        # cost_matrix = cost_matrix/ cost_matrix.max() * 10000
        # cost_matrix = cost_matrix.astype(int)
        # test_row_asses, test_col_asses, _ = lapjv(cost_matrix)
        #
        train_col_asses = train_col_asses[:train_num]
        grid_X_train = np.zeros(X_train_embeddings.shape)
        grid_X_test = np.zeros(X_test_embeddings.shape)
        grid_X_train[:len(train_col_asses)] = grid_train[train_col_asses]
        # grid_X_test[:len(test_col_asses)] = grid_test[test_col_asses]


        return grid_X_train, grid_X_test

    def grid_layout(self):
        support_vectors_index = np.array(self.clf.support_)
        no_support_vectors_index = np.array([i for i in range(self.train_num) if i not in self.clf.support_])
        y_train = self.y_train.astype(int) * 2 + 1
        y_test = self.y_test.astype(int) * 2 + 1
        y_train[support_vectors_index] = y_train[support_vectors_index] - 1
        color_map = plt.get_cmap("tab20")(np.array(y_train.tolist() + y_test.tolist()))
        color_map_train = color_map[:self.X_train.shape[0], :]
        color_map_test = color_map[self.X_train.shape[0]:, :]

        if hasattr(self.data, "embed_X_train"):
            print("using pre-computated embeddings")
            X_train_embeddings = self.data.embed_X_train
            X_test_embeddings = self.data.embed_X_test
            print("X_train_embeddings shape:{}, X_test_embeddings shape:{}"
                  .format(X_train_embeddings.shape, X_test_embeddings.shape))
        else:
            print("calculating tsne now...")
            tsne = TSNE(n_components=2, random_state=123)
            X_embeddings = tsne.fit_transform(X)
            X_train_embeddings = X_embeddings[:self.X_train.shape[0], :]
            X_test_embeddings = X_embeddings[self.X_train.shape[0]:, :]

        grid_X_train, grid_X_test = self.get_grid_layout()

        ax_train = plt.subplot(221)
        ax_train.scatter(X_train_embeddings[no_support_vectors_index, 0],
                         0 - X_train_embeddings[no_support_vectors_index, 1],
                       s=8,
                       marker="o",
                       c=color_map_train[no_support_vectors_index,:],
                       alpha=0.7)
        ax_train.scatter(X_train_embeddings[support_vectors_index, 0],
                         0 - X_train_embeddings[support_vectors_index, 1],
                       s=20,
                       marker="x",
                       c=color_map_train[support_vectors_index,:])

        ax_train = plt.subplot(222)
        ax_train.scatter(grid_X_train[no_support_vectors_index, 0],
                         0 - grid_X_train[no_support_vectors_index, 1],
                       s=8,
                       marker="o",
                       c=color_map_train[no_support_vectors_index,:],
                       alpha=0.7)
        ax_train.scatter(grid_X_train[support_vectors_index, 0],
                         0 - grid_X_train[support_vectors_index, 1],
                       s=20,
                       marker="x",
                       c=color_map_train[support_vectors_index,:])

        ax = plt.subplot(223)
        ax.scatter(X_test_embeddings[:,0], 0 - X_test_embeddings[:, 1],
                   s=8,
                   marker="o",
                   c=color_map_test,
                   alpha=0.7)

        ax = plt.subplot(224)
        ax.scatter(grid_X_test[:, 0], 0 - grid_X_test[:, 1],
                   s=8,
                   marker="o",
                   c=color_map_test,
                   alpha=0.7)
        plt.show()


    def save_grid(self):
        grid_X_train, grid_X_test = self.get_grid_layout()
        mat = {
            config.grid_X_train_name: grid_X_train,
            config.grid_X_test_name: grid_X_test
        }
        filename = os.path.join(config.data_root, self.dataname, config.grid_dataname)
        pickle_save_data(filename, mat)


if __name__ == '__main__':
    cls = GridLayout(config.svhn)
    cls.training(kernel='linear')
    cls.grid_layout()
    # cls.save_grid()
    # cls.get_grid_layout()