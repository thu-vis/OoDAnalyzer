import numpy as np
import os
import sys
import math
import tensorflow as tf
from time import time

from scipy.sparse import csr_matrix
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
from mosek.fusion import *

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.utils.data_utils import Data
from scripts.utils.log_utils import logger
from scripts.Grid import GridLayout

def knn_sparse(dense_cost_matrix, k):
    cost_matrix = dense_cost_matrix
    hole_num = cost_matrix.shape[0]
    radish_num = cost_matrix.shape[1]
    hole_connected_idx = [[0 for _ in range(radish_num)] for __ in range(hole_num)]
    radish_sortdist_index = []
    hole_count = [0 for _ in range(hole_num)]
    radish_toCheck = [k for _ in range(hole_num)]
    for i in range(radish_num):
        col_sort = cost_matrix[:, i].argsort()
        radish_sortdist_index.append(col_sort)
        knn_index = col_sort[:50]
        for j in knn_index:
            hole_count[j] += 1
            hole_connected_idx[j][i] = 1
    logger.info('sort finished')
    for hole in range(hole_num):
        if hole_count[hole] <= k:
            continue
        rs = [i for i, v in enumerate(hole_connected_idx[hole]) if v > 0]
        distances = np.array([cost_matrix[hole][r] for r in rs])
        order = distances.argsort()
        for index in order[k:]:
            radish_idx = rs[index]
            next_hole_to_assign = radish_sortdist_index[radish_idx][radish_toCheck[radish_idx]]
            while hole_connected_idx[next_hole_to_assign][radish_idx] > 0 or hole_count[next_hole_to_assign] >= k:
                radish_toCheck[radish_idx] += 1
                next_hole_to_assign = radish_sortdist_index[radish_idx][radish_toCheck[radish_idx]]
            radish_toCheck[radish_idx] += 1
            hole_count[next_hole_to_assign] += 1
            hole_connected_idx[next_hole_to_assign][radish_idx] = 1
            hole_count[hole] -= 1
            hole_connected_idx[hole][radish_idx] = 0
    res_matrix = [[cost_matrix[hole][radish] if hole_connected_idx[hole][radish] else 0 for radish in
                    range(radish_num)] for hole in range(hole_num)]
    res_matrix = np.array(res_matrix)
    return res_matrix

class GridTest(object):
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

        self.grid_layout = GridLayout(self.dataname)

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

    def native_lap(self):
        if self.clf is None:
            raise ValueError("you need train the svm if you want to get the support vector.")
        X_train_embeddings = self.data.embed_X_train
        X_test_embeddings = self.data.embed_X_test
        print("X_train_embeddings shape:{}, X_test_embeddings shape:{}"
                  .format(X_train_embeddings.shape, X_test_embeddings.shape))
        grid_X_train, grid_X_test, _, _ = self.grid_layout.get_grid_layout()
        # grid_X_train, grid_X_test, _, _ = self.grid_layout.get_grid_layout_native_lap_knn()
        # grid_X_train, grid_X_test, _, _ = self.grid_layout.get_grid_layout_native_lap_inverse_knn()
        support_vectors_index = np.array(self.clf.support_)
        no_support_vectors_index = np.array([i for i in range(self.train_num) if i not in self.clf.support_])
        y_train = self.y_train.astype(int) * 2 + 1
        y_test = self.y_test.astype(int) * 2 + 1
        y_train[support_vectors_index] = y_train[support_vectors_index] - 1
        color_map = plt.get_cmap("tab20")(np.array(y_train.tolist() + y_test.tolist()))
        color_map_train = color_map[:self.X_train.shape[0], :]
        color_map_test = color_map[self.X_train.shape[0]:, :]
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

    def mosek_for_lap(self):
        embed_X = self.data.embed_X.copy()
        embed_X -= embed_X.min(axis=0)
        embed_X /= embed_X.max(axis=0)
        embed_X_train = embed_X[np.array(self.train_idx), :]
        embed_X_test = embed_X[np.array(self.test_idx), :]
        train_num = embed_X_train.shape[0]
        test_num = embed_X_test.shape[0]

        N_sqrt_train = int(np.sqrt(embed_X_train.shape[0])) + 1
        # N_sqrt_train = int(np.sqrt(embed_X_train.shape[0]) * 1.05)
        N_train = N_sqrt_train * N_sqrt_train
        for_embed_X_train = embed_X_train[:N_train]
        N_sqrt_test = int(np.sqrt(embed_X_test.shape[0])) + 1
        # N_sqrt_test = int(np.sqrt(embed_X_test.shape[0]) * 1.05)
        N_test = N_sqrt_test * N_sqrt_test
        for_embed_X_test = embed_X_test[:N_test]

        grid_train = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / N_sqrt_train, N_sqrt_train),
                                           np.linspace(0, 1 - 1.0 / N_sqrt_train, N_sqrt_train))) \
            .reshape(-1, 2)
        grid_test = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / N_sqrt_test, N_sqrt_test),
                                          np.linspace(0, 1 - 1.0 / N_sqrt_test, N_sqrt_test))) \
            .reshape(-1, 2)
        train_original_cost_matrix = cdist(grid_train, for_embed_X_train, "euclidean")
        ####### For debug ###############
        train_original_cost_matrix = train_original_cost_matrix[:train_num, :train_num]
        #################################
        sparse_cost_matrix = csr_matrix(train_original_cost_matrix)
        data = sparse_cost_matrix.data
        indptr = sparse_cost_matrix.indptr
        indices = sparse_cost_matrix.indices
        variable_indices = list(range(len(indices)))
        logger.debug("knn construction is finished, variables num: {}".format(len(data)))
        M = Model("lo1")
        # x = M.variable("x", len(data), Domain.greaterThan(0)); logger.info(">=0")
        x = M.variable("x", len(data), Domain.binary())
        logger.info("binary")
        A = []
        C = []
        hole_num, radish_num = sparse_cost_matrix.shape
        for i in range(hole_num):
            idx = variable_indices[indptr[i]:indptr[i + 1]]
            res = [[i, j, 1] for j in idx]
            # A = A + res
            A.append(res)

        logger.info("constructing A, A length: {}"
                    .format(len(A)))

        for i in range(radish_num):
            idx = [variable_indices[indices[indptr[j]:indptr[j + 1]].tolist().index(i) + indptr[j]]
                   for j in range(hole_num) if i in indices[indptr[j]:indptr[j + 1]]]
            res = [[i, j, 1] for j in idx]
            # C = C + res
            C.append(res)
        logger.info("constructing C, C length: {}".format(len(C)))
        A = [x for j in A for x in j]
        C = [x for j in C for x in j]
        A = list(zip(*A))
        C = list(zip(*C))
        logger.info("finished python's convert sparse matrix")
        A = Matrix.sparse(hole_num, len(data), list(A[0]), list(A[1]), list(A[2]))
        C = Matrix.sparse(radish_num, len(data), list(C[0]), list(C[1]), list(C[2]))
        logger.info("finished mosek's convert sparse matrix")

        logger.info("adding constraints")
        M.constraint(Expr.mul(A, x), Domain.lessThan(1))
        M.constraint(Expr.mul(C, x), Domain.equalsTo(1))

        M.objective("obj", ObjectiveSense.Minimize, Expr.dot(data, x))
        M.setLogHandler(sys.stdout)
        logger.info("begin solving")
        M.solve()
        solutions = x.level()
        rowes = []
        for i in range(train_num):
            tmp_sol = solutions[i * train_num: i * train_num + train_num]
            tmp_idx = tmp_sol.tolist().index(1)
            rowes.append(tmp_idx)
        print(rowes)

        from lapjv import lapjv
        lap_rowes, coles, _ = lapjv(train_original_cost_matrix)
        print(lap_rowes)


    def mosek_for_knn_lap(self):
        embed_X = self.data.embed_X.copy()
        embed_X -= embed_X.min(axis=0)
        embed_X /= embed_X.max(axis=0)
        embed_X_train = embed_X[np.array(self.train_idx), :]
        embed_X_test = embed_X[np.array(self.test_idx), :]
        train_num = embed_X_train.shape[0]
        test_num = embed_X_test.shape[0]

        N_sqrt_train = int(np.sqrt(embed_X_train.shape[0])) + 1
        # N_sqrt_train = int(np.sqrt(embed_X_train.shape[0]) * 1.05)
        N_train = N_sqrt_train * N_sqrt_train
        for_embed_X_train = embed_X_train[:N_train]
        N_sqrt_test = int(np.sqrt(embed_X_test.shape[0])) + 1
        # N_sqrt_test = int(np.sqrt(embed_X_test.shape[0]) * 1.05)
        N_test = N_sqrt_test * N_sqrt_test
        for_embed_X_test = embed_X_test[:N_test]

        grid_train = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / N_sqrt_train, N_sqrt_train),
                                           np.linspace(0, 1 - 1.0 / N_sqrt_train, N_sqrt_train))) \
            .reshape(-1, 2)
        grid_test = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / N_sqrt_test, N_sqrt_test),
                                          np.linspace(0, 1 - 1.0 / N_sqrt_test, N_sqrt_test))) \
            .reshape(-1, 2)


        train_original_cost_matrix = cdist(grid_train, for_embed_X_train, "euclidean")
        sparse_cost_matrix = knn_sparse(train_original_cost_matrix, k=50)
        sparse_cost_matrix = csr_matrix(sparse_cost_matrix)
        data = sparse_cost_matrix.data
        indptr = sparse_cost_matrix.indptr
        indices = sparse_cost_matrix.indices
        variable_indices = list(range(len(indices)))
        logger.debug("knn construction is finished, variables num: {}".format(len(data)))
        M = Model("lo1")
        # x = M.variable("x", len(data), Domain.greaterThan(0)); logger.info(">=0")
        x = M.variable("x", len(data), Domain.binary())
        logger.info("binary")
        A = []
        C = []
        hole_num, radish_num = sparse_cost_matrix.shape
        for i in range(hole_num):
            idx = variable_indices[indptr[i]:indptr[i + 1]]
            res = [[i, j, 1] for j in idx]
            # A = A + res
            A.append(res)

        logger.info("constructing A, A length: {}"
                    .format(len(A)))

        for i in range(radish_num):
            idx = [variable_indices[indices[indptr[j]:indptr[j + 1]].tolist().index(i) + indptr[j]]
                   for j in range(hole_num) if i in indices[indptr[j]:indptr[j + 1]]]
            res = [[i, j, 1] for j in idx]
            # C = C + res
            C.append(res)
        logger.info("constructing C, C length: {}".format(len(C)))
        A = [x for j in A for x in j]
        C = [x for j in C for x in j]
        A = list(zip(*A))
        C = list(zip(*C))
        logger.info("finished python's convert sparse matrix")
        A = Matrix.sparse(hole_num, len(data), list(A[0]), list(A[1]), list(A[2]))
        C = Matrix.sparse(radish_num, len(data), list(C[0]), list(C[1]), list(C[2]))
        logger.info("finished mosek's convert sparse matrix")

        logger.info("adding constraints")
        M.constraint(Expr.mul(A, x), Domain.lessThan(1))
        M.constraint(Expr.mul(C, x), Domain.equalsTo(1))

        M.objective("obj", ObjectiveSense.Minimize, Expr.dot(data, x))
        M.setLogHandler(sys.stdout)
        logger.info("begin solving")
        M.solve()
        solutions = x.level()


if __name__ == '__main__':
    g = GridTest(config.two_dim)
    g.training(kernel='linear')
    # g.native_lap()
    # g.mosek_for_lap()
    g.mosek_for_knn_lap()
    # g._sampling_and_assignment_test()