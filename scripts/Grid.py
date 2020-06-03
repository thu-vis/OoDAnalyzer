import numpy as np
import os
import sys
import ctypes
import math
from time import time

from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from scipy.spatial.distance import cdist
from PIL import Image
from lapjv import lapjv
# from lapjvo import lapjvo
import multiprocessing

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.utils.data_utils import Data
from scripts.utils.log_utils import logger

try:
    from mosek.fusion import *
except:
    None


def funcA(i, radish_num, hole_num):
    res = []
    rg = np.array(range(radish_num * hole_num))
    idx = [0] * i * radish_num + [1] * radish_num + [0] * (hole_num - i - 1) * radish_num
    # A[i,:] = np.array(idx)
    idx = rg[np.array(idx).astype(bool)]
    for j in idx:
        res.append([i, j, 1])
    return res


def func(i, conflict_pair_hole_idx, radish_num, hole_num):
    t = time()
    a, b = conflict_pair_hole_idx[i]
    # print("1 ", time()-t)
    # idx_a = [0] * a * radish_num + [1] * radish_num + \
    #         [0] * (hole_num - a - 1) * radish_num
    # idx_b = [0] * b * radish_num + [1] * radish_num + \
    #         [0] * (hole_num - b - 1) * radish_num
    # idx = [1 if ((t>=a*radish_num and t <(a+1)*radish_num) or (t>=b*radish_num and t<(b+1)*radish_num))
    #        else 0 for t in range(radish_num * hole_num)]
    idx = list(range(a * radish_num, a * radish_num + radish_num)) \
          + list(range(b * radish_num, b * radish_num + radish_num))
    # print("2 ", time()-t)
    # idx = np.array(idx_a) + np.array(idx_b)
    # idx = rg[np.array(idx).astype(bool)]
    # print("3 ", time()-t)
    # for j in idx:
    #     res.append([i, j, 1])
    res = [[i, j, 1] for j in idx]
    # print("4 ", time()-t)
    return res


def knn_sparse(dense_cost_matrix, k):
    cost_matrix = dense_cost_matrix
    hole_num = cost_matrix.shape[0]
    radish_num = cost_matrix.shape[1]
    hole_connected_idx = [[0 for _ in range(radish_num)] for __ in range(hole_num)]
    radish_sortdist_index = []
    hole_count = [0 for _ in range(hole_num)]
    radish_toCheck = [k for _ in range(radish_num)]
    for i in range(radish_num):
        col_sort = cost_matrix[:, i].argsort()
        radish_sortdist_index.append(col_sort)
        knn_index = col_sort[:k]
        for j in knn_index:
            hole_count[j] += 1
            hole_connected_idx[j][i] = 1
    logger.info('sort finished')
    hc = np.array(hole_count)
    hc_sort = hc.argsort().tolist()
    hc_sort.reverse()
    for hole in hc_sort:
        if hole_count[hole] <= k:
            break
        rs = [i for i, v in enumerate(hole_connected_idx[hole]) if v > 0]
        distances = np.array([cost_matrix[hole][r] for r in rs])
        order = distances.argsort().tolist()
        order.reverse()
        for index in order:
            if hole_count[hole] <= k:
                break
            radish_idx = rs[index]
            if radish_toCheck[radish_idx] == hole_num:
                continue
            while True:
                if radish_toCheck[radish_idx] == hole_num:
                    break
                next_hole_to_assign = radish_sortdist_index[radish_idx][radish_toCheck[radish_idx]]
                radish_toCheck[radish_idx] += 1
                if hole_connected_idx[next_hole_to_assign][radish_idx] == 0 and hole_count[next_hole_to_assign] < k:
                    hole_count[next_hole_to_assign] += 1
                    hole_connected_idx[next_hole_to_assign][radish_idx] = 1
                    hole_count[hole] -= 1
                    hole_connected_idx[hole][radish_idx] = 0
                    break
    res_matrix = [[cost_matrix[hole][radish] if hole_connected_idx[hole][radish] else 0 for radish in
                   range(radish_num)] for hole in range(hole_num)]
    res_matrix = np.array(res_matrix)
    logger.info('sparse finished')
    return res_matrix


def knn_sparse_ugly(dense_cost_matrix, k):
    original_cost_matrix = dense_cost_matrix
    cost_matrix = np.zeros((original_cost_matrix.shape))
    hole_num = cost_matrix.shape[0]
    radish_num = cost_matrix.shape[1]
    for j in range(radish_num):
        col = original_cost_matrix[:, j]
        col_sort = col.argsort()
        cost_matrix[col_sort[:k], j] = original_cost_matrix[col_sort[:k], j]
    return cost_matrix

libc = ctypes.cdll.LoadLibrary(os.path.join(config.scripts_root, "c_utils.dll"))

class GridLayout(object):
    def __init__(self, dataname, sampling_num=500):
        self.dataname = dataname
        self.data = Data(self.dataname)
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = self.data.get_data("all")
        self.train_idx = self.data.train_idx
        self.valid_idx = self.data.valid_idx
        self.test_idx = self.data.test_idx
        self.embed_X_train, self.embed_X_valid, self.embed_X_test = self.data.get_embed_X("all")
        self.grid_X_train, self.grid_X_test = None, None
        self.train_row_asses, self.test_row_asses = None, None
        self.train_num, self.feature_num = self.X_train.shape
        self.num_class = self.y_train.max() + 1
        self.grid_filepath = os.path.join(config.data_root, self.dataname, config.grid_dataname)

        self.sampling_num = sampling_num
        self.buffer_mode = False

    def _assignment(self, original_idx):
        t = time()
        original_X = self.data.X[np.array(original_idx)]
        kmeans = KMeans(n_clusters=self.sampling_num, random_state=123).fit(original_X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        sampled_idx = []
        unsampled_idx = []
        for i in range(self.sampling_num):
            selected_indicator = (labels == i)
            selected_X = original_X[selected_indicator]
            selected_idx = np.array(original_idx)[selected_indicator]
            center = centers[i]
            dis = (selected_X - center.reshape(1, -1).repeat(repeats=selected_X.shape[0], axis=0)) ** 2
            dis = dis.sum(axis=1)
            near_the_center_idx = selected_idx[dis.argmax()]
            sampled_idx.append(near_the_center_idx)
            selected_idx_list = selected_idx.tolist()
            selected_idx_list.remove(near_the_center_idx)
            if len(selected_idx_list) >= 1:
                unsampled_idx = unsampled_idx + selected_idx_list
        sampled_idx = np.array(sampled_idx)
        unsampled_idx = np.array(unsampled_idx)
        sampled_X = self.data.X[sampled_idx]
        unsampled_X = self.data.X[unsampled_idx]
        dis_matrix = cdist(unsampled_X, sampled_X, "euclidean")
        assignment_idx_of_idx = dis_matrix.argmin(axis=1)
        assignment_idx = sampled_idx[assignment_idx_of_idx]
        logger.info("kmeans assignment time cost: {}".format(time() - t))
        return sampled_idx, unsampled_idx, assignment_idx

    def _assignment(self, original_idx):
        """
        with random sampling
        :param original_idx:
        :return:
        """
        idx = np.array(range(len(original_idx)))
        np.random.seed(123)
        np.random.shuffle(idx)
        sampled_idx_of_idx = idx[:self.sampling_num]
        unsampled_idx_of_idx = idx[self.sampling_num:]
        sampled_idx = np.array(original_idx)[sampled_idx_of_idx]
        unsampled_idx = np.array(original_idx)[unsampled_idx_of_idx]
        sampled_X = self.data.X[sampled_idx]
        unsampled_X = self.data.X[unsampled_idx]
        dis_matrix = cdist(unsampled_X, sampled_X, "euclidean")
        assignment_idx_of_idx = dis_matrix.argmin(axis=1)
        assignment_idx = sampled_idx[assignment_idx_of_idx]
        return sampled_idx, unsampled_idx, assignment_idx

    def _sampling_and_assignment(self):
        """
        sampling with random sampling
        :return:
        """
        t = time()
        # training data processing
        train_sampled_idx, train_unsampled_idx, train_assignment_idx = self._assignment(self.train_idx)
        # test data processing
        test_sampled_idx, test_unsampled_idx, test_assignment_idx = self._assignment(self.test_idx)
        logger.info("sampling and assignment time cost: {}".format(time() - t))
        return train_sampled_idx, train_unsampled_idx, train_assignment_idx, \
               test_sampled_idx, test_unsampled_idx, test_assignment_idx

    def _sampling_and_assignment_test(self):
        train_sampled_idx, train_unsampled_idx, train_assignment_idx, \
        test_sampled_idx, test_unsampled_idx, test_assignment_idx = self._sampling_and_assignment()
        ax = plt.subplot(111)
        color_map = plt.get_cmap("tab10")(np.array(self.data.y))
        ax.scatter(self.data.embed_X[np.array(self.train_idx), 0],
                   self.data.embed_X[np.array(self.train_idx), 1],
                   s=8,
                   marker="o",
                   c=color_map[np.array(self.train_idx), :],
                   alpha=0.7)
        ax.scatter(self.data.embed_X[np.array(train_sampled_idx), 0],
                   self.data.embed_X[np.array(train_sampled_idx), 1],
                   s=20,
                   marker="x",
                   c=color_map[np.array(train_sampled_idx), :]
                   )
        plt.show()

    def sampling_and_assignment(self):
        """
        sampling with k means
        :return:
        """
        # TODO:
        None

    def hierarchy_layout_lap(self, all_idx, sampled_idx, assignment_idx, square_num):
        t = time()
        pure_grid_time = 0
        histogram_time = 0
        sampled_assigned_num = np.zeros(len(sampled_idx))
        for idx in assignment_idx:
            count_idx = sampled_idx.tolist().index(idx)
            sampled_assigned_num[count_idx] += 1

        histogram_t = time()
        # histogram
        hist, bins = np.histogram(sampled_assigned_num, 100 * square_num, normed=True)
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]
        sampled_assigned_num = np.interp(sampled_assigned_num, bins[:-1], cdf)
        histogram_time = histogram_time + time() - histogram_t
        # visualization
        #########################
        # plt.hist(sampled_assigned_num, bins= 100 * square_num)
        # plt.show()
        # exit()
        ###########################
        sampled_assigned_class = sampled_assigned_num / \
                                 (sampled_assigned_num.max() * 1.01) * square_num
        sampled_assigned_class = sampled_assigned_class.astype(int) + 1

        for size in range(square_num, 0, -1):
            logger.info("{} squares with size of {}*{}"
                        .format(sum(sampled_assigned_class == size), size, size))
        total_grid_num = (sampled_assigned_class ** 2).sum()
        grid_n = int(np.sqrt(total_grid_num) * 1.05)
        grid = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / grid_n, grid_n),
                                     np.linspace(0, 1 - 1.0 / grid_n, grid_n))).reshape(-1, 2)
        min_grid_width = grid[1][0]
        embed_X = self.data.embed_X.copy()
        embed_X -= embed_X.min(axis=0)
        embed_X /= embed_X.max(axis=0)
        mask_matrix = np.ones((grid_n, grid_n)).astype(int) * -1
        final_grid = np.zeros(embed_X.shape)
        for size in range(square_num, 0, -1):
            selected_idx = sampled_idx[sampled_assigned_class == size]
            selected_idx_of_idx = np.array(range(len(sampled_assigned_class)))[sampled_assigned_class == size]
            radish_num = len(selected_idx)
            radish = embed_X[np.array(selected_idx), :]
            holes = []
            for i in range(mask_matrix.shape[0]):
                for j in range(mask_matrix.shape[1]):
                    square = mask_matrix[i:i + size, j:j + size]
                    if (square.sum()) <= (- size * size):
                        holes.append([i, j])
                        mask_matrix[i:i + size, j:j + size] = 0
            hole_num = len(holes)
            if radish_num > hole_num:
                cost_matrix = np.ones((radish_num, radish_num)) * 100000
                raise ValueError("radishes are more than holes")
            else:
                cost_matrix = np.ones((hole_num, hole_num)) * 100000

            for i in range(hole_num):
                for j in range(radish_num):
                    hole_coordinate = grid[holes[i][1] * grid_n + holes[i][0]]
                    hole_coordinate = np.array(hole_coordinate) + \
                                      np.array([min_grid_width * size / 2.0, min_grid_width * size / 2.0])
                    radish_coordinate = radish[j, :]
                    dis = (((hole_coordinate - radish_coordinate) ** 2).sum()) ** 0.5
                    assert dis * 10 < 100000
                    cost_matrix[i, j] = dis

            grid_t = time()
            train_row_asses, train_col_asses, info = lapjv(cost_matrix)
            pure_grid_time = pure_grid_time + time() - grid_t
            if radish_num > hole_num:
                for i in range(radish_num):
                    corres_idx = train_col_asses[i]
                    if corres_idx < hole_num:
                        final_grid[sampled_idx[selected_idx_of_idx[i]]] = \
                            grid[holes[corres_idx][1] * grid_n + holes[corres_idx][0]]
                    else:
                        sampled_assigned_class[selected_idx_of_idx[i]] -= 1
            else:
                for i in range(hole_num):
                    corres_idx = train_col_asses[i]
                    if i < radish_num:
                        final_grid[sampled_idx[selected_idx_of_idx[i]], :] = \
                            grid[holes[corres_idx][1] * grid_n + holes[corres_idx][0]]
                    else:
                        mask_matrix[holes[corres_idx][0]:holes[corres_idx][0] + size,
                        holes[corres_idx][1]:holes[corres_idx][1] + size] = -1

        grid_X = final_grid[np.array(all_idx), :]
        logger.info("single grid layout time cost: {}".format(time() - t))
        logger.info("pure single grid layout time cost: {}".format(pure_grid_time))
        logger.info("histogram time cost: {}".format(histogram_time))
        return grid_X, sampled_assigned_class, mask_matrix, grid_n

    def hierarchy_layout_lp(self, all_idx, sampled_idx, assignment_idx, square_num):
        t = time()
        pure_grid_time = 0
        histogram_time = 0
        sampled_assigned_num = np.zeros(len(sampled_idx))
        for idx in assignment_idx:
            count_idx = sampled_idx.tolist().index(idx)
            sampled_assigned_num[count_idx] += 1

        histogram_t = time()
        # histogram
        hist, bins = np.histogram(sampled_assigned_num, 100 * square_num, normed=True)
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]
        sampled_assigned_num = np.interp(sampled_assigned_num, bins[:-1], cdf)
        histogram_time = histogram_time + time() - histogram_t
        # visualization
        #########################
        # plt.hist(sampled_assigned_num, bins= 100 * square_num)
        # plt.show()
        # exit()
        ###########################
        sampled_assigned_class = sampled_assigned_num / \
                                 (sampled_assigned_num.max() * 1.01) * square_num
        sampled_assigned_class = sampled_assigned_class.astype(int) + 1

        for size in range(square_num, 0, -1):
            logger.info("{} squares with size of {}*{}"
                        .format(sum(sampled_assigned_class == size), size, size))
        total_grid_num = (sampled_assigned_class ** 2).sum()
        grid_n = int(np.sqrt(total_grid_num)) + 1
        grid = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / grid_n, grid_n),
                                     np.linspace(0, 1 - 1.0 / grid_n, grid_n))).reshape(-1, 2)
        min_grid_width = grid[1][0]
        embed_X = self.data.embed_X.copy()
        embed_X -= embed_X.min(axis=0)
        embed_X /= embed_X.max(axis=0)
        mask_matrix = np.ones((grid_n, grid_n)).astype(int) * -1
        final_grid = np.zeros(embed_X.shape)
        for size in range(square_num, 0, -1):
            selected_idx = sampled_idx[sampled_assigned_class == size]
            selected_idx_of_idx = np.array(range(len(sampled_assigned_class)))[sampled_assigned_class == size]
            radish_num = len(selected_idx)
            radish = embed_X[np.array(selected_idx), :]
            holes = []
            for i in range(mask_matrix.shape[0]):
                for j in range(mask_matrix.shape[1]):
                    square = mask_matrix[i:i + size, j:j + size]
                    if (square.sum()) <= (- size * size):
                        holes.append([i, j])
            hole_num = len(holes)
            conflict_pair_hole_idx = []
            for i in range(hole_num):
                for j in range(i + 1, hole_num):
                    if (abs(holes[i][0] - holes[j][0]) < size and abs(holes[i][1] - holes[j][1]) < size):
                        conflict_pair_hole_idx.append([i, j])

            cost_matrix = np.ones((radish_num, hole_num))
            lp_t = time()
            for i in range(radish_num):
                for j in range(hole_num):
                    hole_coordinate = grid[holes[j][1] * grid_n + holes[j][0]]
                    hole_coordinate = np.array(hole_coordinate) + \
                                      np.array([min_grid_width * size / 2.0, min_grid_width * size / 2.0])
                    radish_coordinate = radish[i, :]
                    dis = (((hole_coordinate - radish_coordinate) ** 2).sum()) ** 0.5
                    assert dis * 10 < 100000
                    cost_matrix[i, j] = dis
            linearized_cost_matrix = cost_matrix.reshape(-1)
            M = Model("lo1")
            x = M.variable("x", radish_num * hole_num, Domain.binary())
            # memory error, using sparse matrix instead
            # A = np.zeros((hole_num, radish_num * hole_num))
            # B = np.zeros((len(conflict_pair_hole_idx), radish_num * hole_num))
            # C = np.zeros((radish_num, radish_num * hole_num))
            # A = [[],[],[]]
            # B = [[],[],[]]
            # C = [[],[],[]]
            A = []
            B = []
            C = []

            for i in range(hole_num):
                idx = list(range(i * radish_num, i * radish_num + radish_num))
                res = [[i, j, 1] for j in idx]
                # A = A + res
                A.append(res)

            logger.info("constructing A time cost: {}, A length: {}"
                        .format(time() - lp_t, len(A)))

            logger.info("total loop num: {}".format(len(conflict_pair_hole_idx)))
            for i in range(len(conflict_pair_hole_idx)):
                a, b = conflict_pair_hole_idx[i]
                idx = list(range(a * radish_num, a * radish_num + radish_num)) \
                      + list(range(b * radish_num, b * radish_num + radish_num))
                res = [[i, j, 1] for j in idx]
                # B = B + res
                B.append(res)
                if i % 100 == 0:
                    logger.info("{}-th, time cost: {}".format(i, time() - lp_t))
            logger.info("constructing A and B time cost: {}, B length: {}".format(time() - lp_t, len(B)))
            for i in range(radish_num):
                idx = [t * radish_num + i for t in range(hole_num)]
                res = [[i, j, 1] for j in idx]
                # C = C + res
                C.append(res)
            logger.info("constructing A, B and C time cost: {}".format(time() - lp_t))
            A = [x for j in A for x in j]
            B = [x for j in B for x in j]
            C = [x for j in C for x in j]
            A = list(zip(*A))
            B = list(zip(*B))
            C = list(zip(*C))
            logger.info("convert A, B and C time cost: {}".format(time() - lp_t))
            logger.info("begin convert sparse matrix ... ...")
            A = Matrix.sparse(hole_num, radish_num * hole_num, list(A[0]), list(A[1]), list(A[2]))
            B = Matrix.sparse(len(conflict_pair_hole_idx), radish_num * hole_num, list(B[0]), list(B[1]), list(B[2]))
            C = Matrix.sparse(radish_num, radish_num * hole_num, list(C[0]), list(C[1]), list(C[2]))

            M.constraint(Expr.mul(A, x), Domain.lessThan(1))
            M.constraint(Expr.mul(B, x), Domain.lessThan(1))
            M.constraint(Expr.mul(C, x), Domain.equalsTo(1))

            M.objective("obj", ObjectiveSense.Minimize, Expr.dot(linearized_cost_matrix, x))
            logger.info("begin solving ")
            M.solve()
            logger.info(x)
            logger.info("lp time: {}".format(time() - lp_t))
            exit()

            grid_t = time()
            max_val = 10000000
            cost_matrix = cost_matrix / cost_matrix.max() * max_val
            cost_matrix = (cost_matrix + 0.5).astype(int)
            train_row_asses, train_col_asses, info = lapjv(cost_matrix)
            pure_grid_time = pure_grid_time + time() - grid_t
            if radish_num > hole_num:
                for i in range(radish_num):
                    corres_idx = train_col_asses[i]
                    if corres_idx < hole_num:
                        final_grid[sampled_idx[selected_idx_of_idx[i]]] = \
                            grid[holes[corres_idx][1] * grid_n + holes[corres_idx][0]]
                    else:
                        sampled_assigned_class[selected_idx_of_idx[i]] -= 1
            else:
                for i in range(hole_num):
                    corres_idx = train_col_asses[i]
                    if i < radish_num:
                        final_grid[sampled_idx[selected_idx_of_idx[i]], :] = \
                            grid[holes[corres_idx][1] * grid_n + holes[corres_idx][0]]
                    else:
                        mask_matrix[holes[corres_idx][0]:holes[corres_idx][0] + size,
                        holes[corres_idx][1]:holes[corres_idx][1] + size] = -1

        grid_X = final_grid[np.array(all_idx), :]
        logger.info("single grid layout time cost: {}".format(time() - t))
        logger.info("pure single grid layout time cost: {}".format(pure_grid_time))
        logger.info("histogram time cost: {}".format(histogram_time))
        return grid_X, sampled_assigned_class, mask_matrix, grid_n

    def hierarchy_layout_knn_lp(self, all_idx, sampled_idx, assignment_idx, square_num):
        t = time()
        pure_grid_time = 0
        histogram_time = 0
        sampled_assigned_num = np.zeros(len(sampled_idx))
        for idx in assignment_idx:
            count_idx = sampled_idx.tolist().index(idx)
            sampled_assigned_num[count_idx] += 1

        histogram_t = time()
        # histogram
        hist, bins = np.histogram(sampled_assigned_num, 100 * square_num, normed=True)
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]
        sampled_assigned_num = np.interp(sampled_assigned_num, bins[:-1], cdf)
        histogram_time = histogram_time + time() - histogram_t
        # visualization
        #########################
        # plt.hist(sampled_assigned_num, bins= 100 * square_num)
        # plt.show()
        # exit()
        ###########################
        sampled_assigned_class = sampled_assigned_num / \
                                 (sampled_assigned_num.max() * 1.01) * square_num
        sampled_assigned_class = sampled_assigned_class.astype(int) + 1

        for size in range(square_num, 0, -1):
            logger.info("{} squares with size of {}*{}"
                        .format(sum(sampled_assigned_class == size), size, size))
        total_grid_num = (sampled_assigned_class ** 2).sum()
        grid_n = int(np.sqrt(total_grid_num) * 1.2)
        grid = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / grid_n, grid_n),
                                     np.linspace(0, 1 - 1.0 / grid_n, grid_n))).reshape(-1, 2)
        min_grid_width = grid[1][0]
        embed_X = self.data.embed_X.copy()
        embed_X -= embed_X.min(axis=0)
        embed_X /= embed_X.max(axis=0)
        mask_matrix = np.ones((grid_n, grid_n)).astype(int) * -1
        final_grid = np.zeros(embed_X.shape)
        for size in range(square_num, 0, -1):
            selected_idx = sampled_idx[sampled_assigned_class == size]
            selected_idx_of_idx = np.array(range(len(sampled_assigned_class)))[sampled_assigned_class == size]
            radish_num = len(selected_idx)
            radish = embed_X[np.array(selected_idx), :]
            holes = []
            for i in range(mask_matrix.shape[0]):
                for j in range(mask_matrix.shape[1]):
                    square = mask_matrix[i:i + size, j:j + size]
                    if (square.sum()) <= (- size * size):
                        holes.append([i, j])
            hole_num = len(holes)
            conflict_pair_hole_idx = []
            for i in range(hole_num):
                for j in range(i + 1, hole_num):
                    if (abs(holes[i][0] - holes[j][0]) < size and abs(holes[i][1] - holes[j][1]) < size):
                        conflict_pair_hole_idx.append([i, j])
            logger.info("hole num: {}".format(hole_num))
            cost_matrix = np.ones((radish_num, hole_num))
            lp_t = time()
            for i in range(radish_num):
                for j in range(hole_num):
                    hole_coordinate = grid[holes[j][1] * grid_n + holes[j][0]]
                    hole_coordinate = np.array(hole_coordinate) + \
                                      np.array([min_grid_width * size / 2.0, min_grid_width * size / 2.0])
                    radish_coordinate = radish[i, :]
                    dis = (((hole_coordinate - radish_coordinate) ** 2).sum()) ** 0.5
                    assert dis * 10 < 100000
                    cost_matrix[i, j] = dis
            sparse_cost_matrix = knn_sparse(cost_matrix.transpose(), k=50)
            sparse_cost_matrix = csr_matrix(sparse_cost_matrix)
            data = sparse_cost_matrix.data
            indptr = sparse_cost_matrix.indptr
            indices = sparse_cost_matrix.indices
            variable_indices = list(range(len(indices)))
            logger.debug("knn construction is finished, variables num: {}".format(len(data)))
            M = Model("lo1")
            # x = M.variable("x", len(data), Domain.greaterThan(0)); logger.info(">=0")
            x = M.variable("x", len(data), Domain.binary());
            logger.info("binary")
            A = []
            B = []
            C = []

            for i in range(hole_num):
                idx = variable_indices[indptr[i]:indptr[i + 1]]
                res = [[i, j, 1] for j in idx]
                A.append(res)

            logger.info("constructing A, A length: {}"
                        .format(len(A)))

            logger.info("total loop num: {}".format(len(conflict_pair_hole_idx)))
            for i in range(len(conflict_pair_hole_idx)):
                a, b = conflict_pair_hole_idx[i]
                idx = variable_indices[indptr[a]:indptr[a + 1]] + \
                      variable_indices[indptr[b]:indptr[b + 1]]
                res = [[i, j, 1] for j in idx]
                B.append(res)
                if i % 10000 == 0:
                    logger.debug("{}-th, time cost: {}".format(i, time() - lp_t))
            logger.info("constructing B, B length: {}".format(len(B)))
            for i in range(radish_num):
                idx = [variable_indices[indices[indptr[j]:indptr[j + 1]].tolist().index(i) + indptr[j]]
                       for j in range(hole_num) if i in indices[indptr[j]:indptr[j + 1]]]
                res = [[i, j, 1] for j in idx]
                C.append(res)
            logger.info("constructing C, C length: {}".format(len(B)))
            A = [x for j in A for x in j]
            B = [x for j in B for x in j]
            C = [x for j in C for x in j]
            A = list(zip(*A))
            B = list(zip(*B))
            C = list(zip(*C))
            logger.info("finished python's convert sparse matrix")
            A = Matrix.sparse(hole_num, len(data), list(A[0]), list(A[1]), list(A[2]))
            if len(B) > 1:
                B = Matrix.sparse(len(conflict_pair_hole_idx), len(data), list(B[0]), list(B[1]), list(B[2]))
                M.constraint(Expr.mul(B, x), Domain.lessThan(1))
            C = Matrix.sparse(radish_num, len(data), list(C[0]), list(C[1]), list(C[2]))
            logger.info("finished mosek's convert sparse matrix")

            logger.info("adding constraints")
            M.constraint(Expr.mul(A, x), Domain.lessThan(1))
            M.constraint(Expr.mul(C, x), Domain.equalsTo(1))

            M.objective("obj", ObjectiveSense.Minimize, Expr.dot(data, x))
            M.setLogHandler(sys.stdout)
            logger.info("begin solving")
            M.solve()
            logger.info(x.level())
            logger.info("lp time: {}".format(time() - lp_t))
            solutions = x.level()
            # exit()

            for i in range(hole_num):
                tmp_sol = solutions[indptr[i]:indptr[i + 1]]
                if len(tmp_sol) < 1 or tmp_sol.max() < 0.5:
                    continue
                tmp_idx = (tmp_sol + 0.5).astype(int).tolist().index(1)
                tmp_idx = indices[indptr[i]:indptr[i + 1]][tmp_idx]
                try:
                    final_grid[sampled_idx[selected_idx_of_idx[tmp_idx]]] = \
                        grid[holes[i][1] * grid_n + holes[i][0]]
                except Exception as e:
                    print(e)
                    aa = 1
                mask_matrix[holes[i][0]:holes[i][0] + size, \
                holes[i][1]:holes[i][1] + size] = size

            a = 1

        grid_X = final_grid[np.array(all_idx), :]
        logger.info("single grid layout time cost: {}".format(time() - t))
        logger.info("pure single grid layout time cost: {}".format(pure_grid_time))
        logger.info("histogram time cost: {}".format(histogram_time))
        return grid_X, sampled_assigned_class, mask_matrix, grid_n

    def get_grid_layout(self, embed_method="tsne", debug_request_mask=False):
        """

        :param embed_method:
        :return:
        """
        if self.buffer_mode is True and os.path.exists(self.grid_filepath):
            logger.warn("buffer mode is on, and buffer exists")
            mat = pickle_load_data(self.grid_filepath)
            if not embed_method in mat:
                raise ValueError("embed method {} is not supported now.".format(embed_method))
            embed_mat = mat[embed_method]
            grid_X_train = embed_mat[config.grid_X_train_name]
            grid_X_test = embed_mat[config.grid_X_test_name]
            return grid_X_train, grid_X_test
        train_sampled_idx, train_unsampled_idx, train_assignment_idx, \
        test_sampled_idx, test_unsampled_idx, test_assignment_idx = self._sampling_and_assignment()

        # vis test
        # ax = plt.subplot(121)
        # sampled_embed_X_train = self.data.embed_X[np.array(train_sampled_idx), :]
        # sampled_embed_X_test = self.data.embed_X[np.array(test_sampled_idx), :]
        # color_map = plt.get_cmap("tab10")(np.array(self.data.y))
        # ax.scatter(sampled_embed_X_train[:,0], sampled_embed_X_train[:, 1],
        #            s=8, marker="o", c=color_map[np.array(np.array(train_sampled_idx))])
        # ax_test = plt.subplot(221)
        # ax_test.scatter(sampled_embed_X_test[:,0], sampled_embed_X_test[:, 1],
        #            s=8, marker="o", c=color_map[np.array(np.array(test_sampled_idx))])
        # plt.show()
        # exit()

        self.square_num = 3
        t = time()

        grid_X_train, train_sampled_assigned_class, train_mask_matrix, train_grid_n = \
            self.hierarchy_layout_knn_lp(self.train_idx,
                                         train_sampled_idx, train_assignment_idx, self.square_num)
        grid_X_test, test_sampled_assigned_class, test_mask_matrix, test_grid_n = \
            self.hierarchy_layout_knn_lp(self.test_idx,
                                         test_sampled_idx, test_assignment_idx, self.square_num)

        if debug_request_mask:
            return train_mask_matrix, test_mask_matrix

        train_grid_width = 1.0 / train_grid_n
        test_grid_width = 1.0 / test_grid_n
        embed_X = self.data.embed_X.copy()
        embed_X -= embed_X.min(axis=0)
        embed_X /= embed_X.max(axis=0)
        # sub_layout
        all_width = np.ones(self.data.embed_X.shape[0]) * 0
        all_width[np.array(self.train_idx)] = train_grid_width
        all_width[np.array(self.test_idx)] = test_grid_width
        grid_X = np.zeros(self.data.embed_X.shape)
        grid_X[np.array(self.train_idx), :] = grid_X_train
        grid_X[np.array(self.test_idx), :] = grid_X_test
        for i in range(len(train_sampled_idx)):
            global_idx = train_sampled_idx[i]
            size = train_sampled_assigned_class[i]
            # all_width[global_idx] = size * train_grid_width
            # if size == 1:
            #     continue
            origin_coordinate = grid_X[global_idx, :]
            childrens = train_unsampled_idx[train_assignment_idx == global_idx]
            nodes = childrens.tolist() + [global_idx]
            nodes_num = len(nodes)
            sub_grid_n = int(np.sqrt(nodes_num)) + 1
            width = size * train_grid_width
            height = size * train_grid_width
            sub_grid = np.dstack(np.meshgrid(np.linspace(0, width - width / sub_grid_n, sub_grid_n),
                                             np.linspace(0, height - height / sub_grid_n, sub_grid_n))).reshape(-1, 2)
            sub_width = sub_grid[1][0]
            sub_grid = sub_grid + origin_coordinate
            X_for_grid = embed_X[np.array(nodes)]
            cost_matrix = cdist(sub_grid, X_for_grid)
            assert cost_matrix.max() * 10 < 100
            dummy_vertices = np.ones((cost_matrix.shape[0], cost_matrix.shape[0] - cost_matrix.shape[1])) * 100
            cost_matrix = np.concatenate((cost_matrix, dummy_vertices), axis=1)
            row_asses, col_asses, info = lapjv(cost_matrix)
            grid_X[np.array(nodes)] = sub_grid[col_asses[:nodes_num]]
            all_width[np.array(nodes)] = sub_width

        for i in range(len(test_sampled_idx)):
            global_idx = test_sampled_idx[i]
            size = test_sampled_assigned_class[i]
            # all_width[global_idx] = size * test_grid_width
            # if size == 1:
            #     continue
            origin_coordinate = grid_X[global_idx, :]
            childrens = test_unsampled_idx[test_assignment_idx == global_idx]
            nodes = [global_idx] + childrens.tolist()
            nodes_num = len(nodes)
            sub_grid_n = int(np.sqrt(nodes_num)) + 1
            width = size * test_grid_width
            height = size * test_grid_width
            sub_grid = np.dstack(np.meshgrid(np.linspace(0, width - width / sub_grid_n, sub_grid_n),
                                             np.linspace(0, height - height / sub_grid_n, sub_grid_n))).reshape(-1, 2)
            sub_width = sub_grid[1][0]
            sub_grid = sub_grid + origin_coordinate
            X_for_grid = embed_X[np.array(nodes)]
            cost_matrix = cdist(sub_grid, X_for_grid)
            assert cost_matrix.max() * 10 < 100
            dummy_vertices = np.ones((cost_matrix.shape[0], cost_matrix.shape[0] - cost_matrix.shape[1])) * 100
            cost_matrix = np.concatenate((cost_matrix, dummy_vertices), axis=1)
            row_asses, col_asses, info = lapjv(cost_matrix)
            grid_X[np.array(nodes)] = sub_grid[col_asses[:nodes_num]]
            all_width[np.array(nodes)] = sub_width

        grid_X_train = grid_X[np.array(self.train_idx)]
        grid_X_test = grid_X[np.array(self.test_idx)]
        train_width = all_width[np.array(self.train_idx)]
        test_width = all_width[np.array(self.test_idx)]

        cost_train = ((((grid_X_train - embed_X[np.array(self.train_idx)]) ** 2).sum(axis=1)) ** 0.5).sum()
        cost_test = ((((grid_X_test - embed_X[np.array(self.test_idx)]) ** 2).sum(axis=1)) ** 0.5).sum()
        logger.info("final cost train: {}".format(cost_train))
        logger.info("final cost test: {}".format(cost_test))

        logger.info("grid layout time cost: {}".format(time() - t))
        logger.info("final_grid_train shape: {}".format(grid_X_train.shape))
        logger.info("grid_test shape: {}".format(grid_X_test.shape))
        return grid_X_train, grid_X_test, train_width, test_width

    def get_grid_layout_native_lap(self, embed_method="tsne"):
        if self.buffer_mode is True and os.path.exists(self.grid_filepath):
            logger.warn("buffer mode is on, and buffer exists")
            mat = pickle_load_data(self.grid_filepath)
            # if the embed_method is not supported now
            if not embed_method in mat:
                raise ValueError("embed method {} is not supported now.".format(embed_method))
            embed_mat = mat[embed_method]
            grid_X_train = embed_mat[config.grid_X_train_name]
            grid_X_test = embed_mat[config.grid_X_test_name]
            return grid_X_train, grid_X_test

        # if buffer does not exists\
        logger.warn("buffer mode is on, but buffer does not exist.")
        mat = {}
        embed_X = self.data.embed_X.copy()
        embed_X -= embed_X.min(axis=0)
        embed_X /= embed_X.max(axis=0)
        embed_X_train = embed_X[np.array(self.train_idx), :]
        embed_X_test = embed_X[np.array(self.test_idx), :]
        train_num = embed_X_train.shape[0]
        test_num = embed_X_test.shape[0]

        # N_sqrt_train = int(np.sqrt(embed_X_train.shape[0])) + 1
        N_sqrt_train = int(np.sqrt(embed_X_train.shape[0]) * 1.05)
        N_train = N_sqrt_train * N_sqrt_train
        for_embed_X_train = embed_X_train[:N_train]
        # N_sqrt_test = int(np.sqrt(embed_X_test.shape[0])) + 1
        N_sqrt_test = int(np.sqrt(embed_X_test.shape[0]) * 1.05)
        N_test = N_sqrt_test * N_sqrt_test
        for_embed_X_test = embed_X_test[:N_test]

        grid_train = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / N_sqrt_train, N_sqrt_train),
                                           np.linspace(0, 1 - 1.0 / N_sqrt_train, N_sqrt_train))) \
            .reshape(-1, 2)
        grid_test = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / N_sqrt_test, N_sqrt_test),
                                          np.linspace(0, 1 - 1.0 / N_sqrt_test, N_sqrt_test))) \
            .reshape(-1, 2)
        t = time()
        train_original_cost_matrix = cdist(grid_train, for_embed_X_train, "euclidean")
        cost_matrix = train_original_cost_matrix
        # cost_matrix = train_original_cost_matrix * (1000 / train_original_cost_matrix.max())
        assert cost_matrix.max() * 10 < 100
        dummy_vertices = np.ones((N_train, N_train - cost_matrix.shape[1])) * 100
        cost_matrix = np.concatenate((cost_matrix, dummy_vertices), axis=1)
        train_row_asses, train_col_asses, info = lapjv(cost_matrix)
        # train_cost_1 = cost_matrix[train_col_asses, np.array(range(N_train))].sum()
        # train_cost_2 = cost_matrix[np.array(range(N_train)), train_row_asses].sum()
        train_cost_1 = train_original_cost_matrix[train_col_asses[:train_num],
                                                  np.array(range(N_train))[:train_num]].sum()
        tmp_idx = train_row_asses < train_num
        train_cost_2 = train_original_cost_matrix[np.array(range(N_train))[tmp_idx],
                                                  train_row_asses[tmp_idx]].sum()
        logger.info("train cost: {}".format(train_cost_1))

        test_original_cost_matrix = cdist(grid_test, for_embed_X_test, "euclidean")
        cost_matrix = test_original_cost_matrix
        # cost_matrix = test_original_cost_matrix * (1000 / test_original_cost_matrix.max())
        assert cost_matrix.max() * 10 < 100
        dummy_vertices = np.ones((N_test, N_test - cost_matrix.shape[1])) * 100
        cost_matrix = np.concatenate((cost_matrix, dummy_vertices), axis=1)
        test_row_asses, test_col_asses, info = lapjv(cost_matrix)
        test_cost_1 = test_original_cost_matrix[test_col_asses[:test_num],
                                                np.array(range(N_test))[:test_num]].sum()
        tmp_idx = test_row_asses < test_num
        test_cost_2 = test_original_cost_matrix[np.array(range(N_test))[tmp_idx],
                                                test_row_asses[tmp_idx]].sum()
        print("lapjv time: {}".format(time() - t))
        logger.info("test cost: {}".format(test_cost_1))

        train_col_asses = train_col_asses[:train_num]
        test_col_asses = test_col_asses[:test_num]
        grid_X_train = np.zeros(embed_X_train.shape)
        grid_X_test = np.zeros(embed_X_test.shape)
        grid_X_train[:len(train_col_asses)] = grid_train[train_col_asses]
        grid_X_test[:len(test_col_asses)] = grid_test[test_col_asses]

        cost_train = ((((grid_X_train - embed_X_train) ** 2).sum(axis=1)) ** 0.5).sum()
        cost_test = ((((grid_X_test - embed_X_test) ** 2).sum(axis=1)) ** 0.5).sum()
        logger.info("final cost train: {}".format(cost_train))
        logger.info("final cost train: {}".format(cost_test))
        # pickle_save_data(self.grid_filepath, mat)
        # embed_mat = mat[embed_method]
        # grid_X_train = embed_mat[config.grid_X_train_name]
        # grid_X_test = embed_mat[config.grid_X_test_name]
        return grid_X_train, grid_X_test, None, None

    def get_grid_layout_native_lap_knn_v1_very_slow(self, embed_method="tsne"):
        if self.buffer_mode is True and os.path.exists(self.grid_filepath):
            logger.warn("buffer mode is on, and buffer exists")
            mat = pickle_load_data(self.grid_filepath)
            # if the embed_method is not supported now
            if not embed_method in mat:
                raise ValueError("embed method {} is not supported now.".format(embed_method))
            embed_mat = mat[embed_method]
            grid_X_train = embed_mat[config.grid_X_train_name]
            grid_X_test = embed_mat[config.grid_X_test_name]
            return grid_X_train, grid_X_test

        # if buffer does not exists\
        if self.buffer_mode is True:
            logger.warn("buffer mode is on, but buffer does not exist.")
        else:
            logger.warn("buffer mode is off, and buffer does not exist.")
        mat = {}
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
        cost_matrix = train_original_cost_matrix.copy()
        # knn process
        k = 50
        hole_num = cost_matrix.shape[0]
        radish_num = cost_matrix.shape[1]
        radish_idx = [[]] * radish_num
        hole_idx = []
        hole_ranking = []
        for i in range(hole_num):
            hole_idx.append([])
            hole_ranking.append([])
        for i in range(radish_num):
            col = cost_matrix[:, i].copy()
            col_sort = col.argsort()
            radish_idx[i] = col_sort.tolist()
            for ranking, j in enumerate(col_sort[:k]):
                hole_idx[j].append(i)
                hole_ranking[j].append(ranking)
        while 1:
            hole_idx_lens = []
            radish_lens = []
            for j in range(hole_num):
                hole_idx_lens.append(len(hole_idx[j]))
            num = sum(np.array(hole_idx_lens) > k)
            logger.info("degree of left side: {}, degree of right size: {}".
                        format(radish_num * k, sum(hole_idx_lens)))
            logger.info("nodes with degree larger than k is {}".format(num))
            if num <= 0:
                break
            hole_idx_sorted_idx = np.array(hole_idx_lens).argsort()[::-1]
            for count, j in enumerate(hole_idx_sorted_idx):
                idxs = hole_idx[j]
                ranking = hole_ranking[j]
                if len(idxs) < k:
                    continue
                # logger.info("begin")
                sorted_idx = cost_matrix[j, np.array(idxs)].argsort()
                sorted_idx = np.array(idxs)[sorted_idx]
                preserved_idx = sorted_idx[:k].tolist()
                # logger.info("remove")
                rest = []
                for i in sorted_idx[k:]:
                    if len(radish_idx[i]) <= k:
                        rest.append(i)
                        cost_matrix[j, i] = 0
                        continue
                    radish_idx[i].remove(j)
                    try:
                        added_hole_idx = radish_idx[i][k - 1]
                    except Exception as e:
                        print(e)
                        import IPython;
                        IPython.embed()
                    hole_idx[added_hole_idx].append(i)
                # for i in range(radish_num):
                #     if i not in preserved_idx:
                #         position = radish_idx[i].index(j)
                #         radish_idx[i][position] = -1
                # logger.info("finished")
                hole_idx[j] = sorted_idx[:k].tolist() + rest
                # if count % 1000 == 0:
                #     logger.info(count)
                # hole_idx[j] = sorted_idx.tolist()
                #

        # # cost_matrix[col_sort[:k], i] = train_original_cost_matrix[col_sort[:k], i]
        # for i in range(cost_matrix.shape[0]):
        #     row = train_original_cost_matrix[i,:].copy()
        #     row_sort = row.argsort()
        #     # cost_matrix[i, row_sort[:5]] = train_original_cost_matrix[i, row_sort[:5]]

        cost_matrix = np.ones(train_original_cost_matrix.shape) * 0
        for i in range(radish_num):
            cost_matrix[np.array(radish_idx[i][:k]), i] = \
                train_original_cost_matrix[np.array(radish_idx[i][:k]), i]

        # cost_matrix = train_original_cost_matrix * (1000 / train_original_cost_matrix.max())
        dummy_vertices = np.ones((N_train, N_train - cost_matrix.shape[1])) * 100000
        cost_matrix = np.concatenate((cost_matrix, dummy_vertices), axis=1)
        t = time()
        # train_row_asses, train_col_asses, info = lapjv(cost_matrix)
        csr_cost_matrix = csr_matrix(cost_matrix)
        print("data lens: {}, data shape: {}".format(len(csr_cost_matrix.data),
                                                     cost_matrix.shape))
        info, train_row_asses, train_col_asses = \
            lap_mod(cost_matrix.shape[0], csr_cost_matrix.data,
                    csr_cost_matrix.indptr, csr_cost_matrix.indices)
        # train_cost_1 = cost_matrix[train_col_asses,
        #                             np.array(range(len(train_col_asses)))].sum()
        train_cost_1 = train_original_cost_matrix[train_col_asses[:train_num],
                                                  np.array(range(N_train))[:train_num]].sum()
        train_col_asses = train_col_asses[:train_num]
        grid_X_train = np.zeros(embed_X_train.shape)
        grid_X_train[:len(train_col_asses)] = grid_train[train_col_asses]
        logger.info("train info[0]: {}".format(info))
        logger.info("train cost: {}, time cost: {}".format(train_cost_1, time() - t))

        ##################
        ##### test #######
        ##################
        test_original_cost_matrix = cdist(grid_test, for_embed_X_test, "euclidean")
        cost_matrix = np.ones(test_original_cost_matrix.shape) * 10000
        # cost_matrix = test_original_cost_matrix * (1000 / test_original_cost_matrix.max())
        # knn process
        for i in range(cost_matrix.shape[1]):
            col = test_original_cost_matrix[:, i].copy()
            col_sort = col.argsort()
            cost_matrix[col_sort[:k], i] = test_original_cost_matrix[col_sort[:k], i]
        for i in range(cost_matrix.shape[0]):
            row = test_original_cost_matrix[i, :].copy()
            row_sort = row.argsort()
            cost_matrix[i, row_sort[:5]] = test_original_cost_matrix[i, row_sort[:5]]
        dummy_vertices = np.ones((N_test, N_test - cost_matrix.shape[1])) * 100000
        cost_matrix = np.concatenate((cost_matrix, dummy_vertices), axis=1)
        t = time()
        test_row_asses, test_col_asses, info = lapjv(cost_matrix)
        # csr_cost_matrix = csr_matrix(cost_matrix)
        # info, test_row_asses, test_col_asses = \
        #     lap_mod(cost_matrix.shape[0],csr_cost_matrix.data,
        #             csr_cost_matrix.indptr, csr_cost_matrix.indices)
        test_cost_1 = test_original_cost_matrix[test_col_asses[:test_num],
                                                np.array(range(N_test))[:test_num]].sum()
        logger.info("test cost: {}, time cost: {}".format(test_cost_1, time() - t))

        train_col_asses = train_col_asses[:train_num]
        test_col_asses = test_col_asses[:test_num]
        grid_X_train = np.zeros(embed_X_train.shape)
        grid_X_test = np.zeros(embed_X_test.shape)
        grid_X_train[:len(train_col_asses)] = grid_train[train_col_asses]
        grid_X_test[:len(test_col_asses)] = grid_test[test_col_asses]

        cost_train = ((((grid_X_train - embed_X_train) ** 2).sum(axis=1)) ** 0.5).sum()
        cost_test = ((((grid_X_test - embed_X_test) ** 2).sum(axis=1)) ** 0.5).sum()
        logger.info("final cost train: {}".format(cost_train))
        logger.info("final cost train: {}".format(cost_test))

        # save file
        # pickle_save_data(self.grid_filepath, mat)
        # embed_mat = mat[embed_method]
        # grid_X_train = embed_mat[config.grid_X_train_name]
        # grid_X_test = embed_mat[config.grid_X_test_name]

        print("lap with knn time: {}".format(time() - t))
        return grid_X_train, grid_X_test, None, None

    def get_grid_layout_native_lap_knn(self, embed_method="tsne"):
        if self.buffer_mode is True and os.path.exists(self.grid_filepath):
            logger.warn("buffer mode is on, and buffer exists")
            mat = pickle_load_data(self.grid_filepath)
            # if the embed_method is not supported now
            if not embed_method in mat:
                raise ValueError("embed method {} is not supported now.".format(embed_method))
            embed_mat = mat[embed_method]
            grid_X_train = embed_mat[config.grid_X_train_name]
            grid_X_test = embed_mat[config.grid_X_test_name]
            train_width = embed_mat["train_width"]
            test_width = embed_mat["test_width"]
            return grid_X_train, grid_X_test

        # if buffer does not exists\
        if self.buffer_mode is True:
            logger.warn("buffer mode is on, but buffer does not exist.")
        else:
            logger.warn("buffer mode is off, and buffer does not exist.")
        # knn param
        k = 50

        mat = {}
        embed_X = self.data.embed_X.copy()
        embed_X -= embed_X.min(axis=0)
        embed_X /= embed_X.max(axis=0)
        embed_X_train = embed_X[np.array(self.train_idx), :]
        embed_X_test = embed_X[np.array(self.test_idx), :]
        train_num = embed_X_train.shape[0]
        test_num = embed_X_test.shape[0]

        N_sqrt_train = int(np.sqrt(embed_X_train.shape[0])) + 1
        logger.info("N_sqrt_train: {}".format(N_sqrt_train))
        N_train = N_sqrt_train * N_sqrt_train
        for_embed_X_train = embed_X_train[:N_train]
        N_sqrt_test = int(np.sqrt(embed_X_test.shape[0])) + 1
        logger.info("N_sqrt_test: {}".format(N_sqrt_test))
        N_test = N_sqrt_test * N_sqrt_test
        for_embed_X_test = embed_X_test[:N_test]

        grid_train = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / N_sqrt_train, N_sqrt_train),
                                           np.linspace(0, 1 - 1.0 / N_sqrt_train, N_sqrt_train))) \
            .reshape(-1, 2)
        grid_test = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / N_sqrt_test, N_sqrt_test),
                                          np.linspace(0, 1 - 1.0 / N_sqrt_test, N_sqrt_test))) \
            .reshape(-1, 2)
        train_grid_width = grid_train[1][0]
        test_grid_width = grid_test[1][0]

        ######################
        ##### training #######
        ######################
        t_sum = time()
        train_original_cost_matrix = cdist(grid_train, for_embed_X_train, "euclidean")
        # knn process
        dummy_points = np.ones((N_train - train_original_cost_matrix.shape[1], 2)) * 0.5
        # dummy at [0.5, 0.5]
        dummy_vertices = (1 - cdist(grid_train, dummy_points, "euclidean")) * 100
        cost_matrix = np.concatenate((train_original_cost_matrix, dummy_vertices), axis=1)

        cost_matrix = cost_matrix.astype(np.dtype('d'))
        # rows, cols = cost_matrix.shape
        # cost_matrix_1 = np.asarray(cost_matrix.copy())
        # cost_matrix_2 = np.asarray(cost_matrix.T.copy())
        # ptr1 = cost_matrix_1.ctypes.data_as(ctypes.c_char_p)
        # ptr2 = cost_matrix_2.ctypes.data_as(ctypes.c_char_p)
        # libc.knn_sparse(ptr1, rows, cols, k, False, 0)
        # libc.knn_sparse(ptr2, cols, rows, k, False, 0)
        # cost_matrix_2 = cost_matrix_2.T
        # logger.info("end knn preprocessing")
        #
        # # merge two sub-graph
        # cost_matrix = np.maximum(cost_matrix_1, cost_matrix_2)
        #
        # # binning
        # cost_matrix = cost_matrix / train_original_cost_matrix.max() * 100
        # cost_matrix = cost_matrix.astype(np.int32)
        cost_matrix[cost_matrix == 0] = 10000000

        # begin LAP-JV
        logger.info("begin LAP JV in training")
        t = time()
        train_row_asses, train_col_asses, info = lapjvo(cost_matrix)
        train_cost_1 = train_original_cost_matrix[train_col_asses[:train_num],
                                                  np.array(range(N_train))[:train_num]].sum()
        train_col_asses = train_col_asses[:train_num]
        grid_X_train = np.zeros(embed_X_train.shape)
        grid_X_train[:len(train_col_asses)] = grid_train[train_col_asses]
        logger.info("train cost: {}, time cost: {}".format(train_cost_1, time() - t))

        ##################
        ##### test #######
        ##################
        test_original_cost_matrix = cdist(grid_test, for_embed_X_test, "euclidean")
        dummy_points = np.ones((N_test - test_original_cost_matrix.shape[1], 2)) * 0.5
        dummy_vertices = (1 - cdist(grid_test, dummy_points, "euclidean")) * 100 # dummy [0.5, 0.5]
        cost_matrix = np.concatenate((test_original_cost_matrix, dummy_vertices), axis=1)
        cost_matrix = cost_matrix.astype(np.dtype('d'))
        rows, cols = cost_matrix.shape
        cost_matrix_1 = np.asarray(cost_matrix.copy())
        cost_matrix_2 = np.asarray(cost_matrix.T.copy())
        ptr1 = cost_matrix_1.ctypes.data_as(ctypes.c_char_p)
        ptr2 = cost_matrix_2.ctypes.data_as(ctypes.c_char_p)
        libc.knn_sparse(ptr1, rows, cols, k, False, 0)
        libc.knn_sparse(ptr2, cols, rows, k, False, 0)
        cost_matrix_2 = cost_matrix_2.T
        logger.info("end knn preprocessing")

        # merge two sub-graph
        cost_matrix = np.maximum(cost_matrix_1, cost_matrix_2)

        # binning
        cost_matrix = cost_matrix / test_original_cost_matrix.max() * 100
        cost_matrix = cost_matrix.astype(np.int32)
        cost_matrix[cost_matrix == 0] = 10000000

        # begin LAP-JV
        logger.info("begin LAP JV in test")
        t = time()
        test_row_asses, test_col_asses, info = lapjv(cost_matrix)
        test_cost_1 = test_original_cost_matrix[test_col_asses[:test_num],
                                                np.array(range(N_test))[:test_num]].sum()
        logger.info("test cost: {}, time cost: {}".format(test_cost_1, time() - t))

        train_col_asses = train_col_asses[:train_num]
        test_col_asses = test_col_asses[:test_num]
        grid_X_train = np.zeros(embed_X_train.shape)
        grid_X_test = np.zeros(embed_X_test.shape)
        grid_X_train[:len(train_col_asses)] = grid_train[train_col_asses]
        grid_X_test[:len(test_col_asses)] = grid_test[test_col_asses]

        cost_train = ((((grid_X_train - embed_X_train) ** 2).sum(axis=1)) ** 0.5).sum()
        cost_test = ((((grid_X_test - embed_X_test) ** 2).sum(axis=1)) ** 0.5).sum()
        logger.info("final cost train: {}".format(cost_train))
        logger.info("final cost test: {}".format(cost_test))

        train_width = np.ones(grid_X_train.shape[0]) * train_grid_width
        test_width = np.ones(grid_X_test.shape[0]) * test_grid_width

        # merging adjacent instances
        train_row_asses = train_row_asses.reshape(N_sqrt_train, N_sqrt_train)
        train_mask = np.zeros((N_sqrt_train, N_sqrt_train)).astype(int)
        stride_size = 1
        for i in range(N_sqrt_train-stride_size+1):
            for j in range(N_sqrt_train-stride_size+1):
                idxs = train_row_asses[i:i+stride_size, j:j+stride_size]
                if (idxs < train_num).all() and \
                    (train_mask[i:i+stride_size, j:j+stride_size] == 0).all():
                    train_width[train_row_asses[i, j]] *= stride_size
                    train_width[idxs.reshape(-1)[1:]] = -1
                    train_mask[i:i+stride_size, j:j+stride_size] = 1

        test_row_asses = test_row_asses.reshape(N_sqrt_test, N_sqrt_test)
        test_mask = np.zeros((N_sqrt_test, N_sqrt_test)).astype(int)
        stride_size = 1
        for i in range(N_sqrt_test-stride_size+1):
            for j in range(N_sqrt_test-stride_size+1):
                idxs = test_row_asses[i:i+stride_size, j:j+stride_size]
                if (idxs < test_num).all() and \
                    (test_mask[i:i+stride_size, j:j+stride_size] == 0).all():
                    test_width[test_row_asses[i, j]] *= stride_size
                    test_width[idxs.reshape(-1)[1:]] = -1
                    test_mask[i:i+stride_size, j:j+stride_size] = 1



        # save file
        mat[config.grid_X_train_name] = grid_X_train
        mat[config.grid_X_test_name] = grid_X_test
        mat["train_width"] = train_width
        mat["test_width"] = test_width
        pickle_save_data(self.grid_filepath, mat)
        logger.info("grid layout buffer is saved")

        self.grid_X_train = grid_X_train
        self.grid_X_test = grid_X_test
        self.train_row_asses = train_row_asses.reshape(-1)
        self.test_row_asses = test_row_asses.reshape(-1)

        return grid_X_train, grid_X_test, train_row_asses.reshape(-1), test_row_asses.reshape(-1), train_width, test_width

    def get_grid_layout_native_lap_inverse_knn(self, embed_method="tsne"):
        if self.buffer_mode is True and os.path.exists(self.grid_filepath):
            logger.warn("buffer mode is on, and buffer exists")
            mat = pickle_load_data(self.grid_filepath)
            # if the embed_method is not supported now
            if not embed_method in mat:
                raise ValueError("embed method {} is not supported now.".format(embed_method))
            embed_mat = mat[embed_method]
            grid_X_train = embed_mat[config.grid_X_train_name]
            grid_X_test = embed_mat[config.grid_X_test_name]
            return grid_X_train, grid_X_test

        # if buffer does not exists\
        if self.buffer_mode is True:
            logger.warn("buffer mode is on, but buffer does not exist.")
        else:
            logger.warn("buffer mode is off, and buffer does not exist.")
        mat = {}
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
        cost_matrix = np.ones(train_original_cost_matrix.shape) * 10000
        # knn process
        k = 50
        for j in range(cost_matrix.shape[0]):
            row = train_original_cost_matrix[j, :]
            row_sort = row.argsort()
            cost_matrix[j, row_sort[:k]] = train_original_cost_matrix[j, row_sort[:k]]
        for i in range(cost_matrix.shape[1]):
            col = train_original_cost_matrix[:, i]
            col_sort = col.argsort()
            cost_matrix[col_sort[:k], i] = train_original_cost_matrix[col_sort[:k], i]

        # cost_matrix = train_original_cost_matrix * (1000 / train_original_cost_matrix.max())
        dummy_vertices = np.ones((N_train, N_train - cost_matrix.shape[1])) * 100000
        cost_matrix = np.concatenate((cost_matrix, dummy_vertices), axis=1)
        t = time()
        train_row_asses, train_col_asses, info = lapjv(cost_matrix)
        # csr_cost_matrix = csr_matrix(cost_matrix)
        # print("data lens: {}, data shape: {}".format(len(csr_cost_matrix.data),
        #                                              cost_matrix.shape))
        # info, train_row_asses, train_col_asses = \
        #     lap_mod(cost_matrix.shape[0],csr_cost_matrix.data,
        #             csr_cost_matrix.indptr, csr_cost_matrix.indices)
        # train_cost_1 = cost_matrix[train_col_asses,
        #                             np.array(range(len(train_col_asses)))].sum()
        train_cost_1 = train_original_cost_matrix[train_col_asses[:train_num],
                                                  np.array(range(N_train))[:train_num]].sum()
        train_col_asses = train_col_asses[:train_num]
        grid_X_train = np.zeros(embed_X_train.shape)
        grid_X_train[:len(train_col_asses)] = grid_train[train_col_asses]
        logger.info("train info[0]: {}".format(info))
        logger.info("train cost: {}, time cost: {}".format(train_cost_1, time() - t))

        test_original_cost_matrix = cdist(grid_test, for_embed_X_test, "euclidean")
        cost_matrix = np.ones(test_original_cost_matrix.shape) * 10000
        # cost_matrix = test_original_cost_matrix * (1000 / test_original_cost_matrix.max())
        # knn process
        for i in range(cost_matrix.shape[1]):
            col = test_original_cost_matrix[:, i].copy()
            col_sort = col.argsort()
            cost_matrix[col_sort[:k], i] = test_original_cost_matrix[col_sort[:k], i]
        for i in range(cost_matrix.shape[0]):
            row = test_original_cost_matrix[i, :].copy()
            row_sort = row.argsort()
            cost_matrix[i, row_sort[:5]] = test_original_cost_matrix[i, row_sort[:5]]
        dummy_vertices = np.ones((N_test, N_test - cost_matrix.shape[1])) * 100000
        cost_matrix = np.concatenate((cost_matrix, dummy_vertices), axis=1)
        t = time()
        test_row_asses, test_col_asses, info = lapjv(cost_matrix)
        # csr_cost_matrix = csr_matrix(cost_matrix)
        # info, test_row_asses, test_col_asses = \
        #     lap_mod(cost_matrix.shape[0],csr_cost_matrix.data,
        #             csr_cost_matrix.indptr, csr_cost_matrix.indices)
        test_cost_1 = test_original_cost_matrix[test_col_asses[:test_num],
                                                np.array(range(N_test))[:test_num]].sum()
        logger.info("test cost: {}, time cost: {}".format(test_cost_1, time() - t))

        train_col_asses = train_col_asses[:train_num]
        test_col_asses = test_col_asses[:test_num]
        grid_X_train = np.zeros(embed_X_train.shape)
        grid_X_test = np.zeros(embed_X_test.shape)
        grid_X_train[:len(train_col_asses)] = grid_train[train_col_asses]
        grid_X_test[:len(test_col_asses)] = grid_test[test_col_asses]

        cost_train = ((((grid_X_train - embed_X_train) ** 2).sum(axis=1)) ** 0.5).sum()
        cost_test = ((((grid_X_test - embed_X_test) ** 2).sum(axis=1)) ** 0.5).sum()
        logger.info("final cost train: {}".format(cost_train))
        logger.info("final cost train: {}".format(cost_test))

        # save file
        # pickle_save_data(self.grid_filepath, mat)
        # embed_mat = mat[embed_method]
        # grid_X_train = embed_mat[config.grid_X_train_name]
        # grid_X_test = embed_mat[config.grid_X_test_name]

        print("lap with knn time: {}".format(time() - t))
        return grid_X_train, grid_X_test, None, None

    def get_prediction_results(self):
        train_pred_y = self.data.pred_train
        test_pred_y = self.data.pred_test
        return train_pred_y, test_pred_y

    def get_predict_probability(self):
        train_pred_proba_y = self.data.clf.predict_proba(self.X_train)
        test_pred_proba_y = self.data.clf.predict_proba(self.X_test)
        return train_pred_proba_y, test_pred_proba_y

    def get_decision_boundary(self, data_type):
        def _meanSqaureResidue(x, y, regr):
            if regr == None:
                return 0
            else:
                x = x.reshape(-1, 1)
                y = y
                pred_y = regr.predict(x)
                return np.mean((y - pred_y) ** 2)
        def _getintersection(lr1, lr2):
            if lr1[2] == None and lr2[2] == None:
                return None
            if lr1[2] == None:
                m = lr2[2]
                x = _points[lr1[0]][0]
                y = m.coef_[0] * x + m.intercept_
                return (x, y)
            if lr2[2] == None:
                m = lr1[2]
                x = _points[lr2[0]][0]
                y = m.coef_[0] * x + m.intercept_
                return (x, y)
            m1 = lr1[2]
            m2 = lr2[2]
            if m1.coef_[0] == m2.coef_[0]:
                return None
            else:
                x = (m2.intercept_ - m1.intercept_) / (m1.coef_[0] - m2.coef_[0])
                y = m1.coef_[0] * x + m1.intercept_
                return (x, y)
        def _recursivelyFit(points, l, r):
            regr, s = _regress(points, l, r)
            if s < 1e-3:
                return [[l, r, regr]]
            else:
                mid1 = int((2 * l + r) / 3)
                mid2 = int((l + 2 * r) / 3)
                part1 = _recursivelyFit(points, l, mid1)
                part2 = _recursivelyFit(points, mid1 + 1, mid2)
                part3 = _recursivelyFit(points, mid2 + 1, r)
                part2.extend(part3)
                part1.extend(part2)
                return part1
                # mid = int((l + r) / 2)
                # part1 = _recursivelyFit(points, l, mid)
                # part2 = _recursivelyFit(points, mid + 1, r)
                # part1.extend(part2)
                # return part1
        def _mergeFit(points, segments):
            flag = -1
            new_regr = None
            min_loss = 1
            for i in range(len(segments) - 1):
                l = segments[i][0]
                r = segments[i + 1][1]
                regr, s = _regress(points, l, r)
                if s < 1e-3 and s < min_loss:
                    new_regr = [[l, r, regr]]
                    flag = i
                    min_loss = s

            if flag == -1:
                return segments
            else:
                new_segments = segments[:flag]
                new_segments.extend(new_regr)
                new_segments.extend(segments[flag + 2:])
                return _mergeFit(points, new_segments)
        def _adjustFit(points, segments):
            for i in range(len(segments) - 1):
                # Nearly parallel
                intersection = _getintersection(segments[i], segments[i + 1])
                l = segments[i][0]
                r = segments[i + 1][1]
                p = points[l:r + 1]
                x = p[:, 0]
                y = p[:, 1]
                if intersection == None or \
                        (intersection[0] - np.max(x) > 0.5 * (np.max(x) - np.min(x))) or \
                        (np.min(x) - intersection[0] > 0.5 * (np.max(x) - np.min(x))) or \
                        (intersection[1] - np.max(y) > 0.5 * (np.max(y) - np.min(y))) or \
                        (np.min(y) - intersection[1] > 0.5 * (np.max(y) - np.min(y))):
                    regr, s = _regress(points, l, r)
                    _adjust_result = [[l, r, regr]]
                    new_segments = segments[:i]
                    new_segments.extend(_adjust_result)
                    new_segments.extend(segments[i + 2:])
                    return _adjustFit(points, new_segments)
            return segments
        def _fit(points, l, r):
            result = _recursivelyFit(points, l, r)
            result = _mergeFit(points, result)
            result = _adjustFit(points, result)
            return np.array(result)
        def _getBoundaryPoints():
            # support_vectors_index = np.array(self.clf.support_)
            print(data_type)
            if data_type == "train":
                X = self.X_train
                y = self.y_train
                grid_X = self.grid_X_train
                row_asses = self.train_row_asses
            else:
                X = self.X_test
                y = self.y_test
                grid_X = self.grid_X_test
                row_asses = self.test_row_asses
            N = int(np.sqrt(grid_X.shape[0])) + 1
            print(N)
            boundary_points_idx = []
            grid_train = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / N, N),
                                               np.linspace(0, 1 - 1.0 / N, N))) \
                .reshape(-1, 2)
            predictions = self.clf.predict(X)
            # Define valid indices
            valid_idx = [0 for _ in range(N**2)]
            f_neighbour = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            next_to_visit = 0
            while True:
                while next_to_visit < N**2 and \
                        (valid_idx[next_to_visit] != 0 or row_asses[next_to_visit] >= predictions.shape[0]):
                    next_to_visit += 1
                if next_to_visit >= N**2:
                    break
                head = 0
                tail = 0
                queue = [next_to_visit]
                valid_idx[next_to_visit] = -1
                while head <= tail:
                    cur = queue[head]
                    this_label = predictions[row_asses[cur]]
                    i = cur // N
                    j = cur - i * N
                    for t in range(4):
                        new_i = i + f_neighbour[t][0]
                        new_j = j + f_neighbour[t][1]
                        if new_i < 0 or new_i >= N or new_j < 0 or new_j >= N:
                            continue
                        new_pos = new_i * N + new_j
                        if row_asses[new_pos] >= predictions.shape[0] or \
                                this_label != predictions[row_asses[new_pos]] or \
                                valid_idx[new_pos] != 0:
                            continue
                        queue.append(new_pos)
                        valid_idx[new_pos] = -1
                        tail += 1
                    head += 1
                if tail > 0.01 * (N**2):
                    for idx in queue:
                        valid_idx[idx] = 1

            for i in range(N):
                for j in range(N):
                    idx = i * N + j
                    if valid_idx[idx] <= 0:
                        continue
                    # if y[row_asses[idx]] != predictions[row_asses[idx]]:
                    #     continue
                    this_label = predictions[row_asses[idx]]
                    if i < N - 1:
                        bottom_neighbour = idx + N
                        # if row_asses[bottom_neighbour] < grid_X.shape[0] and \
                        #         y[row_asses[bottom_neighbour]] != this_label and \
                        #         y[row_asses[bottom_neighbour]] == predictions[row_asses[bottom_neighbour]]:
                        if row_asses[bottom_neighbour] < grid_X.shape[0] and \
                                predictions[row_asses[bottom_neighbour]] != this_label and \
                                valid_idx[bottom_neighbour] > 0:
                            boundary_points_idx.append((idx, bottom_neighbour))
                    if j < N - 1:
                        right_neighbour = idx + 1
                        # if row_asses[right_neighbour] < grid_X.shape[0] and \
                        #         y[row_asses[right_neighbour]] != this_label and \
                        #         y[row_asses[right_neighbour]] == predictions[row_asses[right_neighbour]]:
                        if row_asses[right_neighbour] < grid_X.shape[0] and \
                                predictions[row_asses[right_neighbour]] != this_label and \
                                valid_idx[right_neighbour] > 0:
                            boundary_points_idx.append((idx, right_neighbour))
            boundary_points = [(grid_train[i] + grid_train[j]) / 2 for (i, j) in boundary_points_idx]
            # for idx in support_vectors_index:
            #     if y[idx] == predictions[idx]:
            #         boundary_points.append(grid_X[idx])
            return np.array(boundary_points)
        def _sortPoints(points):
            cnt = points.shape[0]
            inf = 1e8
            dist = [inf for _ in range(cnt)]
            visited = [False for _ in range(cnt)]
            # Select seed point
            corner_dist = [min(p[0], 1 - p[0]) + min(p[1], 1 - p[1]) for p in points]
            seed_idx = corner_dist.index(min(corner_dist))
            dist[seed_idx] = 0
            visited[seed_idx] = True
            idx_list = [seed_idx]
            # Add nearest point to idx_list
            for i in range(1, cnt):
                current = idx_list[-1]
                min_dist = inf
                next_idx = -1
                for j in range(cnt):
                    if not visited[j]:
                        dist[j] = min(dist[j], np.sqrt(np.sum(np.square(points[current] - points[j]))))
                        if dist[j] <= min_dist:
                            next_idx = j
                            min_dist = dist[j]
                idx_list.append(next_idx)
                visited[next_idx] = True
            sorted_points = points[idx_list]
            return sorted_points
        def _regress(points, l, r):
            p = points[l:r + 1]
            x = p[:, 0]
            y = p[:, 1]
            flag = False
            for i in range(x.shape[0]):
                if x[i] != x[0]:
                    flag = True
                    break
            if not flag:
                regr = None
            else:
                regr = linear_model.LinearRegression()
                regr.fit(x.reshape(-1, 1), y)
            return regr, _meanSqaureResidue(x, y, regr)
        def _createKeypoints(points, segments):
            lines_to_plot = []
            for i in range(len(segments)):
                l = lr[i][0]
                r = lr[i][1]
                p = _points[l:r + 1]
                x_coords = p[:, 0]
                x = [np.min(x_coords), np.max(x_coords)]
                y_coords = p[:, 1]
                y = [np.min(y_coords), np.max(y_coords)]
                lines_to_plot.append((lr[i][2], np.array(x), np.array(y)))
            lines_orignal = []
            for i, line in enumerate(lines_to_plot):
                regr = line[0]
                x = line[1]
                if regr == None:
                    # yl = lines_to_plot[i - 1][1].predict(lines_to_plot[i - 1][0])[1] if i > 0 else 0
                    # yr = lines_to_plot[i + 1][1].predict(lines_to_plot[i + 1][0])[0] if i < len(lines_to_plot)-1 else 1
                    y = line[2]
                else:
                    y = regr.predict(x.reshape(-1, 1))
                start_point = [x[0], y[0]]
                end_point = [x[1], y[1]]
                if i == 0:
                    start_point_offset = min(start_point[0], 1 - start_point[0], \
                                             abs(start_point[1]), abs(1 - start_point[1]))
                    end_point_offset = min(end_point[0], 1 - end_point[0], \
                                           abs(end_point[1]), abs(1 - end_point[1]))
                else:
                    prev_end_point = lines_orignal[-1][1]
                    start_point_offset = np.sum(np.square(np.array(prev_end_point) - np.array(start_point)))
                    end_point_offset = np.sum(np.square(np.array(prev_end_point) - np.array(end_point)))
                if start_point_offset > end_point_offset:
                    start_point = [x[1], y[1]]
                    end_point = [x[0], y[0]]
                print(i, start_point, end_point)
                k = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0]) \
                    if start_point[0] != end_point[0] else None
                if i == 0:
                    if start_point[0] < end_point[0]:
                        start_point[0] = 0
                        start_point[1] = end_point[1] - k * end_point[0]
                    else:
                        start_point[0] = 1
                        start_point[1] = end_point[1] + k - k * end_point[0]
                if i == len(lines_to_plot) - 1:
                    if end_point[0] < start_point[0]:
                        end_point[0] = 0
                        end_point[1] = start_point[1] - k * start_point[0]
                    else:
                        end_point[0] = 1
                        end_point[1] = start_point[1] + k - k * start_point[0]
                if k == None:
                    if start_point[1] > 1:
                        start_point[1] = 1
                    elif start_point[1] < 0:
                        start_point[1] = 0
                    if end_point[1] > 1:
                        end_point[1] = 1
                    elif end_point[1] < 0:
                        end_point[1] = 0
                else:
                    if start_point[1] > 1:
                        start_point[1] = 1
                        start_point[0] = (1 - end_point[1]) / k + end_point[0]
                    elif start_point[1] < 0:
                        start_point[1] = 0
                        start_point[0] = (0 - end_point[1]) / k + end_point[0]
                    if end_point[1] > 1:
                        end_point[1] = 1
                        end_point[0] = (1 - start_point[1]) / k + start_point[0]
                    elif end_point[1] < 0:
                        end_point[1] = 0
                        end_point[0] = (0 - start_point[1]) / k + start_point[0]
                lines_orignal.append((start_point, end_point))
            kp = []
            for i in range(len(lines_orignal)):
                if i == 0:
                    kp.append(lines_orignal[i][0])
                if i == len(lines_orignal) - 1:
                    kp.append(lines_orignal[i][1])
                else:
                    kp.append([(lines_orignal[i][1][0] + lines_orignal[i + 1][0][0]) / 2, \
                               (lines_orignal[i][1][1] + lines_orignal[i + 1][0][1]) / 2])
            return kp

        points = _getBoundaryPoints()
        _points = _sortPoints(points)
        # color_map = plt.get_cmap('tab20')(np.array([math.floor(i / 5) for i in range(_points.shape[0])]))
        # plt.scatter(_points[:, 0], 0 - _points[:, 1], color=color_map, s=10, marker='+')
        lr = _fit(_points, 0, _points.shape[0] - 1)
        keypoints = _createKeypoints(points, lr)
        return keypoints
