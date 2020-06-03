import numpy as np
import os
import math
# import tensorflow as tf
from time import time

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.svm import SVC
# from sklearn.manifold import TSNE, MDS
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.metrics import classification_report
from scipy.spatial.distance import cdist
from PIL import Image
# from lapjv import lapjv

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.utils.data_utils import Data
from scripts.utils.log_utils import logger
from scripts.Grid import GridLayout

class KNNTest(object):
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

        # self.grid_layout = GridLayout(self.dataname)

    def test_knn(self, embed_method="tsne"):
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
        # knn process
        logger.info('begin')
        k = 50
        hole_num = train_original_cost_matrix.shape[0]
        radish_num = train_original_cost_matrix.shape[1]
        # radish_idx = [[]] * radish_num
        # hole_idx = []
        # hole_ranking = []
        # hole_index_mapping = []
        # for i in range(hole_num):
        #     hole_idx.append([])
        #     hole_ranking.append([])
        #     hole_index_mapping.append({})
        # for i in range(radish_num):
        #     col = cost_matrix[:, i].copy()
        #     col_sort = col.argsort()
        #     radish_idx[i] = col_sort.tolist()
        #     for ranking, j in enumerate(col_sort[:k]):
        #         hole_idx[j].append(i)
        #         hole_ranking[j].append(ranking)
        #     for rankinig, j in enumerate(col_sort):
        #         hole_index_mapping[j][i] = ranking
        # hole_idx_lens = []
        # radish_lens = []
        # for j in range(hole_num):
        #     hole_idx_lens.append(len(hole_idx[j]))
        # num = sum(np.array(hole_idx_lens) > k)
        # logger.info("degree of left side: {}, degree of right size: {}".
        #             format(radish_num * k, sum(hole_idx_lens)))
        # logger.info("nodes with degree larger than k is {}".format(num))
        # hole_idx_sorted_idx = np.array(hole_idx_lens).argsort()[::-1]
        # for count, j in enumerate(hole_idx_sorted_idx):
        #     idxs = hole_idx[j]
        #     ranking = hole_ranking[j]
        #     if len(idxs) < k:
        #         continue
        #     # logger.info("begin")
        #     sorted_idx = cost_matrix[j, np.array(idxs)].argsort()
        #     sorted_idx = np.array(idxs)[sorted_idx]
        #     preserved_idx = sorted_idx[:k].tolist()
        #     # logger.info("remove")
        #     rest = []
        #     for i in sorted_idx[k:]:
        #
        #         hole_idx[added_hole_idx].append(i)
        #     hole_idx[j] = sorted_idx[:k].tolist() + rest
        # hole_connected_idx = np.zeros((hole_num, radish_num))
        hole_connected_idx = [[0 for _ in range(radish_num)] for __ in range(hole_num)]
        radish_sortdist_index = []
        hole_count = [0 for _ in range(hole_num)]
        radish_toCheck = [k for _ in range(hole_num)]
        for i in range(radish_num):
            col_sort = train_original_cost_matrix[:,i].argsort()
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
            distances = np.array([train_original_cost_matrix[hole][r] for r in rs])
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
        cost_matrix = [[train_original_cost_matrix[hole][radish] if hole_connected_idx[hole][radish] else 0 for radish in range(radish_num)] for hole in range(hole_num)]
        cost_matrix = np.array(cost_matrix)
        logger.info('build finished')




if __name__ == '__main__':
    testcase = KNNTest(config.svhn)
    testcase.test_knn()