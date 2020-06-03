import numpy as np
import networkx as nx
import os
import math
import tensorflow as tf
from time import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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
from scripts.Grid import GridLayout


class GraphTest(object):
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

    def graph_vis(self):
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
                                           np.linspace(0, 1 - 1.0 / N_sqrt_train, N_sqrt_train)))\
                                            .reshape(-1, 2)
        grid_test = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / N_sqrt_test, N_sqrt_test),
                                          np.linspace(0, 1 - 1.0 / N_sqrt_test, N_sqrt_test)))\
                                            .reshape(-1, 2)

        train_original_cost_matrix = cdist(grid_train, for_embed_X_train, "euclidean")
        cost_matrix = train_original_cost_matrix.copy()

        G = nx.Graph()
        for i in range(cost_matrix.shape[0]):
            G.add_node("0" + str(i))
        for i in range(cost_matrix.shape[1]):
            G.add_node("1" + str(i))

        #knn process
        knn_idx = []
        inverse_idx = [[]] * cost_matrix.shape[0]
        k = 50
        for i in range(cost_matrix.shape[1]):
            col = cost_matrix[:,i].copy()
            col_sort = col.argsort()
            knn_idx.append(col_sort[:k].tolist())
            for j in col_sort[:k].tolist():
                inverse_idx[j].append(i)
                G.add_edge("1" + str(i), "0" + str(j))

        for i in range(cost_matrix.shape[0]):
            row = cost_matrix[i,:].copy()
            row_sort = row.argsort()
            for j in row_sort[:5].tolist():
                G.add_edge("1" + str(j), "0" + str(i))

        # mask_left = np.ones(cost_matrix.shape[0]).astype(int) * -1
        # mask_right = np.ones(cost_matrix.shape[1]).astype(int) * -1
        # pending_list = [0]
        # mask_left[0] = 0
        # present_cls = 0
        # while((mask_left==-1).sum() > 0):
        #     id_left = pending_list[0]
        #     pending_list.remove(id_left)
        #     knn = knn_idx[id_left]


        nn = nx.number_connected_components(G)
        print("connected components: {}".format(nn))

        # nx.draw(G)
        # plt.savefig("sample.jpg")
        # plt.show()

if __name__ == '__main__':
    g = GraphTest(config.svhn)
    g.graph_vis()