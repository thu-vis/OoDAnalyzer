import numpy as np
import os
import warnings

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ...utils.data_utils import Data


def similarity_according_feature(features):
    norm = ((features ** 2).sum(axis=1)) ** 0.5
    norm = np.dot(norm.reshape(-1, 1), norm.reshape(1, -1))
    print("norm shape: %s" % (str(norm.shape)))
    simi = np.dot(features, features.transpose()) / norm
    simi = simi + (1 - simi.max())
    return simi

def get_tsne_from_similarity_matrix(similarity_matrix):
    print("now begin get tsne result from similarity matrix.")
    tsne = TSNE(n_components=2, metric="precomputed",random_state=15)
    dist = 1 - np.array(similarity_matrix)
    if dist.min() > 1e-3:
        warnings.warn("the max similarity of the matrix is larger than 1+1e-3. "
                      "Please check your input if there are some bugs.")
    dist = dist - dist.min()
    plat_coordinate = tsne.fit_transform(dist)
    print("tsne result got!")
    return plat_coordinate

class ProjectionBase(object):
    def __init__(self, dataname):
        self.dataname = dataname
        self.data = Data(dataname)
        self.X_train, self.y_train, self.X_test, self.y_test = self.data.get_data("all")

    def lda(self, show=False):
        X_train = self.X_train
        p = LinearDiscriminantAnalysis(n_components=2)
        y_train = self.y_train.astype(int)
        p.fit(X_train, y_train)
        p_x = p.transform(X_train)
        y = np.zeros(len(self.y_train))
        for idx, i in enumerate(self.y_train):
            y[idx] = i
        color_map = plt.get_cmap("tab10")(y.astype(int))
        try:
            plt.scatter(p_x[:,0], p_x[:,1], s=3, c=color_map)
        except:
            plt.scatter(p_x[:, 0], p_x[:, 0], s=3, c=color_map)
        if show:
            plt.show()


    def pca(self, show=False):
        X_train = self.X_train[:,[1,5,6,7]]
        p = PCA(n_components=2)
        p.fit(X_train)
        p_x = p.transform(X_train)
        y = np.zeros(len(self.y_train))
        for idx, i in enumerate(self.y_train):
            y[idx] = i
        color_map = plt.get_cmap("tab10")(y.astype(int))
        plt.scatter(p_x[:,0], p_x[:,1], s=3, c=color_map)
        if show:
            plt.show()

    def mds(self, show=False):
        X_train = self.X_train
        p = MDS(n_components=2)
        p_x = p.fit_transform(X_train)
        y = np.zeros(len(self.y_train))
        for idx, i in enumerate(self.y_train):
            y[idx] = i
        color_map = plt.get_cmap("tab10")(y.astype(int))
        plt.scatter(p_x[:,0], p_x[:,1], s=3, c=color_map)
        if show:
            plt.show()


    def t_sne(self, show=False):
        X_train = self.X_train[:,[1,5,6,7]]
        simi = similarity_according_feature(X_train)
        tsne = get_tsne_from_similarity_matrix(simi)
        y = np.zeros(len(self.y_train))
        for idx, i in enumerate(self.y_train):
            y[idx] = i
        color_map = plt.get_cmap("tab10")(y.astype(int))
        plt.scatter(tsne[:,0], tsne[:,1], s=3, c=color_map)
        if show:
            plt.show()

        return tsne
