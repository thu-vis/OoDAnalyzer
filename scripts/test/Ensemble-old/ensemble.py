import numpy as np
import os
import math
import tensorflow as tf
from time import time
import shutil

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
from scripts.utils.log_utils import logger
from scripts.Grid import GridLayout

class Ensemble(object):
    def __init__(self, dataname):
        self.dataname = dataname
        self.data = Data(dataname)
        self.train_idx = self.data.train_idx
        self.valid_idx = self.data.valid_idx
        self.test_idx = self.data.test_idx
        self.cnn_features_dir_name = [
            # "weights.100-0.5465.h5",
            "fine-tune",
            # "inceptionresnet_imagenet",
            # "inceptionv3_imagenet",
            # "mobilenet_imagenet",
            # "resnet50_imagenet",
            # "vgg_imagenet",
            # "xception_imagenet",
            # "sift-200",
            # "HOG",
            # "HOG-kmeans-200",
            # "superpixel-500",
            # "LBP",
            # "LBP-hist",
            # "sift-500",
            # "sift-1000"
        ]
        if self.cnn_features_dir_name[0][:7] == "weights" and dataname == config.dog_cat:
            self.cnn_features_dir_name[0] = "weights.20-0.9922.h5"

    def train_svm(self):
        for weight_name in self.cnn_features_dir_name:
            feature_dir = os.path.join(config.data_root, self.dataname,
                                       "feature", weight_name)
            X = pickle_load_data(os.path.join(feature_dir, "X.pkl"))
            y = self.data.y
            train_X = X[np.array(self.train_idx),:]
            train_y = y[np.array(self.train_idx)]
            test_X = X[np.array(self.test_idx), :]
            test_y = y[np.array(self.test_idx)]
            if weight_name == "sift-no":
                logger.warn("using rbf kernel")
                clf = SVC(kernel="rbf", verbose=1)
            else:
                clf = SVC(kernel="linear", C=1, verbose=1)
                logger.warn("using linear kernel")
            clf.fit(train_X, train_y)
            pickle_save_data(os.path.join(feature_dir, "svm_model.pkl"), clf)
            train_score = clf.score(train_X, train_y)
            test_score = clf.score(test_X, test_y)
            print("\n training acc: {}, test acc: {}."
                  .format(train_score, test_score))

    def debug_sift(self):
        weight_name = "HOG-kmeans-200"
        # weight_name = "weights.20-0.9922.h5"
        feature_dir = os.path.join(config.data_root, self.dataname,
                                   "feature", weight_name)
        X = pickle_load_data(os.path.join(feature_dir, "X.pkl"))
        clf = pickle_load_data(os.path.join(feature_dir, "svm_model.pkl"))
        y = self.data.y
        train_X = X[np.array(self.train_idx), :]
        logger.info(train_X.shape)
        train_y = y[np.array(self.train_idx)]
        test_X = X[np.array(self.test_idx), :]
        test_y = y[np.array(self.test_idx)]
        test_redundant_X = X[np.array(self.data.test_redundant_idx), :]
        test_redundant_y = y[np.array(self.data.test_redundant_idx)]
        train_score = clf.score(train_X, train_y)
        test_score = clf.score(test_X, test_y)
        test_redundant_score = clf.score(test_redundant_X, test_redundant_y)
        test_bias_score = (test_score * len(test_y) -
            test_redundant_score * len(test_redundant_y)) / \
            (len(test_y) - len(test_redundant_y))
        logger.info("train score: {}, test score: {}, test redundant score({}): {}, test bias score({}): {}"
            .format(train_score, test_score, len(test_redundant_y), test_redundant_score, len(test_y) - len(test_redundant_y), test_bias_score))
        w = clf.coef_
        w = np.array(w)
        b = clf.intercept_[0]
        d_train = np.dot(w, train_X.transpose()).reshape(-1) + b
        d_train = abs(d_train)
        d_test = np.dot(w, test_X.transpose()).reshape(-1) + b
        d_test = abs(d_test)
        valided_X = np.concatenate((train_X, test_X), axis=0)
        if not os.path.exists(os.path.join(feature_dir, "embed_X.pkl")):
            logger.info("calculating tsne now...")
            tsne = TSNE(n_components=2, random_state=123)
            X_embeddings = tsne.fit_transform(valided_X)
            embed_X = np.zeros((X.shape[0], 2))
            embed_X[np.array(self.train_idx + self.test_idx), :] = X_embeddings
            pickle_save_data(os.path.join(feature_dir, "embed_X.pkl"), embed_X)
            X_train_embeddings = X_embeddings[:train_X.shape[0], :]
            X_test_embeddings = X_embeddings[train_X.shape[0]:, :]
            logger.info("tsne calculation finished.")
        else:
            logger.warn("embed X buffer exists, loading buffer now.")
            embed_X = pickle_load_data(os.path.join(feature_dir, "embed_X.pkl"))
            X_train_embeddings = embed_X[np.array(self.train_idx), :]
            X_test_embeddings = embed_X[np.array(self.test_idx), :]
        train_num = X_train_embeddings.shape[0]
        support_vectors_index = np.array(clf.support_)
        no_support_vectors_index = np.array([i for i in range(train_num) if i not in clf.support_])
        y_train = np.array(train_y).astype(int) * 2 + 1
        y_test = np.array(test_y).astype(int) * 2 + 1
        y_train[support_vectors_index] = y_train[support_vectors_index] - 1
        color_map = plt.get_cmap("tab20")(np.array(y_train.tolist() + y_test.tolist()))
        color_map_train = color_map[:train_X.shape[0], :]
        color_map_test = color_map[train_X.shape[0]:, :]

        interested_train_idx = np.array(self.train_idx)[X_train_embeddings[:,0] < -30]

        for i in interested_train_idx:
            src = os.path.join(config.data_root, self.dataname,
                               "images", str(i) + ".jpg")
            target = os.path.join(config.data_root, self.dataname,
                                  "tmp", str(i) + ".jpg")
            shutil.copy(src, target)

    def vis(self):
        for weight_name in self.cnn_features_dir_name:
            feature_dir = os.path.join(config.data_root, self.dataname,
                                       "feature", weight_name)
            X = pickle_load_data(os.path.join(feature_dir, "X.pkl"))
            clf = pickle_load_data(os.path.join(feature_dir, "svm_model.pkl"))
            y = self.data.y
            train_X = X[np.array(self.train_idx),:]
            train_y = y[np.array(self.train_idx)]
            test_X = X[np.array(self.test_idx),:]
            test_y = y[np.array(self.test_idx)]
            w = clf.coef_
            w = np.array(w)
            b = clf.intercept_[0]
            d_train = np.dot(w, train_X.transpose()).reshape(-1) + b
            d_train = abs(d_train)
            d_test = np.dot(w, test_X.transpose()).reshape(-1) + b
            d_test = abs(d_test)
            valided_X = np.concatenate((train_X, test_X), axis=0)
            if not os.path.exists(os.path.join(feature_dir, "embed_X.pkl")):
                logger.info("calculating tsne now...")
                tsne = TSNE(n_components=2, random_state=123)
                X_embeddings = tsne.fit_transform(valided_X)
                embed_X = np.zeros((X.shape[0], 2))
                embed_X[np.array(self.train_idx + self.test_idx), :] = X_embeddings
                pickle_save_data(os.path.join(feature_dir, "embed_X.pkl"), embed_X)
                X_train_embeddings = X_embeddings[:train_X.shape[0], :]
                X_test_embeddings = X_embeddings[train_X.shape[0]:, :]
                logger.info("tsne calculation finished.")
            else:
                logger.warn("embed X buffer exists, loading buffer now.")
                embed_X = pickle_load_data(os.path.join(feature_dir, "embed_X.pkl"))
                X_train_embeddings = embed_X[np.array(self.train_idx),:]
                X_test_embeddings = embed_X[np.array(self.test_idx),:]
            train_num = X_train_embeddings.shape[0]
            support_vectors_index = np.array(clf.support_)
            no_support_vectors_index = np.array([i for i in range(train_num) if i not in clf.support_])
            y_train = np.array(train_y).astype(int) * 2 + 1
            y_test = np.array(test_y).astype(int) * 2 + 1
            y_train[support_vectors_index] = y_train[support_vectors_index] - 1
            color_map = plt.get_cmap("tab20")(np.array(y_train.tolist() + y_test.tolist()))
            color_map_train = color_map[:train_X.shape[0], :]
            color_map_test = color_map[train_X.shape[0]:, :]

            # plt.figure(figsize=(400, 200))
            ax_train = plt.subplot(121)
            ax_train.scatter(X_train_embeddings[no_support_vectors_index, 0],
                             0 - X_train_embeddings[no_support_vectors_index, 1],
                             s=8,
                             marker="o",
                             c=color_map_train[no_support_vectors_index, :],
                             alpha=0.7)
            ax_train.scatter(X_train_embeddings[support_vectors_index, 0],
                             0 - X_train_embeddings[support_vectors_index, 1],
                             s=20,
                             marker="x",
                             c=color_map_train[support_vectors_index, :])
            ax = plt.subplot(122)
            ax.scatter(X_test_embeddings[:,0], 0 - X_test_embeddings[:, 1],
                       s=8,
                       marker="o",
                       c=color_map_test,
                       alpha=0.7)
            # plt.savefig(os.path.join(feature_dir, "tsne_result.jpg"))
            fig = plt.gcf()
            fig.set_size_inches(21.5, 10.5)
            fig.savefig(os.path.join(feature_dir, "tsne_result.jpg"))

if __name__ == '__main__':
    e = Ensemble(config.animals)
    e.train_svm()
    e.vis()
    # e.debug_sift()