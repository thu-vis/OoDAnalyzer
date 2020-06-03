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
from scripts.utils.helper_utils import check_dir
from scripts.utils.data_utils import Data


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

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def gaussian_trans(d, d0=0, sigma=1):
    r = -(d - d0)**2
    r = r / 2.0 / sigma**2
    r = np.exp(r)
    return r

class SVM_DecisionBoundary(object):
    def __init__(self, dataname):
        self.dataname = dataname
        self.model_dir_name = "SVM-" + self.dataname
        self.model_dir = os.path.join(config.model_root,
                                      self.model_dir_name)
        check_dir(self.model_dir)
        self.data = Data(self.dataname)
        self.X_train, self.y_train, self.X_test, self.y_test = self.data.get_data("all")

        if self.dataname == config.mnist:
            self.X_train = self.X_train[:5000,:]
            self.y_train = self.y_train[:5000]
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

    def early_stopping_training(self, kernel="linear", C=1, gamma="auto"):
        kernel = kernel
        for max_iter in [1, 2, 5, 20, 100]:
            clf = SVC(kernel=kernel, C=C, gamma=gamma, verbose=1, max_iter=max_iter)
            clf.fit(self.X_train, self.y_train)
            train_score = clf.score(self.X_train, self.y_train)
            test_score = clf.score(self.X_test, self.y_test)
            print("\n max_iter:{}, training acc: {}, test acc:{}".format(max_iter, train_score, test_score))

    def margin_vis(self):
        if self.clf is None:
            raise ValueError("you need train the svm if you want to get the support vector.")
        support_vectors_index = np.array(self.clf.support_)
        if self.kernel != "linear":
            raise ValueError("not support non-linear kernel now")
        w = self.clf.coef_
        w = np.array(w)
        b = self.clf.intercept_[0]
        d_train = np.dot(w, self.X_train.transpose()).reshape(-1) + b
        d_train = abs(d_train)
        d_test = np.dot(w, self.X_test.transpose()).reshape(-1) + b
        d_test = abs(d_test)
        X = np.concatenate((self.X_train, self.X_test), axis=0)
        d = np.array(d_train.tolist() + d_test.tolist())
        d = d / d.max()
        ####################
        if True:
            var = np.var(d)**0.5
            d = gaussian_trans(d, d0=0, sigma=1)
            d = d - d.min()
            d = d / d.max()
            color_map = plt.get_cmap("Greens")(d)
        else:
            color_map = plt.get_cmap("Reds")(d)
        ###################
        if hasattr(self.data, "X_embed_train"):
            print("using pre-computated embeddings")
            X_train_embeddings = self.data.X_embed_train[:]
            X_test_embeddings = self.data.X_embed_test[:]
        else:
            tsne = TSNE(n_components=2, random_state=123)
            X_embeddings = tsne.fit_transform(X)
            X_train_embeddings = X_embeddings[:self.X_train.shape[0], :]
            X_test_embeddings = X_embeddings[self.X_train.shape[0]:, :]
        color_map_train = color_map[:self.X_train.shape[0], :]
        color_map_test = color_map[self.X_train.shape[0]:, :]
        ax_train = plt.subplot(121)
        ax_train.scatter(X_train_embeddings[:,0], X_train_embeddings[:,1], s=8, c=color_map_train)
        # ax_train.title("train")
        # plt.colorbar(as_train)
        ax_test = plt.subplot(122)
        ax_test.scatter(X_test_embeddings[:,0], X_test_embeddings[:,1], s=8, c=color_map_test)
        # ax_test.title("test")
        # plt.colorbar(as_test)
        plt.show()

    def cross_validation(self):
        tuned_parameters=[
            {
                "kernel":["rbf"],
                "gamma": [2**-15, 2**-10, 2**-5, 2**0, 2**3],
                "C": [2**-5, 2**0, 2**5, 2**10, 2**15],
                "max_iter": [10000]
            }
        ]
        scores = ["accuracy"]
        clf = GridSearchCV(SVC(), tuned_parameters, cv=3, scoring="accuracy")
        clf.fit(self.X_train, self.y_train)
        print(clf.best_params_)

    def get_support_vector(self):
        if self.clf is None:
            raise ValueError("you need train the svm if you want to get the support vector.")
        support_vectors_index = np.array(self.clf.support_)
        # support_vectors = self.X_train[support_vectors_index,:]

        support_vectors_image_id = \
            np.array(self.data.selected_training_idx)[support_vectors_index]

        print(support_vectors_index)
        image_dir = os.path.join(config.data_root,
                                 config.mnist_vgg,
                                 "train_all_no_by_categories")
        support_vector_target = self.y_train[support_vectors_index]
        class_1 = []
        class_2 = []
        for idx, i in enumerate(support_vector_target):
            if i == 0:
                class_1.append(support_vectors_image_id[idx])
            else:
                class_2.append(support_vectors_image_id[idx])
        max_len = max(len(class_1), len(class_2))
        ims = [Image.open(os.path.join(image_dir, str(i) + ".jpg")) for i in class_1]
        width, height = ims[0].size
        result_1 = Image.new(ims[0].mode, (width, height * len(ims)))
        for i, im in enumerate(ims):
            result_1.paste(im, box=(0, i * height))
        result_1.save(os.path.join(config.data_root,
                                   self.dataname,
                                   self.kernel + "_1" +".jpg"))


        ims = [Image.open(os.path.join(image_dir, str(i) + ".jpg")) for i in class_2]
        width, height = ims[0].size
        result_2 = Image.new(ims[0].mode, (width, height * len(ims)))
        for i, im in enumerate(ims):
            result_2.paste(im, box=(0, i * height))
        result_2.save(os.path.join(config.data_root,
                                   self.dataname,
                                   self.kernel + "_2" +".jpg"))
        return

    def get_normal_vector(self):
        image_dir = os.path.join(config.data_root,
                                 config.mnist_vgg,
                                 "train_all_no_by_categories")
        support_vectors_image_id = self.data.selected_training_idx[:100]
        support_vector_target = self.y_train[:100]
        class_1 = []
        class_2 = []
        for idx, i in enumerate(support_vector_target):
            if i == 0:
                class_1.append(support_vectors_image_id[idx])
            else:
                class_2.append(support_vectors_image_id[idx])
        max_len = max(len(class_1), len(class_2))
        ims = [Image.open(os.path.join(image_dir, str(i) + ".jpg")) for i in class_1]
        width, height = ims[0].size
        result_1 = Image.new(ims[0].mode, (width, height * len(ims)))
        for i, im in enumerate(ims):
            result_1.paste(im, box=(0, i * height))
        result_1.save(os.path.join(config.data_root,
                                   self.dataname,
                                   "1_normal" + ".jpg"))

        ims = [Image.open(os.path.join(image_dir, str(i) + ".jpg")) for i in class_2]
        width, height = ims[0].size
        result_2 = Image.new(ims[0].mode, (width, height * len(ims)))
        for i, im in enumerate(ims):
            result_2.paste(im, box=(0, i * height))
        result_2.save(os.path.join(config.data_root,
                                   self.dataname,
                                   "2_normal" + ".jpg"))
        return

    def vis_support_vector_3d(self):
        if self.clf is None:
            raise ValueError("you need train the svm if you want to get the support vector.")
        support_vectors_index = np.array(self.clf.support_)
        no_support_vectors_index = np.array([i for i in range(self.train_num) if i not in self.clf.support_])
        color_map = plt.get_cmap("tab20")(self.y_train.astype(int))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.X_train[support_vectors_index, 0],
                   self.X_train[support_vectors_index, 1],
                   self.X_train[support_vectors_index, 2],
                   s=3,
                   marker="+",
                   c=color_map[support_vectors_index,:])
        ax.scatter(self.X_train[no_support_vectors_index, 0],
                   self.X_train[no_support_vectors_index, 1],
                   self.X_train[no_support_vectors_index, 2],
                   s=3,
                   marker="o",
                   c=color_map[no_support_vectors_index,:])
        plt.show()

    def vis_grid(self):
        if self.clf is None:
            raise ValueError("you need train the svm if you want to get the support vector.")
        support_vectors_index = np.array(self.clf.support_)
        no_support_vectors_index = np.array([i for i in range(self.train_num) if i not in self.clf.support_])
        X = np.concatenate((self.X_train, self.X_test), axis=0)
        if hasattr(self.data, "X_embed_train"):
            print("using pre-computated embeddings")
            X_train_embeddings = self.data.X_embed_train
            X_test_embeddings = self.data.X_embed_test
            print("X_train_embeddings shape:{}, X_test_embeddings shape:{}"
                  .format(X_train_embeddings.shape, X_test_embeddings.shape))
        else:
            print("calculating tsne now...")
            tsne = TSNE(n_components=2, random_state=123)
            X_embeddings = tsne.fit_transform(X)
            X_train_embeddings = X_embeddings[:self.X_train.shape[0], :]
            X_test_embeddings = X_embeddings[self.X_train.shape[0]:, :]
        y_train = self.y_train.astype(int) * 2 + 1
        y_test = self.y_test.astype(int) * 2 + 1
        y_train[support_vectors_index] = y_train[support_vectors_index] - 1
        color_map = plt.get_cmap("tab20")(np.array(y_train.tolist() + y_test.tolist()))
        color_map_train = color_map[:self.X_train.shape[0], :]
        color_map_test = color_map[self.X_train.shape[0]:, :]

        N = int(np.sqrt(X_train_embeddings.shape[0]))
        X_train_embeddings = X_train_embeddings[:N * N]
        X_train_embeddings -= X_train_embeddings.min(axis=0)
        X_train_embeddings /= X_train_embeddings.max(axis=0)

        grid = np.dstack(np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))).reshape(-1,2)

        t = time()
        cost_matrix = cdist(grid, X_train_embeddings, "sqeuclidean")
        cost_matrix = cost_matrix * (100000 / cost_matrix.max())
        row_asses, col_asses, _ = lapjv(cost_matrix)
        print("lapjv time: {}".format(time() - t))
        # X_train_embeddings = grid[col_asses]
        ax_train = plt.subplot(121)
        support_vectors_index = support_vectors_index[support_vectors_index<N*N]
        no_support_vectors_index = no_support_vectors_index[no_support_vectors_index<N*N]
        color_map_train = color_map_train[:N*N]
        ax_train.scatter(X_train_embeddings[no_support_vectors_index, 0],
                         X_train_embeddings[no_support_vectors_index, 1],
                       s=8,
                       marker="o",
                       c=color_map_train[no_support_vectors_index,:],
                       alpha=0.7)
        ax_train.scatter(X_train_embeddings[support_vectors_index, 0],
                         X_train_embeddings[support_vectors_index, 1],
                       s=20,
                       marker="x",
                       c=color_map_train[support_vectors_index,:])
        plt.show()

    def vis_support_vector_jointly(self):
        if self.clf is None:
            raise ValueError("you need train the svm if you want to get the support vector.")
        support_vectors_index = np.array(self.clf.support_)
        no_support_vectors_index = np.array([i for i in range(self.train_num) if i not in self.clf.support_])
        X = np.concatenate((self.X_train, self.X_test), axis=0)
        if hasattr(self.data, "X_embed_train"):
            print("using pre-computated embeddings")
            X_train_embeddings = self.data.X_embed_train
            X_test_embeddings = self.data.X_embed_test
            print("X_train_embeddings shape:{}, X_test_embeddings shape:{}"
                  .format(X_train_embeddings.shape, X_test_embeddings.shape))
        else:
            print("calculating tsne now...")
            tsne = TSNE(n_components=2, random_state=123)
            X_embeddings = tsne.fit_transform(X)
            X_train_embeddings = X_embeddings[:self.X_train.shape[0], :]
            X_test_embeddings = X_embeddings[self.X_train.shape[0]:, :]
        y_train = self.y_train.astype(int) * 2 + 1
        y_test = self.y_test.astype(int) * 2 + 1
        y_train[support_vectors_index] = y_train[support_vectors_index] - 1
        color_map = plt.get_cmap("tab20")(np.array(y_train.tolist() + y_test.tolist()))
        color_map_train = color_map[:self.X_train.shape[0], :]
        color_map_test = color_map[self.X_train.shape[0]:, :]
        ax_train = plt.subplot(121)
        ax_train.scatter(X_train_embeddings[no_support_vectors_index, 0],
                         X_train_embeddings[no_support_vectors_index, 1],
                       s=8,
                       marker="o",
                       c=color_map_train[no_support_vectors_index,:],
                       alpha=0.7)
        ax_train.scatter(X_train_embeddings[support_vectors_index, 0],
                         X_train_embeddings[support_vectors_index, 1],
                       s=20,
                       marker="x",
                       c=color_map_train[support_vectors_index,:])
        ax_test = plt.subplot(122)
        ax_test.scatter(X_test_embeddings[:, 0], X_test_embeddings[:, 1], s=8, c=color_map_test)
        plt.show()

    def vis_support_vector(self, method):
        if self.clf is None:
            raise ValueError("you need train the svm if you want to get the support vector.")
        support_vectors_index = np.array(self.clf.support_)
        no_support_vectors_index = np.array([i for i in range(self.train_num) if i not in self.clf.support_])
        y = self.y_train * 2 + 1
        y[support_vectors_index] = y[support_vectors_index] - 1
        color_map = plt.get_cmap("tab20")(y.astype(int))
        if self.feature_num > 2:
            embeddor = method(n_components=2, random_state=123)
            X_embedding = embeddor.fit_transform(self.X_train)
        else:
            X_embedding = self.X_train
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if (len(no_support_vectors_index) < 1):
            None
        else:
            ax.scatter(X_embedding[no_support_vectors_index, 0],
                       X_embedding[no_support_vectors_index, 1],
                       s=8,
                       marker="o",
                       c=color_map[no_support_vectors_index,:],
                       alpha=0.7)
        ax.scatter(X_embedding[support_vectors_index, 0],
                   X_embedding[support_vectors_index, 1],
                   s=20,
                   marker="x",
                   c=color_map[support_vectors_index,:])

        # X0, X1 = X_embedding[:, 0], X_embedding[:, 1]
        # xx, yy = make_meshgrid(X0, X1)
        # plot_contours(ax, self.clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.4)

        plt.show()

    def vis_test(self,method, save_projection_results):
        if self.clf is None:
            raise ValueError("you need train the svm if you want to get the support vector.")
        support_vectors_index = np.array(self.clf.support_)
        no_support_vectors_index = np.array([i for i in range(self.train_num) if i not in self.clf.support_])
        y = self.y_train * 2 + 1
        y[support_vectors_index] = y[support_vectors_index] - 1
        support_vectors_num = len(support_vectors_index)
        y_num = self.X_test.shape[0]
        X = np.zeros((y_num + support_vectors_num, self.X_test.shape[1]))
        X[:y_num, :] = self.X_test
        X[y_num:, :] = self.X_train[support_vectors_index, :]
        y = np.zeros(y_num + support_vectors_num)
        y[:y_num] = self.y_test * 2 + 1
        y[y_num:] = self.y_train[support_vectors_index] * 2

        embeddor = method(n_components=2, random_state=123)
        X_embedding = embeddor.fit_transform(X)
        # X_embedding = embeddor.transform(self.X_test)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        color_map = plt.get_cmap("tab20")(y.astype(int))
        ax.scatter(X_embedding[:y_num, 0], X_embedding[:y_num, 1],
                   s=8, marker="o", c=color_map[:y_num,:], alpha=0.4)
        ax.scatter(X_embedding[y_num:, 0], X_embedding[y_num:, 1],
                   s=20, marker="x", c=color_map[y_num:,:])

        if save_projection_results:
            np.save(os.path.join(config.data_root,
                                 self.dataname,
                                 "tsne_normal_test_sv.npy"), X_embedding)
            np.save(os.path.join(config.data_root,
                                 self.dataname,
                                 "label_normal_test_sv.npy"), y)

        plt.show()


if __name__ == '__main__':
    s = SVM_DecisionBoundary(config.mnist_3_5_vgg)
    # for i in [0.01, 0.1, 1,10,100,1000]:
    #     s.training(C=i)
    # s.training(kernel="rbf", C=5, gamma=5e-10)
    # s.early_stopping_training(kernel="linear")
    s.training(kernel='linear')
    # for gamma in [1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13]:
    # s.training(kernel="rbf", gamma=1, C=20)
    # s.cross_validation()
    # s.get_support_vector()
    # s.vis_support_vector_jointly()
    s.vis_grid()
    # s.get_normal_vector()

    # s.margin_vis()

    # s.vis_test(TSNE, save_projection_results=True)