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
    def __init__(self, dataname, suffix=""):
        self.dataname = dataname
        self.model_dir_name = "SVM-" + self.dataname
        self.model_dir = os.path.join(config.model_root,
                                      self.model_dir_name)
        check_dir(self.model_dir)
        self.data = Data(self.dataname,suffix)
        self.X_train, self.y_train,_ ,_ , self.X_test, self.y_test = self.data.get_data("all")


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
        IoD_idx = [i for i in self.data.test_idx if i not in self.data.test_redundant_idx]
        OoD_idx = self.data.test_redundant_idx
        IoD_idx = np.array(IoD_idx)
        OoD_idx = np.array(OoD_idx)
        self.clf.fit(self.X_train, self.y_train)
        train_score = self.clf.score(self.X_train, self.y_train)
        test_score = self.clf.score(self.X_test, self.y_test)
        IoD_score = self.clf.score(self.data.X[IoD_idx,:], self.data.y[IoD_idx])
        OoD_score = self.clf.score(self.data.X[OoD_idx,:], self.data.y[OoD_idx])
        if kernel == "linear":
            weights = self.clf.coef_
            margin = 1.0 / ((weights**2).sum())**0.5
        else:
            margin = "not defined"
        print("\n training acc: {}, test acc: {}, margin value: {}."
              .format(train_score, test_score, margin))
        print("IoD score: {}, OoD score: {}".format(IoD_score, OoD_score))
        print("IoD len: {}, OoD len: {}".format(len(IoD_idx), len(OoD_idx)))

if __name__ == '__main__':
    a = SVM_DecisionBoundary(dataname=config.dog_cat, suffix="")
    a.training()