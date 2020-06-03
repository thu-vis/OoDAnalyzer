import numpy as np
import os
import math
import tensorflow as tf
from time import time

from scipy.stats import entropy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
# import joblib

from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc

from scipy.spatial.distance import cdist
from PIL import Image
from lapjv import lapjv

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.utils.data_utils import Data
from scripts.utils.log_utils import logger
from scripts.Grid import GridLayout
def TPR95(x, y):
    return 0
    x = x / x.max()
    gap = (x.max() - x.min()) / 100000
    total = 0.0
    flag = 1
    for delta in np.arange(x.min(), x.max(), gap):
        # tpr = np.sum(np.sum(x > delta)) / len(x
        y_pred = (x > delta).astype(int)
        tn, fp, fn, tp = confusion_matrix(y,y_pred).ravel()
        tpr = tp / (tp+fn)
        if tpr < 0.9505:
            print("tpr", tpr)
            return fp / (fp + tn)

def DetectionError(x, y):
    return 0
    x = x / x.max()
    gap = (x.max() - x.min()) / 100000
    total = 0.0
    for delta in np.arange(x.min(), x.max(), gap):
        # tpr = np.sum(np.sum(x > delta)) / len(x
        y_pred = (x > delta).astype(int)
        tn, fp, fn, tp = confusion_matrix(y,y_pred).ravel()
        tpr = tp / (tp+fn)
        if tpr < 0.9505:
            print("tpr", tpr)
            return (sum(y_pred!=y) / len(y))

def AUROC(x, y):
    x = x / x.max()
    return roc_auc_score(y, x)

def AUPR(x, y):
    x = x / x.max()
    precision, recall, thresholds = precision_recall_curve(y, x)
    area = auc(recall, precision)
    return area

def TOP_K(x, y, k = 200):
    x = x / x.max()
    idx = x.argsort()[::-1][:k]
    return sum(y[idx] == 1) / k

def metrics(x, y):
    tpr95 = TPR95(x, y)
    detection_error = DetectionError(x, y)
    auroc = AUROC(x, y)
    aupr = AUPR(x, y)
    top_10 = TOP_K(x,y, k=10)
    top_50 = TOP_K(x,y, k=50)
    top_100 = TOP_K(x,y, k=100)
    top_200 = TOP_K(x,y, k=200)
    print("FPR at 95%TPR\tDetection Error\tAUROC\tAUPR\ttop_10_prec\ttop_50_prec\ttop_100_prec\ttop_200_prec")
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
          .format(tpr95, detection_error, auroc, aupr, top_10, top_50, top_100, top_200))


class Ensemble(object):

    def __init__(self, dataname):
        config.data_root = "H:/backup"
        self.dataname = dataname
        logger.info("using dataset {} now".format(self.dataname))
        self.data = Data(dataname)
        self.train_idx = self.data.train_idx
        self.valid_idx = self.data.valid_idx
        self.test_idx = self.data.test_idx
        self.test_redundant_idx = self.data.test_redundant_idx
        self.all_entropy = None
        self.cnn_features_dir_name = [
            # "weights.20-0.5167.h5",
            "weights.20-0.5199.h5",
            "weights.20-0.5203.h5",
            "weights.20-0.5205.h5",
            "weights.20-0.5237.h5",
            "weights.100-0.5465.h5",
            # "inceptionresnet_imagenet",
            # "inceptionv3_imagenet",
            # "mobilenet_imagenet",
            # "resnet50_imagenet",
            # # "vgg_imagenet",
            # # "xception_imagenet",
            # "sift-200",
            # # "HOG-kmeans-200",
            # # "LBP-hist",
            # "superpixel",
            # "orb-200",
            # "brief-200"
        ]

    def grid_search_logistic(self):
        for weight_name in self.cnn_features_dir_name:
            feature_dir = os.path.join(config.data_root, self.dataname,
                                           "feature", weight_name)
            X = pickle_load_data(os.path.join(feature_dir, "X.pkl"))
            print(X.shape)
            # clf = pickle_load_data(os.path.join(feature_dir, model_name + "_model.pkl"))
            y = self.data.y
            train_X = X[np.array(self.train_idx), :]
            train_y = y[np.array(self.train_idx)]
            test_X = X[np.array(self.test_idx), :]
            test_y = y[np.array(self.test_idx)]
            grid_search_dir = os.path.join(feature_dir, "logistic")
            check_dir(grid_search_dir)
            for C in [1e0]:
                print("C:", C)
                clf = LogisticRegression(C=C)
                clf.fit(train_X, train_y)
                model_path = os.path.join(os.path.join(grid_search_dir), str(C))
                pickle_save_data(model_path, clf)

    def ensemble_result_logistic(self, subset):
        test_predys = np.zeros((len(self.test_idx), 2))
        hard_predys = np.zeros((len(self.test_idx), 2))
        for weight_name in self.cnn_features_dir_name:
            feature_dir = os.path.join(config.data_root, self.dataname,
                                           "feature", weight_name)
            X = pickle_load_data(os.path.join(feature_dir, "X.pkl"))
            print(X.shape)
            # clf = pickle_load_data(os.path.join(feature_dir, model_name + "_model.pkl"))
            y = self.data.y
            train_X = X[np.array(self.train_idx), :]
            train_y = y[np.array(self.train_idx)]
            test_X = X[np.array(self.test_idx), :]
            test_y = y[np.array(self.test_idx)]
            grid_search_dir = os.path.join(feature_dir, "logistic")
            # for C in [1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5]:
            # for C in [1e-5, 1e0, 1e5]:
            for C in subset:
                print("C:", C)
                model_path = os.path.join(os.path.join(grid_search_dir), str(C))
                clf = pickle_load_data(model_path)
                test_predy = clf.predict_proba(test_X)
                hard_test_predy = clf.predict(test_X)
                test_predys = test_predys + test_predy
                hard_predys[:, 1] = hard_predys[:, 1] + hard_test_predy
                hard_predys[:, 0] = hard_predys[:, 0] + 1 - hard_test_predy
        test_entropy = entropy(hard_predys.T)
        all_entropy = np.zeros(self.data.X.shape[0])
        all_entropy[np.array(self.test_idx)] = test_entropy

        all_idx = np.zeros(max(self.test_idx) + 100)
        all_idx[np.array(self.test_idx)] = np.array(range(len(self.test_idx)))
        bias_idx = all_idx[np.array(self.test_redundant_idx)].astype(int)
        normal_idx = [i for i in range(len(self.test_idx)) if i not in bias_idx]
        bias_idx = bias_idx
        all_idx = np.array(normal_idx + bias_idx.tolist())
        y = np.zeros(len(self.test_idx)).astype(int)
        y[bias_idx] = 1
        # our_en = self.data.entropy[np.array(self.test_idx)]
        our_en = all_entropy[np.array(self.test_idx)]
        our_en = our_en[all_idx]
        metrics(our_en, y)

    def grid_search_rf(self):
        for weight_name in self.cnn_features_dir_name:
            feature_dir = os.path.join(config.data_root, self.dataname,
                                           "feature", weight_name)
            X = pickle_load_data(os.path.join(feature_dir, "X.pkl"))
            print(X.shape)
            # clf = pickle_load_data(os.path.join(feature_dir, model_name + "_model.pkl"))
            y = self.data.y
            train_X = X[np.array(self.train_idx), :]
            train_y = y[np.array(self.train_idx)]
            test_X = X[np.array(self.test_idx), :]
            test_y = y[np.array(self.test_idx)]
            grid_search_dir = os.path.join(feature_dir, "rf")
            check_dir(grid_search_dir)
            for max_depth in [100, 300, 500, 700, 1000]:
                for n_estimator in [10, 100, 200, 500]:
                    print("max_depth_{}, n_estimator_{}".format(max_depth, n_estimator))
                    model_path = os.path.join(os.path.join(grid_search_dir),
                                              "max_depth_{}, n_estimator_{}".format(max_depth, n_estimator))
                    if os.path.exists(model_path):
                        continue
                    clf = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth)
                    clf.fit(train_X, train_y)
                    pickle_save_data(model_path, clf)

    def grid_search_svm(self):
        for weight_name in self.cnn_features_dir_name:
            feature_dir = os.path.join(config.data_root, self.dataname,
                                           "feature", weight_name)
            X = pickle_load_data(os.path.join(feature_dir, "X.pkl"))
            print(X.shape)
            # clf = pickle_load_data(os.path.join(feature_dir, model_name + "_model.pkl"))
            y = self.data.y
            train_X = X[np.array(self.train_idx), :]
            train_y = y[np.array(self.train_idx)]
            test_X = X[np.array(self.test_idx), :]
            test_y = y[np.array(self.test_idx)]
            grid_search_dir = os.path.join(feature_dir, "svm")
            check_dir(grid_search_dir)
            for C in [2e-7, 2e-5, 2e-3, 2e-1, 2e1, 2e3, 2e5, 2e7]:
                        for gamma in [2e-5, 2e-3, 2e-1, 2e1, 2e3]:
                            clf = SVC(kernel="rbf", C=C, gamma=gamma)
                            model_name = "kernel_{}_C_{}_gamma{}".format("rbf", C, gamma)
                            if os.path.exists(model_name):
                                continue
                            clf.fit(train_X, train_y)
                            model_path = os.path.join(grid_search_dir, model_name)
                            pickle_save_data(model_path, clf)


if __name__ == '__main__':
    d = Ensemble(config.svhn)
    d.grid_search_logistic()
    # d.grid_search_svm()
    # d.ensemble_result_logistic([1e0])
    # d.ensemble_result_logistic([1e0, 1e1])
    d.ensemble_result_logistic([1e-1, 1e0, 1e1])
    # d.grid_search_rf()
    # d.grid_search_svm()
    # exit()

