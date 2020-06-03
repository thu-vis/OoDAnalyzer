import numpy as np
import os
import math
import tensorflow as tf
from time import time
import shutil

from scipy.stats import entropy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
from scipy.spatial.distance import cdist
from PIL import Image
from lapjv import lapjv


from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.utils.data_utils import Data
from scripts.utils.log_utils import logger
from scripts.Grid import GridLayout
from scripts.Ensemble import Ensemble

def TPR95(x, y):
    return 0
    t = time()
    x = x / x.max()
    gap = (x.max() - x.min()) / 1e10
    total = 0.0
    flag = 1
    for delta in np.arange(x.min(), x.max(), gap):
        # tpr = np.sum(np.sum(x > delta)) / len(x
        y_pred = (x > delta).astype(int)
        tn, fp, fn, tp = confusion_matrix(y,y_pred).ravel()
        tpr = tp / (tp+fn)
        if tpr < 0.9505:
            return fp / (fp + tn)
    print("TPR95:", time() - t)

def DetectionError(x, y):
    return 0
    t = time()
    x = x / x.max()
    gap = (x.max() - x.min()) / 1e10
    total = 0.0
    for delta in np.arange(x.min(), x.max(), gap):
        # tpr = np.sum(np.sum(x > delta)) / len(x
        y_pred = (x > delta).astype(int)
        tn, fp, fn, tp = confusion_matrix(y,y_pred).ravel()
        tpr = tp / (tp+fn)
        if tpr < 0.9505:
            return (sum(y_pred!=y) / len(y))
    print("DetectionError:", time() - t)

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



class Comparison(object):
    def __init__(self, dataname):
        config.data_root = "H:/backup"
        self.dataname = dataname
        self.data = Data(dataname)
        self.test_idx = self.data.test_idx
        self.test_redundant_idx = self.data.test_redundant_idx
        self.data_dir = r"H:\backup\Lipsticks"
        self.feature_dir = os.path.join(self.data_dir, "feature")
        self.feature_dir_name = [
            "weights.20-0.9516.h5",
            "weights.10-0.9562.h5",
            "weights.20-0.9551.h5",
            "weights.20-0.9552.h5",
            "weights.10-0.9513.h5",
        ]
        self._load_data()

    def _load_data(self):
        all_pred_y = []
        diff_y = []
        for feature_dir_name in self.feature_dir_name:
            feature_dir = os.path.join(self.feature_dir, feature_dir_name)
            test_data = pickle_load_data(os.path.join(feature_dir, "normal_test.pkl"))
            X = pickle_load_data(os.path.join(feature_dir, "X.pkl"))
            y = self.data.y
            test_X = X[np.array(self.test_idx), :]
            test_y = y[np.array(self.test_idx)]
            pred_y = np.zeros(len(test_y))
            filename = test_data["filename"]
            all_pred_y_in_test = test_data["features"][1]
            for idx, name in enumerate(filename):
                name = name.replace("\\", "/")
                cls, img_name = name.split("/")
                img_id, _ = img_name.split(".")
                img_id = int(img_id)
                if img_id in self.test_idx:
                    pred_y[self.test_idx.index(img_id)] = all_pred_y_in_test[idx]
            hard_pred_y = (pred_y > 0.5).astype(int)
            print("score:", sum(hard_pred_y==test_y)/len(test_y))
            all_pred_y.append(pred_y)
            diff_y.append(pred_y)
        _all_pred_y = np.array(all_pred_y).mean(axis=0)
        all_pred_y = np.zeros((_all_pred_y.shape[0], 2))
        all_pred_y[:,0] = 1 - _all_pred_y
        all_pred_y[:,1] = _all_pred_y
        ent = entropy(all_pred_y.transpose())
        all_idx = np.zeros(max(self.test_idx) + 100)
        all_idx[np.array(self.test_idx)] = np.array(range(len(self.test_idx)))
        bias_idx = all_idx[np.array(self.test_redundant_idx)].astype(int)
        normal_idx = [i for i in range(len(self.test_idx)) if i not in bias_idx]

        matrix = np.zeros((len(self.feature_dir), len(self.feature_dir)))
        for a, a_y in enumerate(diff_y):
            for b, b_y in enumerate(diff_y):
                a_hard = (a_y > 0.5).astype(int)
                b_hard = (b_y > 0.5).astype(int)
                s = sum(a_hard == b_hard)
                matrix[a, b] = s
        print(matrix)

        idx = np.array(range(18039))
        np.random.seed(123)
        np.random.shuffle(idx)
        normal_idx = np.array(normal_idx)[idx[:2000]].tolist()
        all_idx = np.array(normal_idx[:] + bias_idx.tolist())
        print(ent[np.array(normal_idx)].mean(), ent[np.array(bias_idx)].mean())
        y = np.zeros(len(self.test_idx)).astype(int)
        y[bias_idx] = 1
        ent = ent[all_idx]; y = y[all_idx]
        metrics(ent, y)

        # self.entropy_analysis(ent[np.array(y == 0)], ent[np.array(y == 1)])

        our_en = self.data.entropy[np.array(self.test_idx)]
        our_en = our_en[all_idx]

        # self.entropy_analysis(our_en[np.array(y == 0)], our_en[np.array(y == 1)])
        # print(our_en[np.array(normal_idx)].mean(), our_en[np.array(bias_idx)].mean())
        metrics(our_en, y)

    def entropy_analysis(self, normal_entropy, test_redundant_entropy):
        ax = plt.subplot(121)
        ax.hist(normal_entropy, 20)
        ax.set_title("IoD entropy histogram ({})".format(self.dataname))
        ax.set_ylabel("count")
        ax.set_xlabel("entropy")
        ax = plt.subplot(122)
        ax.set_title("OoD entropy histogram ({})".format(self.dataname))
        ax.set_ylabel("count")
        ax.set_xlabel("entropy")
        ax.hist(test_redundant_entropy, 20)
        plt.show()


if __name__ == '__main__':
    c = Comparison(config.lipsticks)
