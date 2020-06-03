import pickle
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, roc_auc_score, \
    precision_recall_curve, auc, roc_curve

# Pickle loading and saving
def pickle_save_data(filename, data):
    try:
        pickle.dump(data, open(filename, "wb"))
    except Exception as e:
        print(e, end=" ")
        print("So we use the highest protocol.")
        pickle.dump(data, open(filename, "wb"), protocol=4)
    return True


def pickle_load_data(filename):
    try:
        mat = pickle.load(open(filename, "rb"))
    except Exception as e:
        mat = pickle.load(open(filename, "rb"))
    return mat


# metrics
def accuracy(y_true, y_pred, weights=None):
    score = (y_true == y_pred)
    return np.average(score, weights=weights)


# json loading and saving
def json_save_data(filename, data):
    open(filename, "w").write(json.dumps(data))
    return True


def json_load_data(filename):
    return json.load(open(filename, "r"))


# directory
def check_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return True


# normalization
def unit_norm_for_each_col(X):
    X -= X.min(axis=0)
    X /= X.max(axis=0)
    return X


def TPR95(x, y):
    return 0
    # x = x / x.max()
    # gap = (x.max() - x.min()) / 10000000
    # total = 0.0
    # flag = 1
    # for delta in np.arange(x.min(), x.max(), gap):
    #     # tpr = np.sum(np.sum(x > delta)) / len(x
    #     y_pred = (x > delta).astype(int)
    #     tn, fp, fn, tp = confusion_matrix(y,y_pred).ravel()
    #     tpr = tp / (tp+fn)
    #     if tpr < 0.9505:
    #         return fp / (fp + tn)

def DetectionError(x, y):
    return 0
    # x = x / x.max()
    # gap = (x.max() - x.min()) / 10000000
    # total = 0.0
    # for delta in np.arange(x.min(), x.max(), gap):
    #     # tpr = np.sum(np.sum(x > delta)) / len(x
    #     y_pred = (x > delta).astype(int)
    #     tn, fp, fn, tp = confusion_matrix(y,y_pred).ravel()
    #     tpr = tp / (tp+fn)
    #     if tpr < 0.9505:
    #         return (sum(y_pred!=y) / len(y))

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



def OoD_metrics(x, y):
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
