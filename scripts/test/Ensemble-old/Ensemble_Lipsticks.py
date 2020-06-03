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
        config.data_root = "H:/backup"
        self.dataname = dataname
        logger.info("using dataset {} now".format(self.dataname))
        self.data = Data(dataname)
        self.train_idx = self.data.train_idx
        self.valid_idx = self.data.valid_idx
        self.test_idx = self.data.test_idx
        self.test_redundant_idx = self.data.test_redundant_idx
        self.all_entropy = None
        if dataname == config.rea:
            self.cnn_features_dir_name = [
                # "unet_nopre_mix_3_0.001_unet-wce-dice-bce",
                # "unet_nopre_mix_33_0.001_unet-wce-eldice-bce",
                # "unet_nested_nopre_mix_3_0.001_unet_nested-wce-dice-bce",
                # "unet_nested_nopre_mix_33_0.001_unet_nested-wce-eldice-bce",
                # "unet_nested_dilated_nopre_mix_3_0.001_unet_nested_dilated-wce-dice-bce",
                # "unet_nested_dilated_nopre_mix_33_0.001_unet_nested_dilated-wce-eldice-bce"
                "fine-tune"
            ]
        elif 1:
            self.cnn_features_dir_name = [
                "weights.10-0.9562.h5",
                "inceptionresnet_imagenet",
                "inceptionv3_imagenet",
                "mobilenet_imagenet",
                "resnet50_imagenet",
                "vgg_imagenet",
                "xception_imagenet",
                "sift-200",
                "HOG-200",
                "LBP-hist",
            ]

    def uncertainty(self, model_list=None):
        train_predys = []
        test_predys = []
        for model_name in model_list:
            for weight_name in self.cnn_features_dir_name:
                logger.info("processing {}".format(weight_name))
                feature_dir = os.path.join(config.data_root, self.dataname,
                                           "feature", weight_name)
                X = pickle_load_data(os.path.join(feature_dir, "X.pkl"))
                clf = pickle_load_data(os.path.join(feature_dir, model_name + "_model.pkl"))
                y = self.data.y
                train_X = X[np.array(self.train_idx), :]
                train_y = y[np.array(self.train_idx)]
                test_X = X[np.array(self.test_idx), :]
                test_y = y[np.array(self.test_idx)]
                train_predy = clf.predict(train_X)
                test_predy = clf.predict(test_X)
                train_predys.append(train_predy)
                test_predys.append(test_predy)
                logger.info("processing {} finished".format(weight_name))

        train_predys = np.array(train_predys).transpose()
        test_predys = np.array(test_predys).transpose()

        method_num = len(self.cnn_features_dir_name)
        coherent_matrix = np.zeros((method_num, method_num))
        for i in range(method_num):
            for j in range(method_num):
                corrected = (train_predys[:,i]==train_predys[:,j]).sum() + \
                    (test_predys[:,i] == test_predys[:,j]).sum()
                coherent_matrix[i,j] = corrected
        print(coherent_matrix)
        train_entropy = []
        test_entropy = []
        for i in range(train_predys.shape[0]):
            tmp = np.bincount(train_predys[i,:])
            if np.count_nonzero(tmp) == 1:
                train_entropy.append(0)
            else:
                train_entropy.append(entropy(tmp))
        for i in range(test_predys.shape[0]):
            tmp = np.bincount(test_predys[i,:])
            if np.count_nonzero(tmp) == 1:
                test_entropy.append(0)
            else:
                test_entropy.append(entropy(tmp))
        all_entropy = np.zeros(self.data.X.shape[0])
        all_entropy[np.array(self.train_idx)] = train_entropy
        all_entropy[np.array(self.test_idx)] = test_entropy
        print(os.path.join(config.data_root,
                                        self.dataname,
                                        "all_entropy.pkl"))
        if len(model_list) == 1:
            suffix = model_list[0]
        else:
            suffix = ""
        pickle_save_data(os.path.join(config.data_root,
                                        self.dataname,
                                        "all_entropy" + suffix +".pkl"), all_entropy)
        return

    def train_ensemble_models(self, model_name="svm"):
        for weight_name in self.cnn_features_dir_name:
            logger.info("processing {}".format(weight_name))
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
            print("train_X.shape: {}, test_X.shape: {}".format(train_X.shape,
                                                               test_X.shape))

            # #For quick test
            # print("warning***************************************** for quick test")
            # idx = np.array(range(train_X.shape[0]))
            # np.random.seed(123)
            # np.random.shuffle(idx)
            # train_X = train_X[idx[:5000]]
            # train_y = train_y[idx[:5000]]

            if model_name == "svm":
                clf = SVC(kernel="linear", verbose=1)
            elif model_name == "rf":
                clf = RandomForestClassifier(n_estimators=100)
            elif model_name == "xgboost":
                clf = XGBClassifier()
            elif model_name == "logistic":
                clf = LogisticRegression()

            clf.fit(train_X, train_y)
            print(weight_name, clf.score(train_X, train_y), clf.score(test_X, test_y))
            pickle_save_data(os.path.join(feature_dir, model_name + "_model.pkl"), clf)



    def get_entropy(self):
        if not os.path.exists(os.path.join(config.data_root,
                                                    self.dataname,
                                                    "all_entropy.pkl")):
            self.uncertainty()

        all_entropy = pickle_load_data(os.path.join(config.data_root,
                                                    self.dataname,
                                                    "all_entropy.pkl"))
        self.all_entropy = all_entropy
        train_entropy = all_entropy[np.array(self.train_idx)]
        test_entropy = all_entropy[np.array(self.test_idx)]
        return train_entropy, test_entropy

    def entropy_analysis(self):
        all_entropy = pickle_load_data(os.path.join(config.data_root,
                                                    self.dataname,
                                                    "all_entropy.pkl"))
        bias_test_idx = [i for i in self.test_idx if i not in self.test_redundant_idx]
        logger.info("bias_test len: {}, redundant_test len: {}"
                    .format(len(bias_test_idx), len(self.test_redundant_idx)))
        bias_test_entropy = all_entropy[bias_test_idx]
        test_redundant_entropy = all_entropy[np.array(self.test_redundant_idx)]
        logger.info("bias_test entropy mean: {}. test redundant entropy mean: {}"
                    .format(bias_test_entropy.mean(), test_redundant_entropy.mean()))

        ax = plt.subplot(121)
        ax.hist(bias_test_entropy, 20)
        ax.set_title("IoD entropy histogram ({})".format(self.dataname))
        ax.set_ylabel("count")
        ax.set_xlabel("entropy")
        ax = plt.subplot(122)
        ax.set_title("OoD entropy histogram ({})".format(self.dataname))
        ax.set_ylabel("count")
        ax.set_xlabel("entropy")
        ax.hist(test_redundant_entropy, 20)
        plt.show()

    def get_similar(self, id, k):
        #TODO
        if id in self.data.train_idx:
            all_idx = self.data.train_idx.copy()
        elif id in self.data.test_idx:
            all_idx = self.data.test_idx.copy()
        all_feature = self.data.X
        center_feature = self.data.X[id]
        distances = [np.sum(np.square(center_feature - all_feature[idx])) for idx in all_idx]
        index = [all_idx[i] for i in sorted(range(len(all_idx)), key=distances.__getitem__)[:k]]
        return index

if __name__ == '__main__':
    e = Ensemble(config.lipsticks)
    # for model_name in ["svm","logistic", "xgboost", "rf"]:
    #     e.train_ensemble_models(model_name)
    # e.train_ensemble_models("logistic")
    e.uncertainty(["logistic"])
    # e.entropy_analysis()
