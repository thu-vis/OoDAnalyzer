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
            "weights.100-0.5465.h5",
            "inceptionresnet_imagenet",
            "inceptionv3_imagenet",
            "mobilenet_imagenet",
            "resnet50_imagenet",
            "vgg_imagenet",
            "xception_imagenet",
            "sift-200",
            # "HOG",
            "HOG-kmeans-200",
            # "superpixel-500",
            # "LBP",
            # "LBP-hist",
            # "sift-500",
            # "sift-1000"
        ]
        if self.cnn_features_dir_name[0][:7] == "weights" and \
            (dataname == config.dog_cat or dataname == config.dog_cat_extension):
            self.cnn_features_dir_name[0] = "weights.20-0.9922.h5"


    def uncertainty(self):
        test_predys = []
        for weight_name in self.cnn_features_dir_name:
            logger.info("processing {}".format(weight_name))
            svm_dir = os.path.join(config.data_root, config.dog_cat, "feature", weight_name)
            feature_dir = os.path.join(config.data_root, self.dataname, "feature", weight_name)
            X = pickle_load_data(os.path.join(feature_dir, "X.pkl"))
            clf = pickle_load_data(os.path.join(svm_dir, "svm_model.pkl"))
            y = self.data.y
            test_X = X[np.array(self.test_idx), :]
            test_y = y[np.array(self.test_idx)]
            test_predy = clf.predict(test_X)
            test_predys.append(test_predy)
            logger.info("processing {} finished".format(weight_name))

        test_predys = np.array(test_predys).transpose()

        method_num = len(self.cnn_features_dir_name)

        test_entropy = []
        for i in range(test_predys.shape[0]):
            tmp = np.bincount(test_predys[i,:])
            if np.count_nonzero(tmp) == 1:
                test_entropy.append(0)
            else:
                test_entropy.append(entropy(tmp))
        all_entropy = np.zeros(self.data.X.shape[0])
        all_entropy[np.array(self.test_idx)] = test_entropy
        pickle_save_data(os.path.join(config.data_root,
                                        self.dataname,
                                        "all_entropy.pkl"), all_entropy)
        return

    def get_entropy(self):
        if not os.path.exists(os.path.join(config.data_root,
                                                    self.dataname,
                                                    "all_entropy.pkl")):
            self.uncertainty()

        all_entropy = pickle_load_data(os.path.join(config.data_root,
                                                    self.dataname,
                                                    "all_entropy.pkl"))
        test_entropy = all_entropy[np.array(self.test_idx)]
        return test_entropy

    def entropy_analysis(self):
        all_entropy = pickle_load_data(os.path.join(config.data_root,
                                                    self.dataname,
                                                    "all_entropy.pkl"))

        self.test_redundant_idx = self.data.test_redundant_idx
        # self.test_redundant_idx = []
        bias_test_idx = [i for i in self.test_idx if i not in self.test_redundant_idx]
        logger.info("bias_test len: {}, redundant_test len: {}"
                    .format(len(bias_test_idx), len(self.test_redundant_idx)))
        bias_test_entropy = all_entropy[bias_test_idx]
        test_redundant_entropy = all_entropy[np.array(self.test_redundant_idx)]
        logger.info("bias_test entropy mean: {}. test redundant entropy mean: {}"
                    .format(bias_test_entropy.mean(), test_redundant_entropy.mean()))

        # ax = plt.subplot(121)
        # ax.hist(bias_test_entropy, 20)
        # ax.set_title("IoD entropy histogram ({})".format(self.dataname))
        # ax.set_ylabel("count")
        # ax.set_xlabel("entropy")
        ax = plt.subplot(122)
        ax.set_title("OoD entropy histogram ({})".format(self.dataname))
        ax.set_ylabel("count")
        ax.set_xlabel("entropy")
        ax.hist(test_redundant_entropy, 20)
        plt.show()
if __name__ == '__main__':
    d = Ensemble(config.dog_cat_extension)
    e = d.get_entropy()
    print(np.array(e).mean())
    d.entropy_analysis()