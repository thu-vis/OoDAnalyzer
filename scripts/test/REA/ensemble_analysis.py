import numpy as np
import os
import scipy.io as sio
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from PIL import Image
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Input, MaxPooling2D
from keras import applications
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_save_data, pickle_load_data
from scripts.database.database import DataBase
from scripts.utils.log_utils import logger
from scripts.utils.embedder_utils import Embedder
import shutil
from scipy.stats import entropy


def matrix_entropy(X):
    dim = X.shape[0]
    score = X.mean(axis=0)
    dist = np.zeros((X.shape[1], 2))
    dist[:,0] = score
    dist[:,1] = 1 - score
    en = entropy(dist.T)

    return en


class REAAnalysis(DataBase):
    def __init__(self):
        dataname = config.rea
        super(REAAnalysis, self).__init__(dataname)

    def load_data(self, loading_from_buffer=True):
        super(REAAnalysis, self).load_data(loading_from_buffer)
        self.class_name = self.all_data["class_name"]
        self.class_name_encoding = self.all_data["class_name_encoding"]
        self.X = self.all_data["X"]
        self.y = self.all_data["y"]
        self.train_idx = self.all_data["train_idx"]
        self.train_redundant_idx = self.all_data["train_redundant_idx"]
        self.valid_idx = self.all_data["valid_idx"]
        self.valid_redundant_idx = self.all_data["valid_redundant_idx"]
        self.test_idx = self.all_data["test_idx"]
        self.test_redundant_idx = self.all_data["test_redundant_idx"]

    def get_all_pred(self):
        features_dir_name = [
            "unet_nopre_mix_3_0.001_unet-wce-dice-bce",
            "unet_nopre_mix_33_0.001_unet-wce-eldice-bce",
            "unet_nested_nopre_mix_3_0.001_unet_nested-wce-dice-bce",
            "unet_nested_nopre_mix_33_0.001_unet_nested-wce-eldice-bce",
            "unet_nested_dilated_nopre_mix_3_0.001_unet_nested_dilated-wce-dice-bce",
            "unet_nested_dilated_nopre_mix_33_0.001_unet_nested_dilated-wce-eldice-bce"
        ]
        all_pred_y_prob = []
        for feature_dir_name in features_dir_name:
            feature_dir = os.path.join(self.feature_dir, feature_dir_name)
            train_info = pickle_load_data(os.path.join(feature_dir, "train_feature.npy"))
            val_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
            test_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
            X = np.concatenate((train_info["features"],
                                val_info["features"],
                                test_info["features"]), axis=0)
            pred_y_prob = np.concatenate((train_info["pred_cls"],
                                val_info["pred_cls"],
                                test_info["pred_cls"]), axis=0)
            y = np.concatenate((train_info["label_cls"],
                                val_info["label_cls"],
                                test_info["label_cls"]), axis=0)

            pred_y_prob = pred_y_prob[:,0]
            all_pred_y_prob.append(pred_y_prob)
        pred_y_prob = np.array(all_pred_y_prob)
        return pred_y_prob

    def analysis_entropy(self):
        pred_y_prob = self.get_all_pred()
        all_entropy = matrix_entropy(pred_y_prob)
        all_entropy[:70*128] = 0


        # a = [9782, 9783, 9784, 9785, 9789, 9790, 9792,
        #      9794, 9795, 9796, 9797, 9798,
        #      9803, 9805, 9807, 10599, 9786, 9787, 9788, 9781
        #      ]

        # a for cresent_image
        a = [
            9236, 9237, 9243, 9244, 9245, 9249,
            9251, 9252, 9326, 9329, 9334,
            9234, 9247, 9254, 9330,
            9336,9339, 9235,
            9230, 9335, 9332, 9248, 9246, 9242, 9231,
            9233, 9331, 9250, 9253, 9238
        ]

        # # for HRF
        # a = [
        #     9376, 9475, 9375, 9487, 9373, 9381, 9498,
        #     9473, 9475, 9494, 9375,
        #     9348, 9496, 9354,
        #     9379, 9489,
        #     9491, 9352, 9495, 9474, 9477, 9372,
        #     9351, 9355, 9384, 9374,
        #     9366, 9472, 9356, 9357, 9377,
        #     9358, 9363, 9361, 9362, 9490, 9483,
        #     9486, 9484, 9368, 9480, 9429, 9478
        # ]

        a = list(set(a))

        a = np.array(a)
        a_pred = pred_y_prob.T[a,:]

        # a_pred[a_pred<0.5] = 1 - a_pred[a_pred<0.5]
        pred_y_prob[pred_y_prob<0.5] = 1 - pred_y_prob[pred_y_prob<0.5]
        print(a_pred.mean())
        print(pred_y_prob.mean())

        all_entropy[:70*128] = 0

    def save_entropy(self):
        pred_y_prob = self.get_all_pred()
        all_entropy = matrix_entropy(pred_y_prob)
        # exit()
        pickle_save_data(os.path.join(config.data_root,
                     self.dataname,
                     "all_entropy.pkl"), all_entropy)

    def inplace_process_data(self):
        features_dir_name = [
            "unet_nopre_mix_3_0.001_unet-wce-dice-bce",
            "unet_nopre_mix_33_0.001_unet-wce-eldice-bce",
            "unet_nested_nopre_mix_3_0.001_unet_nested-wce-dice-bce",
            "unet_nested_nopre_mix_33_0.001_unet_nested-wce-eldice-bce",
            "unet_nested_dilated_nopre_mix_3_0.001_unet_nested_dilated-wce-dice-bce",
            "unet_nested_dilated_nopre_mix_33_0.001_unet_nested_dilated-wce-eldice-bce"
        ]
        redundant_pred_y = []
        redundant_y = []
        redundant_pred_y_pro = []
        redundant_idx = np.array(self.test_redundant_idx)
        test_pred_y_pro = []
        for feature_dir_name in features_dir_name:
            feature_dir = os.path.join(self.feature_dir, feature_dir_name)
            train_info = pickle_load_data(os.path.join(feature_dir, "train_feature.npy"))
            val_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
            test_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
            X = np.concatenate((train_info["features"],
                                val_info["features"],
                                test_info["features"]), axis=0)
            pred_y_prob = np.concatenate((train_info["pred_cls"],
                                val_info["pred_cls"],
                                test_info["pred_cls"]), axis=0)
            y = np.concatenate((train_info["label_cls"],
                                val_info["label_cls"],
                                test_info["label_cls"]), axis=0)
            redundant_pred_y_pro.append(pred_y_prob[:,0][redundant_idx])
            pred_y = (pred_y_prob[:,0]>0.5).astype(int)
            y = (y[:,0]).astype(int)
            redundant_pred_y.append(pred_y[redundant_idx])
            redundant_y.append(y[redundant_idx])
            test_pred_y_pro.append(pred_y_prob[:,0][np.array(self.test_idx)])
        redundant_pred_y_pro = np.array(redundant_pred_y_pro)
        redundant_pred_y = np.array(redundant_pred_y)
        redundant_y = np.array(redundant_y)
        test_pred_y_pro = np.array(test_pred_y_pro)
        test_redundant_entropy = matrix_entropy(redundant_pred_y_pro)
        test_entropy = matrix_entropy(test_pred_y_pro)
        print("redundant_pred_y_pro", redundant_pred_y_pro.shape, test_redundant_entropy.mean())
        print("test_pred_y_pro",test_pred_y_pro.shape, test_entropy.mean())

        for idx, en in enumerate(test_redundant_entropy):
            id = redundant_idx[idx]
            _pred_y = pred_y[redundant_idx][idx]
            _y = y[redundant_idx][idx]
            print("{}: {}, {}, {}".format(id, round(en, 3), _pred_y, _y))

        a = 1


if __name__ == '__main__':
    d = REAAnalysis()
    d.load_data()
    d.analysis_entropy()