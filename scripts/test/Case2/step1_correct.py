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
from scripts.utils.helper_utils import OoD_metrics as metrics
from scripts.database.database import DataBase
from scripts.utils.data_utils import Data
from scripts.utils.log_utils import logger
from scripts.utils.embedder_utils import Embedder
import shutil
from scipy.stats import entropy
import matplotlib.pyplot as plt

class REAAcc(DataBase):
    def __init__(self):
        dataname = config.rea
        super(REAAcc, self).__init__(dataname)

    def load_data(self, loading_from_buffer=True):
        super(REAAcc, self).load_data(loading_from_buffer)
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

    def save_acc(self):
        feature_dir_name = \
            "unet_nested_dilated_nopre_mix_33_0.001_adding_new_samples_3"

        feature_dir = os.path.join(self.feature_dir, feature_dir_name)
        train_info = pickle_load_data(os.path.join(feature_dir, "train_feature.npy"))
        val_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
        test_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
        redundant_idx = np.array(self.test_redundant_idx)
        pred_y_prob = np.concatenate((train_info["pred_cls"],
                                      val_info["pred_cls"],
                                      test_info["pred_cls"]), axis=0)
        y = np.concatenate((train_info["label_cls"],
                            val_info["label_cls"],
                            test_info["label_cls"]), axis=0)
        redundant_pred_y_pro = pred_y_prob[:,0][redundant_idx]
        redundant_pred_y = (redundant_pred_y_pro > 0.5).astype(int)
        redundant_y = y[:,0][redundant_idx]

        prediction = (pred_y_prob[:,0] > 0.5).astype(int)

        print("correct: {}, total: {}".format((redundant_pred_y==redundant_y).sum(),
                                              len(redundant_y)))
        prediction_path = os.path.join(config.data_root, self.dataname, "prediction.pkl")
        pickle_save_data(prediction_path, prediction)


    def get_pred(self, feature_dir_name):
        feature_dir = os.path.join(self.feature_dir, feature_dir_name)
        try:
            train_info = pickle_load_data(os.path.join(feature_dir, "train_feature.npy"))
        except:
            print("training_feature.npy does not exists")
            train_info = {
                "pred_cls": np.zeros((8960,3)),
                "label_cls": np.ones((8960,3))
            }
        val_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
        test_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))

        pred_y_prob = np.concatenate((train_info["pred_cls"],
                                      val_info["pred_cls"],
                                      test_info["pred_cls"]), axis=0)
        y = np.concatenate((train_info["label_cls"],
                            val_info["label_cls"],
                            test_info["label_cls"]), axis=0)

        return pred_y_prob, y

    def acc_comparing(self):
        feature_dir_name_1 = \
            "unet_nested_dilated_nopre_mix_33_0.001_unet_nested_dilated-wce-eldice-bce"
        feature_dir_name_2 = \
            "unet_nested_dilated_nopre_mix_33_0.0005_step1_more"
        feature_dir_name_3 = \
            "unet_nested_dilated_nopre_mix_33_0.0002_step2-HRF"
        pred_y_prob_1, y = self.get_pred(feature_dir_name_1)
        pred_y_prob_2, y = self.get_pred(feature_dir_name_2)
        pred_y_prob_3, y = self.get_pred(feature_dir_name_3)

        # pred_y_1 = pred_y_prob_1[:,0]
        # y = y[:,0]
        # a = [1152, 1943, 1944, 1955, 7424, 9345, 9346, 9347, 9348, 9353, 9354, 9355, 9356,
        #      9357, 9358, 9359, 9361, 9362, 9363, 9364, 9365, 9366, 9368, 9382, 9384, 9392, 9472,
        #      9473,9474, 9478, 9480, 9481, 9482,
        #      9484, 9485, 9491, 9493, 9495, 9496, 9498, 9986, 9987]
        # a = np.array(a)
        #
        # y[:, 0][9344:9386] = 1
        # y[:, 0][9472:9474] = 1

        # a = range(8960 + 68, 8960 + 79 + 1)
        # a = np.array(a)
        # print("acc:", sum((pred_y_prob_1[a,0] > 0.5).astype(int) == y[a,0]) / len(a))
        #
        # a = range(8960 + 6 * 128 + 53, 8960 + 6 * 128 + 68 + 1)
        # a = np.array(a)
        # print("acc:", sum((pred_y_prob_1[a,0] > 0.5).astype(int) == y[a,0]) / len(a))
        #
        # a = range(8960 + 13 * 128 + 56, 8960 + 13 * 128 + 79 + 1)
        # a = np.array(a)
        # print("acc:", sum((pred_y_prob_1[a,0] > 0.5).astype(int) == y[a,0]) / len(a))
        #
        # a = range(8960 + 14 * 128 + 55, 8960 + 14 * 128 + 77 + 1)
        # a = np.array(a)
        # print("acc:", sum((pred_y_prob_1[a,0] > 0.5).astype(int) == y[a,0]) / len(a))




        p1 = pred_y_prob_1[:,0][np.array(self.test_idx)]
        p2 = pred_y_prob_2[:,0][np.array(self.test_idx)]
        p3 = pred_y_prob_3[:,0][np.array(self.test_idx)]
        y = y[:,0][np.array(self.test_idx)]

        metrics(p1, y)
        metrics(p2, y)
        metrics(p3, y)

        # for threshold in np.arange(0.5, 1, 0.05):
        for threshold in [0.5]:
            hard_p1 = (p1>threshold).astype(int)
            hard_p2 = (p2>threshold).astype(int)
            hard_p3 = (p3>threshold).astype(int)
            print("{}\t{}\t{}\t{}"
                  .format(threshold, sum(hard_p1==y)/len(y), sum(hard_p2==y)/len(y), sum(hard_p3==y)/len(y)))

        # self.analysis(p2, y)
        # redundant_pred_y_pro = pred_y_prob[:,0][redundant_idx]
        # redundant_pred_y = (redundant_pred_y_pro > 0.5).astype(int)
        # redundant_y = y[:,0][redundant_idx]
        # print("correct: {}, total: {}".format((redundant_pred_y==redundant_y).sum(),
        #                                       len(redundant_y)))

    def analysis(self, p, y):
        neg_p = p[y==0]
        pos_p = p[y==1]
        ax = plt.subplot(121)
        ax.hist(neg_p, 20)
        ax.set_title("negative (ground truth) instance ({})".format(""))
        ax.set_ylabel("count")
        ax.set_xlabel("entropy")
        ax = plt.subplot(122)
        ax.set_title("negative (ground truth) instance ({})".format(""))
        ax.set_ylabel("count")
        ax.set_xlabel("entropy")
        ax.hist(pos_p, 20)
        plt.show()
    # def svm_acc(self):
    #     data = Data(self.dataname)
    #     prediction = data.prediction
    #     # feature_dir_name = \
    #     #     "unet_nested_dilated_nopre_mix_33_0.001_adding_new_samples_3"
    #     feature_dir_name = \
    #         "fine-tune"
    #     feature_dir = os.path.join(self.feature_dir, feature_dir_name)
    #     train_info = pickle_load_data(os.path.join(feature_dir, "train_feature.npy"))
    #     val_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
    #     test_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
    #     redundant_idx = np.array(self.test_redundant_idx)
    #     pred_y_prob = np.concatenate((train_info["pred_cls"],
    #                                   val_info["pred_cls"],
    #                                   test_info["pred_cls"]), axis=0)
    #     y = np.concatenate((train_info["label_cls"],
    #                         val_info["label_cls"],
    #                         test_info["label_cls"]), axis=0)
    #     redundant_pred_y_pro = pred_y_prob[:, 0][redundant_idx]
    #     # redundant_pred_y = (redundant_pred_y_pro > 0.5).astype(int)
    #     redundant_pred_y = prediction[redundant_idx]
    #     redundant_y = y[:, 0][redundant_idx]
    #
    #     # redundant_pred_y = prediction[70*128:(70+15)*128]
    #     # redundant_y = y[:, 0][70*128:(70+15)*128]
    #
    #     print("correct: {}, total: {}".format((redundant_pred_y==redundant_y).sum(),
    #                                           len(redundant_y)))

    def pick_images(self):
        # feature_dir_name = "unet_nested_dilated_nopre_mix_33_0.0005_step1_more"
        feature_dir_name = "unet_nested_dilated_nopre_mix_33_0.0005_step1_more"
        pred_y, y = self.get_pred(feature_dir_name)
        redundant_idx = np.array(self.test_redundant_idx)
        redundant_pred_y = pred_y[:,0][redundant_idx]
        redundant_hard_pred_y = (redundant_pred_y>0.5).astype(int)
        redundant_y = y[:,0][redundant_idx]

        correct_idx = np.array(redundant_idx)[redundant_hard_pred_y == redundant_y]
        print(len(correct_idx))

    def HRF_info(self):
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

        a = [9119, 9120, 9121, 9122, 9124, 9168, 9169, 9170, 9172, 9174, 9176, 9181, 9186, 9187,
            9256, 9348, 9349, 9350, 9352, 9355, 9356, 9358, 9359, 9363, 9366, 9367, 9369, 9371, 9372,
            9376, 9378, 9379, 9380, 9381, 9382, 9383, 9384, 9385, 9399, 9475, 9476, 9477, 8478,
            9479, 9480, 9481, 9482, 9485, 9486, 9487, 9488, 9489, 9490, 9495, 9496, 9497, 9498,]

        a = np.array(a)

        # feature_dir_name = "unet_nested_dilated_nopre_mix_33_0.001_unet_nested_dilated-wce-eldice-bce"
        # feature_dir_name = "unet_nested_dilated_nopre_mix_33_0.001_adding_new_samples_3"
        feature_dir_name = "unet_nested_dilated_nopre_mix_33_0.0002_step2-HRF"
        pred_y, y = self.get_pred(feature_dir_name)
        selected_pred_y = pred_y[:,0][a]
        selected_hard_pred_y = (selected_pred_y>0.5).astype(int)
        selected_y = y[:,0][a]
        print(sum(selected_hard_pred_y==selected_y)/len(a))
        print(sum(selected_hard_pred_y==selected_y), len(a))

    def prediction_analysis(self):
        feature_dir_name = "fine-tune"
        # feature_dir_name = "unet_nested_dilated_nopre_mix_33_0.0002_step2-HRF"
        # feature_dir_name = "unet_nested_dilated_nopre_mix_33_0.0005_step1_more"

        a = [9782, 9783, 9784, 9785, 9786, 9787, 9788, 9789, 9790, 9791, 9792, 9793, 9794, 9795,
        9796, 9797, 9798, 9799, 9800, 9801,
        10686, 10687, 10688, 10689, 10690, 10691, 10692, 10693, 10694, 10695, 10696,
        10697, 10698, 10699, 10700, 10701, 10702, 10703, 10704,
        10814, 10815, 10816, 10817, 10818, 10819, 10820, 10821, 10822, 10823, 10824,
        10825, 10826, 10827, 10828, 10829,
        9028, 9029, 9030, 9031, 9032, 9033, 9034, 9035, 9036, 9037, 9038,
        9019, 9020, 9021, 9022, 9023, 9024, 9025, 9026, 9027,]

        # a = [9119, 9120, 9121, 9122, 9124, 9168, 9169, 9170, 9172, 9174, 9176, 9181, 9186, 9187,
        #     9256, 9348, 9349, 9350, 9352, 9355, 9356, 9358, 9359, 9363, 9366, 9367, 9369, 9371, 9372,
        #     9376, 9378, 9379, 9380, 9381, 9382, 9383, 9384, 9385, 9399, 9475, 9476, 9477, 8478,
        #     9479, 9480, 9481, 9482, 9485, 9486, 9487, 9488, 9489, 9490, 9495, 9496, 9497, 9498,]

        # a = self.test_redundant_idx

        pred_y,y = self.get_pred(feature_dir_name)
        pred_y = pred_y[:,0][np.array(a)]
        y = y[:,0][np.array(a)]
        correct_num = (pred_y > 0.5).astype(int) == y
        print(correct_num.sum(), len(y))
        pred_y[pred_y<0.5] = 1 - pred_y[pred_y<0.5]
        print("confidence mean: ", pred_y.mean())
        ax = plt.subplot(111)
        ax.hist(pred_y, 20)
        plt.show()

if __name__ == '__main__':
    d = REAAcc()
    d.load_data()
    # d.save_acc()
    # d.acc_comparing()
    d.prediction_analysis()