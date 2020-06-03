import numpy as np
import os
import scipy.io as sio
from time import time
import warnings
import math

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_mldata
import tensorflow as tf
from scipy.interpolate import interp1d
from PIL import Image
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import keras
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Input
from keras import applications
from keras import optimizers
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from multiprocessing import Pool
from skimage.feature import hog, local_binary_pattern
from skimage import color
from sklearn.manifold import TSNE, MDS

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_save_data, pickle_load_data
from scripts.database.database import DataBase
from scripts.database.animals import DataAnimals
from scripts.utils.log_utils import logger
from scripts.utils.embedder_utils import Embedder
import shutil

class DataAnimalsStep1(DataAnimals):
    def __init__(self, dataname=None, suffix="", class_num=5):
        if dataname is None:
            dataname = config.animals_step1
        self.class_num = class_num
        self.father_dataname = config.animals
        super(DataAnimalsStep1, self).__init__(dataname, suffix)

    def preprocessing_data(self):
        all_data_name = os.path.join(config.data_root, self.father_dataname, config.all_data_cache_name)
        all_data = pickle_load_data(all_data_name)
        class_name = all_data[config.class_name][:5]
        X = all_data["X"]
        y = all_data["y"]
        y = np.array(y)
        train_idx = all_data["train_idx"]
        train_redundant_idx = all_data["train_redundant_idx"]
        valid_idx = all_data["valid_idx"]
        valid_redundant_idx = all_data["valid_redundant_idx"]
        test_idx = all_data["test_idx"]
        test_redundant_idx = all_data["test_redundant_idx"]
        test_sub_y = all_data["test_sub_y"]
        all_sub_y = np.ones(len(y)) * -1
        all_sub_y[np.array(test_idx)] = test_sub_y

        # check
        for red_idx in test_redundant_idx:
            if red_idx not in test_idx:
                raise ValueError("ERROR")

        leopard_idx = test_idx[-1000:]
        test_idx = test_idx[:-1000]
        test_redundant_idx = test_redundant_idx[:-1000]

        processed_data = pickle_load_data(os.path.join(config.data_root,
                                                       self.father_dataname,
                                                       config.processed_dataname))
        X = processed_data[config.X_name]

        print("test_idx: {}. test_redundant_idx: {}"
              .format(len(test_idx), len(test_redundant_idx)))

        reserved_cartoon = []
        for idx, sub_y in enumerate(test_sub_y):
            selected_idx = test_idx[idx]
            if sub_y == 6:
                reserved_cartoon.append(selected_idx)
                test_idx.remove(selected_idx)
                test_redundant_idx.remove(selected_idx)
            if len(reserved_cartoon) > 25:
                break
        for idx, sub_y in enumerate(test_sub_y):
            selected_idx = test_idx[idx]
            if sub_y == 12:
                reserved_cartoon.append(selected_idx)
                test_idx.remove(selected_idx)
                test_redundant_idx.remove(selected_idx)
            if len(reserved_cartoon) > 50:
                break

        # check
        for red_idx in test_redundant_idx:
            if red_idx not in test_idx:
                raise ValueError("ERROR 2")

        print("reserved_cartoon: {}, test_idx: {}. test_redundant_idx: {}"
              .format(len(reserved_cartoon), len(test_idx), len(test_redundant_idx)))

        # split data
        idx = np.array(range(len(test_idx)))
        np.random.seed(123)
        np.random.shuffle(idx)
        split_point = int(len(idx) * 2 / 3)
        step_1_test_idx = np.array(test_idx)[idx[:split_point]].tolist()
        step_1_test_idx.sort()
        step_2_test_idx = np.array(test_idx)[idx[split_point:]].tolist()
        step_2_test_idx.sort()

        # check
        for red_idx in test_redundant_idx:
            if red_idx not in step_1_test_idx and red_idx not in step_2_test_idx:
                raise ValueError("ERROR 3")

        print("step_1_test_idx:{}, step_2_test_idx:{}, test_idx: {}"
              .format(len(step_1_test_idx), len(step_2_test_idx), len(test_idx)))
        step_1_test_redundant_idx = []
        step_2_test_redundant_idx = []
        for idx in test_redundant_idx:
            if idx in step_1_test_idx:
                step_1_test_redundant_idx.append(idx)
            elif idx in step_2_test_idx:
                step_2_test_redundant_idx.append(idx)
            else:
                print("*************************bug********************")
        print("step_1_test_redundant_idx: {}, step_2_test_redundant_idx: {}, test_redundant_idx: {}"
              .format(len(step_1_test_redundant_idx),
                      len(step_2_test_redundant_idx),
                      len(test_redundant_idx)))

        step_2_test_redundant_idx = step_2_test_redundant_idx + leopard_idx
        step_2_test_idx = step_2_test_idx + leopard_idx

        # save split info
        split_info_file = os.path.join(config.data_root, self.father_dataname, "split_info.pkl")
        pickle_save_data(split_info_file, {
            "step_1_test_idx": step_1_test_idx,
            "step_2_test_idx": step_2_test_idx,
            "step_1_test_redundant_idx":step_1_test_redundant_idx,
            "step_2_test_redundant_idx": step_2_test_redundant_idx,
            "reserved_cartoon": reserved_cartoon
        })


        self.all_data = {
            "class_name": class_name,
            "class_name_encoding": [],
            "X": None,
            "y": y,
            "train_idx": train_idx,
            "train_redundant_idx": train_redundant_idx,
            "valid_idx": valid_idx,
            "valid_redundant_idx": valid_redundant_idx,
            "test_idx": step_1_test_idx,
            "test_redundant_idx": step_1_test_redundant_idx,
            "all_sub_y": all_sub_y
        }
        self.save_cache();exit()


        self.class_name = class_name
        self.X = X
        self.y = y
        self.train_idx = train_idx
        self.train_redundant_idx = train_redundant_idx
        self.valid_idx = valid_idx
        self.valid_redundant_idx = valid_redundant_idx
        self.test_idx = step_1_test_idx
        self.test_redundant_idx = step_1_test_redundant_idx
        self.add_info = {}

        super(DataAnimalsStep1, self).postprocess_data("", embedding=True)

        self.save_file()

    def move_image(self):
        dirs = [self.test_data_dir]
        for idx, data_dir in enumerate(dirs):
            selected_idx = self.test_idx
            for cls, cls_name in enumerate(self.class_name):
                cls_dir = os.path.join(data_dir, cls_name)
                check_dir(cls_dir)
                if len(selected_idx) == 0:
                    logger.info("selected_idx is empty")
                    continue
                selected_y = np.array(self.y)[np.array(selected_idx)]
                cls_idx = np.array(selected_idx)[selected_y==cls]
                for i in cls_idx:
                    src = os.path.join(config.data_root, self.father_dataname, "images", str(i) + ".jpg")
                    target = os.path.join(cls_dir, str(i) + ".jpg")
                    shutil.copy(src, target)

    def load_data(self, loading_from_buffer=True):
        super(DataAnimalsStep1, self).load_data(loading_from_buffer)
        # self.class_name = self.all_data["class_name"]
        # self.class_name_encoding = self.all_data["class_name_encoding"]
        # self.X = self.all_data["X"]
        # self.y = self.all_data["y"]
        # self.y = np.array(self.y)
        # self.train_idx = self.all_data["train_idx"]
        # self.train_redundant_idx = self.all_data["train_redundant_idx"]
        # self.valid_idx = self.all_data["valid_idx"]
        # self.valid_redundant_idx = self.all_data["valid_redundant_idx"]
        # self.test_idx = self.all_data["test_idx"]
        # self.test_redundant_idx = self.all_data["test_redundant_idx"]
        # self.all_sub_y = self.all_data["all_sub_y"]
        mat = pickle_load_data(os.path.join(self.data_dir, config.processed_dataname))
        self.y = mat[config.y_name]
        self.train_idx = mat[config.train_idx_name]
        self.test_idx = mat[config.test_idx_name]
        self.train_redundant_idx = mat[config.train_redundant_idx_name]
        self.test_redundant_idx = mat[config.test_redundant_idx_name]


    def save_cartoon_backup(self):
        split_info_file = os.path.join(config.data_root, self.father_dataname, "split_info.pkl")
        mat = pickle_load_data(split_info_file)
        reserved_cartoon = mat["reserved_cartoon"]
        cartoon_dir = os.path.join(self.data_dir, "cartoon")
        check_dir(cartoon_dir)
        for i in reserved_cartoon:
            src = os.path.join(config.data_root, self.father_dataname, "images", str(i) + ".jpg")
            target = os.path.join(cartoon_dir, str(i) + ".jpg")
            shutil.copy(src, target)

    def postprocess_data(self, weight_name, if_return=False, embedding=True):
        feature_dir = os.path.join(self.feature_dir, weight_name)
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        X = None
        prediction = None
        count  = 0
        new_points = []
        for data_dir in dirs:
            file_prefix = os.path.split(data_dir)[1].split(".")[0]
            data_filename = os.path.join(feature_dir, file_prefix + config.pkl_ext)
            if not os.path.exists(data_filename):
                logger.warn("{} does not exist, skip!".format(data_filename))
                continue
            mat = pickle_load_data(data_filename)
            features = mat["features"][0]
            pred_y = mat["features"][1]
            filenames = mat["filename"]
            for idx, name in enumerate(filenames):
                name = name.replace("\\", "/")
                cls, img_name = name.split("/")
                img_id, _ = img_name.split(".")
                try:
                    img_id = int(img_id)
                except:
                    img_id = 1000000
                if img_id >= 100000:
                    continue
                if data_dir == self.train_data_dir and img_id not in self.train_idx:
                    if img_id in self.test_idx:
                        print(img_id)
                    else:
                        count = count + 1
                        self.train_idx.append(img_id)
                        new_points.append(img_id)

                if len(features.shape) > 2:
                    features = features.reshape(features.shape[0], -1)
                if X is None:
                    X = np.zeros((self.y.shape[0], features.shape[1]))
                    prediction = np.ones(self.y.shape[0]) * -1
                X[img_id, :] = features[idx, :]
                try:
                    prediction[img_id] = pred_y[idx,:].argmax()
                except:
                    prediction[img_id] = -1

        # a = pickle_load_data(os.path.join(config.data_root,
        #                               self.dataname,
        #                               "processed_data.pkl"))
        # print(len(self.train_idx))
        # a[config.train_idx_name] = self.train_idx
        # pickle_save_data(os.path.join(config.data_root, self.dataname, "processed_data.pkl"),a)
        # exit()


        self.X = X
        # pickle_save_data(os.path.join(feature_dir, "X.pkl"), X);exit()
        self.add_info = {
            "prediction": prediction
        }
        # print(prediction.shape)
        if if_return:
            logger.info("'if return' flag is enabled. Returning immediately!")
            return X

        pre_step_data = pickle_load_data(os.path.join(config.data_root,
                                                      self.dataname,
                                                      "processed_data_baseline_backup.pkl"))
        old_prediction = pickle_load_data(os.path.join(config.data_root,
                                                      "Animals",
                                                      "prediction.pkl"))
        now_prediction = pickle_load_data(os.path.join(config.data_root,
                                                      "Animals-step1",
                                                      "prediction.pkl"))


        embedding_X = pre_step_data["embed_X"]
        for idx in new_points:
            print(idx)
            x = X[idx,:].reshape(1,-1)
            dis = ((X - x.repeat(axis=0, repeats=X.shape[0]))**2).sum(axis=1)
            dis[idx] = 0
            argmax = dis.argmin()
            assert argmax in (self.train_idx + self.test_idx)
            embedding_X[idx,:] = embedding_X[argmax, :]

        print("solve new predictions")

        count = 0

        for idx in self.test_idx:
            if old_prediction[idx] != now_prediction[idx]:
                x = X[idx, :].reshape(1, -1)
                dis = ((X - x.repeat(axis=0, repeats=X.shape[0])) ** 2).sum(axis=1)
                nn_idx = np.argsort(dis)[1:6]
                embedding = embedding_X[idx, :] - embedding_X[idx, :]
                for i in nn_idx:
                    embedding += embedding_X[i, :]
                embedding_X[idx, :] = embedding / 5
                count += 1
                print(count, idx, old_prediction[idx], now_prediction[idx])


        print("begin tsne")

        initial_train_X = embedding_X[np.array(np.array(self.train_idx + self.test_idx)), :]
        print(initial_train_X.shape)

        valided_data_idx = self.train_idx + self.valid_idx + self.test_idx
        logger.info("info confirm, valided data num: {}".format(len(valided_data_idx)))
        if embedding is False:
            self.embed_X = None
            self.all_embed_X = None
            return
        projected_X = self.X[np.array(valided_data_idx), :]
        projection_method = ["tsne", "pca"]
        default_method = "tsne"
        self.all_embed_X = {}
        self.embed_X = []
        embedder = TSNE(n_components=2, random_state=123,
                        # early_exaggeration=1.0,
                        init=initial_train_X)
        # embedder = TSNE(n_components=2, random_state=123)
        partial_embed_X = embedder.fit_transform(projected_X)
        embed_X = np.zeros((len(self.y), 2))
        embed_X[np.array(valided_data_idx), :] = partial_embed_X
        self.all_embed_X["tsne"] = embed_X
        self.embed_X = embed_X

    def accuracy_test(self, processed_data_name):
        processed_data = pickle_load_data(os.path.join(config.data_root,
                                                      self.dataname,
                                                      processed_data_name))
        prediction = processed_data[config.add_info_name]["prediction"]


        name_list = ["black cat", "white cat", "black dog", "white dog",
                     "cat and human", "cat cage",
                     "cat cartoon", "cat in dress",
                     "cat indoor", "cat outdoor",
                     "dog and human", "dog cage",
                     "dog cartoon", "dog in dress",
                     "dog indoor", "dog outdoor",
                     "two cat", "two dog",
                     "rabbit", "wolf", "tiger", "tiger-cat", "husky", "leopard"]

        X = processed_data[config.X_name]
        y = processed_data[config.y_name]
        X_train = X[np.array(self.train_idx),:]
        y_train = y[np.array(self.train_idx)]
        X_test = X[np.array(self.test_idx),:]
        y_test = y[np.array(self.test_idx)]

        clf = SVC(kernel="linear", verbose=1, max_iter=5000)
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        print("training score: {}, test score: {}".format(train_score, test_score))
        return clf.predict(X_test)
        #
        # for i, name in enumerate(name_list):
        #     sub_idx = self.all_sub_y == i
        #     sub_X = X[sub_idx]
        #     sub_y = y[sub_idx]
        #     score = clf.score(sub_X, sub_y)
        #     print(name, ":", score)


    def inplace_process_data(self):
        cnn_features_dir_name = [
                # "weights.20-0.8054.h5",
                # "weights.19-0.8028.h5",
                # "weights.32-0.8073.h5",
                # "inceptionresnet_imagenet",
                # "inceptionv3_imagenet",
                # "mobilenet_imagenet",
                # "resnet50_imagenet",
                # "vgg_imagenet",
                # "xception_imagenet",
                # "sift-200",
                # "brief-200",
                "orb-200",
                "surf-200"
                # "superpixel"

        ]
        for weight_name in cnn_features_dir_name:
            logger.info(weight_name)
            X = self.postprocess_data(weight_name, if_return=True)
            filename = os.path.join(self.feature_dir, weight_name, "X.pkl")
            pickle_save_data(filename, X)

if __name__ == '__main__':
    d = DataAnimalsStep1()
    # d.preprocessing_data()
    d.load_data()
    # d.inplace_process_data()
    d.process_superpixel()
    # d.save_cartoo n_backup()
    # d.postprocess_data("weights.40-0.8031.h5", embedding=False)
    # d.save_file()
    prev_label = d.accuracy_test("processed_data_ADD_100.pkl")
    now_label = d.accuracy_test("processed_data_ADD_100_50cartoon.pkl")
    onchange = prev_label != now_label
    count = 0
    for id, i in enumerate(onchange):
        if i == True:
            if now_label[id] < 2 and prev_label[id] < 2:
                count += 1
                print(d.test_idx[id], now_label[id])

    print(count)
