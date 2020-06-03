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

class DataAnimalsStep2(DataAnimals):
    def __init__(self, suffix="", class_num=6):
        dataname = config.animals_step2
        self.class_num = class_num
        self.father_dataname = config.animals
        super(DataAnimalsStep2, self).__init__(dataname, suffix)

    def preprocessing_data(self):
        mat = pickle_load_data(os.path.join(self.data_dir, "processed_data_step1.pkl"))
        self.y = mat[config.y_name]
        self.train_idx = mat[config.train_idx_name]
        self.test_idx = mat[config.test_idx_name]
        self.train_redundant_idx = mat[config.train_redundant_idx_name]
        self.test_redundant_idx = mat[config.test_redundant_idx_name]

        split_info = pickle_load_data(os.path.join(self.data_dir, "split_info.pkl"))

        self.test_idx = split_info["step_1_test_idx"] + split_info["step_2_test_idx"]
        self.test_redundant_idx = split_info["step_1_test_redundant_idx"] + \
                                  split_info["step_2_test_redundant_idx"]
        print(len(self.test_idx))

        global_id = 45057
        for idx, cls_name in enumerate(["leopard"]):
            cls_dir = os.path.join(r"H:\backup\RawData", self.father_dataname, cls_name)
            img_name_list = os.listdir(cls_dir)
            for img_name in img_name_list:
                src = os.path.join(cls_dir, img_name)
                target = os.path.join(self.images_dir, str(global_id) + ".jpg")

        feature_path = os.path.join(self.feature_dir, "weights.30-0.8628.h5")
        name_encoding = {}
        encoding_count = 46057
        feature_dir = os.path.join(self.feature_dir, "resnet50_imagenet")
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        self.y = np.array(self.y.tolist() + [-1 for i in range(10000)])
        X = None
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
                    # print(name)
                    img_id = encoding_count
                    name_encoding[name] = encoding_count
                    encoding_count = encoding_count + 1
                    if data_dir == self.train_data_dir:
                        self.train_idx.append(img_id)

                if len(features.shape) > 2:
                    features = features.reshape(features.shape[0], -1)
                if X is None:
                    X = np.zeros((self.y.shape[0], features.shape[1]))
                X[img_id, :] = features[idx,:]

        print(len(self.train_idx), len(self.test_idx))

        self.all_data = {
            "class_name": self.class_name,
            "class_name_encoding": self.class_name_encoding,
            "X": X,
            "y": self.y,
            "train_idx": self.train_idx,
            "train_redundant_idx": self.train_redundant_idx,
            "valid_idx": self.valid_idx,
            "valid_redundant_idx": self.valid_redundant_idx,
            "test_idx": self.test_idx,
            "test_redundant_idx": self.test_redundant_idx,
            "name_encoding": name_encoding
        }
        self.save_cache()

    def inplace_process_data(self):
        cnn_features_dir_name = [
            "weights.30-0.8604.h5",
            "weights.30-0.8628.h5",
            "weights.50-0.8507.h5",
            "inceptionresnet_imagenet",
            "inceptionv3_imagenet",
            "mobilenet_imagenet",
            "resnet50_imagenet",
            "vgg_imagenet",
            "xception_imagenet",
            "sift-200",
            "brief-200",
            "orb-200",
            "surf-200",
            # "superpixel"

        ]
        for weight_name in cnn_features_dir_name:
            logger.info(weight_name)
            X = self.postprocess_data(weight_name, if_return=True)
            filename = os.path.join(self.feature_dir, weight_name, "X.pkl")
            pickle_save_data(filename, X)


    def postprocess_data(self, weight_name, if_return=False, embedding=True):
        feature_dir = os.path.join(self.feature_dir, weight_name)
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        # X = np.zeros((self.y.shape[0], 1024))
        X = None
        prediction = None
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
                    # print(name)
                    img_id = self.all_data["name_encoding"][name]
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

        self.X = X
        self.add_info = {
            "prediction": prediction
        }
        if if_return:
            logger.info("'if return' flag is enabled. Returning immediately!")
            return X

        super(DataAnimals, self).postprocess_data(weight_name, embedding)



    def tsne_postprocess_data(self, weight_name=None, if_return=False, embedding=True):
        self.load_data()
        # TODO: add incremental t-SNE here

        pre_step_data = pickle_load_data(os.path.join(config.data_root,
                                                      config.animals_step1,
                                                      "processed_data.pkl"))
        old_prediction = pickle_load_data(os.path.join(config.data_root,
                                                      config.animals_step1,
                                                      "prediction.pkl"))
        now_prediction = pickle_load_data(os.path.join(config.data_root,
                                                      self.dataname,
                                                      "prediction.pkl"))

        pre_data_num = pre_step_data["X_name"].shape[0]
        now_data_num = self.X.shape[0]

        new_points = []

        for idx in self.train_idx:
            if idx not in pre_step_data['train_idx']:
                new_points.append(idx)

        for idx in self.test_idx:
            if idx not in pre_step_data['test_idx']:
                new_points.append(idx)

        all_feature = self.X
        valid_feature = all_feature[:pre_data_num]
        X = pre_step_data["X_name"][pre_step_data["train_idx"]]


        embedding_X = np.zeros((self.X.shape[0], 2))
        embedding_X[:pre_data_num] = pre_step_data["embed_X"]


        count = 0
        max_dis = 1e12
        for idx in new_points:
            count += 1
            print(count, idx)
            x = all_feature[idx, :].reshape(1, -1)
            dis = ((valid_feature - x.repeat(axis=0, repeats=valid_feature.shape[0])) ** 2).sum(axis=1)
            if idx < pre_data_num:
                dis[idx] = max_dis
            argmax = dis.argmin()
            assert argmax in (pre_step_data['train_idx'] + pre_step_data['test_idx'])
            embedding_X[idx, :] = embedding_X[argmax, :]

        print("solve new predictions")

        count = 0

        for idx in self.test_idx:
            if idx > pre_data_num or old_prediction[idx] != now_prediction[idx]:
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
                        early_exaggeration=1.0,
                        init=initial_train_X)
        # embedder = TSNE(n_components=2, random_state=123)
        partial_embed_X = embedder.fit_transform(projected_X)
        embed_X = np.zeros((len(self.y), 2))
        embed_X[np.array(valided_data_idx), :] = partial_embed_X
        self.all_embed_X["tsne"] = embed_X
        self.embed_X = embed_X

        # # fake embed_X
        # self.all_embed_X = None
        # self.embed_X = None

        self.save_file()

    def process_superpixel(self):

        superpixel_feature_dir = os.path.join(self.feature_dir, "superpixel")
        X = np.zeros((len(self.y), 200))
        processed_list = []
        processed_name = []
        exclude_list = []
        unprocessed_list = []
        for file_name in ["binary_sp_features_train.pkl",
                          "binary_sp_features_test.pkl"]:
            mat = pickle_load_data(os.path.join(superpixel_feature_dir, file_name))
            a = 1
            for name in mat:
                feature = mat[name]
                img_id = name.split(".")[1]
                try:
                    img_id = int(img_id)
                except:
                    # print(name)
                    try:
                        name_key = name.replace(".", "/") + ".JPEG"
                        img_id = self.all_data["name_encoding"][name_key]
                    except:
                        print("double exception",name)
                if (img_id in self.train_idx) or (img_id in self.test_idx):
                    X[img_id] = feature
                    if img_id in processed_list:
                        print("dumplicated: ", name, processed_name[processed_list.index(img_id)])
                    else:
                        processed_list.append(img_id)
                        processed_name.append(name)
                else:
                    print("{} should not be in the training and test".format(name))
                    exclude_list.append(img_id)
        filename = os.path.join(self.feature_dir, "superpixel", "X.pkl")
        print(len(exclude_list), len(unprocessed_list))
        pickle_save_data(filename, X)

    def save_additional_images(self):
        self.load_data()
        name_encoding = self.all_data["name_encoding"]
        for name in name_encoding:
            id = name_encoding[name]
            cls, img_name = name.split("/")
            if cls == "tiger":
                continue
            print(cls, img_name)
            src = os.path.join(self.data_dir, "raw_images", img_name)
            target = os.path.join(self.data_dir, "images", str(id) + ".jpg")
            shutil.copy(src, target)


if __name__ == '__main__':
    d = DataAnimalsStep2()
    # d.save_additional_images()
    # d.preprocessing_data()
    # d.inplace_process_data()
    # d.postprocess_data()
    # d.process_superpixel()
    d.tsne_postprocess_data()