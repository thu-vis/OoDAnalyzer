# -*- coding:utf-8 -*-
import numpy as np
import os
from abc import abstractmethod
import math
import cv2
from time import time

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Input
from keras import applications
from keras import optimizers
from multiprocessing import Pool
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from skimage.feature import hog, local_binary_pattern
from skimage import color
from PIL import Image

from ..utils.config_utils import config
from ..utils.helper_utils import pickle_load_data, pickle_save_data, check_dir
from ..utils.log_utils import logger
from ..utils.embedder_utils import Embedder

gradient = np.linspace(0, 1, 256)
gradient = np.vstack(gradient).reshape(-1)
color_gradient = plt.get_cmap("OrRd")(gradient)
color_gradient = (color_gradient[:, :3] * 255).astype(np.uint8)
color_gradient = color_gradient[:, np.array([2, 1, 0])]
color_gradient = color_gradient.reshape(color_gradient.shape[0], 1, -1)



def list_of_groups(init_list, children_list_len):
    if len(init_list) == 0:
        return []
    sum = 0
    res = []
    for lens in children_list_len:
        res.append(init_list[sum: sum + lens])
        sum = sum + lens
    return res

def orb_descriptor(img):
    orb = cv2.ORB_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    return des

def sift_descriptor(img):
    sift = cv2.xfeatures2d.SIFT_create()
    _, d = sift.detectAndCompute(img, None)
    return d

def brief_descriptor(img):
    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp = star.detect(img,None)
    kp, des = brief.compute(img, kp)
    return des

def surf_descriptor(img):
    surf = cv2.xfeatures2d.SURF_create()
    _, d = surf.detectAndCompute(img, None)
    return d

def _sift_train(start, end, filename, data_dir, descriptor):
    train_des = []
    train_des_num = []
    default_dim = 32
    if descriptor.__name__ == "sift_descriptor":
        default_dim = 128
    elif descriptor.__name__ == "surf_descriptor":
        default_dim = 64
    for idx in range(start, end):
        img_name = filename[idx]
        img = cv2.imread(os.path.join(data_dir, img_name))
        try:
            d = descriptor(img)
        except Exception as e:
            print(e)
            print("********************************************")
            d = np.zeros(default_dim).astype(int).reshape(1, default_dim)
        if d is None:
            print("**************None**************************")
            d = np.zeros(default_dim).astype(int).reshape(1, default_dim)
        train_des = train_des + d.astype(int).tolist()
        train_des_num.append(d.shape[0])
        if (idx - start + 1) % 50 == 0:
            print("start: {}, end: {}, now processed: {}. d shape: {}"
                  .format(start, end, (idx-start+1)/(end-start+1), d.shape))
    return train_des, train_des_num

def _sift_prediction(start, end, filename, data_dir, km, descriptor):
    labels = []
    des_num = []
    default_dim = 32
    if descriptor.__name__ == "sift_descriptor":
        default_dim = 128
    elif descriptor.__name__ == "surf_descriptor":
        default_dim = 64
    for idx in range(start, end):
        img_name = filename[idx]
        img = cv2.imread(os.path.join(data_dir, img_name))
        try:
            d = descriptor(img)
        except Exception as e:
            print(e)
            d = np.zeros(default_dim).astype(int).reshape(1, default_dim)
        if d is None:
            d = np.zeros(default_dim).astype(int).reshape(1, default_dim)

        labels = labels + km.predict(d).tolist()
        des_num.append(d.shape[0])
        if (idx - start + 1) % 50 == 0:
            print("start: {}, end: {}, now processed: {}"
                  .format(start, end, (idx - start + 1) / (end - start + 1)))
    return labels, des_num

def _hog_train(start, end, filename, data_dir):
    train_des = []
    train_des_num = []
    size = 0
    for idx in range(start, end):
        img_name = filename[idx]
        img = Image.open(os.path.join(data_dir, img_name))
        if min(img.size) < 64:
            img = img.resize((512, 512))
            logger.warn("new img size: {}".format(img.size))
        try:
            img_data = color.rgb2gray(np.array(img))
            fd = hog(img_data, orientations=8,
                     pixels_per_cell=(16, 16),
                     cells_per_block=(4, 4),
                     block_norm='L2', feature_vector=False)
            fd = fd.reshape(fd.shape[0] * fd.shape[1], -1)
            size = fd.shape[1]
        except Exception as e:
            fd = np.zeros(size).astype(int).reshape(1, size)
        train_des = train_des + fd.astype(int).tolist()
        train_des_num.append(fd.shape[0])
        if (idx - start + 1) % 50 == 0:
            print("start: {}, end: {}, now processed: {}"
                  .format(start, end, (idx - start + 1) / (end - start + 1)))
    return train_des, train_des_num

def _hog_prediction(start, end, filename, data_dir, km):
    labels = []
    des_num = []
    size = 0
    for idx in range(start, end):
        img_name = filename[idx]
        img = Image.open(os.path.join(data_dir, img_name))
        if min(img.size) < 64:
            img = img.resize((512, 512))
            logger.warn("new img size: {}".format(img.size))
        try:
            img_data = color.rgb2gray(np.array(img))
            fd = hog(img_data, orientations=8,
                     pixels_per_cell=(16, 16),
                     cells_per_block=(4, 4),
                     block_norm='L2', feature_vector=False)
            fd = fd.reshape(fd.shape[0] * fd.shape[1], -1)
            size = fd.shape[1]
        except Exception as e:
            fd = np.zeros(size).astype(int).reshape(1, size)

        labels = labels + km.predict(fd).tolist()
        des_num.append(fd.shape[0])
        if (idx - start + 1) % 50 == 0:
            print("start: {}, end: {}, now processed: {}"
                  .format(start, end, (idx - start + 1) / (end - start + 1)))
    return labels, des_num

def _lbp_prediction(start, end, filename, data_dir):
    X = []
    for idx in range(start, end):
        img_name = filename[idx]
        img = Image.open(os.path.join(data_dir, img_name))
        try:
            img_data = color.rgb2gray(np.array(img))
        except:
            img_data = np.array(img).mean(axis=2).astype(np.uint8)
        fd = local_binary_pattern(img_data, n_points, radius, METHOD)
        fd = fd.reshape(-1).astype(int)
        bincount = np.bincount(np.array(fd))
        X.append(bincount)
        if (idx - start + 1) % 50 == 0:
            print("start: {}, end: {}, now processed: {}"
                  .format(start, end, (idx - start + 1) / (end - start + 1)))
    return X


class DataBase(object):
    def __init__(self, dataname, suffix="", data_dir=None):
        """
        this the parent class for data pre-processing class
        :param dataname: dataname from config
        """
        self.dataname = dataname
        if data_dir is None:
            self.data_dir = os.path.join(config.data_root, dataname)
        check_dir(self.data_dir)
        self.raw_data_dir = os.path.join(config.raw_data_root, dataname)
        check_dir(self.raw_data_dir)
        # this variable is used to store all un-processed data and
        # should not appear in the final buffer.
        self.all_data = {}

        self.class_name = []
        self.class_name_encoding = {}
        # Note: X_train_redundant is not included in X_train, but X_test_redundant is included in X_test.
        self.X = None
        self.embed_X = None
        self.y = None
        self.train_idx = []
        self.train_redundant_idx = []
        self.valid_idx = []
        self.valid_redundant_idx = []
        self.test_idx = []
        self.test_redundant_idx = []
        # additional information can be store here
        self.add_info = {}

        # model related
        self.img_width = 224
        self.img_height = 224
        self.model = None
        self.top_model = None
        self.output_model = None
        self.bottleneck_output = None
        self.last_but_one_output = None
        self.last_output = None

        # data dir
        self.images_dir = os.path.join(self.data_dir,
                                       "images")
        self.saliency_map_dir = os.path.join(self.data_dir,
                                             "saliency-map")
        check_dir(self.images_dir)
        check_dir(self.saliency_map_dir)
        self.feature_dir = os.path.join(self.data_dir,
                                        "feature" + suffix)
        check_dir(self.feature_dir)
        self.model_weight_dir = os.path.join(self.data_dir,
                                             "weights" + suffix)
        check_dir(self.model_weight_dir)
        self.top_model_weights_path = os.path.join(self.model_weight_dir,
                                                   "bottleneck_fc_model.h5")
        self.train_data_dir = os.path.join(self.data_dir,
                                           "bias_train" + suffix)
        check_dir(self.train_data_dir)
        self.valid_data_dir = os.path.join(self.data_dir,
                                           "normal_test")
        check_dir(self.valid_data_dir)
        self.test_data_dir = os.path.join(self.data_dir,
                                          "normal_test")
        check_dir(self.test_data_dir)
        self.epochs = 50
        self.batch_size = 16

    @abstractmethod
    def preprocessing_data(self):
        logger.warn("this function should be overrided and "
                    "you should not see this message.")

    def save_cache(self):
        """
        this function save all data (variables in self.all_data) in "RawData/all_data.pkl"
        !!!this function should not be overrided
        :return:
        """
        logger.warn("begin saving unprocessed cache of {}".format(self.dataname))
        all_data_name = os.path.join(self.data_dir, config.all_data_cache_name)
        pickle_save_data(all_data_name, self.all_data)
        logger.info("cache saving done!")

    def load_data(self, loading_from_buffer=False):
        all_data_name = os.path.join(self.data_dir, config.all_data_cache_name)
        if os.path.exists(all_data_name) and loading_from_buffer:
            logger.warn("all data cache exists. load data from cache ... ...".format(all_data_name))
            self.all_data = pickle_load_data(all_data_name)
            logger.info("cache loading done!")
            return True
        logger.info("all data cache does not exists.")

    @abstractmethod
    def process_data(self):
        logger.warn("this function should be overrided and "
                    "you should not see this message.")

    def postprocess_data(self, weight_name, embedding=True):
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
        for pm in projection_method:
            embedder = Embedder(pm, n_components=2, random_state=123)
            partial_embed_X = embedder.fit_transform(projected_X, None)
            embed_X = np.zeros((len(self.y), 2))
            embed_X[np.array(valided_data_idx), :] = partial_embed_X
            self.all_embed_X[pm] = embed_X
            if pm == default_method:
                self.embed_X = embed_X

    def save_file(self, suffix=""):
        filename = os.path.join(self.data_dir, "processed_data" + suffix + config.pkl_ext)
        # TODO: add time information and warn users when loading
        logger.warn("save processed data in {}".format(filename))
        mat = {}
        # TODO: class_name
        mat[config.class_name] = self.class_name
        mat[config.X_name] = self.X
        mat[config.embed_X_name] = self.embed_X
        mat[config.all_embed_X_name] = self.all_embed_X
        mat[config.y_name] = self.y
        mat[config.train_idx_name] = self.train_idx
        mat[config.train_redundant_idx_name] = self.train_redundant_idx
        mat[config.valid_idx_name] = self.valid_idx
        mat[config.valid_redundant_idx_name] = self.valid_redundant_idx
        mat[config.test_idx_name] = self.test_idx
        mat[config.test_redundant_idx_name] = self.test_redundant_idx
        mat[config.add_info_name] = self.add_info
        pickle_save_data(filename, mat)
        logger.info("saved processed data.")

    # def lowlevel_features(self, method_name="sift"):
    #     feature_dir = os.path.join(self.feature_dir, str(method_name))
    #     check_dir(feature_dir)
    #     dirs = [self.train_data_dir, self.test_data_dir]
    #     filename_list = []
    #     y_list = []
    #     for data_dir in dirs:
    #         datagen = ImageDataGenerator(rescale=1. / 255)
    #         generator = datagen.flow_from_directory(
    #             data_dir,
    #             target_size=(self.img_width, self.img_height),
    #             batch_size=self.batch_size,
    #             class_mode="binary",
    #             shuffle=False
    #         )
    #         y = generator.classes[generator.index_array]
    #         y_list.append(y)
    #         filename = generator.filenames
    #         filename_list.append(filename)
    #     train_filename, test_filename = filename_list
    #     train_y, test_y = y_list
    #     logger.info("train instance: {}".format(len(train_filename)))
    #
    #     train_des = []
    #     train_des_num = []
    #     test_des = []
    #     test_des_num = []
    #     if len(method_name) >=5 and method_name[:5] == "sift-":
    #         n_cluster = int(method_name.split("-")[1])
    #         logger.info("number of clusters is {}".format(n_cluster))
    #         sift = cv2.xfeatures2d.SIFT_create()
    #         if os.path.exists(os.path.join(feature_dir, "raw_sift_feature.pkl")):
    #             logger.info("raw_sift_feature.pkl exists, loading...")
    #             train_des, train_des_num, test_des, test_des_num = \
    #                 pickle_load_data(os.path.join(feature_dir, "raw_sift_feature.pkl"))
    #             logger.info("raw_sift_feature.pkl is loaded.")
    #         else:
    #             for idx, img_name in enumerate(train_filename):
    #                 img = cv2.imread(os.path.join(self.train_data_dir, img_name))
    #                 try:
    #                     _, d = sift.detectAndCompute(img, None)
    #                 except Exception as e:
    #                     print(e)
    #                     d = np.zeros(128).reshape(1,128)
    #                 if d is None:
    #                     d = np.zeros(128).reshape(1,128)
    #                 if d is None:
    #                     d = np.zeros(128).reshape(1,128)
    #                 train_des = train_des + d.tolist()
    #                 train_des_num.append(d.shape[0])
    #             train_des = np.array(train_des)
    #             for idx, img_name in enumerate(test_filename):
    #                 img = cv2.imread(os.path.join(self.test_data_dir, img_name), 0)
    #                 try:
    #                     _, d = sift.detectAndCompute(img, None)
    #                 except Exception as e:
    #                     print(e)
    #                     d = np.zeros(128).reshape(1,128)
    #                 if d is None:
    #                     d = np.zeros(128).reshape(1,128)
    #                 if d is None:
    #                     d = np.zeros(128).reshape(1,128)
    #                 test_des = test_des + d.tolist()
    #                 test_des_num.append(d.shape[0])
    #             logger.info("sift transform finished")
    #             pickle_save_data(os.path.join(feature_dir, "raw_sift_feature.pkl"),
    #                              [train_des, train_des_num, test_des, test_des_num])
    #
    #
    #         if not os.path.exists(os.path.join(feature_dir, "km.pkl")):
    #             logger.info("kmeans model does not exist, training kmeans ......")
    #             km = MiniBatchKMeans(n_clusters= n_cluster, batch_size=5000,
    #                                  max_no_improvement=10, reassignment_ratio=1e-3, verbose=1, random_state=1223)
    #             km.fit(train_des)
    #             pickle_save_data(os.path.join(feature_dir, "km.pkl"), km)
    #         else:
    #             logger.info("kmeans model exists")
    #             km = pickle_load_data(os.path.join(os.path.join(feature_dir, "km.pkl")))
    #
    #         logger.info("des shape: {}".format(train_des.shape))
    #         if len(train_des) == 0:
    #             train_cluster_labels = np.array([])
    #         else:
    #             train_cluster_labels = km.predict(train_des)
    #         if len(test_des) == 0:
    #             test_cluster_labels = np.array([])
    #         else:
    #             test_cluster_labels = km.predict(test_des)
    #         train_labels_lists = list_of_groups(train_cluster_labels.tolist(), train_des_num)
    #         test_labels_lists = list_of_groups(test_cluster_labels.tolist(), test_des_num)
    #
    #         train_X = np.zeros((len(train_labels_lists), n_cluster))
    #         for i in range(len(train_labels_lists)):
    #             bincount = np.bincount(np.array(train_labels_lists[i]))
    #             # bincount = bincount.astype(np.float32) / bincount.sum()
    #             train_X[i, :len(bincount)] = bincount
    #         test_X = np.zeros((len(test_labels_lists), n_cluster))
    #         for i in range(len(test_labels_lists)):
    #             bincount = np.bincount(np.array(test_labels_lists[i]))
    #             # bincount = bincount.astype(np.float32) / bincount.sum()
    #             test_X[i, :len(bincount)] = bincount
    #
    #         if os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0]==0:
    #             logger.info("scaler.pkl exists, loading...")
    #             scaler = pickle_load_data(os.path.join(feature_dir, "scaler.pkl"))
    #             logger.info("scaler.pkl is loaded.")
    #             # train_X = np.array([])
    #         elif not os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0]!=0:
    #             logger.info("scaler.pkl does not exist, training...")
    #             scaler = StandardScaler().fit(train_X)
    #             train_X = scaler.transform(train_X)
    #             pickle_save_data(os.path.join(feature_dir, "scaler.pkl"), scaler)
    #             logger.info("scaler is trained and saved in {}".format(feature_dir))
    #         else:
    #             logger.error("training examples do not exist, but are used to train a scaler.")
    #             exit(0)
    #         test_X = scaler.transform(test_X)
    #
    #     elif method_name == "sift":
    #         sift = cv2.xfeatures2d.SIFT_create()
    #         for idx, img_name in enumerate(train_filename):
    #             img = cv2.imread(os.path.join(self.train_data_dir, img_name))
    #             _, d = sift.detectAndCompute(img, None)
    #             train_des = train_des + d.tolist()
    #             train_des_num.append(d.shape[0])
    #         train_des = np.array(train_des)
    #         for idx, img_name in enumerate(test_filename):
    #             img = cv2.imread(os.path.join(self.test_data_dir, img_name), 0)
    #             _, d = sift.detectAndCompute(img, None)
    #             test_des = test_des + d.tolist()
    #             test_des_num.append(d.shape[0])
    #         logger.info("sift transform finished")
    #
    #         if not os.path.exists(os.path.join(feature_dir, "km.pkl")):
    #             logger.info("kmeans model does not exist, training kmeans ......")
    #             km = MiniBatchKMeans(n_clusters=1000, batch_size=5000,
    #                                  max_no_improvement=10, reassignment_ratio=1e-3, verbose=1)
    #             km.fit(train_des)
    #             pickle_save_data(os.path.join(feature_dir, "km.pkl"), km)
    #         else:
    #             logger.info("kmeans model exists")
    #             km = pickle_load_data(os.path.join(self.raw_data_dir, "km_" + method_name + ".pkl"))
    #
    #         logger.info("des shape: {}".format(train_des.shape))
    #         train_cluster_labels = km.predict(train_des)
    #         test_cluster_labels = km.predict(test_des)
    #         train_labels_lists = list_of_groups(train_cluster_labels.tolist(), train_des_num)
    #         test_labels_lists = list_of_groups(test_cluster_labels.tolist(), test_des_num)
    #
    #         train_X = np.zeros((len(train_labels_lists), 1000))
    #         for i in range(len(train_labels_lists)):
    #             bincount = np.bincount(np.array(train_labels_lists[i]))
    #             # bincount = bincount.astype(np.float32) / bincount.sum()
    #             train_X[i, :len(bincount)] = bincount
    #         test_X = np.zeros((len(test_labels_lists), 1000))
    #         for i in range(len(test_labels_lists)):
    #             bincount = np.bincount(np.array(test_labels_lists[i]))
    #             # bincount = bincount.astype(np.float32) / bincount.sum()
    #             test_X[i, :len(bincount)] = bincount
    #         scaler = StandardScaler().fit(train_X)
    #         train_X = scaler.transform(train_X)
    #         test_X = scaler.transform(test_X)
    #
    #     elif len(method_name) >=11 and method_name[:11] == "superpixel-":
    #         n_cluster = int(method_name.split("-")[1])
    #         logger.info("number of clusters is {}".format(n_cluster))
    #         train_pkl = os.path.join(feature_dir, "bow_train_" + str(n_cluster) + ".pkl")
    #         test_pkl = os.path.join(feature_dir, "bow_test_" + str(n_cluster) + ".pkl")
    #         train_feature = pickle_load_data(train_pkl)
    #         test_feature = pickle_load_data(test_pkl)
    #         train_X = []
    #         for name in train_filename:
    #             name = name.replace("\\","/").strip(".jpg")
    #             name = name.split("/")
    #             name = name[0] + "." + name[1]
    #             train_X.append(train_feature[name])
    #             a = 1
    #         train_X = np.array(train_X)
    #         test_X = []
    #         for name in test_filename:
    #             name = name.replace("\\","/").strip(".jpg")
    #             name = name.split("/")
    #             name = name[0] + "." + name[1]
    #             test_X.append(test_feature[name])
    #         test_X = np.array(test_X)
    #
    #     elif method_name == "HOG":
    #         for idx, img_name in enumerate(train_filename):
    #             img = Image.open(os.path.join(self.train_data_dir, img_name))
    #             img = img.resize((128, 128))
    #             img_data = color.rgb2gray(np.array(img))
    #             try:
    #                 fd, hog_image = hog(img_data, orientations=8,
    #                                     pixels_per_cell=(16, 16),
    #                                     cells_per_block=(4, 4),
    #                                     block_norm='L2', visualise=True)
    #             except Exception as e:
    #                 print(e)
    #                 import IPython; IPython.embed()
    #                 exit()
    #             if idx % 100 == 0:
    #                 print(idx, end=", ")
    #             train_des.append(fd)
    #
    #         for idx, img_name in enumerate(test_filename):
    #             img = Image.open(os.path.join(self.test_data_dir, img_name))
    #             img = img.resize((128, 128))
    #             img_data = color.rgb2gray(np.array(img))
    #             try:
    #                 fd, hog_image = hog(img_data, orientations=8,
    #                                     pixels_per_cell=(16, 16),
    #                                     cells_per_block=(4, 4),
    #                                     block_norm='L2', visualise=True)
    #             except Exception as e:
    #                 print(e)
    #                 import IPython; IPython.embed()
    #                 exit()
    #             if idx % 100 == 0:
    #                 print(idx, end=", ")
    #             test_des.append(fd)
    #
    #         train_X = np.array(train_des)
    #         test_X = np.array(test_des)
    #
    #     elif len(method_name)>=11 and method_name[:11] == "HOG-kmeans-":
    #         n_cluster = int(method_name.split("-")[2])
    #         logger.info("number of clusters is {}".format(n_cluster))
    #         if os.path.exists(os.path.join(feature_dir, "raw_sift_feature.pkl")):
    #             logger.info("raw_sift_feature.pkl exists, loading.")
    #             train_des, train_des_num, test_des, test_des_num = \
    #                 pickle_load_data(os.path.join(feature_dir, "raw_sift_feature.pkl"))
    #         else:
    #             for idx, img_name in enumerate(train_filename):
    #                 img = Image.open(os.path.join(self.train_data_dir, img_name))
    #                 if min(img.size) < 64:
    #                     img = img.resize((512, 512))
    #                     logger.warn("new img size: {}".format(img.size))
    #                 img_data = color.rgb2gray(np.array(img))
    #                 try:
    #                     fd = hog(img_data, orientations=8,
    #                                         pixels_per_cell=(16, 16),
    #                                         cells_per_block=(4, 4),
    #                                         block_norm='L2', feature_vector=False)
    #                     fd = fd.reshape(fd.shape[0]*fd.shape[1], -1)
    #                 except Exception as e:
    #                     print(e)
    #                     import IPython; IPython.embed()
    #                     exit()
    #                 if (idx+1) % 100 == 0:
    #                     print(idx, end=", ")
    #                 train_des = train_des + fd.tolist()
    #                 train_des_num.append(fd.shape[0])
    #             train_des = np.array(train_des)
    #             for idx, img_name in enumerate(test_filename):
    #                 img = Image.open(os.path.join(self.test_data_dir, img_name))
    #                 if min(img.size) < 64:
    #                     img = img.resize((512, 512))
    #                     logger.warn("new img size: {}".format(img.size))
    #                 try:
    #                     img_data = color.rgb2gray(np.array(img))
    #                 except Exception as e:
    #                     logger.info(e)
    #                     print(img_name)
    #                     img_data = np.array(img).mean(axis=2).astype(np.uint8)
    #                 try:
    #                     fd = hog(img_data, orientations=8,
    #                                         pixels_per_cell=(16, 16),
    #                                         cells_per_block=(4, 4),
    #                                         block_norm='L2', feature_vector=False)
    #                     fd = fd.reshape(fd.shape[0]*fd.shape[1], -1)
    #                 except Exception as e:
    #                     print(e)
    #                     import IPython; IPython.embed()
    #                     exit()
    #                 if (idx+1) % 100 == 0:
    #                     print(idx, end=", ")
    #                 test_des = test_des + fd.tolist()
    #                 test_des_num.append(fd.shape[0])
    #             test_des = np.array(test_des)
    #             logger.info("HOG transform finished")
    #             pickle_save_data(os.path.join(feature_dir, "raw_sift_feature.pkl"),
    #                              [train_des, train_des_num, test_des, test_des_num])
    #
    #         if not os.path.exists(os.path.join(feature_dir, "km.pkl")):
    #             logger.info("kmeans model does not exist, training kmeans ......")
    #             km = MiniBatchKMeans(n_clusters=1000, batch_size=5000,
    #                                  max_no_improvement=10, reassignment_ratio=1e-3, verbose=1)
    #             km.fit(train_des)
    #             pickle_save_data(os.path.join(feature_dir, "km.pkl"), km)
    #         else:
    #             logger.info("kmeans model exists")
    #             km = pickle_load_data(os.path.join(feature_dir, "km.pkl"))
    #
    #         logger.info("des shape: {}".format(train_des.shape))
    #         if len(train_des) == 0:
    #             train_cluster_labels = np.array([])
    #         else:
    #             train_cluster_labels = km.predict(train_des)
    #         if len(test_des) == 0:
    #             test_cluster_labels = np.array([])
    #         else:
    #             test_cluster_labels = km.predict(test_des)
    #         train_labels_lists = list_of_groups(train_cluster_labels.tolist(), train_des_num)
    #         test_labels_lists = list_of_groups(test_cluster_labels.tolist(), test_des_num)
    #
    #         train_X = np.zeros((len(train_labels_lists), 1000))
    #         for i in range(len(train_labels_lists)):
    #             bincount = np.bincount(np.array(train_labels_lists[i]))
    #             # bincount = bincount.astype(np.float32) / bincount.sum()
    #             train_X[i, :len(bincount)] = bincount
    #         test_X = np.zeros((len(test_labels_lists), 1000))
    #         for i in range(len(test_labels_lists)):
    #             bincount = np.bincount(np.array(test_labels_lists[i]))
    #             # bincount = bincount.astype(np.float32) / bincount.sum()
    #             test_X[i, :len(bincount)] = bincount
    #         if os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0]==0:
    #             logger.info("scaler.pkl exists, loading...")
    #             scaler = pickle_load_data(os.path.join(feature_dir, "scaler.pkl"))
    #             logger.info("scaler.pkl is loaded.")
    #             # train_X = np.array([])
    #         elif train_X.shape[0]!=0:
    #             logger.info("scaler.pkl does not exist, training...")
    #             scaler = StandardScaler().fit(train_X)
    #             train_X = scaler.transform(train_X)
    #             pickle_save_data(os.path.join(feature_dir, "scaler.pkl"), scaler)
    #             logger.info("scaler is trained and saved in {}".format(feature_dir))
    #         else:
    #             logger.error("training examples do not exist, but are used to train a scaler.")
    #             exit(0)
    #         test_X = scaler.transform(test_X)
    #
    #     elif method_name == "LBP":
    #         METHOD="uniform"
    #         radius = 3
    #         n_points = 8 * radius
    #         for idx, img_name in enumerate(train_filename):
    #             img = Image.open(os.path.join(self.train_data_dir, img_name))
    #             img = img.resize((256, 256))
    #             try:
    #                 img_data = color.rgb2gray(np.array(img))
    #             except:
    #                 img_data = np.array(img).mean(axis=2).astype(np.uint8)
    #             fd = local_binary_pattern(img_data, n_points, radius, METHOD)
    #             train_des.append(fd)
    #
    #         for idx, img_name in enumerate(test_filename):
    #             img = Image.open(os.path.join(self.test_data_dir, img_name))
    #             img = img.resize((256, 256))
    #             try:
    #                 img_data = color.rgb2gray(np.array(img))
    #             except:
    #                 img_data = np.array(img).mean(axis=2).astype(np.uint8)
    #             fd = local_binary_pattern(img_data, n_points, radius, METHOD)
    #             test_des.append(fd)
    #
    #         train_X = np.array(train_des)
    #         test_X = np.array(test_des)
    #
    #     elif method_name == "LBP-hist":
    #         METHOD="uniform"
    #         radius = 3
    #         n_points = 8 * radius
    #         if os.path.exists(os.path.join(feature_dir, "raw_sift_feature.pkl")):
    #             logger.info("raw_sift_feature.pkl exists, loading.")
    #             train_des, test_des = \
    #                 pickle_load_data(os.path.join(feature_dir, "raw_sift_feature.pkl"))
    #         else:
    #             for idx, img_name in enumerate(train_filename):
    #                 img = Image.open(os.path.join(self.train_data_dir, img_name))
    #                 try:
    #                     img_data = color.rgb2gray(np.array(img))
    #                 except:
    #                     img_data = np.array(img).mean(axis=2).astype(np.uint8)
    #                 fd = local_binary_pattern(img_data, n_points, radius, METHOD)
    #                 train_des.append(fd.reshape(-1).astype(int))
    #                 # if idx > 100:
    #                 #     break
    #
    #             for idx, img_name in enumerate(test_filename):
    #                 img = Image.open(os.path.join(self.test_data_dir, img_name))
    #                 try:
    #                     img_data = color.rgb2gray(np.array(img))
    #                 except:
    #                     img_data = np.array(img).mean(axis=2).astype(np.uint8)
    #                 fd = local_binary_pattern(img_data, n_points, radius, METHOD)
    #                 test_des.append(fd.reshape(-1).astype(int))
    #                 # if idx > 100:
    #                 #     break
    #             pickle_save_data(os.path.join(feature_dir, "raw_sift_feature.pkl"),
    #                              [train_des, test_des])
    #
    #         train_X = []
    #         for i in range(len(train_des)):
    #             bincount = np.bincount(np.array(train_des[i]))
    #             # bincount = bincount.astype(np.float32) / bincount.sum()
    #             train_X.append(bincount)
    #         test_X = []
    #         for i in range(len(test_des)):
    #             bincount = np.bincount(np.array(test_des[i]))
    #             # bincount = bincount.astype(np.float32) / bincount.sum()
    #             test_X.append(bincount)
    #         train_X = np.array(train_X)
    #         test_X = np.array(test_X)
    #         if os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0]==0:
    #             logger.info("scaler.pkl exists, loading...")
    #             scaler = pickle_load_data(os.path.join(feature_dir, "scaler.pkl"))
    #             logger.info("scaler.pkl is loaded.")
    #             # train_X = np.array([])
    #         elif train_X.shape[0]!=0:
    #             logger.info("scaler.pkl does not exist, training...")
    #             scaler = StandardScaler().fit(train_X)
    #             train_X = scaler.transform(train_X)
    #             pickle_save_data(os.path.join(feature_dir, "scaler.pkl"), scaler)
    #             logger.info("scaler is trained and saved in {}".format(feature_dir))
    #         else:
    #             logger.error("training examples do not exist, but are used to train a scaler.")
    #             exit(0)
    #         test_X = [i.tolist() for i in test_X]
    #         tmp = []
    #         for i in test_X:
    #             if len(i) < 26:
    #                 for j in range(26-len(i)):
    #                     i.append(0)
    #             tmp.append(i)
    #         test_X = np.array(test_X)
    #         test_X = scaler.transform(test_X)
    #
    #
    #     else:
    #         raise ValueError("{} method is not support now".format(method_name))
    #
    #
    #     train_mat = {
    #         "features": [train_X, 0],
    #         "y": train_y,
    #         "filename": train_filename
    #     }
    #     test_mat = {
    #         "features": [test_X, 0],
    #         "y": test_y,
    #         "filename": test_filename
    #     }
    #     check_dir(feature_dir)
    #     pickle_save_data(os.path.join(feature_dir, "bias_train.pkl"), train_mat)
    #     pickle_save_data(os.path.join(feature_dir, "normal_test.pkl"), test_mat)
    #     logger.info("process finished")

    def sift_features(self, method_name):
        feature_dir = os.path.join(self.feature_dir, str(method_name))
        descriptor_name = method_name.split("-")[0]
        descriptor = None
        if descriptor_name == "sift":
            descriptor = sift_descriptor
        elif descriptor_name == "orb":
            descriptor = orb_descriptor
        elif descriptor_name == "brief":
            descriptor = brief_descriptor
        elif descriptor_name == "surf":
            descriptor = surf_descriptor
        else:
            raise ValueError("not supported method")
        n_cluster = int(method_name.split("-")[1])
        check_dir(feature_dir)
        dirs = [self.train_data_dir, self.test_data_dir]
        filename_list = []
        y_list = []
        for data_dir in dirs:
            datagen = ImageDataGenerator(rescale=1. / 255)
            generator = datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                class_mode="binary",
                shuffle=False
            )
            y = generator.classes[generator.index_array]
            y_list.append(y)
            filename = generator.filenames
            filename_list.append(filename)
        train_filename, test_filename = filename_list
        train_y, test_y = y_list
        logger.info("train instance: {}".format(len(train_filename)))


        cpu_kernel = 40
        train_des = []
        train_des_num = []
        test_des = []
        test_des_num = []
        logger.info("number of clusters is {}".format(n_cluster))
        sift = cv2.xfeatures2d.SIFT_create()
        if os.path.exists(os.path.join(feature_dir, "train_raw_feature.pkl")):
            logger.info("train_raw_sift_feature.pkl exists, loading...")
            train_des, train_des_num = pickle_load_data(
                os.path.join(feature_dir, "train_raw_feature.pkl")
            )
        else:
            t1 = time()
            step_size = math.ceil(len(train_filename) / cpu_kernel)
            print("step_size: ", step_size)
            start_ends = []
            train_des_res = [None for i in range(cpu_kernel)]
            train_des_num_res = [None for i in range(cpu_kernel)]
            for i in range(cpu_kernel):
                start_ends.append([i*step_size,
                                   min((i+1)*step_size, len(train_filename))])
            pool = Pool()
            res = [pool.apply_async(_sift_train,
                                    (start, end, train_filename, self.train_data_dir, descriptor))
                   for start, end in start_ends]
            for idx, r in enumerate(res):
                train_des_res[idx], train_des_num_res[idx] = \
                    r.get()
            print("extraction finished: ", time() - t1)
            for l in train_des_res:
                train_des.extend(l)
            for l in train_des_num_res:
                train_des_num.extend(l)
            print("aggregation finished: ", time() - t1)
            pickle_save_data(os.path.join(feature_dir, "train_raw_feature.pkl"),
                            [train_des, train_des_num])
            print("save finished: ", time() - t1)

        # import IPython; IPython.embed()
        if not os.path.exists(os.path.join(feature_dir, "km.pkl")):
            logger.info("kmeans model does not exist, training kmeans ......")
            km = MiniBatchKMeans(n_clusters=n_cluster, batch_size=5000,
                                 max_no_improvement=10, reassignment_ratio=1e-3, verbose=1, random_state=1223)
            km.fit(train_des)
            pickle_save_data(os.path.join(feature_dir, "km.pkl"), km)
        else:
            logger.info("kmeans model exists")
            km = pickle_load_data(os.path.join(os.path.join(feature_dir, "km.pkl")))
        print("train_filename", len(train_filename))
        km_res = []
        for filename, data_dir in zip([train_filename, test_filename], [self.train_data_dir, self.test_data_dir]):
            t1 = time()
            step_size = math.ceil(len(filename) / cpu_kernel)
            print("step_size: ", step_size)
            start_ends = []
            cluster_labels = []
            des_num = []
            cluster_labels_res = [None for i in range(cpu_kernel)]
            des_num_res = [None for i in range(cpu_kernel)]
            for i in range(cpu_kernel):
                start_ends.append([i * step_size,
                                   min((i + 1) * step_size, len(filename))])
            pool = Pool()
            res = [pool.apply_async(_sift_prediction,
                                    (start, end, filename, data_dir, km, descriptor))
                   for start, end in start_ends]
            for idx, r in enumerate(res):
                cluster_labels_res[idx], des_num_res[idx] = \
                    r.get()
            print("extraction finished: ", time() - t1)
            for l in cluster_labels_res:
                cluster_labels.extend(l)
            for l in des_num_res:
                des_num.extend(l)
            print("aggregation finished: ", time() - t1)
            km_res.append([cluster_labels, des_num])

        train_labels_lists = list_of_groups(km_res[0][0], km_res[0][1])
        test_labels_lists = list_of_groups(km_res[1][0], km_res[1][1])

        train_X = np.zeros((len(train_labels_lists), n_cluster))
        for i in range(len(train_labels_lists)):
            bincount = np.bincount(np.array(train_labels_lists[i]))
            # bincount = bincount.astype(np.float32) / bincount.sum()
            train_X[i, :len(bincount)] = bincount
        test_X = np.zeros((len(test_labels_lists), n_cluster))
        for i in range(len(test_labels_lists)):
            bincount = np.bincount(np.array(test_labels_lists[i]))
            # bincount = bincount.astype(np.float32) / bincount.sum()
            test_X[i, :len(bincount)] = bincount

        if os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0] == 0:
            logger.info("scaler.pkl exists, loading...")
            scaler = pickle_load_data(os.path.join(feature_dir, "scaler.pkl"))
            logger.info("scaler.pkl is loaded.")
            # train_X = np.array([])
        elif not os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0] != 0:
            logger.info("scaler.pkl does not exist, training...")
            scaler = StandardScaler().fit(train_X)
            train_X = scaler.transform(train_X)
            pickle_save_data(os.path.join(feature_dir, "scaler.pkl"), scaler)
            logger.info("scaler is trained and saved in {}".format(feature_dir))
        else:
            logger.error("training examples do not exist, but are used to train a scaler.")
            exit(0)
        test_X = scaler.transform(test_X)
        print("bin finished: ", time() - t1)

        train_mat = {
            "features": [train_X, 0],
            "y": train_y,
            "filename": train_filename
        }
        test_mat = {
            "features": [test_X, 0],
            "y": test_y,
            "filename": test_filename
        }
        check_dir(feature_dir)
        pickle_save_data(os.path.join(feature_dir, "bias_train.pkl"), train_mat)
        pickle_save_data(os.path.join(feature_dir, "normal_test.pkl"), test_mat)
        logger.info("process finished")

    def HOG_features(self, method_name):
        feature_dir = os.path.join(self.feature_dir, str(method_name))
        n_cluster = int(method_name.split("-")[1])
        check_dir(feature_dir)
        dirs = [self.train_data_dir, self.test_data_dir]
        filename_list = []
        y_list = []
        for data_dir in dirs:
            datagen = ImageDataGenerator(rescale=1. / 255)
            generator = datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                class_mode="binary",
                shuffle=False
            )
            y = generator.classes[generator.index_array]
            y_list.append(y)
            filename = generator.filenames
            filename_list.append(filename)
        train_filename, test_filename = filename_list
        train_y, test_y = y_list
        logger.info("train instance: {}".format(len(train_filename)))


        cpu_kernel = 40
        train_des = []
        train_des_num = []
        test_des = []
        test_des_num = []
        logger.info("number of clusters is {}".format(n_cluster))
        if os.path.exists(os.path.join(feature_dir, "train_raw_feature.pkl")):
            logger.info("train_raw_sift_feature.pkl exists, loading...")
            train_des, train_des_num = pickle_load_data(
                os.path.join(feature_dir, "train_raw_feature.pkl")
            )
        else:
            t1 = time()
            step_size = math.ceil(len(train_filename) / cpu_kernel)
            print("step_size: ", step_size)
            start_ends = []
            train_des_res = [None for i in range(cpu_kernel)]
            train_des_num_res = [None for i in range(cpu_kernel)]
            for i in range(cpu_kernel):
                start_ends.append([i*step_size,
                                   min((i+1)*step_size, len(train_filename))])
            pool = Pool()
            res = [pool.apply_async(_hog_train,
                                    (start, end, train_filename, self.train_data_dir))
                   for start, end in start_ends]
            for idx, r in enumerate(res):
                train_des_res[idx], train_des_num_res[idx] = \
                    r.get()
            print("extraction finished: ", time() - t1)
            for l in train_des_res:
                train_des.extend(l)
            for l in train_des_num_res:
                train_des_num.extend(l)
            print("aggregation finished: ", time() - t1)
            pickle_save_data(os.path.join(feature_dir, "train_raw_feature.pkl"),
                            [train_des, train_des_num])
            print("save finished: ", time() - t1)

        if not os.path.exists(os.path.join(feature_dir, "km.pkl")):
            logger.info("kmeans model does not exist, training kmeans ......")
            km = MiniBatchKMeans(n_clusters=n_cluster, batch_size=5000,
                                 max_no_improvement=10, reassignment_ratio=1e-3, verbose=1, random_state=1223)
            km.fit(train_des)
            pickle_save_data(os.path.join(feature_dir, "km.pkl"), km)
        else:
            logger.info("kmeans model exists")
            km = pickle_load_data(os.path.join(os.path.join(feature_dir, "km.pkl")))

        km_res = []
        for filename, data_dir in zip([train_filename, test_filename], [self.train_data_dir, self.test_data_dir]):
            t1 = time()
            step_size = math.ceil(len(filename) / cpu_kernel)
            print("step_size: ", step_size)
            start_ends = []
            cluster_labels = []
            des_num = []
            cluster_labels_res = [None for i in range(cpu_kernel)]
            des_num_res = [None for i in range(cpu_kernel)]
            for i in range(cpu_kernel):
                start_ends.append([i * step_size,
                                   min((i + 1) * step_size, len(filename))])
            pool = Pool()
            res = [pool.apply_async(_hog_prediction,
                                    (start, end, filename, data_dir, km))
                   for start, end in start_ends]
            for idx, r in enumerate(res):
                cluster_labels_res[idx], des_num_res[idx] = \
                    r.get()
            print("extraction finished: ", time() - t1)
            for l in cluster_labels_res:
                cluster_labels.extend(l)
            for l in des_num_res:
                des_num.extend(l)
            print("aggregation finished: ", time() - t1)
            km_res.append([cluster_labels, des_num])

        train_labels_lists = list_of_groups(km_res[0][0], km_res[0][1])
        test_labels_lists = list_of_groups(km_res[1][0], km_res[1][1])

        train_X = np.zeros((len(train_labels_lists), n_cluster))
        for i in range(len(train_labels_lists)):
            bincount = np.bincount(np.array(train_labels_lists[i]))
            # bincount = bincount.astype(np.float32) / bincount.sum()
            train_X[i, :len(bincount)] = bincount
        test_X = np.zeros((len(test_labels_lists), n_cluster))
        for i in range(len(test_labels_lists)):
            bincount = np.bincount(np.array(test_labels_lists[i]))
            # bincount = bincount.astype(np.float32) / bincount.sum()
            test_X[i, :len(bincount)] = bincount

        if os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0] == 0:
            logger.info("scaler.pkl exists, loading...")
            scaler = pickle_load_data(os.path.join(feature_dir, "scaler.pkl"))
            logger.info("scaler.pkl is loaded.")
            # train_X = np.array([])
        elif not os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0] != 0:
            logger.info("scaler.pkl does not exist, training...")
            scaler = StandardScaler().fit(train_X)
            train_X = scaler.transform(train_X)
            pickle_save_data(os.path.join(feature_dir, "scaler.pkl"), scaler)
            logger.info("scaler is trained and saved in {}".format(feature_dir))
        else:
            logger.error("training examples do not exist, but are used to train a scaler.")
            exit(0)
        test_X = scaler.transform(test_X)
        print("bin finished: ", time() - t1)

        train_mat = {
            "features": [train_X, 0],
            "y": train_y,
            "filename": train_filename
        }
        test_mat = {
            "features": [test_X, 0],
            "y": test_y,
            "filename": test_filename
        }
        check_dir(feature_dir)
        pickle_save_data(os.path.join(feature_dir, "bias_train.pkl"), train_mat)
        pickle_save_data(os.path.join(feature_dir, "normal_test.pkl"), test_mat)
        logger.info("process finished")

    def LBP_features(self, method_name):
        feature_dir = os.path.join(self.feature_dir, str(method_name))
        check_dir(feature_dir)
        dirs = [self.train_data_dir, self.test_data_dir]
        filename_list = []
        y_list = []
        for data_dir in dirs:
            datagen = ImageDataGenerator(rescale=1. / 255)
            generator = datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                class_mode="binary",
                shuffle=False
            )
            y = generator.classes[generator.index_array]
            y_list.append(y)
            filename = generator.filenames
            filename_list.append(filename)
        train_filename, test_filename = filename_list
        train_y, test_y = y_list
        logger.info("train instance: {}".format(len(train_filename)))

        cpu_kernel = 40
        km_res = []
        for filename, data_dir in zip([train_filename, test_filename], [self.train_data_dir, self.test_data_dir]):
            t1 = time()
            step_size = math.ceil(len(filename) / cpu_kernel)
            print("step_size: ", step_size)
            start_ends = []
            X = []
            X_res = [None for i in range(cpu_kernel)]
            for i in range(cpu_kernel):
                start_ends.append([i * step_size,
                                   min((i + 1) * step_size, len(filename))])
            pool = Pool()
            res = [pool.apply_async(_lbp_prediction,
                                    (start, end, filename, data_dir))
                   for start, end in start_ends]
            for idx, r in enumerate(res):
                X_res[idx] = \
                    r.get()
            print("extraction finished: ", time() - t1)
            for l in X_res:
                X.extend(l)
            print("aggregation finished: ", time() - t1)
            km_res.append(X)

        train_X = km_res[0]
        tmp_test_X = km_res[1]
        test_X = []
        train_X = np.array(train_X)
        for x in tmp_test_X:
            if len(x) < 26:
                tmp_x = np.zeros(26)
                tmp_x[:len(x)] = x
                x = tmp_x
            test_X.append(x)
        test_X = np.array(test_X)
        # import IPython; IPython.embed()
        if os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0] == 0:
            logger.info("scaler.pkl exists, loading...")
            scaler = pickle_load_data(os.path.join(feature_dir, "scaler.pkl"))
            logger.info("scaler.pkl is loaded.")
            # train_X = np.array([])
        elif not os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0] != 0:
            logger.info("scaler.pkl does not exist, training...")
            scaler = StandardScaler().fit(train_X)
            train_X = scaler.transform(train_X)
            pickle_save_data(os.path.join(feature_dir, "scaler.pkl"), scaler)
            logger.info("scaler is trained and saved in {}".format(feature_dir))
        else:
            logger.error("training examples do not exist, but are used to train a scaler.")
            exit(0)
        test_X = scaler.transform(test_X)
        print("bin finished: ", time() - t1)

        train_mat = {
            "features": [train_X, 0],
            "y": train_y,
            "filename": train_filename
        }
        test_mat = {
            "features": [test_X, 0],
            "y": test_y,
            "filename": test_filename
        }
        check_dir(feature_dir)
        pickle_save_data(os.path.join(feature_dir, "bias_train.pkl"), train_mat)
        pickle_save_data(os.path.join(feature_dir, "normal_test.pkl"), test_mat)
        logger.info("process finished")

    def pretrain_get_model(self, model_name="vgg", weight="imagenet"):
        input_shape = (self.img_width, self.img_height, 3)
        logger.warn("the model input is {}".format(input_shape))
        if model_name == "vgg":
            model = applications.VGG16(include_top=True, input_shape=input_shape, weights=weight)
            model = Model(inputs=model.input, outputs=[model.get_layer("fc2").output,
                                                       model.get_layer("predictions").output])
        elif model_name == "resnet50":
            model = applications.ResNet50(include_top=True, input_shape=input_shape, weights=weight)
            model = Model(inputs=model.input, outputs=[model.get_layer("avg_pool").output,
                                                       model.get_layer("fc1000").output])
        elif model_name == "xception":
            model = applications.Xception(include_top=True, input_shape=input_shape, weights=weight)
            model = Model(inputs=model.input, outputs=[model.get_layer("avg_pool").output,
                                                       model.get_layer("predictions").output])
        elif model_name == "inceptionv3":
            model = applications.InceptionV3(include_top=True, input_shape=input_shape, weights=weight)
            model = Model(inputs=model.input, outputs=[model.get_layer("avg_pool").output,
                                                       model.get_layer("predictions").output])
        elif model_name == "inceptionresnet":
            model = applications.InceptionResNetV2(include_top=True, input_shape=input_shape, weights=weight)
            model = Model(inputs=model.input, outputs=[model.get_layer("avg_pool").output,
                                                       model.get_layer("predictions").output])
        elif model_name == "mobilenet":
            model = applications.MobileNet(include_top=True, input_shape=input_shape, weights=weight)
            model = Model(inputs=model.input, outputs=[model.get_layer("global_average_pooling2d_1").output,
                                                       model.get_layer("reshape_2").output])
        else:
            raise ValueError("{} model is not support now".format(model_name))
        feature_model = model
        feature_dir = os.path.join(self.feature_dir, str(model_name) + "_" + str(weight))
        check_dir(feature_dir)
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        for data_dir in dirs:
            datagen = ImageDataGenerator(rescale=1. / 255)
            generator = datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                class_mode="binary",
                shuffle=False
            )
            y = generator.classes[generator.index_array]
            filename = generator.filenames
            nb_samples = y.reshape(-1).shape[0]
            if nb_samples == 0:
                continue
            file_prefix = os.path.split(data_dir)[1].split(".")[0]
            logger.info("{} sampling number: {}".format(file_prefix, nb_samples))
            features = feature_model.predict_generator(
                generator, math.ceil(nb_samples / self.batch_size))
            logger.info("{} feature shape: {}".format(file_prefix, features[0].shape))
            mat = {
                "features": features,
                "y": y,
                "filename": filename
            }
            pickle_save_data(os.path.join(feature_dir, file_prefix + config.pkl_ext), mat)

    def ft_get_model(self):
        """
        create the model and according outputs
        :return: None
        """
        if self.model is not None:
            # TODO: anything else
            return
        input_shape = (self.img_width, self.img_height, 3)
        logger.warn("the model input is {}".format(input_shape))
        model = applications.VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
        inputs = Input(shape=input_shape)
        tmp_x = inputs
        for layer in model.layers[:-1]:
            tmp_x = layer(tmp_x)
        bottleneck_output = tmp_x
        bottleneck_output_shape = bottleneck_output.shape.as_list()
        a = Input(shape=bottleneck_output_shape[1:])
        b = Conv2D(1024, (3, 3), activation="relu", padding="same")(a)
        last_but_one_output = GlobalAveragePooling2D()(b)
        top_outputs = Dense(1, activation="sigmoid")(last_but_one_output)

        ##############################################################
        self.top_model = Model(inputs=a, outputs=top_outputs)  #######
        ##############################################################

        outputs = self.top_model(bottleneck_output)

        ############################################################
        self.model = Model(inputs=inputs, outputs=outputs)  ########
        ############################################################

        tmp_model = Model(inputs=a, outputs=[last_but_one_output, b, top_outputs])
        last_but_one_output, b, outputs = tmp_model(bottleneck_output)


        #######################################################################################
        self.output_model = Model(inputs=inputs,  ##############################################
                                  outputs=[bottleneck_output, last_but_one_output, outputs])  ###
        #######################################################################################

        self.saliency_map_model = Model(inputs=inputs,
                                        outputs=[b])

    def ft_save_bottleneck(self):
        self.ft_get_model()
        model = self.output_model

        dirs = [self.train_data_dir, self.valid_data_dir]
        for data_dir in dirs:
            datagen = ImageDataGenerator(rescale=1. / 255)
            generator = datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                class_mode="binary",
                shuffle=False
            )
            file_prefix = os.path.split(data_dir)[1].split(".")[0]
            y = generator.classes[generator.index_array]
            nb_samples = y.reshape(-1).shape[0]
            logger.info("{}, total instances: {}".format(file_prefix, nb_samples))
            features = model.predict_generator(generator, math.ceil(nb_samples / self.batch_size))
            bottleneck_output = features[0]
            mat = {
                "features": bottleneck_output,
                "y": y
            }
            pickle_save_data(os.path.join(self.feature_dir, file_prefix +
                                  "_bottleneck.pkl"), mat)
            logger.info("{} bottleneck file saved.".format(file_prefix))

    def ft_train_top_model(self):
        self.ft_get_model()
        dirs = [self.train_data_dir, self.valid_data_dir]

        # training top model
        model = self.top_model
        # get training data
        file_prefix = os.path.split(self.train_data_dir)[1].split(".")[0]
        data = pickle_load_data(os.path.join(self.feature_dir, file_prefix +
                                    "_bottleneck.pkl"))
        train_data = data["features"]
        train_labels = data["y"].reshape(-1, 1)
        logger.info("top model training: input shape: {}".format(train_data.shape))
        file_prefix = os.path.split(self.valid_data_dir)[1].split(".")[0]
        data = pickle_load_data(os.path.join(self.feature_dir, file_prefix +
                                    "_bottleneck.pkl"))
        valid_data = data["features"]
        valid_labels = data["y"].reshape(-1, 1)
        logger.info("top model training: input shape: {}".format(valid_data.shape))

        # shuffle
        np.random.seed(123)
        train_index = np.random.permutation(train_labels.shape[0])
        valid_index = np.random.permutation(valid_labels.shape[0])
        train_data = train_data[train_index, :]
        train_labels = train_labels[train_index, :]
        valid_data = valid_data[valid_index, :]
        valid_labels = valid_labels[valid_index, :]

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=2e-4, momentum=0.9),
                      # optimizer = "rmsprop",
                      metrics=['accuracy'])

        checkpointer = ModelCheckpoint(filepath=os.path.join(self.model_weight_dir,
                                                             "top_weights.{epoch:02d}-{val_acc:.4f}.h5"),
                                       verbose=1)
        model.fit(train_data, train_labels,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  validation_data=(valid_data, valid_labels),
                  callbacks=[checkpointer])
        model.save_weights(self.top_model_weights_path)

        # test: the following codes are designed for debugging
        # model = self.output_model
        # self.top_model.load_weights(self.top_model_weights_path)
        # datagen = ImageDataGenerator(rescale=1. / 255)
        # generator = datagen.flow_from_directory(
        #     self.valid_data_dir,
        #     target_size=(self.img_width, self.img_height),
        #     batch_size=self.batch_size,
        #     class_mode="binary",
        #     shuffle=False
        # )
        # y = generator.classes[generator.index_array]
        # filename = generator.filenames
        # nb_samples = y.reshape(-1).shape[0]
        # features = model.predict_generator(
        #     generator, math.ceil(nb_samples/self.batch_size))
        # pred_y = features[2].reshape(-1)
        # pred_y = (pred_y > 0.5).astype(int)
        # valid_acc = sum(y.reshape(-1) == pred_y) / nb_samples
        # logger.warn("test valid acc: {}".format(valid_acc))

    def ft_train_conv_layer(self):
        self.ft_get_model()
        model = self.model
        try:
            self.top_model.load_weights(self.top_model_weights_path)
        except Exception as e:
            logger.warn(e)
            logger.warn("training conv layers without pretrained top models")
        # model.summary()

        # freezing top layers of model
        for layer in model.layers[:16]:
            print(layer)
            layer.trainable = False

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=2e-4, momentum=0.9),
                      # optimizer = "rmsprop",
                      metrics=['accuracy'])

        # prepare data augmentation configuration
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=0.2,
            horizontal_flip=True)
        valid_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary')
        train_y = train_generator.classes[train_generator.index_array]
        nb_train_samples = train_y.reshape(-1).shape[0]
        valid_generator = valid_datagen.flow_from_directory(
            self.valid_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary')
        valid_y = valid_generator.classes[valid_generator.index_array]
        nb_valid_samples = valid_y.reshape(-1).shape[0]

        checkpointer = ModelCheckpoint(filepath=os.path.join(self.model_weight_dir,
                                                             "weights.{epoch:02d}-{val_acc:.4f}.h5"),
                                       verbose=1)
        model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // self.batch_size,
            epochs=self.epochs,
            validation_data=valid_generator,
            validation_steps=nb_valid_samples // self.batch_size,
            callbacks=[checkpointer])

    def save_features_and_results(self, weight_name):
        logger.info("begin save features and results.")
        feature_dir = os.path.join(self.feature_dir, weight_name)
        check_dir(feature_dir)
        weight_path = os.path.join(self.model_weight_dir, weight_name)
        self.ft_get_model()
        output_model = self.output_model
        output_model.load_weights(weight_path)

        data_idxs = [self.train_idx, self.valid_idx, self.test_idx]
        prefixs = ["train", "valid", "test"]
        for data_idx, prefix in zip(data_idxs, prefixs):
            filename = [str(i) + ".jpg" for i in data_idx]
            y = [self.y[i] for i in data_idx]
            mat = {
                "filename": filename,
                "class": y
            }
            dataframe = pd.DataFrame(mat)
            datagen = ImageDataGenerator(rescale=1. / 255)
            generator = datagen.flow_from_dataframe(
                dataframe,
                self.images_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                class_mode="binary",
                shuffle=False
            )
            nb_samples = np.array(y).reshape(-1).shape[0]
            logger.info("{} sampling number: {}".format(prefix, nb_samples))
            features = output_model.predict_generator(
                generator, math.ceil(nb_samples / self.batch_size))
            features = features[1:]
            logger.info("{} feature shape: {}".format(prefix, features[0].shape))
            mat = {
                "features": features,
                "y": y,
                "filename": filename
            }
            pickle_save_data(os.path.join(feature_dir, prefix + config.pkl_ext), mat)

    def _save_features_and_results(self, weight_name):
        logger.info("begin save features and results.")
        feature_dir = os.path.join(self.feature_dir, weight_name)
        check_dir(feature_dir)
        print("feature_dir:", feature_dir)
        weight_path = os.path.join(self.model_weight_dir, weight_name)
        print("weight_name:", weight_path)
        self.ft_get_model()
        output_model = self.output_model
        output_model.load_weights(weight_path)

        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        for data_dir in dirs:
            datagen = ImageDataGenerator(rescale=1. / 255)
            generator = datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                class_mode="binary",
                shuffle=False
            )
            y = generator.classes[generator.index_array]
            filename = generator.filenames
            nb_samples = y.reshape(-1).shape[0]
            if nb_samples == 0:
                continue
            file_prefix = os.path.split(data_dir)[1].split(".")[0]
            logger.info("{} sampling number: {}".format(file_prefix, nb_samples))
            features = output_model.predict_generator(
                generator, math.ceil(nb_samples / self.batch_size))
            features = features[1:]
            logger.info("{} feature shape: {}".format(file_prefix, features[0].shape))
            mat = {
                "features": features,
                "y": y,
                "filename": filename
            }
            pickle_save_data(os.path.join(feature_dir, file_prefix + config.pkl_ext), mat)

    def saliency_map_process(self, weight_name):
        """

        :param weight_name:
        :return:
        """
        feature_dir = os.path.join(self.feature_dir, weight_name)
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]

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
                if self.dataname == config.dog_cat:
                    img_id = self.class_name_encoding[cls] * 12500 + int(img_id)
                img_id = int(img_id)



    def imagenet_pretrained_model_feature(self):
        None

    def process_superpixel(self):
        superpixel_feature_dir = os.path.join(self.feature_dir, "superpixel")
        X = np.zeros((len(self.y), 200))
        for file_name in ["binary_sp_features_train.pkl",
                          "binary_sp_features_test.pkl"]:
            mat = pickle_load_data(os.path.join(superpixel_feature_dir, file_name))
            for name in mat:
                feature = mat[name]
                img_id = name.split(".")[1]
                img_id = int(img_id)
                X[img_id] = feature

        filename = os.path.join(self.feature_dir, "superpixel", "X.pkl")
        pickle_save_data(filename, X)