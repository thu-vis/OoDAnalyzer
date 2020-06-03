import numpy as np
import os
import scipy.io as sio
from time import time
import warnings
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_mldata
from sklearn.manifold import TSNE
from scipy.interpolate import interp1d
import tensorflow as tf
from PIL import Image
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Input
from keras import applications
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from multiprocessing import Pool

from scripts.utils.config_utils import config
from scripts.utils.log_utils import logger
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.database.database import DataBase
import shutil

pre_processing_func = lambda x: preprocess_input(x, mode="tf")



taxo = {
    'train_male_lip': [],
    'train_female_lip': [],
    'train_male_nolip': [],
    'train_female_nolip': [],
    'valid_male_lip': [],
    'valid_female_lip': [],
    'valid_male_nolip': [],
    'valid_female_nolip': [],
    'test_male_lip': [],
    'test_female_lip': [],
    'test_male_nolip': [],
    'test_female_nolip': []
}

gradient = np.linspace(0, 1, 256)
gradient = np.vstack(gradient).reshape(-1)
color_gradient = plt.get_cmap("OrRd")(gradient)
color_gradient = (color_gradient[:,:3] * 255).astype(np.uint8)
color_gradient = color_gradient[:,np.array([2,1,0])]
color_gradient = color_gradient.reshape(color_gradient.shape[0],1,-1)



def list_of_groups(init_list, children_list_len):
    if len(init_list) == 0:
        return []
    sum = 0
    res = []
    for lens in children_list_len:
        res.append(init_list[sum: sum + lens])
        sum = sum + lens
    return res


def _sift_train(start, end, filename, data_dir):
    train_des = []
    train_des_num = []
    sift = cv2.xfeatures2d.SIFT_create()
    for idx in range(start, end):
        img_name = filename[idx]
        img = cv2.imread(os.path.join(data_dir, img_name))
        try:
            _, d = sift.detectAndCompute(img, None)
        except Exception as e:
            print(e)
            d = np.zeros(128).astype(int).reshape(1, 128)
        if d is None:
            d = np.zeros(128).astype(int).reshape(1, 128)
        if d is None:
            d = np.zeros(128).astype(int).reshape(1, 128)
        train_des = train_des + d.astype(int).tolist()
        train_des_num.append(d.shape[0])
        if (idx - start + 1) % 50 == 0:
            print("start: {}, end: {}, now processed: {}"
                  .format(start, end, (idx-start+1)/(end-start+1)))
    return train_des, train_des_num

def _sift_prediction(start, end, filename, data_dir, km):
    labels = []
    des_num = []
    sift = cv2.xfeatures2d.SIFT_create()
    for idx in range(start, end):
        img_name = filename[idx]
        img = cv2.imread(os.path.join(data_dir, img_name))
        try:
            _, d = sift.detectAndCompute(img, None)
        except Exception as e:
            print(e)
            d = np.zeros(128).astype(int).reshape(1, 128)
        if d is None:
            d = np.zeros(128).astype(int).reshape(1, 128)
        if d is None:
            d = np.zeros(128).astype(int).reshape(1, 128)

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

class DataLipsticks(DataBase):
    def __init__(self, suffix=""):
        dataname = config.lipsticks
        print("************data dir reordered****************")
        self.data_dir = os.path.join("H:/backup", dataname)
        print("new data_dir: ",self.data_dir)
        super(DataLipsticks, self).__init__(dataname, suffix, self.data_dir)

    def preprocessing_data(self):
        self.class_name = ["nolip", "lip"]
        self.class_name_encoding = {
            self.class_name[0]: 0,
            self.class_name[1]: 1
        }
        data = [['train_male_nolip', 'train_female_lip'],
            ['valid_male_nolip', 'valid_female_lip'],
            ['test_male_nolip', 'test_female_nolip', 'test_male_lip', 'test_female_lip'],
            ['train_female_nolip','train_male_lip'],
            ['valid_female_nolip', 'valid_male_lip'],
            ['test_female_nolip', 'test_male_lip'],
        ]
        idx_lists = [self.train_idx, self.valid_idx, self.test_idx,
                     self.train_redundant_idx, self.valid_redundant_idx, self.test_redundant_idx]
        y = np.ones(202599) * -1
        y = y.astype(int)
        all_lists = []
        for categories in data:
            idx_list = []
            for filename in categories:
                cls_name = filename.split("_")[2]
                cls = self.class_name_encoding[cls_name]
                filename = os.path.join(self.data_dir,
                                        "Anno",
                                        filename + ".txt")
                images_names = open(filename, "r").read()
                image_names = images_names.split("\n")[:-1]
                assert int(image_names[0]) == len(image_names[1:])
                image_names = image_names[1:]
                logger.info("total {} images".format(len(image_names)))
                for img_name in image_names:
                    img_name = img_name.split(".")[0] + ".png"
                    src_img = os.path.join(self.raw_data_dir, "all_data", img_name)
                    target_img_id = int(img_name.split(".")[0]) - 1
                    y[target_img_id] = cls
                    target_img = os.path.join(self.images_dir, str(target_img_id) + ".jpg")
                    idx_list.append(target_img_id)
            all_lists.append(idx_list)
        assert sum(y >= 0) == 202599
        self.all_data = {
            "class_name": self.class_name,
            "class_name_encoding": self.class_name_encoding,
            "X": None,
            "y": y,
            "train_idx": all_lists[0],
            "valid_idx": all_lists[1],
            "test_idx": all_lists[2],
            "train_redundant_idx": all_lists[3],
            "valid_redundant_idx": all_lists[4],
            "test_redundant_idx": all_lists[5]
        }
        self.save_cache()

    def load_data(self, loading_from_buffer=True):
        super(DataLipsticks, self).load_data(loading_from_buffer)
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

        a = 1
    # def process_data(self):
    #     # self.ft_save_bottleneck()
    #     # self.ft_train_top_model()
    #     self.ft_train_conv_layer()
    #
    # def postprocess_data(self, weight_name):
    #     feature_dir = os.path.join(self.feature_dir, weight_name)
    #     dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
    #     # TODO:
    #     X = np.zeros((self.y.shape[0], 1024))
    #     for data_dir in dirs:
    #         file_prefix = os.path.split(data_dir)[1].split(".")[0]
    #         data_filename = os.path.join(feature_dir, file_prefix + config.pkl_ext)
    #         if not os.path.exists(data_filename):
    #             logger.warn("{} does not exist, skip!".format(data_filename))
    #             continue
    #         mat = pickle_load_data(data_filename)
    #         features = mat["features"][0]
    #         pred_y = mat["features"][1]
    #         filenames = mat["filename"]
    #         for idx, name in enumerate(filenames):
    #             cls, img_name = name.split("/")
    #             img_id, _ = img_name.split(".")
    #             img_id = int(img_id) - 1
    #             img_id = int(img_id)
    #             X[img_id,:] = features[idx,:]
    #     self.X = X
    #     super(DataLipsticks, self).postprocess_data(weight_name)

    def sift_features(self, method_name):
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

    def pretrain_get_features(self):
        model_names = ["vgg", "resnet50", "xception", "inceptionv3",
                       "inceptionresnet", "mobilenet"]
        # model_names = ["mobilenet", "nasnet"]
        for model_name in model_names:
            super(DataLipsticks, self).pretrain_get_model(model_name=model_name)

    def inplace_process_data(self):
        cnn_features_dir_name = [
            # "weights.10-0.9562.h5",
            # "weights.10-0.9513.h5",
            # "weights.20-0.9516.h5",
            # "weights.20-0.9551.h5",
            # "weights.20-0.9552.h5",
            # "inceptionresnet_imagenet",
            # "inceptionv3_imagenet",
            # "mobilenet_imagenet",
            # "resnet50_imagenet",
            # "vgg_imagenet",
            # "xception_imagenet",
            # "sift-200",
            # # # "HOG",
            # "HOG-200",
            # # "LBP",
            # "LBP-hist",
            # # "superpixel-500",
            # # "sift-1000",
            "orb-200",
            "brief-200"

        ]
        for weight_name in cnn_features_dir_name:
            logger.info(weight_name)
            X = self.postprocess_data(weight_name, if_return=True, embedding=False)
            filename = os.path.join(self.feature_dir, weight_name, "X.pkl")
            pickle_save_data(filename, X)

    def postprocess_data(self, weight_name, if_return=False, embedding=False):
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
                img_id = int(img_id)
                # if img_id >= 100000:
                #     continue
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

        super(DataLipsticks, self).postprocess_data(weight_name, embedding)


if __name__ == '__main__':
    d = DataLipsticks()
    d.preprocessing_data()
    # d.process_data()
    # d.load_data()
    # d.inplace_process_data()
    # d.pretrain_get_features()
    # d.sift_features("sift-200")
    # d.HOG_features("HOG-200")
    # d.LBP_features("LBP-hist")
    # d._save_features_and_results("weights.20-0.9943.h5")
    # d.postprocess_data("weights.10-0.9562.h5", embedding=False)
    # d.save_file()
