import numpy as np
import os
import scipy.io as sio
from time import time
import warnings
import math

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

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_save_data, pickle_load_data
from scripts.database.database import DataBase
from scripts.utils.log_utils import logger
from scripts.utils.embedder_utils import Embedder
import shutil

class NewAccTest(DataBase):
    def __init__(self):
        dataname = config.animals_add4
        super(NewAccTest, self).__init__(dataname)

    def add4_test_acc(self,weight_name):
        # mat = pickle_load_data(os.path.join(self.data_dir, "processed_data_pred.pkl"))
        mat = pickle_load_data(os.path.join(self.data_dir, "processed_data_pred_removing_images.pkl"))
        test_idx = mat["test_idx"]
        print("test_idx len:", len(test_idx))
        feature_dir = weight_name
        dirs = [self.test_data_dir]
        prediction = []
        groundtruth = []
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
            y = mat["y"].reshape(-1)
            if y.max()+1 != pred_y.shape[1]:
                y[y==2] == 5
                y[y>2] = y[y>2] - 1
            for idx, name in enumerate(filenames):
                name = name.replace("\\", "/")
                cls, img_name = name.split("/")
                img_id, _ = img_name.split(".")
                try:
                    img_id = int(img_id)
                except:
                    print(img_id)
                    continue
                if img_id in test_idx:
                    prediction.append(pred_y[idx])
                    groundtruth.append(y[idx])
        print(len(prediction), len(groundtruth))
        prediction = np.array(prediction).argmax(axis=1)
        acc = (prediction == groundtruth).sum() / len(groundtruth)
        print("acc: ", acc)

    def leopard_test_acc(self, weight_name):
        # mat = pickle_load_data(os.path.join(config.data_root, config.animals_leopard, "processed_data_pred.pkl"))
        # mat = pickle_load_data(os.path.join(config.data_root, config.animals_leopard, "processed_data_pred_removing_images.pkl"))
        mat = pickle_load_data(os.path.join(config.data_root, config.animals_add4, "processed_data_pred_removing_images.pkl"))
        # mat = pickle_load_data(os.path.join(config.data_root, config.animals_add4, "processed_data_pred_removing_images.pkl"))
        test_idx = mat["test_idx"]
        print("test_idx len:", len(test_idx))

        # load nam_encoding
        all_data = pickle_load_data(os.path.join(config.data_root, config.animals_step2, "all_data_cache.pkl"))
        name_encoding = all_data["name_encoding"]

        feature_dir = weight_name
        dirs = [self.test_data_dir]
        prediction = []
        groundtruth = []
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
            y = mat["y"].reshape(-1)
            if y.max()+1 != pred_y.shape[1]:
                y[y==2] == 5
                y[y>2] = y[y>2] - 1
            for idx, name in enumerate(filenames):
                name = name.replace("\\", "/")
                cls, img_name = name.split("/")
                img_id, _ = img_name.split(".")
                img_id = int(img_id)
                if img_id in test_idx:
                    prediction.append(pred_y[idx])
                    groundtruth.append(y[idx])
        print(len(prediction), len(groundtruth))
        prediction = np.array(prediction).argmax(axis=1)
        acc = (prediction == groundtruth).sum() / len(groundtruth)
        print("acc: ", acc)

if __name__ == '__main__':
    n = NewAccTest()
    # n.leopard_test_acc(os.path.join(config.data_root,
    #                         config.animals,
    #                         "feature",
    #                         "weights.40-0.8519.h5-leopard"))


    n.add4_test_acc(os.path.join(config.data_root,
                            config.animals,
                            "feature",
                            "weights.50-0.8507.h5-100-51-husky-tigercat-4"))
