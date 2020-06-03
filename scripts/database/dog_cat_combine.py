import numpy as np
import os
import scipy.io as sio
from time import time
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_mldata
import tensorflow as tf
from tensorflow import keras
from scipy.interpolate import interp1d
from PIL import Image
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

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir
from scripts.database.database import DataBase
import shutil

pre_processing_func = lambda x: preprocess_input(x, mode="tf")

class DataDogCatCombine(DataBase):
    def __init__(self):
        dataname = config.dog_cat_combine
        super(DataDogCatCombine, self).__init__(dataname)


    def load_data(self):
        print("load data from disk.")

        bias_train_filename = np.load(os.path.join(config.data_root,
                                                   self.dataname,
                                                   "feature",
                                                   "bias_train_filename.npy")).tolist()
        bias_train_data_ft = np.load(os.path.join(config.data_root,
                                                   self.dataname,
                                                   "feature",
                                                   "train.npy"))
        all_test_filename = np.load(os.path.join(config.data_root,
                                                 self.dataname,
                                                 "feature",
                                                 "normal_test_filename.npy")).tolist()
        outside_all_filename = np.load(os.path.join(config.data_root,
                                                 self.dataname,
                                                 "feature",
                                                 "outside_all_filename.npy")).tolist()
        bias_test_filename = np.load(os.path.join(config.data_root,
                             self.dataname,
                             "feature",
                             "bias_test_filename.npy")).tolist()
        all_test_data_ft = np.load(os.path.join(config.data_root,
                                                   self.dataname,
                                                   "feature",
                                                   "normal_test.npy"))
        all_filename = np.load(os.path.join(config.data_root,
                                                 self.dataname,
                                                 "feature",
                                                 "filename.npy")).tolist()
        all_data = np.load(os.path.join(config.data_root,
                                                 self.dataname,
                                                 "feature",
                                                 "feature.npy"))
        all_y = np.load(os.path.join(config.data_root,
                                     self.dataname,
                                     "feature",
                                     "y.npy")).reshape(-1)
        bias_train_data = []
        bias_train_y = []
        all_test_data = []
        all_test_y = []

        for i, name in enumerate(bias_train_filename[:2576]):
            # try:
            idx = all_filename.index(name)
            if idx >= 24992:
                continue
            a = all_data[idx,:]
            a = a / ((a**2).sum())**0.5
            b = bias_train_data_ft[i,:]
            b = b / ((b**2).sum())**0.5
            bias_train_data.append(a.tolist() + b.tolist())
            bias_train_y.append(all_y[idx])
            # except:
            #     print(name)
        bias_train_data = np.array(bias_train_data)
        bias_train_y = np.array(bias_train_y)

        for i, name in enumerate(all_test_filename[:4992]):
            # try:
            idx = all_filename.index(name)
            # or (name not in outside_all_filename and name not in bias_test_filename):
            # in order to selected black (white)dog and black (white) cat
            # if idx >= 24992 or (name not in outside_all_filename and name not in bias_test_filename):
            if idx >= 24992:
                continue
            a = all_data[idx,:]
            a = a / ((a**2).sum())**0.5
            b = all_test_data_ft[i, :]
            b = b / ((b**2).sum())**0.5
            all_test_data.append(a.tolist() + b.tolist())
            all_test_y.append(all_y[idx])
            # except:
            #     print(name)
        all_test_data = np.array(all_test_data)
        all_test_y = np.array(all_test_y)

        self.X_train = bias_train_data
        self.y_train = bias_train_y
        self.X_test = all_test_data
        self.y_test = all_test_y

        print("data loaded!!")
        print("train data num: %s, test data num: %s" % (len(self.X_train), len(self.X_test)))

        self.save_cache()

    def process_data(self):
        super(DataDogCatCombine, self).process_data()

if __name__ == '__main__':
    d = DataDogCatCombine()
    d.load_data()
    d.process_data()
    d.save_file()