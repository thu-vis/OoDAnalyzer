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

class DataDogCatImagenet(DataBase):
    def __init__(self):
        dataname = config.dog_cat_imagenet
        super(DataDogCatImagenet, self).__init__(dataname)


    def load_data(self):
        print("load data from disk.")

        bias_train_filename = np.load(os.path.join(config.data_root,
                                                   self.dataname,
                                                   "feature",
                                                   "normal_train_filename.npy")).tolist()
        all_test_filename = np.load(os.path.join(config.data_root,
                                                 self.dataname,
                                                 "feature",
                                                 "normal_test_filename.npy")).tolist()
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

        for name in bias_train_filename:
            # try:
            idx = all_filename.index(name)
            if idx >= 24992:
                continue
            bias_train_data.append(all_data[idx,:].tolist())
            bias_train_y.append(all_y[idx])
            # except:
            #     print(name)
        bias_train_data = np.array(bias_train_data)
        bias_train_y = np.array(bias_train_y)

        for name in all_test_filename:
            # try:
            idx = all_filename.index(name)
            if idx >= 24992:
                continue
            all_test_data.append(all_data[idx,:].tolist())
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
        super(DataDogCatImagenet, self).process_data()

if __name__ == '__main__':
    d = DataDogCatImagenet()
    d.load_data()
    d.process_data()
    d.save_file()