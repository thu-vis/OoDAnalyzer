import numpy as np
import os
import scipy.io as sio
from time import time
import warnings
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_mldata
from keras.datasets import cifar10
import tensorflow as tf
from scipy.interpolate import interp1d
from PIL import Image
# import matplotlib as mpl
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt
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

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_save_data, pickle_load_data
from scripts.database.database import DataBase
from scripts.utils.log_utils import logger
from scripts.utils.embedder_utils import Embedder
import shutil

class DataCIFAR10(DataBase):
    def __init__(self):
        dataname = config.cifar10
        super(DataCIFAR10, self).__init__(dataname)

    def save_raw_data(self):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        pickle_save_data(os.path.join(self.data_dir, "raw_data.pkl"),
                         [X_train, y_train, X_test, y_test])

    def preprocessing_data(self):
        self.class_name = ["airplane", "automobile", "bird",
                           "cat", "deer", "dog", "frog",
                           "horse", "ship", "truck"]
        self.class_name_encoding = {
            self.class_name[0]: 0,
            self.class_name[1]: 1,
            self.class_name[2]: 2,
            self.class_name[3]: 3,
            self.class_name[4]: 4,
            self.class_name[5]: 5,
            self.class_name[6]: 6,
            self.class_name[7]: 7,
            self.class_name[8]: 8,
            self.class_name[9]: 9
        }

        X_train, y_train, X_test, y_test = pickle_load_data(os.path.join(self.data_dir,
                                                                         "raw_data.pkl"))
        self.train_idx = []
        self.train_redundant_idx = []
        self.valid_idx = []
        self.valid_redundant_idx = []
        self.test_idx = []
        self.test_redundant_idx = []
        self.train_idx = np.array(range(50000)).tolist()
        self.test_idx = np.array(range(50000, 60000)).tolist()


        X = np.concatenate((X_train, X_test), axis=0)
        self.y = np.array(y_train.tolist() + y_test.tolist())
        for i in range(X.shape[0]):
            x_data = X[i,:,:,:]
            img = Image.fromarray(x_data)
            img = img.resize((64, 64), Image.ANTIALIAS)
            img.save(os.path.join(self.images_dir, str(i) + ".jpg"))
            # exit()

        self.all_data = {
            "class_name": self.class_name,
            "class_name_encoding": self.class_name_encoding,
            "X": None,
            "y": self.y,
            "train_idx": self.train_idx,
            "train_redundant_idx": self.train_redundant_idx,
            "valid_idx": self.valid_idx,
            "valid_redundant_idx": self.valid_redundant_idx,
            "test_idx": self.test_idx,
            "test_redundant_idx": self.test_redundant_idx
        }
        self.save_cache()

        a = 1

    def save_images(self):
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        for idx, data_dir in enumerate(dirs):
            if idx == 0:
                selected_idx = self.train_idx
            elif idx == 1:
                selected_idx = self.valid_idx
            elif idx == 2:
                selected_idx = self.test_idx
            for cls, cls_name in enumerate(self.class_name):
                cls_dir = os.path.join(data_dir, cls_name)
                check_dir(cls_dir)
                if len(selected_idx) == 0:
                    continue
                selected_y = np.array(self.y)[np.array(selected_idx)]
                print(max(selected_y))
                # import IPython; IPython.embed()
                cls_idx = np.array(np.array(selected_idx))[selected_y==cls]
                for i in cls_idx:
                    src = os.path.join(self.images_dir, str(i) + ".jpg")
                    target = os.path.join(cls_dir, str(i) + ".jpg")
                    shutil.copy(src, target)


    def postprocess_data(self, weight_name, if_return=False, embedding=True):
        feature_dir = os.path.join(self.feature_dir, weight_name)
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        # X = np.zeros((self.y.shape[0], 1024))
        self.y = np.zeros(60000)
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
        pickle_save_data(os.path.join(config.data_root, self.dataname, "X.pkl"), X)
        # self.train_idx = np.array(range(50000)).tolist()
        # self.test_idx = np.array(range(50000, 60000)).tolist()
        self.add_info = {
            "prediction": prediction
        }
        if if_return:
            logger.info("'if return' flag is enabled. Returning immediately!")
            return X

        super(DataCIFAR10, self).postprocess_data(weight_name, embedding)



if __name__ == '__main__':
    d = DataCIFAR10()
    # d.preprocessing_data()
    # d.save_images()
    d.load_data()
    weight_name = "weights.40-0.8709.h5"
    d.postprocess_data(weight_name)
    d.save_file()