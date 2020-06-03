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
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, \
    GlobalAveragePooling2D, Conv2D, Input, MaxPooling2D, BatchNormalization
from keras import applications
from keras import optimizers
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from multiprocessing import Pool
from skimage.feature import hog, local_binary_pattern
from skimage import color
from keras.datasets import mnist

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_save_data, pickle_load_data
from scripts.database.database import DataBase
from scripts.utils.log_utils import logger
from scripts.utils.embedder_utils import Embedder
import shutil


class DataMNISTmlp(DataBase):
    def __init__(self, suffix=""):
        dataname = "MNIST_mlp"
        super(DataMNISTmlp, self).__init__(dataname, suffix)

    def save_data(self):
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        y_ood = np.zeros((18726, 10))
        y_ood[0,:] = 1
        y_test = np.concatenate((y_test, y_ood), axis=0)
        print(y_test.shape)

        x_ood = np.zeros((20000, 784))
        global_id = 0
        notMNNIST_dir = os.path.join(config.data_root,
                                     self.dataname,
                                     "notMNIST_small")
        for sub_dir_name in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
            sub_dir = os.path.join(notMNNIST_dir, sub_dir_name)
            img_name_list = os.listdir(sub_dir)
            for img_name in img_name_list:
                img_path = os.path.join(sub_dir, img_name)
                img = cv2.imread(img_path)
                try:
                    data = np.array(img).mean(axis=2).reshape(-1)
                except:
                    print(data.shape)
                    data = np.array(img).reshape(-1)
                try:
                    x_ood[global_id, :] = data
                    global_id = global_id + 1
                except:
                    print(img_path)

        x_test = np.concatenate((x_test, x_ood[:global_id, :]), axis=0).astype("float32")
        print(x_test.shape)

        pickle_save_data(os.path.join(self.data_dir, "raw_data.pkl"),
                         [x_train, y_train, x_test, y_test])


    def train_model(self):
        batch_size = 128
        num_classes = 10
        epochs = 20

        x_train, y_train, x_test, y_test = \
            pickle_load_data(os.path.join(self.data_dir, "raw_data.pkl"))

        # the data, split between train and test sets
        x_train /= 255
        x_test /= 255
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        model = Sequential()
        model.add(Dense(200, activation='relu', input_shape=(784,)))
        model.add(Dense(200, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(200, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(),
                      metrics=['accuracy'])

        checkpointer = ModelCheckpoint(filepath=os.path.join(self.model_weight_dir,
                                                             "top_weights.{epoch:02d}-{val_acc:.4f}.h5"),
                                       verbose=1)

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            callbacks=[checkpointer])
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

if __name__ == '__main__':
    d = DataMNISTmlp("MNIST_mlp")
    d.save_data(); exit()
    for idx, suffix in enumerate(["_repeat_1",
                                  "_repeat_2",
                                  "_repeat_3",
                                  "_repeat_4",
                                  "_repeat_5"]):
        # weights_name = ["weights.20-0.7654.h5",
        #                 "weights.20-0.8087.h5",
        #                 "weights.20-0.8259.h5"]
        d = DataMNISTmlp(suffix)
        # # exit()
        # d.ft_save_bottleneck()
        # d.ft_train_top_model()
        # d.ft_train_conv_layer()
        # d._save_features_and_results(weights_name[idx])
        d.train_model()