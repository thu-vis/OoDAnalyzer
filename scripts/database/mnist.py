import numpy as np
import os
import pandas as pd
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
from scripts.database.database import DataBase
from scripts.utils.log_utils import logger
from scripts.utils.embedder_utils import Embedder
import shutil


def interpolate(a, x_extend, y_extend):
    x = np.array(range(a.shape[0]))
    xnew = np.linspace(x.min(), x.max(), x_extend)
    f = interp1d(x, a, axis=0)
    a = f(xnew)
    a = a.transpose()
    x = np.array(range(a.shape[0]))
    xnew = np.linspace(x.min(), x.max(), x_extend)
    f = interp1d(x, a, axis=0)
    a = f(xnew)
    a = a.transpose()
    return a

class DataMNIST(DataBase):
    def __init__(self):
        dataname = config.mnist
        super(DataMNIST, self).__init__(dataname)

    def load_data(self, loading_from_buffer=False):
        print("loading data from sklearn!")
        mnist = fetch_mldata("MNIST original")
        target = mnist["target"]
        data = mnist["data"]
        data = data
        index = np.array(range(len(target)))
        np.random.seed(123)
        np.random.shuffle(index)
        target = target[index]
        data = data[index, :]

        print("data 5000")
        data = data[:5000,:]
        projection_method =["tsne", "pca", "mds"]
        default_method = "tsne"
        for pm in projection_method:
            embedder = Embedder(pm, n_components=2, random_state=123)
            embedder.fit_transform(data, None)

        print("data 2000")
        data = data[:2000, :]
        projection_method = ["tsne", "pca", "mds"]
        default_method = "tsne"
        for pm in projection_method:
            embedder = Embedder(pm, n_components=2, random_state=123)
            embedder.fit_transform(data, None)

        exit()

        x_extend = 56
        y_extend = 56
        data_extend = np.zeros((data.shape[0],x_extend, y_extend))
        for i in range(data.shape[0]):
            x = data[i,:].reshape(28, 28)
            data_extend[i,:,:] = interpolate(x, x_extend, y_extend)
        data = data_extend

        ################ test #################
        # x = data[2,:,:].astype(np.uint8)
        # x = x.reshape(x_extend, x_extend, 1).repeat(repeats=3, axis=2)
        # img = Image.fromarray(x)
        # img.show()
        ################ test #################

        # # test
        # x = data[0,:] * 255.0
        # x = x.reshape(28,28).astype(np.uint8)
        # # img_origin = x.reshape(28,28,1).repeat(repeats=3, axis=2)
        # # img_origin = Image.fromarray(img_origin)
        # # img_origin.show()
        # x_extend = interpolate(x, 56*10, 56*10).astype(np.uint8)
        # img_extend = x_extend.reshape(56*10, 56*10, 1).repeat(repeats=3, axis=2)
        # img_extend = Image.fromarray(img_extend)
        # img_extend.show()
        # exit()

        data  = data.reshape(-1,x_extend,y_extend,1).repeat(repeats=3, axis=3)
        print("data shape:{}".format(data.shape))
        img_width, img_height = x_extend, y_extend
        batch_size = 7
        input_shape = (img_width, img_height, 3)
        base_model = keras.applications.VGG16(include_top=False, input_shape=input_shape, weights="imagenet")
        print("Model loaded.")
        print("Model output shape:",base_model.output_shape)
        print(base_model.summary())

        # dataset = tf.data.Dataset.from_tensor_slices((data, target))
        # dataset = dataset.batch(batch_size)

        data = base_model.predict(data[:,:,:,:])
        print(data.shape)
        self.X_train = data[:60000, :]
        self.X_test = data[60000:, :]
        self.y_train = target[:60000]
        self.y_test = target[60000:]


    def process_data(self):
        super(DataMNIST, self).process_data()

    def postprocess_data(self, weight_name):
        self.class_dname = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.X = np.concatenate((self.X_train, self.X_test), axis=0)
        self.y = np.array(self.y_train.tolist() + self.y_test.tolist())
        self.embed_X = []
        self.train_idx = np.array(range(60000))
        self.train_redundant_idx = []
        self.valid_idx = []
        self.valid_redundant_idx = []
        self.test_idx = []
        self.test_redundant_idx = []

    def process_superpixel(self):
        superpixel_feature_dir = os.path.join(self.feature_dir, "superpixel")
        X = np.zeros((70000, 200))
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

if __name__ == '__main__':
    d = DataMNIST()
    # d.load_data()
    d.process_superpixel()
    # d.process_data()
    # d.postprocess_data(None)
    # d.save_file()