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


class DataSVHN(DataBase):
    def __init__(self):
        dataname = config.svhn
        config.data_root = r"H:\backup"
        super(DataSVHN, self).__init__(dataname)

    def preprocessing_data(self):
        train_label_dir = os.path.join(config.raw_data_root,
                                       self.dataname,
                                       "two_peak")
        test_label_dir = os.path.join(config.raw_data_root,
                                      self.dataname,
                                      "two_peak_test")
        X_train = np.load(os.path.join(self.data_dir, "train_x.npy"))
        y_train = np.load(os.path.join(self.data_dir, "train_y.npy")).reshape(-1).astype(int)
        X_test = np.load(os.path.join(self.data_dir, "test_x.npy"))
        y_test = np.load(os.path.join(self.data_dir, "test_y.npy")).reshape(-1).astype(int)
        np.random.seed(123)
        self.class_name = ["3", "5"]
        train_idx = []
        train_redundant_idx = []
        for idx, class_name in enumerate([3,5]):
            class_dir = os.path.join(train_label_dir, str(class_name))
            class_idx = np.load(os.path.join(class_dir, "final_" + str(idx) + ".npy"))
            train_idx = train_idx + class_idx.reshape(-1).tolist()
            class_redundant_idx = np.load(os.path.join(class_dir, "final_" + str(1-idx) + ".npy"))
            train_redundant_idx = train_redundant_idx + class_redundant_idx.reshape(-1).tolist()

        test_idx = []
        test_redundant_idx = []
        for idx, class_name in enumerate([3,5]):
            class_dir = os.path.join(test_label_dir, str(class_name))
            class_idx_0 = np.load(os.path.join(class_dir, "final_0.npy")).reshape(-1).tolist()
            class_idx_1 = np.load(os.path.join(class_dir, "final_1.npy")).reshape(-1).tolist()
            test_idx = test_idx + class_idx_0 + class_idx_1
            if idx == 0:
                test_redundant_idx = test_redundant_idx + class_idx_1
            else:
                test_redundant_idx = test_redundant_idx + class_idx_0

        self.X = np.concatenate((X_train, X_test), axis=0)
        self.y = np.array(y_train.tolist() + y_test.tolist())
        self.real_y = self.y.copy()
        y = []
        for i in self.y:
            if i == 3:
                y.append(0)
            elif i == 5:
                y.append(1)
            else:
                y.append(2)
        self.y = np.array(y)
        test_idx = np.array(test_idx) + X_train.shape[0]
        test_redundant_idx = np.array(test_redundant_idx) + X_train.shape[0]
        # for i in range(self.X.shape[0]):
        #     x = self.X[i,:,:,:]
        #     img = Image.fromarray(x)
        #     img_path = os.path.join(self.images_dir, str(i) + ".jpg")
        #     img.save(img_path)
        self.all_data = {
            "class_name": self.class_name,
            "class_name_encoding": self.class_name_encoding,
            "X": None,
            "y": self.y,
            "real_y": self.real_y,
            "train_idx": train_idx,
            "train_redundant_idx": train_redundant_idx,
            "valid_idx": [],
            "valid_redundant_idx": [],
            "test_idx": test_idx.tolist(),
            "test_redundant_idx": test_redundant_idx.tolist()
        }

        self.save_cache()

    def load_data(self, loading_from_buffer=True):
        super(DataSVHN, self).load_data(loading_from_buffer)
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

    def process_data(self):
        None


    def inplace_process_data(self):
        cnn_features_dir_name = [
            # "weights.20-0.5167.h5",
            # "weights.20-0.5199.h5",
            # "weights.20-0.5203.h5",
            # "weights.20-0.5205.h5",
            # "weights.20-0.5237.h5",
            # "weights.100-0.5465.h5",
            # "inceptionresnet_imagenet",
            # "inceptionv3_imagenet",
            # "mobilenet_imagenet",
            # "resnet50_imagenet",
            # "vgg_imagenet",
            # # "xception_imagenet",
            # "sift-200",
            # # "HOG",
            # "HOG-kmeans-200",
            # # "LBP",
            # "LBP-hist",
            # "superpixel-500",
            # "sift-1000"，
            "orb-200",
            "brief-200"

        ]
        for weight_name in cnn_features_dir_name:
            X = self.postprocess_data(weight_name, if_return=True)
            filename = os.path.join(self.feature_dir, weight_name, "X.pkl")
            pickle_save_data(filename, X)

    def postprocess_data(self, weight_name, if_return=False):
        """

        :return:
        """
        feature_dir = os.path.join(self.feature_dir, weight_name)
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
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
                img_id = int(img_id)
                if data_dir == self.test_data_dir:
                    img_id = img_id + 73257
                if len(features.shape) > 2:
                    features = features.reshape(features.shape[0], -1)
                if X is None:
                    X = np.zeros((self.y.shape[0], features.shape[1]))
                try:
                    X[img_id,:] = features[idx,:]
                except Exception as e:
                    print(e)
                    a = 1

        self.X = X
        if if_return:
            logger.info("'if return' flag is enabled, returning immediately!")
            return X

        valided_data_idx = self.train_idx + self.valid_idx + self.test_idx
        logger.info("info confirm, valided data num: {}".format(len(valided_data_idx)))
        projected_X = self.X[np.array(valided_data_idx),:]
        projection_method =["tsne"]
        default_method = "tsne"
        self.all_embed_X = {}
        self.embed_X = []
        random_state = 123
        if self.dataname == config.svhn:
            random_state = 124
        for pm in projection_method:
            embedder = Embedder(pm, n_components=2, random_state=random_state)
            partial_embed_X = embedder.fit_transform(projected_X, None)
            embed_X = np.zeros((len(self.y), 2))
            embed_X[np.array(valided_data_idx),:] = partial_embed_X
            self.all_embed_X[pm] = embed_X
            if pm == default_method:
                self.embed_X = embed_X

    def pretrain_get_features(self):
        model_names = ["vgg", "resnet50", "xception", "inceptionv3",
                       "inceptionresnet", "mobilenet"]
        # model_names = ["mobilenet", "nasnet"]
        for model_name in model_names:
            super(DataSVHN, self).pretrain_get_model(model_name=model_name)




if __name__ == '__main__':
    d = DataSVHN()
    # for i in ["HOG-kmeans-200", "LBP-hist"]:
    #     d.lowlevel_features(i)
    # d.preprocessing_data()
    d.load_data()
    d.inplace_process_data()
    # d.pretrain_get_features()
    # # d._save_features_and_results("weights.100-0.5465.h5")
    # d.postprocess_data("weights.100-0.5465.h5")
    # d.save_file()
    # d.ft_save_bottleneck()
    # d.ft_train_conv_layer()