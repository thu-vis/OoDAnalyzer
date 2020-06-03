import numpy as np
import os
import scipy.io as sio
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_mldata
from PIL import Image

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import pickle_load_data, pickle_save_data
from scripts.utils.log_utils import logger
from scripts.utils.embedder_utils import Embedder
from scripts.database.database import DataBase
from scripts.utils.data_utils import Data

class DataMNIST_3_5_VGG(DataBase):
    def __init__(self):
        dataname = config.mnist_3_5_vgg
        super(DataMNIST_3_5_VGG, self).__init__(dataname)

    def preprocessing_data(self):
        self.class_name = ["3", "5"]
        self.class_name_encoding = {
            self.class_name[0]: 0,
            self.class_name[1]: 1
        }

        mnist = fetch_mldata("MNIST original")
        target = mnist["target"]
        data = mnist["data"]
        data = data
        index = np.array(range(len(target)))
        np.random.seed(123)
        np.random.shuffle(index)
        target = target[index]
        X = data[index, :]

        # save images in image_dir
        images_dir = self.images_dir
        feature_dir = os.path.join(self.feature_dir, "mnist-all")
        all_features_filename = os.path.join(feature_dir, config.processed_dataname)
        data = pickle_load_data(all_features_filename)
        y = data[config.y_name]
        assert sum(y==target) == len(y)

        classes = [3,5]
        selected_training_idx = [idx for idx, c in enumerate(target[:60000]) if c in classes]
        selected_test_idx = [idx + 60000 for idx, c in enumerate(target[60000:]) if c in classes]
        X_train = X[selected_training_idx, :]
        y_train = target[selected_training_idx]
        X_test = X[selected_test_idx, :]
        y_test = target[selected_test_idx]
        y_train = (y_train // 2 - 1).astype(int)
        y_test = (y_test // 2 - 1).astype(int)
        X = np.concatenate((X_train, X_test), axis=0)
        target = np.array(y_train.tolist() + y_test.tolist())

        for i in range(X.shape[0]):
            x = X[i,:]
            target_path = os.path.join(images_dir, str(i) + ".jpg")
            x = x.reshape(28, 28, 1).repeat(repeats=3, axis=2)
            x = x.astype(np.uint8)
            img = Image.fromarray(x)
            img.save(target_path)

        None

    def load_data(self):
        # TODO
        None


    def process_data(self):
        super(DataMNIST_3_5_VGG, self).process_data()
        # TODO


    def postprocess_data(self, weight_name):
        feature_dir = os.path.join(self.feature_dir, weight_name)
        all_features_filename = os.path.join(feature_dir, config.processed_dataname)
        data = pickle_load_data(all_features_filename)
        X_train = data[config.train_x_name]
        y_train = data[config.train_y_name]
        X_test = data[config.test_x_name]
        y_test = data[config.test_y_name]
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.array(y_train.tolist() + y_test.tolist())
        self.train_idx = np.array(range(X_train.shape[0])).tolist()
        self.train_redundant_idx = []
        self.valid_idx = []
        self.valid_redundant_idx = []
        self.test_idx = X_train.shape[0] + np.array(range(X_test.shape[0]))
        self.test_idx = self.test_idx.tolist()
        self.test_redundant_idx = []
        self.X = X
        self.y = y
        logger.info("train num: {}".format(len(self.train_idx)))

        valided_data_idx = self.train_idx + self.valid_idx + self.test_idx
        logger.info("info confirm, valided data num: {}".format(len(valided_data_idx)))
        # projected_X = np.zeros((len(valided_data_idx), 1024))
        projected_X = X[np.array(valided_data_idx), :]
        t = time()
        logger.info("embedding begin, all data shape:{}".format(projected_X.shape))
        embedder = Embedder("tsne", n_components=2, random_state=123)
        partial_embed_X = embedder.fit_transform(projected_X, None)
        logger.info("embedding time: {}".format(time() - t))
        embed_X = np.zeros((len(self.y), 2))
        embed_X[np.array(valided_data_idx), :] = partial_embed_X
        self.embed_X = embed_X



if __name__ == '__main__':
    d = DataMNIST_3_5_VGG()
    d.preprocessing_data()
    # d.load_data()
    # d.process_data()
    # d.postprocess_data("mnist")
    # d.save_file()
