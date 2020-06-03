import numpy as np
import os
import scipy.io as sio
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_mldata

from PIL import Image
from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir
from scripts.database.database import DataBase
from scripts.utils.data_utils import Data

class DataMNIST_1_2_VGG(DataBase):
    def __init__(self):
        dataname = config.mnist_1_2_vgg
        super(DataMNIST_1_2_VGG, self).__init__(dataname)


    def load_data(self):
        print("loading data from {}!".format(config.mnist_vgg))
        d = Data(config.mnist_vgg)
        X_train, y_train, X_test, y_test = d.get_data("all")
        print("vgg training data shape:{}".format(X_train.shape))
        data = np.concatenate((X_train, X_test), axis=0)
        target = np.zeros(len(y_train) + len(y_test))
        target[:len(y_train)] = y_train
        target[len(y_train):] = y_test
        classes = [1,2]

        selected_training_idx = [idx for idx, c in enumerate(target[:60000]) if c in classes]
        selected_test_idx = [idx + 60000 for idx, c in enumerate(target[60000:]) if c in classes]
        self.X_train = data[selected_training_idx,:]
        self.y_train = target[selected_training_idx]
        self.X_test = data[selected_test_idx,:]
        self.y_test = target[selected_test_idx]
        self.y_train = (self.y_train - 1).astype(int)
        self.y_test = (self.y_test - 1).astype(int)
        self.selected_training_idx = selected_training_idx
        self.selected_test_idx = selected_training_idx

        # selected_idx = [idx for idx, c in enumerate(target) if c in classes]
        # selected_idx = np.array(selected_idx)
        # selected_target = target[selected_idx]
        # selected_target = selected_target - 1
        # selected_data = data[selected_idx, :]
        # index = np.array(range(len(selected_target)))
        # np.random.seed(123)
        # np.random.shuffle(index)
        # y_train = selected_target[index]
        # X_train = selected_data[index, :]
        # split_num = int(len(selected_target) * 5 / 6.0)
        # self.X_train = X_train[:split_num, :]
        # self.y_train = y_train[:split_num]
        # self.X_test = X_train[split_num:,:]
        # self.y_test = y_train[split_num:]

        print("data loaded!!")
        print("train data num: %s, test data num: %s" % (len(self.X_train), len(self.X_test)))

        self.save_cache()

    def process_data(self):
        super(DataMNIST_1_2_VGG, self).process_data()


if __name__ == '__main__':
    d = DataMNIST_1_2_VGG()
    d.load_data()
    d.process_data()
    d.save_file()
