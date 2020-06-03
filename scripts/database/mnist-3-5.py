import numpy as np
import os
import scipy.io as sio
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_mldata

from scripts.utils.config_utils import config
from scripts.database.database import DataBase

class DataMNIST_3_5(DataBase):
    def __init__(self):
        dataname = config.mnist_3_5
        super(DataMNIST_3_5, self).__init__(dataname)

    def load_data(self):
        print("loading data from sklearn!")
        mnist = fetch_mldata("MNIST original")
        target = mnist["target"]
        data = mnist["data"]
        classes = [3,5]
        selected_idx = [idx for idx, c in enumerate(target) if c in classes]
        selected_idx = np.array(selected_idx)
        selected_target = target[selected_idx]
        selected_target = (selected_target - 2) // 2
        selected_data = data[selected_idx, :]
        index = np.array(range(len(selected_target)))
        np.random.seed(123)
        np.random.shuffle(index)
        y_train = selected_target[index]
        X_train = selected_data[index, :]
        split_num = int(len(selected_target) * 5 / 6.0)
        self.X_train = X_train[:split_num, :]
        self.y_train = y_train[:split_num]
        self.X_test = X_train[split_num:,:]
        self.y_test = y_train[split_num:]

        print("data loaded!!")
        print("train data num: %s, test data num: %s" % (len(self.X_train), len(self.X_test)))

        self.save_cache()

    def process_data(self):
        super(DataMNIST_3_5, self).process_data()


if __name__ == '__main__':
    d = DataMNIST_3_5()
    d.load_data()
    d.process_data()
    d.save_file()
