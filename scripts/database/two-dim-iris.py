import numpy as np
import os

from sklearn.datasets import load_iris

from scripts.utils.config_utils import config
from scripts.database.database import DataBase

class DataTwoDimIris(DataBase):
    def __init__(self):
        dataname = config.two_dim_iris
        super(DataTwoDimIris, self).__init__(dataname)

    def load_data(self):
        print("Loading data from sklearn!!!")
        iris = load_iris()
        data = iris.data
        target = iris.target
        data = data[:, :2]
        data = data[target > 0]
        target = target[target > 0]
        target = target - 1

        self.X_train = data
        self.y_train = target
        self.X_test = data
        self.y_test = target

        # # shuffle and split training set and test set
        # idx = np.array(range(len(y)))
        # np.random.seed(0)
        # np.random.shuffle(idx)
        # np.random.seed()
        # X_train = data[idx,:]
        # y_train = y[idx]
        # split_num = int(len(y_train) * 0.8)
        # self.X_train = X_train[:split_num, :]
        # self.y_train = y_train[:split_num]
        # self.X_test = X_train[split_num:,:]
        # self.y_test = y_train[split_num:]

        print("data is successfully processed!")
        print("train data num: %s, test data num: %s" % (len(self.X_train), len(self.X_test)))

        self.save_cache()

        return True

    def process_data(self):
        super(DataTwoDimIris, self).process_data()

if __name__ == '__main__':
    d = DataTwoDimIris()
    d.load_data()
    d.process_data()
    d.save_file()