import numpy as np
import os
import math
import tensorflow as tf
from time import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, accuracy
from scripts.utils.data_utils import Data

class RandomForest_DecisionBoundary(object):
    def __init__(self, dataname):
        self.dataname = dataname
        self.data = Data(self.dataname)
        self.X_train, self.y_train, self.X_test, self.y_test = self.data.get_data("all")
        self.train_num, self.feature_num = self.X_train.shape

    def training(self):
        self.r = RandomForestClassifier(n_estimators=100)
        self.r.fit(self.X_train, self.y_train)
        acc = accuracy(self.y_train, self.r.predict(self.X_train))
        # acc = accuracy(self.y_test, self.r.predict(self.X_test))
        print("accuracy: {}".format(acc))

if __name__ == '__main__':
    m = RandomForest_DecisionBoundary(config.three_dim)
    m.training()