import numpy as np
import os

from scripts.utils.config_utils import config
from scripts.database.database import DataBase

class DataThreeDim(DataBase):
    def __init__(self):
        dataname = config.three_dim
        super(DataThreeDim, self).__init__(dataname)

    def load_data(self):
        print("generating data!!!")
        anchor_points = [[0,0,0],[1,0,0], [0,1,1]]
        num_per_class = 500
        data = None
        y = None
        for idx, point in enumerate(anchor_points):
            noise = np.random.normal(0,0.5,(num_per_class, 3))
            local_data = np.array(point).reshape(1,-1) + noise
            local_y = np.array([idx for _ in range(num_per_class)]).reshape(-1,1)
            if data is None:
                data = local_data.copy()
                y = local_y
            else:
                data = np.concatenate((data, local_data), axis=0)
                y = np.concatenate((y, local_y), axis=0)
        y = y.reshape(-1)

        # shuffle and split training set and test set
        idx = np.array(range(len(y)))
        np.random.seed(0)
        np.random.shuffle(idx)
        np.random.seed()
        X_train = data[idx,:]
        y_train = y[idx]
        split_num = int(len(y_train) * 0.8)
        self.X_train = X_train[:split_num, :]
        self.y_train = y_train[:split_num]
        self.X_test = X_train[split_num:,:]
        self.y_test = y_train[split_num:]

        print("data is successfully processed!")
        print("train data num: %s, test data num: %s" % (len(self.X_train), len(self.X_test)))

        self.save_cache()

        return True

    def process_data(self):
        super(DataThreeDim, self).process_data()

if __name__ == '__main__':
    d = DataThreeDim()
    d.load_data()
    d.process_data()
    d.save_file()