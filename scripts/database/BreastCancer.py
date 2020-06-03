import numpy as np
import os
import pandas as pd
import scipy.io as sio
from time import time

from sklearn.preprocessing import MinMaxScaler

from scripts.utils.config_utils import config
from scripts.database.database import DataBase

class DataBreastCancer(DataBase):
    def __init__(self):
        dataname = config.breast_cancer
        super(DataBreastCancer, self).__init__(dataname)

    def load_data(self, loading_from_buffer = False):
        print("loading data from original data!!!")
        data_path = os.path.join(config.raw_data_root,
                                 self.dataname)
        file_path = os.path.join(data_path, "wdbc.data")
        # loading data
        df = pd.read_csv(file_path)
        col_name = df.columns
        X = pd.DataFrame(df, columns=col_name[2:])
        X = np.array(X)
        str_y = pd.DataFrame(df, columns=col_name[1:2])
        map = {
            "B": 0,
            "M": 1
        }
        y = np.zeros(len(str_y))
        for i in range(len(str_y)):
            y[i] = map[str_y.values[i][0]]
        y = np.array(y).reshape(-1)

        # shuffle and split training set and test set
        idx = np.array(range(len(y)))
        np.random.seed(0)
        np.random.shuffle(idx)
        np.random.seed()
        X_train = X[idx,:]
        y_train = y[idx]
        split_num = int(len(y_train) * 0.8)
        self.X_train = X_train[:split_num, :]
        self.y_train = y_train[:split_num]
        self.X_test = X_train[split_num:,:]
        self.y_test = y_train[split_num:]

        # normalizing
        scaler = MinMaxScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        print("data loaded!!")
        print("train data num: %s, test data num: %s" % (len(self.X_train), len(self.X_test)))

        self.save_cache()

        return True

    def process_data(self):
        super(DataBreastCancer, self).process_data()

if __name__ == '__main__':
    d = DataBreastCancer()
    d.load_data()
    d.process_data()
    d.save_file()