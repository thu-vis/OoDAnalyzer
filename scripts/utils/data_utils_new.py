import os
import numpy as np

from .helper_utils import pickle_save_data, pickle_load_data, unit_norm_for_each_col
from .config_utils import config
from .log_utils import logger


class Data(object):
    def __init__(self, dataname):
        self.dataname = dataname

        self.class_name = []
        self.class_name_encoding = {}
        # Note: X_train_redundant is not included in X_train, but X_test_redundant is included in X_test.
        self.X = None
        self.embed_X = None
        self.y = None
        self.train_idx = []
        self.train_redundant_idx = []
        self.valid_idx = []
        self.valid_redundant_idx = []
        self.test_idx = []
        self.test_redundant_idx = []
        # additional information can be store here
        self.add_info = {}

        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None

        self._load_data()

    def _load_data(self):
        filename = os.path.join(config.data_root,
                                self.dataname,
                                config.processed_dataname)
        mat = pickle_load_data(filename)

        self.mat = mat

        self.X = mat[config.X_name]
        self.y = mat[config.y_name].reshape(-1)
        self.embed_X = mat[config.embed_X_name]
        self.train_idx = mat[config.train_idx_name]
        self.train_redundant_idx = mat[config.train_redundant_idx_name]
        self.valid_idx = mat[config.valid_idx_name]
        self.valid_redundant_idx = mat[config.valid_redundant_idx_name]
        self.test_idx = mat[config.test_idx_name]
        self.test_redundant_idx = mat[config.test_redundant_idx_name]

        self.X_train = self.X[np.array(self.train_idx), :]
        self.y_train = self.y[np.array(self.train_idx)]
        if len(self.valid_idx) > 0:
            self.X_valid = self.X[np.array(self.valid_idx), :]
            self.y_valid = self.y[np.array(self.valid_idx)]
        else:
            self.X_valid = np.array([])
            self.y_valid = np.array([])
        self.X_test = self.X[np.array(self.test_idx), :]
        self.y_test = self.y[np.array(self.test_idx)]

        self.embed_X_train = self.embed_X[np.array(self.train_idx), :]
        self.embed_X_train = unit_norm_for_each_col(self.embed_X_train)
        if len(self.valid_idx) > 0:
            self.embed_X_valid = self.embed_X[np.array(self.valid_idx), :]
            self.embed_X_valid = unit_norm_for_each_col(self.embed_X_valid)
        else:
            self.embed_X_valid = np.array([])
        self.embed_X_test = self.embed_X[np.array(self.test_idx), :]
        self.embed_X_test = unit_norm_for_each_col(self.embed_X_test)
        return True


    def if_embed_data_exist(self):
        """
        whether embed data is precomputed
        :return:
        """
        # TODO:
        return True


    def get_embed_X(self, data_type="train"):
        """
        read embedding data if it exists
        :param data_type:
        :return:
        """
        if data_type == "train":
            return self.embed_X_train
        elif data_type == "test":
            return self.embed_X_test
        elif data_type == "valid":
            return self.embed_X_valid
        elif data_type == "all":
            return [self.embed_X_train, self.embed_X_valid, self.embed_X_test]
        else:
            logger.warn("data type doest match!")
            raise ValueError

    def get_data(self, data_type="train"):
        """
        read data
        :param data_type: "train", "valid", "test", or "all"
        :return:
        """
        if data_type == "train":
            return [self.X_train, self.y_train]
        elif data_type == "test":
            return [self.X_test, self.y_test]
        elif data_type == "valid":
            return [self.X_valid, self.y_valid]
        elif data_type == "all":
            return [self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test]
        else:
            logger.warn("data type doest match!")
            raise ValueError

    def get_manifest(self):
        manifest = {
            config.train_instance_num_name: len(self.train_idx),
            config.valid_instance_num_name: len(self.valid_idx),
            config.test_instance_num_name: len(self.test_idx),
            config.label_names_name: self.mat[config.class_name],
            config.feature_dim_name: self.X.shape[1]
        }
        return manifest

if __name__ == '__main__':
    # data test
    d = Data(config.dog_cat)
