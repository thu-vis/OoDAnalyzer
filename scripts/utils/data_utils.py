import os
import numpy as np

from sklearn.svm import SVC

from .helper_utils import pickle_save_data, pickle_load_data, unit_norm_for_each_col
from .config_utils import config
from .log_utils import logger


class Data(object):
    def __init__(self, dataname, suffix=""):
        self.dataname = dataname

        self.class_name = []
        self.class_name_encoding = {}
        # Note: X_train_redundant is not included in X_train, but X_test_redundant is included in X_test.
        self.X = None
        self.embed_X = None
        self.y = None
        self.train_idx = []
        self.test_idx = []
        # additional information can be store here
        self.add_info = {}
        self.suffix = suffix

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.clf = None
        self.prediction = None
        self.pred_prob = None
        self.pred_train = None
        self.pred_test = None

        self._load_data()


    def _load_data(self):
        filename = os.path.join(config.data_root,
                                self.dataname,
                                "data" + self.suffix + config.pkl_ext)
        mat = pickle_load_data(filename)
        self.mat = mat

        self.X = mat["X"]
        self.y = np.array(mat["y"]).reshape(-1)
        self.train_idx = mat["train_idx"]
        self.test_idx = mat["test_idx"]
        self.pred_prob = mat["pred_y"]
        self.confidence = np.max(self.pred_prob, axis=1)

        self.prediction = self.pred_prob.argmax(axis=1)
        self.pred_train = self.prediction[np.array(self.train_idx)]
        self.pred_test = self.prediction[np.array(self.test_idx)]

        if len(self.train_idx) > 0:
            self.X_train = self.X[np.array(self.train_idx), :]
            self.y_train = self.y[np.array(self.train_idx)]
        else:
            self.X_train = np.array([])
            self.y_train = np.array([])

        self.X_test = self.X[np.array(self.test_idx), :]
        self.y_test = self.y[np.array(self.test_idx)]

        self._load_embed_X()

        if len(self.train_idx) > 0:
            try:
                self.embed_X_train = self.embed_X[np.array(self.train_idx), :]
                self.embed_X_train = unit_norm_for_each_col(self.embed_X_train)
            except:
                self.embed_X_train = None
        else:
            self.embed_X_train = np.array([])

        try:
            self.embed_X_test = self.embed_X[np.array(self.test_idx), :]
            self.embed_X_test = unit_norm_for_each_col(self.embed_X_test)
        except:
            self.embed_X_test = None

        try:
            filename = os.path.join(config.data_root,
                                    self.dataname,
                                    "ood_score" + self.suffix + config.pkl_ext)
            self.entropy = pickle_load_data(filename)
        except:
            None

        self._OoD_norm_by_confidence()

        return True

    def _load_embed_X(self):
        filename = os.path.join(config.data_root,
                                self.dataname, "embed_X.pkl")
        if os.path.exists(filename):
            self.embed_X = pickle_load_data(filename)
        else:
            # TODO:
            None

    def if_embed_data_exist(self):
        """
        whether embed data is precomputed
        :return:
        """
        # TODO:
        return True

    def get_embed_X(self, data_type="train", embed_method="tsne"):
        """
        read embedding data if it exists
        :param data_type:
        :return:
        """
        if embed_method == "tsne":
            return [self.embed_X_train, np.array([]), self.embed_X_test]
        elif embed_method == "all":
            return self.all_embed_X

    def get_data(self, data_type="train"):
        return [self.X_train, self.y_train, np.array([]), np.array([]), self.X_test, self.y_test]

    def get_prediction(self, data_type="train"):
        return [self.pred_train, np.array([]), self.pred_test]

    def get_confidence(self):
        return [self.confidence[np.array(self.train_idx)], self.confidence[np.array(self.test_idx)]]

    def get_acc(self):
        acc = (self.pred_train == self.y_train).sum() / len(self.pred_train)
        return acc

    def get_entropy(self):
        all_entropy = self.entropy
        train_entropy = all_entropy[np.array(self.train_idx)]
        test_entropy = all_entropy[np.array(self.test_idx)]
        # test_entropy /= test_entropy.max()
        return train_entropy, test_entropy

    def get_manifest(self):
        train_acc = self.get_acc()
        manifest = {
            config.train_instance_num_name: len(self.train_idx),
            config.test_instance_num_name: len(self.test_idx),
            config.label_names_name: self.mat[config.class_name],
            config.feature_dim_name: self.X.shape[1],
            "train-acc": train_acc
        }
        return manifest

    def get_similar(self, id, k):
        all_idx = self.train_idx + self.test_idx
        all_feature = self.X
        center_feature = self.X[id]
        distances = [np.sum(np.square(center_feature - all_feature[idx])) for idx in all_idx]
        index = [all_idx[i] for i in sorted(range(len(all_idx)), key=distances.__getitem__)[:k]]
        return index

    def _OoD_norm_by_confidence(self):
        conf_threshold = 0.95
        test_ent = self.entropy[np.array(self.test_idx)]
        test_conf = self.confidence[np.array(self.test_idx)]
        for label in range(len(self.mat['class_name'])):
            ent = test_ent[self.y_test == label]
            conf = test_conf[self.y_test == label]
            ent_high_conf = ent[conf > conf_threshold]
            ent_high_conf = (ent_high_conf - ent_high_conf.min()) / (ent_high_conf.max() - ent_high_conf.min()) + 0.4
            ent_low_conf = ent[conf <= conf_threshold]
            ent_low_conf = (ent_low_conf - ent_low_conf.min()) / (ent_low_conf.max() - ent_low_conf.min())
            ent[conf > conf_threshold] = ent_high_conf
            ent[conf <= conf_threshold] = ent_low_conf
            test_ent[self.y_test == label] = ent

        # test_ent *= test_conf
        self.entropy[np.array(self.test_idx)] = test_ent


        self.entropy[np.array(self.test_idx)] = (self.entropy[np.array(self.test_idx)] - self.entropy[np.array(self.test_idx)].min()) / (self.entropy[np.array(self.test_idx)].max() - self.entropy[np.array(self.test_idx)].min())

if __name__ == '__main__':
    # data test
    d = Data(config.dog_cat)
