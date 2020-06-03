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
        self.train_redundant_idx = []
        self.valid_idx = []
        self.valid_redundant_idx = []
        self.test_idx = []
        self.test_redundant_idx = []
        # additional information can be store here
        self.add_info = {}
        self.suffix = suffix

        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None

        self.clf = None
        self.prediction = None
        self.pred_prob = None
        self.pred_train = None
        self.pred_valid = None
        self.pred_test = None

        self._load_data()
        # self._training()


    def _load_data(self):
        filename = os.path.join(config.data_root,
                                self.dataname,
                                "processed_data" + self.suffix + config.pkl_ext)
        mat = pickle_load_data(filename)

        self.mat = mat
        if self.dataname == config.rea:
            self.mat[config.class_name] = ["REA-free", "REA"]
        elif self.dataname == config.animals or self.dataname=="Animals-leopard":
            self.mat[config.class_name] = ['cat', 'dog', 'rabbit', 'wolf', 'tiger']
            # self.y[9344:9386] = 1
            # self.y[9472:9474] = 1

        self.X = mat[config.X_name]
        self.y = np.array(mat[config.y_name]).reshape(-1)
        self.embed_X = mat[config.embed_X_name]
        self.all_embed_X = mat[config.all_embed_X_name]
        self.train_idx = mat[config.train_idx_name]
        self.train_redundant_idx = mat[config.train_redundant_idx_name]
        self.valid_idx = mat[config.valid_idx_name]
        self.valid_redundant_idx = mat[config.valid_redundant_idx_name]
        self.test_idx = mat[config.test_idx_name]
        self.test_redundant_idx = mat[config.test_redundant_idx_name]
        self.pred_prob = mat["prediction"]
        self.confidence = np.max(self.pred_prob, axis=1)

        self.prediction = self.pred_prob.argmax(axis=1)
        self.pred_train = self.prediction[np.array(self.train_idx)]
        self.pred_valid = np.array([])
        self.pred_test = self.prediction[np.array(self.test_idx)]

        if len(self.train_idx) > 0:
            self.X_train = self.X[np.array(self.train_idx), :]
            self.y_train = self.y[np.array(self.train_idx)]
        else:
            self.X_train = np.array([])
            self.y_train = np.array([])
        if len(self.valid_idx) > 0:
            self.X_valid = self.X[np.array(self.valid_idx), :]
            self.y_valid = self.y[np.array(self.valid_idx)]
        else:
            self.X_valid = np.array([])
            self.y_valid = np.array([])
        self.X_test = self.X[np.array(self.test_idx), :]
        self.y_test = self.y[np.array(self.test_idx)]

        if len(self.train_idx) > 0:
            try:
                self.embed_X_train = self.embed_X[np.array(self.train_idx), :]
                self.embed_X_train = unit_norm_for_each_col(self.embed_X_train)
            except:
                self.embed_X_train = None
        else:
            self.embed_X_train = np.array([])
        if len(self.valid_idx) > 0:
            try:
                self.embed_X_valid = self.embed_X[np.array(self.valid_idx), :]
                self.embed_X_valid = unit_norm_for_each_col(self.embed_X_valid)
            except:
                self.embed_X_valid = None
        else:
            self.embed_X_valid = np.array([])
        try:
            self.embed_X_test = self.embed_X[np.array(self.test_idx), :]
            self.embed_X_test = unit_norm_for_each_col(self.embed_X_test)
        except:
            self.embed_X_test = None

        try:
            filename = os.path.join(config.data_root,
                                    self.dataname,
                                    "all_entropy" + self.suffix + config.pkl_ext)
            self.entropy = pickle_load_data(filename)
        except:
            None

        self._OoD_norm_by_confidence()

        return True

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
            return [self.embed_X_train, self.embed_X_valid, self.embed_X_test]
        elif embed_method == "all":
            return self.all_embed_X

    def get_data(self, data_type="train"):
        """
        read data
        :param data_type: "train", "valid", "test", or "all"
        :return:
        """
        return [self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test]

    def get_prediction(self, data_type="train"):
        return [self.pred_train, self.pred_valid, self.pred_test]

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
            config.valid_instance_num_name: len(self.valid_idx),
            config.test_instance_num_name: len(self.test_idx),
            config.label_names_name: self.mat[config.class_name],
            config.feature_dim_name: self.X.shape[1],
            "train-acc": train_acc
        }
        return manifest

    def _training(self, kernel="linear", C=1, gamma="auto"):
        if self.clf is not None:
            logger.warn("the clf is trained, skip training process!!!")
            return

        svm_model_path = os.path.join(config.data_root, self.dataname, "svm_model.pkl")
        prediction_path = os.path.join(config.data_root, self.dataname, "prediction.pkl")
        if os.path.exists(svm_model_path):
            self.clf = pickle_load_data(svm_model_path)
            logger.info("svm model exists, loading")
        else:
            logger.info("svm model does not exist, training...")
            self.clf = SVC(kernel=kernel,
                           C=C, gamma=gamma,
                           verbose=1, max_iter=-1, probability=True)
            print("parameter (grid.training):", self.clf.get_params())
            print("training data shape:{}, test data shape: {}".format(self.X_train.shape, self.X_test.shape))
            self.clf.fit(self.X_train, self.y_train)
            train_score = self.clf.score(self.X_train, self.y_train)
            test_score = self.clf.score(self.X_test, self.y_test)
            if kernel == "linear":
                weights = self.clf.coef_
                margin = 1.0 / ((weights ** 2).sum()) ** 0.5
            else:
                margin = "not defined"
            print("\n training acc: {}, test acc: {}, margin value: {}."
                  .format(train_score, test_score, margin))
            pickle_save_data(svm_model_path, self.clf)
            logger.info("svm model training process finished, saving...")

        if os.path.exists(prediction_path):
            logger.info("prediction result exists, loading...")
            self.prediction = pickle_load_data(prediction_path)
            redundant_idx = np.array(self.test_redundant_idx)
            try:
                redundant_pred_y = self.prediction[redundant_idx]
                redundant_y = self.y[redundant_idx]
                print("correct: {}, total: {}".format((redundant_pred_y == redundant_y).sum(),
                                                      len(redundant_y)))
            except Exception as e:
                print(e)
        else:
            logger.info("svm model predicting...")
            self.prediction = self.clf.predict(self.X)
            pickle_save_data(prediction_path, self.prediction)

        self.pred_train = self.prediction[np.array(self.train_idx)]
        try:
            self.pred_valid = self.prediction[np.array(self.valid_idx)]
        except Exception as e:
            print(e)
            self.pred_valid = np.array([])
        self.pred_test = self.prediction[np.array(self.test_idx)]

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
