import numpy as np
import os
from scipy.stats import entropy
from scripts.utils.config_utils import config
from scripts.utils.log_utils import logger
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.utils.helper_utils import OoD_metrics as metrics
from scripts.database.database import DataBase
from scripts.test.Ensemble.src_class.Baseline_class import BaselineEnsemble


class BaselineEnsembleMNIST(BaselineEnsemble):
    def __init__(self):
        dataname = "MNIST_mlp"
        feature_dir_name = {
            "feature_repeat_1",
            "feature_repeat_2",
            "feature_repeat_3",
            "feature_repeat_4",
            "feature_repeat_5"
        }

        self.feature_dir_name = feature_dir_name
        self.data_dir = os.path.join(config.data_root, dataname)
        self.feature_dir = os.path.join(self.data_dir, "feature")
        self.test_idx = np.array(range(50000, 70000))
        self.test_redundant_idx = np.array(range(60000, 70000))

        all_idx = np.zeros(max(self.test_idx) + 100)
        all_idx[np.array(self.test_idx)] = np.array(range(len(self.test_idx)))
        bias_idx = all_idx[np.array(self.test_redundant_idx)].astype(int)
        normal_idx = [i for i in range(len(self.test_idx)) if i not in bias_idx]


        self.bias_idx = bias_idx
        self.normal_idx = normal_idx


    def process_data(self):
        all_pred_y = []
        for feature_dir_name in self.feature_dir_name:
            feature_dir = os.path.join(self.feature_dir, feature_dir_name)
            test_data = pickle_load_data(os.path.join(feature_dir, "res.pkl"))
            pred_y = test_data
            all_pred_y.append(pred_y)
        all_individual_pred_y = np.array(all_pred_y).transpose(1, 2, 0)
        all_pred_y = np.array(all_pred_y).transpose(1, 2, 0).mean(axis=2)
        ent = entropy(all_pred_y.transpose())
        all_idx = np.zeros(max(self.test_idx) + 100)
        all_idx[np.array(self.test_idx)] = np.array(range(len(self.test_idx)))
        bias_idx = all_idx[np.array(self.test_redundant_idx)].astype(int)
        normal_idx = [i for i in range(len(self.test_idx)) if i not in bias_idx]

        all_idx = np.array(normal_idx[:] + bias_idx.tolist())
        print(ent[np.array(normal_idx)].mean(), ent[np.array(bias_idx)].mean())
        y = np.zeros(len(self.test_idx)).astype(int)
        y[bias_idx] = 1
        ent = ent[all_idx];
        y = y[all_idx]

        # self.entropy_analysis(ent[np.array(normal_idx)], ent[np.array(bias_idx)])

        metrics(ent, y)

if __name__ == '__main__':
    d = BaselineEnsembleMNIST()
    d.process_data()