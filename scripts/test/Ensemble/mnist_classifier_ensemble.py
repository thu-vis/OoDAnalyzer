import numpy as np
import os
from scripts.utils.config_utils import config
from scripts.utils.log_utils import logger
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.utils.data_utils import Data
from scripts.database.database import DataBase
from scripts.test.Ensemble.src_class.Ensemble_classifier import EnsembleClassifier


class EnsembleClassifierMNIST(EnsembleClassifier):
    def __init__(self):
        dataname = config.mnist
        feature_dir_name = {
            "weights.13-0.9571.h5",
            "weights.17-0.9592.h5",
            "weights.11-0.9437.h5",
            # # "inceptionresnet_imagenet",
            "inceptionv3_imagenet",
            "mobilenet_imagenet",
            "resnet50_imagenet",
            # # "vgg_imagenet",
            # # "xception_imagenet",
            "sift-200",
            # # # "HOG",
            # "HOG-200",
            # # "LBP",
            # "LBP-hist",
            "superpixel",
            "orb-200",
            "brief-200"
        }
        self.dataname = dataname
        self.data = Data(self.dataname)
        self.feature_dir_name = feature_dir_name
        self.data_dir = os.path.join(config.data_root, dataname)
        self.feature_dir = os.path.join(self.data_dir, "feature")
        self.train_idx = np.array(range(50000))
        self.test_idx = np.array(range(50000, 70000))
        self.test_redundant_idx = np.array(range(60000, 70000))
        all_idx = np.zeros(max(self.test_idx) + 100)
        all_idx[np.array(self.test_idx)] = np.array(range(len(self.test_idx)))
        bias_idx = all_idx[np.array(self.test_redundant_idx)].astype(int)
        normal_idx = [i for i in range(len(self.test_idx)) if i not in bias_idx]

        self.bias_idx = bias_idx
        self.normal_idx = normal_idx

    def merge_result(self, test_predys, _test_predy):
        if test_predys is None:
            test_predys = _test_predy
        else:
            test_predys = test_predys + _test_predy
        return test_predys

if __name__ == '__main__':
    d = EnsembleClassifierMNIST()
    d.ensemble_result_logistic([1e0])
    d.ensemble_result_logistic([1e-5, 1e0, 1e5])