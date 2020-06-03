import numpy as np
from scripts.utils.config_utils import config
from scripts.utils.log_utils import logger
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.database.database import DataBase
from scripts.test.Ensemble.src_class.Ensemble_classifier import EnsembleClassifier


class EnsembleClassifierDogCat(EnsembleClassifier):
    def __init__(self):
        dataname = config.dog_cat
        feature_dir_name = {
            # "weights.20-0.7408.h5",
            # "weights.20-0.7350.h5",
            "weights.20-0.7344.h5",
            "weights.20-0.7147.h5",
            "weights.20-0.9922.h5",
            # "inceptionresnet_imagenet",
            "inceptionv3_imagenet",
            "mobilenet_imagenet",
            "resnet50_imagenet",
            # "vgg_imagenet",
            # "xception_imagenet",
            "sift-200",
            # "HOG-200",
            # "LBP-hist",
            "superpixel-500",
            "orb-200",
            "brief-200"
        }
        super(EnsembleClassifierDogCat, self).__init__(dataname,
                                               feature_dir_name)

        all_idx = np.zeros(max(self.test_idx) + 100)
        all_idx[np.array(self.test_idx)] = np.array(range(len(self.test_idx)))
        bias_idx = all_idx[np.array(self.test_redundant_idx)].astype(int)
        normal_idx = [i for i in range(len(self.test_idx)) if i not in bias_idx]

        idx = np.array(range(len(bias_idx)))
        np.random.seed(4)
        np.random.shuffle(idx)
        bias_idx = bias_idx[idx[:640]]

        self.bias_idx = bias_idx
        self.normal_idx = normal_idx

    def merge_result(self, test_predys, _test_predy):
        if test_predys is None:
            test_predys = _test_predy
        else:
            test_predys = test_predys + _test_predy
        return test_predys

if __name__ == '__main__':
    d = EnsembleClassifierDogCat()
    d.ensemble_result_logistic([1e0])
    d.ensemble_result_logistic([1e-5,1e0, 1e5])