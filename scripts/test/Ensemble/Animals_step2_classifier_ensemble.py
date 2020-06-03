import numpy as np
import os
from scripts.utils.config_utils import config
from scripts.utils.log_utils import logger
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.database.database import DataBase
from scripts.test.Ensemble.src_class.Ensemble_classifier import EnsembleClassifier


class EnsembleClassifierAnimalsStep2(EnsembleClassifier):
    def __init__(self, feature_dir_name = None):
        dataname = config.animals + "-step2"
        if feature_dir_name is None:
            feature_dir_name = [
                    "weights.30-0.8604.h5",
                    # "weights.30-0.8628.h5",
                    # "weights.50-0.8507.h5",
                    # "inceptionresnet_imagenet",
                    # "inceptionv3_imagenet",
                    "mobilenet_imagenet",
                    # "resnet50_imagenet",
                    # "vgg_imagenet",
                    "xception_imagenet",
                    # "sift-200",
                    # "brief-200",
                    # "orb-200",
                    "surf-200",
                    # "superpixel"
            ]
        super(EnsembleClassifierAnimalsStep2, self).__init__(dataname,
                                               feature_dir_name)

        all_idx = np.zeros(max(self.test_idx) + 100)
        all_idx[np.array(self.test_idx)] = np.array(range(len(self.test_idx)))
        bias_idx = all_idx[np.array(self.test_redundant_idx)].astype(int)
        normal_idx = [i for i in range(len(self.test_idx)) if i not in bias_idx]

        self.bias_idx = bias_idx
        self.normal_idx = normal_idx

    # def get_all_entropy(self, subset, M3V=False):
    #     entropy_filename = os.path.join(self.data_dir, "all_entropystep1.pkl")
    #     return pickle_load_data(entropy_filename)

    def merge_result(self, test_predys, _test_predy):
        if test_predys is None:
            test_predys = _test_predy
        else:
            test_predys = test_predys + _test_predy
        return test_predys

if __name__ == '__main__':
    d = EnsembleClassifierAnimalsStep2()
    d.grid_search_logistic([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    d.ensemble_result_save([1e-7, 1e-5, 1e-3, 1e0], suffix="step2_4_feture")
    # d.ensemble_result_logistic([1e0])
    # d.ensemble_result_logistic([1e-7, 1e-5, 1e-3, 1e0], hist=False)
    # d.ensemble_result_logistic([1e-7, 1e-5, 1e-3, 1e0], hist=False)



    # feature_dir_name_list = [
    #     "weights.30-0.8604.h5",
    #     "weights.30-0.8628.h5",
    #     "weights.50-0.8507.h5",
    #     "inceptionresnet_imagenet",
    #     "inceptionv3_imagenet",
    #     "mobilenet_imagenet",
    #     "resnet50_imagenet",
    #     "vgg_imagenet",
    #     "xception_imagenet",
    #     "sift-200",
    #     "brief-200",
    #     "orb-200",
    #     "surf-200",
    #     "superpixel"
    # ]
    #
    # import itertools
    #
    # for number in range(4, len(feature_dir_name_list)):
    #     for s in itertools.combinations(range(len(feature_dir_name_list)), number):
    #         subset = [feature_dir_name_list[i] for i in s]
    #         print(subset)
    #         d = EnsembleClassifierAnimalsStep2(subset)
    #         d.ensemble_result_logistic([1e-7, 1e-5, 1e-3, 1e0],hist=False)
