import numpy as np
from scripts.utils.config_utils import config
from scripts.utils.log_utils import logger
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.database.database import DataBase
from scripts.test.Ensemble.src_class.Ensemble_classifier import EnsembleClassifier


class EnsembleClassifierREA(EnsembleClassifier):
    def __init__(self):
        dataname = config.rea
        feature_dir_name = [
                "unet_nested_dilated_nopre_mix_33_0.001_repeat_1",
                "unet_nested_dilated_nopre_mix_33_0.001_repeat_2",
                "unet_nested_dilated_nopre_mix_33_0.001_repeat_3",
                "unet_nested_dilated_nopre_mix_33_0.001_repeat_4",
                "unet_nested_dilated_nopre_mix_33_0.001_unet_nested_dilated-wce-eldice-bce",
                # "inceptionresnet_imagenet",
                # "inceptionv3_imagenet",
                # "mobilenet_imagenet",
                # "resnet50_imagenet",
                # "vgg_imagenet",
                # "xception_imagenet",
                # "sift-200",
                # "brief-200",
                # "orb-200",
                # "surf-200",
                # "superpixel",
        ]
        super(EnsembleClassifierREA, self).__init__(dataname,
                                               feature_dir_name)

        all_idx = np.zeros(max(self.test_idx) + 100)
        all_idx[np.array(self.test_idx)] = np.array(range(len(self.test_idx)))
        self.test_redundant_idx = self.test_redundant_idx + \
            [9119, 9120, 9121, 9122, 9124, 9168, 9169, 9170, 9172, 9174, 9176, 9181, 9186, 9187,
            9256, 9348, 9349, 9350, 9352, 9355, 9356, 9358, 9359, 9363, 9366, 9367, 9369, 9371, 9372,
            9376, 9378, 9379, 9380, 9381, 9382, 9383, 9384, 9385, 9399, 9475, 9476, 9477, 8478,
            9479, 9480, 9481, 9482, 9485, 9486, 9487, 9488, 9489, 9490, 9495, 9496, 9497, 9498,]
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
    d = EnsembleClassifierREA()
    # d.grid_search_logistic([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    # d.ensemble_result_save([1e-5, 1e0, 1e5], "step1")
    # d.ensemble_result_logistic([1e0])

