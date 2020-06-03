import numpy as np
import os

from scripts.utils.config_utils import config
from scripts.utils.log_utils import logger
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.utils.helper_utils import OoD_metrics as metrics
from scripts.database.database import DataBase
from scripts.test.Ensemble.src_class.Baseline_class import BaselineEnsemble
from scipy.stats import entropy


class BaselineEnsembleREA(BaselineEnsemble):
    def __init__(self, feature_dir_name=None):
        dataname = config.rea
        if feature_dir_name is None:
            feature_dir_name = [
                    "unet_nested_dilated_nopre_mix_33_0.001_repeat_1",
                    "unet_nested_dilated_nopre_mix_33_0.001_repeat_2",
                    "unet_nested_dilated_nopre_mix_33_0.001_repeat_3",
                    "unet_nested_dilated_nopre_mix_33_0.001_repeat_4",
                    # "unet_nested_dilated_nopre_mix_33_0.001_unet_nested",
                    "unet_nested_dilated_nopre_mix_33_0.001_unet_nested_dilated-wce-eldice-bce",
            ]
        super(BaselineEnsembleREA, self).__init__(dataname,
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

    def get_all_entropy(self, M3V=False):
        all_pred_y_prob = []
        for feature_dir_name in self.feature_dir_name:
            feature_dir = os.path.join(self.feature_dir, feature_dir_name)
            train_info = pickle_load_data(os.path.join(feature_dir, "train_feature.npy"))
            val_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
            test_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
            X = np.concatenate((train_info["features"],
                                val_info["features"],
                                test_info["features"]), axis=0)
            pred_y_prob = np.concatenate((train_info["pred_cls"],
                                val_info["pred_cls"],
                                test_info["pred_cls"]), axis=0)
            y = np.concatenate((train_info["label_cls"],
                                val_info["label_cls"],
                                test_info["label_cls"]), axis=0)

            pred_y_prob = pred_y_prob[:,0]
            all_pred_y_prob.append(pred_y_prob)
        pred_y_prob = np.array(all_pred_y_prob)
        print(pred_y_prob.shape)
        if M3V is True:
            None

        score = pred_y_prob.mean(axis=0)
        dist = np.zeros((pred_y_prob.shape[1], 2))
        dist[:,0] = score
        dist[:,1] = 1 - score
        all_entropy = entropy(dist.T)
        return all_entropy



if __name__ == '__main__':
    d = BaselineEnsembleREA()
    # d.process_data(hist=True)
    # d.ensemble_result_save("_step1_12345")
    # d.process_data()
    feature_dir_name = [
        "unet_nested_dilated_nopre_mix_33_0.001_repeat_1",
        "unet_nested_dilated_nopre_mix_33_0.001_repeat_2",
        "unet_nested_dilated_nopre_mix_33_0.001_repeat_3",
        "unet_nested_dilated_nopre_mix_33_0.001_repeat_4",
        "unet_nested_dilated_nopre_mix_33_0.001_repeat_1-40",
        "unet_nested_dilated_nopre_mix_33_0.001_repeat_2-40",
        "unet_nested_dilated_nopre_mix_33_0.001_repeat_3-30",
        "unet_nested_dilated_nopre_mix_33_0.001_repeat_4-30",
        "unet_nested_dilated_nopre_mix_33_0.001_unet_nested_dilated-wce-eldice-bce",
    ]
    import itertools
    for num in range(3,7):
        for f_d_n in itertools.combinations(range(len(feature_dir_name)), num):
            dir = [feature_dir_name[i] for i in f_d_n]
            print(dir)
            d = BaselineEnsembleREA(dir)
            d.process_data()
