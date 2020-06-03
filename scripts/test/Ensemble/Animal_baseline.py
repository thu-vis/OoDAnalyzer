import numpy as np
import os

from scripts.utils.config_utils import config
from scripts.utils.log_utils import logger
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.database.database import DataBase
from scripts.test.Ensemble.src_class.Baseline_class import BaselineEnsemble


class BaselineEnsembleDogCat(BaselineEnsemble):
    def __init__(self):
        dataname = config.animals
        feature_dir_name = [
            "weights.40-0.7544.h5",
            "weights.40-0.7456.h5",
            "weights.30-0.7442.h5",
            "weights.30-0.7411.h5",
            # "weights.20-0.9922.h5",
        ]
        super(BaselineEnsembleDogCat, self).__init__(dataname,
                                               feature_dir_name)

        all_idx = np.zeros(max(self.test_idx) + 100)
        all_idx[np.array(self.test_idx)] = np.array(range(len(self.test_idx)))
        bias_idx = all_idx[np.array(self.test_redundant_idx)].astype(int)
        normal_idx = [i for i in range(len(self.test_idx)) if i not in bias_idx]


        self.bias_idx = bias_idx
        self.normal_idx = normal_idx

if __name__ == '__main__':
    d = BaselineEnsembleDogCat()
    # d.process_data(hist=True)
    d.ensemble_result_save("_step0_baseline")