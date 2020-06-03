import numpy as np
from scripts.utils.config_utils import config
from scripts.utils.log_utils import logger
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.database.database import DataBase
from scripts.test.Ensemble.src_class.Baseline_class import BaselineEnsemble


class BaselineEnsembleLipsticks(BaselineEnsemble):
    def __init__(self):

        feature_dir_name = {
            "weights.20-0.9516.h5",
            "weights.10-0.9562.h5",
            "weights.20-0.9551.h5",
            "weights.20-0.9552.h5",
            "weights.10-0.9513.h5",
        }
        super(BaselineEnsembleLipsticks, self).__init__(config.lipsticks,
                                               feature_dir_name)

        all_idx = np.zeros(max(self.test_idx) + 100)
        all_idx[np.array(self.test_idx)] = np.array(range(len(self.test_idx)))
        bias_idx = all_idx[np.array(self.test_redundant_idx)].astype(int)
        normal_idx = [i for i in range(len(self.test_idx)) if i not in bias_idx]

        idx = np.array(range(18039))
        np.random.seed(123)
        np.random.shuffle(idx)
        normal_idx = np.array(normal_idx)[idx[:2000]].tolist()

        self.bias_idx = bias_idx
        self.normal_idx = normal_idx



if __name__ == '__main__':
    d = BaselineEnsembleLipsticks()
    d.process_data(hist=True)