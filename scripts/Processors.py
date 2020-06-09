import numpy as np
import os
import sys
import ctypes
import math
from time import time

from .utils.helper_utils import pickle_save_data, pickle_load_data, unit_norm_for_each_col
from .utils.config_utils import config
from .utils.log_utils import logger

class Processor(object):
    def __init__(self, dataname=None):
        self.dataname = dataname

    def from_old_to_new(self):
        filename = os.path.join(config.data_root,
                                self.dataname,
                                "processed_data" + config.pkl_ext)
        mat = pickle_load_data(filename)
        filename = os.path.join(config.data_root,
                                    self.dataname,
                                    "all_entropy" + config.pkl_ext)
        entropy = pickle_load_data(filename)

        print(mat.keys())
        print(mat["class_name"])
        print(mat["X_name"].shape)
        print(entropy.shape)
        train_idx = mat["train_idx"]
        test_idx = mat["test_idx"]
        all_idx = train_idx + test_idx
        pickle_save_data(os.path.join(config.data_root, self.dataname, "all_idx.pkl"), all_idx)
        exit()
        X = mat["X_name"]
        y = mat["y_name"].reshape(-1)
        embed_X = mat["embed_X"]
        prediction = mat["prediction"]
        # print(prediction.shape)
        print(embed_X.shape)

        r_X = X[np.array(all_idx)]
        r_y = y[np.array(all_idx)]
        r_embed_X = embed_X[np.array(all_idx)]
        r_p = prediction[np.array(all_idx)]
        r_train_idx = list(range(len(train_idx)))
        r_test_idx = list(range(len(train_idx), len(all_idx)))
        r_entropy = entropy[all_idx]

        r_mat = {
            "class_name": mat["class_name"],
            "X": r_X,
            "y": r_y,
            "pred_y": r_p,
            "train_idx": r_train_idx,
            "test_idx": r_test_idx
        }

        pickle_save_data(os.path.join(config.data_root,
                                self.dataname, "data.pkl"), r_mat)
                                
        pickle_save_data(os.path.join(config.data_root,
                                self.dataname, "ood_score.pkl"), r_entropy)
    
        pickle_save_data(os.path.join(config.data_root,
                                self.dataname, "embed_X.pkl"), r_embed_X)

    def process_ood_score(self):
        None
    
    def process_embedding(self):
        None
