import numpy as np
import os
import scipy.io as sio
from time import time
import warnings
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_mldata
from sklearn.manifold import TSNE
from scipy.interpolate import interp1d
import tensorflow as tf
from PIL import Image
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Input
from keras import applications
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from multiprocessing import Pool

from scripts.utils.config_utils import config
from scripts.utils.log_utils import logger
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.database.database import DataBase
from scripts.database.animals import DataAnimals



class step2EntropyDebug(DataAnimals):
    def __init__(self, dataname=None, suffix="", class_num=5):
        dataname = config.animals_step2
        super(step2EntropyDebug, self).__init__(dataname, suffix)

    def entropy_analysis(self):
        ent = pickle_load_data(os.path.join(self.data_dir, "all_entropystep0_4_feature_7.pkl"))
        processed_data = pickle_load_data(os.path.join(self.data_dir, "all_data_cache.pkl"))
        all_sub_y = processed_data["all_sub_y"].astype(int)
        # black_cat_ent = ent[all_sub_y==0]
        # white_dog_ent = ent[all_sub_y==3]
        test_categories_idx = [[] for i in range(22)]
        for idx in processed_data["test_redundant_idx"]:
            try:
                test_categories_idx[all_sub_y[idx]].append(idx)
            except:
                print(idx)

        for idxs in test_categories_idx:
            if len(idxs) ==0:
                print(0)
            else:
                print(len(idxs),ent[np.array(idxs)].mean())
        a = 1

    def entropy_remove_others(self):
        # to get all_sub_y
        tmp_data = pickle_load_data(os.path.join(config.data_root, config.animals_step1
                                                       , "all_data_cache.pkl"))
        all_sub_y = tmp_data["all_sub_y"].astype(int)

        processed_data = pickle_load_data(os.path.join(self.data_dir, "processed_data_pred.pkl"))
        test_idx = processed_data["test_idx"]
        removed_categories = [4,5,7,8,9,10,11,13,14,15,16,17]
        for i in removed_categories:
            removed_idx = np.array(range(len(all_sub_y)))[all_sub_y==i]
            for idx in removed_idx:
                try:
                    test_idx.remove(removed_idx)
                except:
                    print(i, idx)

        processed_data["test_idx"] = test_idx
        pickle_save_data(os.path.join(self.data_dir,"processed_data_remove_cage_and_others.pkl"), processed_data)


if __name__ == '__main__':
    s = step2EntropyDebug()
    s.entropy_remove_others()