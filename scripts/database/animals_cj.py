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

class DataAnimalsShadow(DataAnimals):
    def __init__(self, suffix="", class_num=5):
        dataname = config.animals_step2
        self.class_num = class_num
        super(DataAnimalsShadow, self).__init__(dataname, suffix)

    def modify_processed_data(self):
        mat = pickle_load_data(os.path.join(self.data_dir,
                    "processed_data.pkl"))
        mat["class_name"] = mat["class_name"][:5]
        pickle_save_data(os.path.join(self.data_dir,
                                      config.processed_dataname), mat)
        a = 1

if __name__ == '__main__':
    suffix = ""
    d = DataAnimalsShadow(suffix)
    # d.load_data()
    # d.preprocessing_data(); exit()
    # for i in ["sift-200", "HOG-kmeans-200", "LBP-hist"]:
    # for i in ["sift-200"]:
    #     d.lowlevel_features(i)
    # d.pretrain_get_features()
    # d.sift_features("sift-200")
    # d.sift_features("orb-200")
    # d.sift_features("surf-200")
    # weight_name = ""
    # d._save_features_and_results(weight_name)
    d.modify_processed_data()