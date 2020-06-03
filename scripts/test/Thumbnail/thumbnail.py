import numpy as np
import os
import scipy.io as sio
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from PIL import Image
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Input, MaxPooling2D
from keras import applications
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_save_data, pickle_load_data
from scripts.database.database import DataBase
from scripts.utils.data_utils import Data
from scripts.utils.log_utils import logger
from scripts.utils.embedder_utils import Embedder
import shutil
from scipy.stats import entropy


class thumbnail():
    def __init__(self, dataname):
        self.dataname = dataname

    def save_thumbnail(self):
        thumbnail_dir = os.path.join(config.data_root,
                                     self.dataname,
                                     "thumbnail")
        check_dir(thumbnail_dir)
        image_dir = os.path.join(config.data_root,
                                 self.dataname,
                                 "images")
        img_name_list = os.listdir(image_dir)
        for img_name in img_name_list:
            original_image_path = os.path.join(image_dir, img_name)
            img = Image.open(original_image_path)
            img = img.resize((30,30), Image.ANTIALIAS)
            target_thumbnail_path = os.path.join(thumbnail_dir, img_name)
            img.save(target_thumbnail_path)


if __name__ == '__main__':
    d =  thumbnail(config.animals)
    d.save_thumbnail()