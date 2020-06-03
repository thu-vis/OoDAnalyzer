import numpy as np
import os
import scipy.io as sio
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_mldata
import tensorflow as tf
from tensorflow import keras
from scipy.interpolate import interp1d
from PIL import Image
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

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, accuracy
from scripts.database.database import DataBase

def interpolate(a, x_extend, y_extend):
    x = np.array(range(a.shape[0]))
    xnew = np.linspace(x.min(), x.max(), x_extend)
    f = interp1d(x, a, axis=0)
    a = f(xnew)
    a = a.transpose()
    x = np.array(range(a.shape[0]))
    xnew = np.linspace(x.min(), x.max(), y_extend)
    f = interp1d(x, a, axis=0)
    a = f(xnew)
    a = a.transpose()
    return a


class DataTest(DataBase):
    def __init__(self):
        dataname = config.dog_cat
        super(DataTest, self).__init__(dataname)


if __name__ == '__main__':
    d = DataTest()
    # weight_path = os.path.join(d.model_weight_dir, "weights.20-0.9922.h5")
    d.save_features_and_results("weights.20-0.9922.h5")
    import IPython; IPython.embed()