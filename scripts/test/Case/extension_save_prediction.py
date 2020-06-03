import numpy as np
import os
import math
import tensorflow as tf
from time import time
import shutil

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from scipy.spatial.distance import cdist
from PIL import Image
from lapjv import lapjv

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.utils.data_utils import Data

data = Data(config.dog_cat)
train_idx = data.train_idx
src = os.path.join(config.data_root, config.dog_cat, "images")
target = os.path.join(config.data_root, config.dog_cat, "train_data")
for idx in train_idx:
    img_src = os.path.join(src, str(idx) + ".jpg")
    img_target = os.path.join(target, str(idx) + ".jpg")
    shutil.copy(img_src, img_target)