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
X_train, y_train, X_valid, y_valid, X_test, y_test = data.get_data("all")
print(X_train.shape)
pickle_save_data(os.path.join(r"H:\for kelei", "train.pkl"), X_train)

data = Data(config.dog_cat_extension)
X_train, y_train, X_valid, y_valid, X_test, y_test = data.get_data("all")
print(X_test.shape)
pickle_save_data(os.path.join(r"H:\for kelei", "test.pkl"), X_test)