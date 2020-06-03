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

suffix = "_ADD_100_50cartoon_400tigercat"

name_list = ["black cat", "white cat", "black dog", "white dog",
    "cat and human", "cat cage",
        "cat cartoon", "cat in dress",
        "cat indoor", "cat outdoor",
        "dog and human", "dog cage",
        "dog cartoon", "dog in dress",
        "dog indoor", "dog outdoor",
        "two cat", "two dog",
             "rabbit", "wolf", "tiger", "tiger-cat", "husky", "leopard"]

data = Data(config.animals, suffix)
X_train, y_train,_ ,_ , X_test, y_test = \
    data.get_data("all")
sub_y_data_name = os.path.join(config.data_root, config.animals, config.all_data_cache_name)
sub_y_data = pickle_load_data(sub_y_data_name)
sub_y = sub_y_data["test_sub_y"]

clf = SVC(kernel="linear", verbose=1, max_iter=5000)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print("training score: {}, test score: {}".format(train_score, test_score))
sub_y = np.array(sub_y)
for i, name in enumerate(name_list):

    X = X_test[sub_y==i]
    y = y_test[sub_y==i]
    # if name == "husky":
    #     aaaa = clf.predict(X)
    #     print(aaaa)
    score = clf.score(X, y)
    print(name,":", score)
