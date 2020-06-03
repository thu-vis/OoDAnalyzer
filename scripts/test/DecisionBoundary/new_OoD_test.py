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

suffix = "_add_70"

name_list = ["cat and human", "cat cage",
        "cat cartoon", "cat in dress",
        "cat indoor", "cat outdoor",
        "dog and human", "dog cage",
        "dog cartoon", "dog in dress",
        "dog indoor", "dog outdoor",
        "two cat", "two dog"]
class_name_list = ["and human", "cage", "cartoon", "in dress", "indoor", "outdoor", "wo"]


dog_cat_data = Data(config.dog_cat, suffix)
X_train, y_train,_ ,_ , X_test, y_test = \
    dog_cat_data.get_data("all")

dog_cat_extension_data = Data(config.dog_cat_extension, suffix)
OoD_X_train, OoD_y_train,_ ,_, OoD_X_test, OoD_y_test = \
    dog_cat_extension_data.get_data("all")
all_data_name = os.path.join(config.data_root, config.dog_cat_extension, config.all_data_cache_name)
all_data = pickle_load_data(all_data_name)
sub_y = all_data["sub_y"]
sub_y = np.array(sub_y)
print(OoD_X_test.shape)

rearrage_sub_y = []
for idx, i in enumerate(sub_y):
    name = name_list[i]
    name = name.strip("cat").strip("dog").strip(" ")
    s = class_name_list.index(name)
    if s == 2:
        rearrage_sub_y.append(OoD_y_test[idx])
    else:
        rearrage_sub_y.append(7)
rearrage_sub_y = np.array(rearrage_sub_y)

clf = SVC(kernel="linear", verbose=1, max_iter=5000)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
OoD_score = clf.score(OoD_X_test, OoD_y_test)
average_score = (test_score * len(y_test) + OoD_score * len(OoD_y_test)) / \
                (len(y_test)+ len(OoD_y_test))
print("\n training acc: {}, test acc: {}, OoD acc: {}, average acc: {}"
      .format(train_score, test_score, OoD_score, average_score))

train_num = X_train.shape[0]
#
# embed_path = os.path.join(config.data_root, config.dog_cat_extension,
#                           "embed" + suffix + config.pkl_ext)
# if not os.path.exists(embed_path):
#     X = np.concatenate((X_train, X_test))
#     tsne = TSNE(n_components=2, random_state=123)
#     X_embeddings = tsne.fit_transform(X)
#     pickle_save_data(embed_path, X_embeddings)
# else:
#     print(embed_path, "exists")
#     X_embeddings = pickle_load_data(embed_path)
# color_map = plt.get_cmap("tab10")(np.array(y_train))
# ax = plt.subplot(221)
# ax.set_title("training instances")
# ax.scatter(X_embeddings[:train_num,0], X_embeddings[:train_num, 1],
#            c=color_map,s=8)
# color_map = plt.get_cmap("tab10")(np.array(OoD_y_test))
# ax = plt.subplot(222)
# ax.set_title("all test instances")
# ax.scatter(X_embeddings[train_num:,0], X_embeddings[train_num:, 1],
#            c=color_map,s=8)
# color_map = plt.get_cmap("tab10")(np.array(rearrage_sub_y))
# color_map[rearrage_sub_y==7] = 0
# ax = plt.subplot(224)
# ax.set_title("cartoon instance")
# ax.scatter(X_embeddings[train_num:,0], X_embeddings[train_num:, 1],
#            c=color_map,s=8)
# plt.show()
#
#
# exit()

img_root = os.path.join(config.data_root, config.dog_cat_extension, "images")
error_root = os.path.join(config.data_root, config.dog_cat_extension,
                          "right-error", "set" +suffix ,"error")
right_root = os.path.join(config.data_root, config.dog_cat_extension,
                          "right-error", "set" +suffix ,"right")

sub_score = []
sub_num = np.zeros(len(class_name_list))
scores = np.zeros(len(class_name_list))
for i, name in enumerate(name_list):
    X = OoD_X_test[sub_y==i]
    y = OoD_y_test[sub_y==i]
    idx = np.array(range(len(OoD_y_test)))[sub_y==i]
    # pred_y = clf.predict(X)
    score = clf.score(X, y)
    print(score)
    num = len(y)
    cls_name = name.strip("cat").strip("dog").strip(" ")
    cls_name_idx = class_name_list.index(cls_name)
    scores[cls_name_idx] = ( scores[cls_name_idx] * sub_num[cls_name_idx] \
                            + score * num ) / (sub_num[cls_name_idx] + num)
    sub_num[cls_name_idx] += num
print("*************************************************")
for s in scores:
    print(s)

# for i, name in enumerate(name_list):
#     X = OoD_X_test[sub_y==i]
#     y = OoD_y_test[sub_y
#     idx = np.array(range(len(OoD_y_test)))[sub_y==i]
#     pred_y = clf.predict(X)
#     sub_error_root = os.path.join(error_root, name)
#     check_dir(sub_error_root)
#     sub_right_root = os.path.join(right_root, name)
#     check_dir(sub_right_root)
#     for j in range(len(pred_y)):
#         if pred_y[j] != y[j]:
#             id = idx[j]
#             src = os.path.join(img_root, str(id) + ".jpg")
#             target = os.path.join(sub_error_root, str(id) + ".jpg")
#             shutil.copy(src, target)
#         else:
#             id = idx[j]
#             src = os.path.join(img_root, str(id) + ".jpg")
#             target = os.path.join(sub_right_root, str(id) + ".jpg")
#             shutil.copy(src, target)