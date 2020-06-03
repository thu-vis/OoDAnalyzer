import numpy as np
import os
import math
import tensorflow as tf
from time import time

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
from scipy import interpolate
from sklearn import linear_model
import cv2

from multiprocessing import Pool

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.utils.data_utils import Data
from scripts.Grid import GridLayout


Root = r"H:\ImageNet"
dir_list = ["n02123159", "n02123597", "n02128385", "n02129604", "n02325366"]

def extraction(dir_name):
    print(dir_name)
    sift = cv2.xfeatures2d.SIFT_create()
    img_dir = os.path.join(Root, dir_name)
    img_names = os.listdir(img_dir)
    data_des = []
    data_des_num = []
    for img_name in img_names:
        img = cv2.imread(os.path.join(img_dir, img_name))
        try:
            _, d = sift.detectAndCompute(img, None)
        except Exception as e:
            print("custom error: ", e)
            d = np.zeros(128).reshape(1, 128)
        if d is None:
            d = np.zeros(128).reshape(1, 128)
        data_des = data_des + d.tolist()
        data_des_num.append(d.shape[0])
    pickle_save_data(os.path.join(Root, dir_name + ".pkl"), [data_des, data_des_num])

def main():
    # pool =  Pool()
    # res = [pool.apply_async(extraction, (dn,)) for dn in dir_list]
    # for idx, r in enumerate(res):
    #     r.get()
    for dn in dir_list:
        extraction(dn)


if __name__ == '__main__':
    main()
    # extraction("n02123159")