import numpy as np
import os
import scipy.io as sio
from time import time
import warnings
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_mldata
import tensorflow as tf
from scipy.interpolate import interp1d
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

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_save_data, pickle_load_data
from scripts.database.database import DataBase
from scripts.utils.log_utils import logger
from scripts.utils.embedder_utils import Embedder
import shutil


gradient = np.linspace(0, 1, 256)
gradient = np.vstack(gradient).reshape(-1)
color_gradient = plt.get_cmap("RdYlGn")(gradient)
color_gradient = (color_gradient[:,:3] * 255).astype(np.uint8)
# color_gradient = color_gradient[:,np.array([2,1,0])]
color_gradient = color_gradient.reshape(color_gradient.shape[0],1,-1)

class DataDogCat(DataBase):
    def __init__(self, suffix=""):
        dataname = config.dog_cat
        super(DataDogCat, self).__init__(dataname, suffix)

    def preprocessing_data(self):
        self.class_name = ["cat", "dog"]
        self.class_name_encoding = {
            self.class_name[0]: 0,
            self.class_name[1]: 1
        }
        # save images in image_dir
        images_dir = self.images_dir
        if len(os.listdir(images_dir)) == 25000:
            logger.warn("skip 'saving images to image_dir' process.")
        else:
            src_image_dir = os.path.join(config.raw_data_root, self.dataname, "train_full")
            src_image_names = os.listdir(src_image_dir)
            for src_name in src_image_names:
                name = src_name.split(".")
                target_id = int(self.class_name_encoding[name[0]]) * 25000 / 2 + int(name[1])
                target_id = int(target_id)
                target_name = str(target_id) + ".jpg"
                src = os.path.join(src_image_dir, src_name)
                target = os.path.join(images_dir, target_name)
                img = Image.open(src)
                img = img.resize((512, 512))
                # shutil.copy(src, target)
                img.save(target)

        # split train, valid, test
        all_test_dir = os.path.join(config.raw_data_root, self.dataname, "normal_test")
        self.test_idx = []
        for class_name in self.class_name:
            origin_dir = os.path.join(all_test_dir, class_name)
            origin_imgs = os.listdir(origin_dir)
            for img_name in origin_imgs:
                id = int(img_name.split(".")[0]) + 12500 * self.class_name_encoding[class_name]
                self.test_idx.append(id)
        logger.warn("info confirm: total test instances number: {}".format(len(self.test_idx)))

        all_bias_dir = os.path.join(config.raw_data_root, self.dataname, "all_bias")
        bias_idx = []
        for class_name in self.class_name:
            origin_dir = os.path.join(all_bias_dir, class_name)
            origin_imgs = os.listdir(origin_dir)
            for img_name in origin_imgs:
                id = int(img_name.split(".")[0]) + 12500 * self.class_name_encoding[class_name]
                bias_idx.append(id)
        logger.warn("info confirm: total bias instances number: {}".format(len(bias_idx)))

        self.y = np.array([0]*12500 + [1]*12500).reshape(-1).astype(int)
        self.train_idx = []
        self.train_redundant_idx = []
        self.valid_idx = []
        self.valid_redundant_idx = []
        self.test_redundant_idx = []
        for i in range(25000):
            if (not self.test_idx.count(i)) and (bias_idx.count(i)):
                self.train_idx.append(i)
            elif (not self.test_idx.count(i)) and (not bias_idx.count(i)):
                self.train_redundant_idx.append(i)
            elif (self.test_idx.count(i)) and (not bias_idx.count(i)):
                self.test_redundant_idx.append(i)
        logger.warn("info confirm: train num: {}, train_redundant num:{}, test_redundant num:{}"
                    .format(len(self.train_idx), len(self.train_redundant_idx), len(self.test_redundant_idx)))

        self.all_data = {
            "class_name": self.class_name,
            "class_name_encoding": self.class_name_encoding,
            "X": None,
            "y": self.y,
            "train_idx": self.train_idx,
            "train_redundant_idx": self.train_redundant_idx,
            "valid_idx": self.valid_idx,
            "valid_redundant_idx": self.valid_redundant_idx,
            "test_idx": self.test_idx,
            "test_redundant_idx": self.test_redundant_idx
        }
        self.save_cache()

    def load_data(self, loading_from_buffer=True):
        super(DataDogCat, self).load_data(loading_from_buffer)
        self.class_name = self.all_data["class_name"]
        self.class_name_encoding = self.all_data["class_name_encoding"]
        self.X = self.all_data["X"]
        self.y = self.all_data["y"]
        self.train_idx = self.all_data["train_idx"]
        self.train_redundant_idx = self.all_data["train_redundant_idx"]
        self.valid_idx = self.all_data["valid_idx"]
        self.valid_redundant_idx = self.all_data["valid_redundant_idx"]
        self.test_idx = self.all_data["test_idx"]
        self.test_redundant_idx = self.all_data["test_redundant_idx"]

    def process_data(self):
        """
        fine tune here
        :return:
        """


    def inplace_process_data(self):
        cnn_features_dir_name = [
            # "weights.20-0.7408.h5",
            # "weights.20-0.7350.h5",
            # "weights.20-0.7344.h5",
            # "weights.20-0.7147.h5",
            # "inceptionresnet_imagenet",
            # "inceptionv3_imagenet",
            # "mobilenet_imagenet",
            # "resnet50_imagenet",
            # "vgg_imagenet",
            # "xception_imagenet",
            # "sift",
            # "HOG",
            # "HOG-kmeans-200",
            # "LBP",
            # "LBP-hist",
            # "superpixel-500",
            # "sift-1000"
            "orb-200",
            "brief-200"

        ]
        for weight_name in cnn_features_dir_name:
            X = self.postprocess_data(weight_name, if_return=True)
            filename = os.path.join(self.feature_dir, weight_name, "X.pkl")
            pickle_save_data(filename, X)

    def postprocess_data(self, weight_name, if_return=False):
        """

        :return:
        """
        feature_dir = os.path.join(self.feature_dir, weight_name)
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        # X = np.zeros((self.y.shape[0], 1024))
        X = None
        # tmp = np.zeros(30000)
        # tmp[:25000] = np.array(self.y)
        # self.y = tmp
        all_pred_y = np.zeros(30000)
        for data_dir in dirs:
            file_prefix = os.path.split(data_dir)[1].split(".")[0]
            data_filename = os.path.join(feature_dir, file_prefix + config.pkl_ext)
            if not os.path.exists(data_filename):
                logger.warn("{} does not exist, skip!".format(data_filename))
                continue
            mat = pickle_load_data(data_filename)
            features = mat["features"][0]
            # pred_y = mat["features"][1].reshape(-1)
            filenames = mat["filename"]
            for idx, name in enumerate(filenames):
                name = name.replace("\\", "/")
                cls, img_name = name.split("/")
                img_id, _ = img_name.split(".")
                if int(img_id) <25000:
                    img_id = self.class_name_encoding[cls] * 12500 + int(img_id)
                else:
                    if int(img_id) > 26000:
                        self.y[int(img_id)] = 1
                    else:
                        self.y[int(img_id)] = 0
                # all_pred_y[int(img_id)] = pred_y[idx]
                img_id = int(img_id)
                if len(features.shape) > 2:
                    features = features.reshape(features.shape[0], -1)
                if X is None:
                    X = np.zeros((len(self.y), features.shape[1]))
                X[img_id,:] = features[idx,:]
                if data_dir == self.train_data_dir and self.train_idx.count(img_id) == 0:
                    self.train_idx.append(img_id)
            print("new train_idx len: ", len(self.train_idx))
        self.X = X
        if if_return:
            logger.info("'if return' flag is enabled. Returning immediately!")
            return X

        super(DataDogCat, self).postprocess_data(weight_name)

    def pretrain_get_features(self):
        model_names = ["vgg", "resnet50", "xception", "inceptionv3",
                       "inceptionresnet", "mobilenet"]
        # model_names = ["mobilenet", "nasnet"]
        for model_name in model_names:
            super(DataDogCat, self).pretrain_get_model(model_name=model_name)

    def save_saliency_map_data(self, weight_name):
        logger.info("begin save saliency map data")
        feature_dir = os.path.join(self.feature_dir, weight_name)
        check_dir(feature_dir)
        weight_path = os.path.join(self.model_weight_dir, weight_name)
        self.ft_get_model()
        output_model = self.saliency_map_model
        output_model.summary()
        # import IPython; IPython.embed()
        output_model.load_weights(weight_path)
        weights = output_model.get_layer("model_3").get_layer(name="dense_1").get_weights()[0]

        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        for data_dir in dirs:
            datagen = ImageDataGenerator(rescale=1. / 255)
            generator = datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                class_mode="binary",
                shuffle=False
            )
            y = generator.classes[generator.index_array]
            filename = generator.filenames
            nb_samples = y.reshape(-1).shape[0]
            if nb_samples == 0:
                continue
            file_prefix = os.path.split(data_dir)[1].split(".")[0]
            logger.info("{} sampling number: {}".format(file_prefix, nb_samples))
            features = output_model.predict_generator(
                generator, math.ceil(nb_samples / self.batch_size))
            saliency_map = features
            print(saliency_map.shape)
            mat = {
                "weights": weights,
                "saliency_map": saliency_map,
                "filename": filename
            }
            pickle_save_data(os.path.join(feature_dir, file_prefix + "_saliency_map" + config.pkl_ext),
                             mat)

    def postprocess_saliency_map(self, weight_name):
        saliency_map_dir = self.saliency_map_dir
        feature_dir = os.path.join(self.feature_dir, weight_name)
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        for data_dir in dirs:
            file_prefix = os.path.split(data_dir)[1].split(".")[0]
            data_filename = os.path.join(feature_dir, file_prefix + "_saliency_map" + config.pkl_ext)
            if not os.path.exists(data_filename):
                logger.warn("{} does not exist, skip!".format(data_filename))
                continue
            mat = pickle_load_data(data_filename)
            saliency_map_data = mat["saliency_map"]
            filenames = mat["filename"]
            weights = mat["weights"]
            pred_y = pickle_load_data(
                os.path.join(feature_dir, file_prefix + config.pkl_ext))["features"][1]
            pred_y = np.array(pred_y).reshape(-1)
            for idx, name in enumerate(filenames):
                name = name.replace("\\", "/")
                cls, img_name = name.split("/")
                img_id, _ = img_name.split(".")
                img_id = self.class_name_encoding[cls] * 12500 + int(img_id)
                img_id = int(img_id)
                img_path = os.path.join(saliency_map_dir, str(img_id) + ".jpg")
                original_img_path = os.path.join(self.images_dir, str(img_id) + ".jpg")
                cam = saliency_map_data[idx, :,:,:]
                w,h = cam.shape[:2]
                cam = cam.reshape(-1, cam.shape[2])
                cam = np.dot(cam, weights).reshape(w,h)
                cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
                cam = cam.astype(np.uint8)
                cam_img = Image.fromarray(cam)
                cam_img = cam_img.resize((512, 512), Image.ANTIALIAS)
                cam_img = np.array(cam_img)
                if pred_y[idx] < 0.5:
                    cam_img = 255 - cam_img
                mask = np.zeros(cam_img.shape)
                mask[cam_img>150] = 255
                img_data = Image.open(original_img_path)
                img_data = img_data.resize((512, 512), Image.ANTIALIAS)
                img_data = np.array(img_data).astype(float)
                img_data = (img_data + mask[:,:,None]) / 2.0
                img_data = img_data.astype(np.uint8)
                cam_img = Image.fromarray(img_data)
                cam_img.save(img_path)



if __name__ == '__main__':
    d = DataDogCat()
    # for i in ["sift-200", "HOG-200", "LBP-hist"]:
    # for i in ["LBP-hist"]:
    #     d.lowlevel_features(i)
    #     exit()
    d.load_data()
    # d.postprocess_data("weights.20-0.9922.h5")
    # d.save_saliency_map_data("weights.20-0.9922.h5")
    d.inplace_process_data()
    # d.pretrain_get_features()
    # d.pretrain_get_model()
    # d.preprocessing_data()
    # d._save_features_and_results("weights.20-0.9922.h5")
    # d.load_data()
    # d.postprocess_data("weights.20-0.9922.h5")
    # d.save_file()
    # for idx, suffix in enumerate(["_add_10", "_add_20", "_add_30"]):
    # for idx, suffix in enumerate(["_repeat_1", "_repeat_2", "_repeat_3", "_repeat_4"]):
        # weights_name = ["weights.20-0.7654.h5",
        #                 "weights.20-0.8087.h5",
        #                 "weights.20-0.8259.h5"]
        # weights_name = ["weights.20-0.8852.h5",
        #                 "weights.20-0.8526.h5",
        #                 "weights.20-0.8942.h5",
        #                 "weights.20-0.8934.h5"]
        # d = DataDogCat(suffix)
        # # # exit()
        # d.ft_save_bottleneck()
        # d.ft_train_top_model()
        # d.ft_train_conv_layer()
        # d._save_features_and_results(weights_name[idx])
        # d.load_data()
        # d.postprocess_data(weights_name[idx])
        # d.save_file(suffix)