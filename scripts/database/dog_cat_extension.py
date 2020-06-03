import numpy as np
import os
import scipy.io as sio
from time import time
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_mldata
import tensorflow as tf
from scipy.interpolate import interp1d
from PIL import Image
import matplotlib as mpl
import math
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

class DataDogCatExtension(DataBase):
    def __init__(self, suffix=""):
        dataname = config.dog_cat_extension
        super(DataDogCatExtension, self).__init__(dataname, suffix)

    def preprocessing_data(self):
        self.class_name = ["cat", "dog"]
        self.class_name_encoding = {
            self.class_name[0]: 0,
            self.class_name[1]: 1
        }
        X = []
        y = []
        sub_y = []
        count = 0
        origin_data_dir = os.path.join(self.raw_data_dir, "origin_data")
        sub_class_name_list = os.listdir(origin_data_dir)
        for idx, sub_class_name in enumerate(sub_class_name_list):
            sub_class_dir = os.path.join(origin_data_dir, sub_class_name)
            cls = None
            if sub_class_name.count("cat") > 0:
                cls = 0
            elif sub_class_name.count("dog") > 0:
                cls = 1
            else:
                raise ValueError("ERROR")
            img_name_list = os.listdir(sub_class_dir)
            print(sub_class_name, ": ", len(img_name_list))
            for img_name in img_name_list:
                src = os.path.join(sub_class_dir, img_name)
                target = os.path.join(self.images_dir, str(count) + ".jpg")
                count = count + 1
                y.append(cls)
                sub_y.append(idx)
                try:
                    shutil.copy(src, target)
                except Exception as e:
                    print(e)
        assert len(y) == count
        assert len(sub_y) == count

        self.y = y
        self.train_idx = []
        self.train_redundant_idx = []
        self.valid_idx = []
        self.valid_redundant_idx = []
        self.test_idx = list(range(len(y)))
        self.test_redundant_idx = list(range(len(y)))

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
            "test_redundant_idx": self.test_redundant_idx,
            "sub_y": sub_y
        }
        self.save_cache()

    def load_data(self, loading_from_buffer=True):
        super(DataDogCatExtension, self).load_data(loading_from_buffer)
        self.class_name = self.all_data["class_name"]
        self.class_name_encoding = self.all_data["class_name_encoding"]
        self.X = self.all_data["X"]
        self.y = self.all_data["y"]
        self.y = np.array(self.y)
        self.train_idx = self.all_data["train_idx"]
        self.train_redundant_idx = self.all_data["train_redundant_idx"]
        self.valid_idx = self.all_data["valid_idx"]
        self.valid_redundant_idx = self.all_data["valid_redundant_idx"]
        self.test_idx = self.all_data["test_idx"]
        self.test_redundant_idx = self.all_data["test_redundant_idx"]
        self.sub_y = self.all_data["sub_y"]

    def feature_extraction(self):
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        for idx, data_dir in enumerate(dirs):
            if idx == 0:
                selected_idx = self.train_idx
            elif idx == 1:
                selected_idx = self.valid_idx
            elif idx == 2:
                selected_idx = self.test_idx
            for cls, cls_name in enumerate(self.class_name):
                cls_dir = os.path.join(data_dir, cls_name)
                check_dir(cls_dir)
                if len(selected_idx) == 0:
                    continue
                selected_y = np.array(self.y)[np.array(selected_idx)]
                cls_idx = np.array(selected_idx)[selected_y==cls]
                for i in cls_idx:
                    src = os.path.join(self.images_dir, str(i) + ".jpg")
                    target = os.path.join(cls_dir, str(i) + ".jpg")
                    shutil.copy(src, target)

    def img_format_checking(self):
        img_name_list = os.listdir(self.images_dir)
        for img_name in img_name_list:
            src = os.path.join(self.images_dir, img_name)
            try:
                Image.open(src)
            except Exception as e:
                print(e)
                data = np.array(cv2.imread(src))
                img = Image.fromarray(data)
                img.save(src)
            # try:
            #     img_data = Image.open(src)
            #     img_data = img_data.resize((512, 512), Image.ANTIALIAS)
            #     img_data.save(src)
            # except Exception as e:
            #     print(e)
            #     print(src)

    def pretrain_get_features(self):
        model_names = ["vgg", "resnet50", "xception", "inceptionv3",
                       "inceptionresnet", "mobilenet"]
        # model_names = ["mobilenet", "nasnet"]
        for model_name in model_names:
            super(DataDogCatExtension, self).pretrain_get_model(model_name=model_name)


    def postprocess_data(self, weight_name, if_return=False):
        feature_dir = os.path.join(self.feature_dir, weight_name)
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        # X = np.zeros((self.y.shape[0], 1024))
        X = None
        all_pred_y = np.zeros(30000)
        for data_dir in dirs:
            file_prefix = os.path.split(data_dir)[1].split(".")[0]
            data_filename = os.path.join(feature_dir, file_prefix + config.pkl_ext)
            if not os.path.exists(data_filename):
                logger.warn("{} does not exist, skip!".format(data_filename))
                continue
            mat = pickle_load_data(data_filename)
            features = mat["features"][0]
            pred_y = mat["features"][1].reshape(-1)
            filenames = mat["filename"]
            for idx, name in enumerate(filenames):
                name = name.replace("\\", "/")
                cls, img_name = name.split("/")
                img_id, _ = img_name.split(".")
                img_id = int(img_id)
                all_pred_y[int(img_id)] = pred_y[idx]
                if len(features.shape) > 2:
                    features = features.reshape(features.shape[0], -1)
                if X is None:
                    X = np.zeros((self.y.shape[0], features.shape[1]))
                X[img_id, :] = features[idx, :]

        self.X = X
        if if_return:
            logger.info("'if return' flag is enabled. Returning immediately!")
            return X

        super(DataDogCatExtension, self).postprocess_data(weight_name)

    def inplace_process_data(self):
        cnn_features_dir_name = [
            # "weights.20-0.9922.h5",
            # "inceptionresnet_imagenet",
            # "inceptionv3_imagenet",
            # "mobilenet_imagenet",
            # "resnet50_imagenet",
            # "vgg_imagenet",
            # "xception_imagenet",
            # "sift-200",
            # # "HOG",
            # "HOG-kmeans-200",
            # "LBP",
            "LBP-hist",
            # "superpixel-500",
            # "sift-1000"

        ]
        for weight_name in cnn_features_dir_name:
            X = self.postprocess_data(weight_name, if_return=True)
            filename = os.path.join(self.feature_dir, weight_name, "X.pkl")
            pickle_save_data(filename, X)

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
                mask = np.ones(cam_img.shape) * 0.3
                mask[cam_img>150] = 1.0
                img_data = Image.open(original_img_path)
                img_data = img_data.resize((512, 512), Image.ANTIALIAS)
                img_data = np.array(img_data).astype(float)
                img_data = (img_data * mask[:,:,None])
                img_data = img_data.astype(np.uint8)
                cam_img = Image.fromarray(img_data)
                cam_img.save(img_path)

    def _postprocess_saliency_map(self, weight_name):
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
            for idx, name in enumerate(filenames):
                name = name.replace("\\", "/")
                cls, img_name = name.split("/")
                img_id, _ = img_name.split(".")
                img_id = int(img_id)
                img_path = os.path.join(saliency_map_dir, str(img_id) + ".jpg")
                cam = saliency_map_data[idx, :,:,:]
                w,h = cam.shape[:2]
                cam = cam.reshape(-1, cam.shape[2])
                cam = np.dot(cam, weights).reshape(w,h)
                cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
                cam = cam.astype(np.uint8)
                cam_img = Image.fromarray(cam)
                cam_img = cam_img.resize((512, 512), Image.ANTIALIAS)
                cam = np.array(cam_img)
                cam = cv2.applyColorMap(cam, color_gradient).astype(np.uint8)
                cam_img = Image.fromarray(cam)
                cam_img.save(img_path)

    def save_prediction(self, weight_name, if_return=False):
        feature_dir = os.path.join(self.feature_dir, weight_name)
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        # X = np.zeros((self.y.shape[0], 1024))
        X = None
        all_pred_y = np.zeros(30000)
        for data_dir in dirs:
            file_prefix = os.path.split(data_dir)[1].split(".")[0]
            data_filename = os.path.join(feature_dir, file_prefix + config.pkl_ext)
            if not os.path.exists(data_filename):
                logger.warn("{} does not exist, skip!".format(data_filename))
                continue
            mat = pickle_load_data(data_filename)
            features = mat["features"][0]
            pred_y = mat["features"][1].reshape(-1)
            filenames = mat["filename"]
            pred_file = open(os.path.join(self.data_dir, "prediction.txt"),"w")
            for idx, y in enumerate(pred_y):
                pred_file.writelines("%d: %.4f\n"%(idx, y))



if __name__ == '__main__':
    # d = DataDogCatExtension()
    # # for i in ["LBP-hist"]:
    # #     d.lowlevel_features(i)
    # #     exit()
    # # d.preprocessing_data()
    # d.load_data()
    # # d.inplace_process_data()
    # d.postprocess_saliency_map("weights.20-0.9922.h5")
    # d.save_saliency_map_data("weights.20-0.9922.h5")
    # d.feature_extraction()
    # d._save_features_and_results("weights.20-0.9922.h5")
    # d.postprocess_data("weights.20-0.9922.h5")
    # d.save_file()
    # d.img_format_checking()
    for idx, suffix in enumerate([""]):
        # weights_name = ["weights.20-0.7654.h5",
        #                 "weights.20-0.8087.h5",
        #                 "weights.20-0.8259.h5"]
        weights_name = ["weights.20-0.7350.h5"]
        d = DataDogCatExtension(suffix)
        # # exit()
        # d.ft_save_bottleneck()
        # d.ft_train_top_model()
        # d.ft_train_conv_layer()
        # d._save_features_and_results(weights_name[idx])
        d.load_data()
        # d.save_prediction(weights_name[idx])
        # d.save_file(suffix)
        # d.save_saliency_map_data(weights_name[idx])
        d.postprocess_saliency_map(weights_name[idx])