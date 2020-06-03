import numpy as np
import os
import scipy.io as sio
from time import time
import warnings
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_mldata
from keras.datasets import cifar10
import tensorflow as tf
from scipy.interpolate import interp1d
from PIL import Image
# import matplotlib as mpl
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt
import cv2
import keras
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, \
    GlobalAveragePooling2D, Conv2D, Input, MaxPooling2D
from keras import applications
from keras import optimizers
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from multiprocessing import Pool
from skimage.feature import hog, local_binary_pattern
from skimage import color

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_save_data, pickle_load_data
from scripts.database.database import DataBase
from scripts.utils.log_utils import logger
from scripts.utils.embedder_utils import Embedder
import shutil


METHOD = "uniform"
radius = 3
n_points = 8 * radius

def list_of_groups(init_list, children_list_len):
    if len(init_list) == 0:
        return []
    sum = 0
    res = []
    for lens in children_list_len:
        res.append(init_list[sum: sum + lens])
        sum = sum + lens
    return res


def _sift_train(start, end, filename, data_dir):
    train_des = []
    train_des_num = []
    sift = cv2.xfeatures2d.SIFT_create()
    for idx in range(start, end):
        img_name = filename[idx]
        img = cv2.imread(os.path.join(data_dir, img_name))
        try:
            _, d = sift.detectAndCompute(img, None)
        except Exception as e:
            print(e)
            d = np.zeros(128).astype(int).reshape(1, 128)
        if d is None:
            d = np.zeros(128).astype(int).reshape(1, 128)
        if d is None:
            d = np.zeros(128).astype(int).reshape(1, 128)
        train_des = train_des + d.astype(int).tolist()
        train_des_num.append(d.shape[0])
        if (idx - start + 1) % 50 == 0:
            print("start: {}, end: {}, now processed: {}"
                  .format(start, end, (idx-start+1)/(end-start+1)))
    return train_des, train_des_num

def _sift_prediction(start, end, filename, data_dir, km):
    labels = []
    des_num = []
    sift = cv2.xfeatures2d.SIFT_create()
    for idx in range(start, end):
        img_name = filename[idx]
        img = cv2.imread(os.path.join(data_dir, img_name))
        try:
            _, d = sift.detectAndCompute(img, None)
        except Exception as e:
            print(e)
            d = np.zeros(128).astype(int).reshape(1, 128)
        if d is None:
            d = np.zeros(128).astype(int).reshape(1, 128)
        if d is None:
            d = np.zeros(128).astype(int).reshape(1, 128)

        labels = labels + km.predict(d).tolist()
        des_num.append(d.shape[0])
        if (idx - start + 1) % 50 == 0:
            print("start: {}, end: {}, now processed: {}"
                  .format(start, end, (idx - start + 1) / (end - start + 1)))
    return labels, des_num

def _hog_train(start, end, filename, data_dir):
    train_des = []
    train_des_num = []
    size = 0
    for idx in range(start, end):
        img_name = filename[idx]
        img = Image.open(os.path.join(data_dir, img_name))
        if min(img.size) < 64:
            img = img.resize((512, 512))
            logger.warn("new img size: {}".format(img.size))
        try:
            img_data = color.rgb2gray(np.array(img))
            fd = hog(img_data, orientations=8,
                     pixels_per_cell=(16, 16),
                     cells_per_block=(4, 4),
                     block_norm='L2', feature_vector=False)
            fd = fd.reshape(fd.shape[0] * fd.shape[1], -1)
            size = fd.shape[1]
        except Exception as e:
            fd = np.zeros(size).astype(int).reshape(1, size)
        train_des = train_des + fd.astype(int).tolist()
        train_des_num.append(fd.shape[0])
        if (idx - start + 1) % 50 == 0:
            print("start: {}, end: {}, now processed: {}"
                  .format(start, end, (idx - start + 1) / (end - start + 1)))
    return train_des, train_des_num

def _hog_prediction(start, end, filename, data_dir, km):
    labels = []
    des_num = []
    size = 0
    for idx in range(start, end):
        img_name = filename[idx]
        img = Image.open(os.path.join(data_dir, img_name))
        if min(img.size) < 64:
            img = img.resize((512, 512))
            logger.warn("new img size: {}".format(img.size))
        try:
            img_data = color.rgb2gray(np.array(img))
            fd = hog(img_data, orientations=8,
                     pixels_per_cell=(16, 16),
                     cells_per_block=(4, 4),
                     block_norm='L2', feature_vector=False)
            fd = fd.reshape(fd.shape[0] * fd.shape[1], -1)
            size = fd.shape[1]
        except Exception as e:
            fd = np.zeros(size).astype(int).reshape(1, size)

        labels = labels + km.predict(fd).tolist()
        des_num.append(fd.shape[0])
        if (idx - start + 1) % 50 == 0:
            print("start: {}, end: {}, now processed: {}"
                  .format(start, end, (idx - start + 1) / (end - start + 1)))
    return labels, des_num

def _lbp_prediction(start, end, filename, data_dir):
    X = []
    for idx in range(start, end):
        img_name = filename[idx]
        img = Image.open(os.path.join(data_dir, img_name))
        try:
            img_data = color.rgb2gray(np.array(img))
        except:
            img_data = np.array(img).mean(axis=2).astype(np.uint8)
        fd = local_binary_pattern(img_data, n_points, radius, METHOD)
        fd = fd.reshape(-1).astype(int)
        bincount = np.bincount(np.array(fd))
        X.append(bincount)
        if (idx - start + 1) % 50 == 0:
            print("start: {}, end: {}, now processed: {}"
                  .format(start, end, (idx - start + 1) / (end - start + 1)))
    return X


class DataMNIST(DataBase):
    def __init__(self, suffix=""):
        dataname = config.mnist
        super(DataMNIST, self).__init__(dataname, suffix)

    def save_raw_data(self):
        # (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        # pickle_save_data(os.path.join(self.data_dir, "raw_data.pkl"),
        #                  [X_train, y_train, X_test, y_test])
        mnist = fetch_mldata("MNIST original")
        target = mnist["target"]
        X = mnist["data"]
        idx = np.array(range(len(target)))
        np.random.seed(123)
        np.random.shuffle(idx)
        X_train = X[idx[:60000]]
        y_train = target[idx[:60000]]
        X_test = X[idx[60000:]]
        y_test = target[idx[60000:]]
        pickle_save_data(os.path.join(self.data_dir, "raw_data.pkl"),
                         [X_train, y_train, X_test, y_test])



    def preprocessing_data(self):
        self.class_name = ["airplane", "automobile", "bird",
                           "cat", "deer", "dog", "frog",
                           "horse", "ship", "truck"]
        self.class_name_encoding = {
            self.class_name[0]: 0,
            self.class_name[1]: 1,
            self.class_name[2]: 2,
            self.class_name[3]: 3,
            self.class_name[4]: 4,
            self.class_name[5]: 5,
            self.class_name[6]: 6,
            self.class_name[7]: 7,
            self.class_name[8]: 8,
            self.class_name[9]: 9
        }

        X_train, y_train, X_test, y_test = pickle_load_data(os.path.join(self.data_dir,
                                                                         "raw_data.pkl"))
        self.train_idx = []
        self.train_redundant_idx = []
        self.valid_idx = []
        self.valid_redundant_idx = []
        self.test_idx = []
        self.test_redundant_idx = []
        self.train_idx = np.array(range(50000)).tolist()
        self.test_idx = np.array(range(50000, 70000)).tolist()
        self.test_redundant_idx = np.array(range(60000, 70000)).tolist()


        global_count = 60000
        bias_test_data_dir = os.path.join(self.data_dir, "normal_test_bias")
        for i in range(10):
            sub_dir = os.path.join(bias_test_data_dir, str(i))
            img_name_list = os.listdir(sub_dir)
            print(i, len(img_name_list))
            for idx, img_name in enumerate(img_name_list):
                img_path = os.path.join(sub_dir, img_name)
                img = Image.open(img_path)
                img = img.resize((64, 64), Image.ANTIALIAS)
                img.save(os.path.join(self.images_dir, str(global_count) + ".jpg"))
                target_image_path = os.path.join(self.test_data_dir, str(i), str(global_count) + ".jpg")
                img.save(target_image_path)
                global_count = global_count + 1

        print("final global count: ",global_count)


        X = np.concatenate((X_train, X_test), axis=0)
        self.y = np.array(y_train.tolist() + y_test.tolist())
        for i in range(X.shape[0]):
            x_data = X[i,:,:,:]
            img = Image.fromarray(x_data)
            img = img.resize((64, 64), Image.ANTIALIAS)
            img.save(os.path.join(self.images_dir, str(i) + ".jpg"))
            # exit()

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

        a = 1


    def load_data(self, loading_from_buffer=True):
        super(DataMNIST, self).load_data(loading_from_buffer)
        self.class_name = self.all_data["class_name"]
        self.class_name_encoding = self.all_data["class_name_encoding"]
        self.X = self.all_data["X"]
        self.y = self.all_data["y"]
        self.y = np.array(self.y).reshape(-1)
        self.train_idx = self.all_data["train_idx"]
        self.train_redundant_idx = self.all_data["train_redundant_idx"]
        self.valid_idx = self.all_data["valid_idx"]
        self.valid_redundant_idx = self.all_data["valid_redundant_idx"]
        self.test_idx = self.all_data["test_idx"]
        self.test_redundant_idx = self.all_data["test_redundant_idx"]

    def ft_get_model(self):
        """
        create the model and according outputs
        :return: None
        """
        if self.model is not None:
            # TODO: anything else
            return

        input_shape = (self.img_width, self.img_height, 3)
        logger.warn("the model input is {}".format(input_shape))

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        self.model = model
        model.summary()
        # import IPython; IPython.embed()
        self.feature_model = Model(inputs=model.input,
                                output = [model.get_layer("dense_1").output,
                                          model.get_layer("dense_2").output])



    def ft_train_conv_layer(self):
        self.ft_get_model()
        model = self.model

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adadelta(),
                      # optimizer = "rmsprop",
                      metrics=['accuracy'])

        # prepare data augmentation configuration
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=0.2,
            horizontal_flip=True)
        valid_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical')
        train_y = train_generator.classes[train_generator.index_array]
        nb_train_samples = train_y.reshape(-1).shape[0]
        valid_generator = valid_datagen.flow_from_directory(
            self.valid_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical')
        valid_y = valid_generator.classes[valid_generator.index_array]
        nb_valid_samples = valid_y.reshape(-1).shape[0]

        checkpointer = ModelCheckpoint(filepath=os.path.join(self.model_weight_dir,
                                                             "weights.{epoch:02d}-{val_acc:.4f}.h5"),
                                       verbose=1)
        model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // self.batch_size,
            epochs=20,
            validation_data=valid_generator,
            validation_steps=nb_valid_samples // self.batch_size,
            callbacks=[checkpointer])

    def _save_features_and_results(self, weight_name):
        logger.info("begin save features and results.")
        feature_dir = os.path.join(self.feature_dir, weight_name)
        check_dir(feature_dir)
        print("feature_dir:", feature_dir)
        weight_path = os.path.join(self.model_weight_dir, weight_name)
        print("weight_name:", weight_path)
        self.ft_get_model()
        output_model = self.feature_model
        output_model.summary()
        # exit()
        output_model.load_weights(weight_path)

        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        for data_dir in dirs:
            datagen = ImageDataGenerator(rescale=1. / 255)
            generator = datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                class_mode="categorical",
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
            # features = features[1:]
            features = features
            logger.info("{} feature shape: {}".format(file_prefix, features[0].shape))
            mat = {
                "features": features,
                "y": y,
                "filename": filename
            }
            pickle_save_data(os.path.join(feature_dir, file_prefix + config.pkl_ext), mat)


    def save_images(self):
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
                # print(max(selected_y))
                # import IPython; IPython.embed()
                cls_idx = np.array(np.array(selected_idx))[selected_y==cls]
                for i in cls_idx:
                    src = os.path.join(self.images_dir, str(i) + ".jpg")
                    target = os.path.join(cls_dir, str(i) + ".jpg")
                    shutil.copy(src, target)


    def sift_features(self, method_name):
        feature_dir = os.path.join(self.feature_dir, str(method_name))
        n_cluster = int(method_name.split("-")[1])
        check_dir(feature_dir)
        dirs = [self.train_data_dir, self.test_data_dir]
        filename_list = []
        y_list = []
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
            y_list.append(y)
            filename = generator.filenames
            filename_list.append(filename)
        train_filename, test_filename = filename_list
        train_y, test_y = y_list
        logger.info("train instance: {}".format(len(train_filename)))


        cpu_kernel = 40
        train_des = []
        train_des_num = []
        test_des = []
        test_des_num = []
        logger.info("number of clusters is {}".format(n_cluster))
        sift = cv2.xfeatures2d.SIFT_create()
        if os.path.exists(os.path.join(feature_dir, "train_raw_feature.pkl")):
            logger.info("train_raw_sift_feature.pkl exists, loading...")
            train_des, train_des_num = pickle_load_data(
                os.path.join(feature_dir, "train_raw_feature.pkl")
            )
        else:
            t1 = time()
            step_size = math.ceil(len(train_filename) / cpu_kernel)
            print("step_size: ", step_size)
            start_ends = []
            train_des_res = [None for i in range(cpu_kernel)]
            train_des_num_res = [None for i in range(cpu_kernel)]
            for i in range(cpu_kernel):
                start_ends.append([i*step_size,
                                   min((i+1)*step_size, len(train_filename))])
            pool = Pool()
            res = [pool.apply_async(_sift_train,
                                    (start, end, train_filename, self.train_data_dir))
                   for start, end in start_ends]
            for idx, r in enumerate(res):
                train_des_res[idx], train_des_num_res[idx] = \
                    r.get()
            print("extraction finished: ", time() - t1)
            for l in train_des_res:
                train_des.extend(l)
            for l in train_des_num_res:
                train_des_num.extend(l)
            print("aggregation finished: ", time() - t1)
            pickle_save_data(os.path.join(feature_dir, "train_raw_feature.pkl"),
                            [train_des, train_des_num])
            print("save finished: ", time() - t1)

        if not os.path.exists(os.path.join(feature_dir, "km.pkl")):
            logger.info("kmeans model does not exist, training kmeans ......")
            km = MiniBatchKMeans(n_clusters=n_cluster, batch_size=5000,
                                 max_no_improvement=10, reassignment_ratio=1e-3, verbose=1, random_state=1223)
            km.fit(train_des)
            pickle_save_data(os.path.join(feature_dir, "km.pkl"), km)
        else:
            logger.info("kmeans model exists")
            km = pickle_load_data(os.path.join(os.path.join(feature_dir, "km.pkl")))
        print("train_filename", len(train_filename))
        km_res = []
        for filename, data_dir in zip([train_filename, test_filename], [self.train_data_dir, self.test_data_dir]):
            t1 = time()
            step_size = math.ceil(len(filename) / cpu_kernel)
            print("step_size: ", step_size)
            start_ends = []
            cluster_labels = []
            des_num = []
            cluster_labels_res = [None for i in range(cpu_kernel)]
            des_num_res = [None for i in range(cpu_kernel)]
            for i in range(cpu_kernel):
                start_ends.append([i * step_size,
                                   min((i + 1) * step_size, len(filename))])
            pool = Pool()
            res = [pool.apply_async(_sift_prediction,
                                    (start, end, filename, data_dir, km))
                   for start, end in start_ends]
            for idx, r in enumerate(res):
                cluster_labels_res[idx], des_num_res[idx] = \
                    r.get()
            print("extraction finished: ", time() - t1)
            for l in cluster_labels_res:
                cluster_labels.extend(l)
            for l in des_num_res:
                des_num.extend(l)
            print("aggregation finished: ", time() - t1)
            km_res.append([cluster_labels, des_num])

        train_labels_lists = list_of_groups(km_res[0][0], km_res[0][1])
        test_labels_lists = list_of_groups(km_res[1][0], km_res[1][1])

        train_X = np.zeros((len(train_labels_lists), n_cluster))
        for i in range(len(train_labels_lists)):
            bincount = np.bincount(np.array(train_labels_lists[i]))
            # bincount = bincount.astype(np.float32) / bincount.sum()
            train_X[i, :len(bincount)] = bincount
        test_X = np.zeros((len(test_labels_lists), n_cluster))
        for i in range(len(test_labels_lists)):
            bincount = np.bincount(np.array(test_labels_lists[i]))
            # bincount = bincount.astype(np.float32) / bincount.sum()
            test_X[i, :len(bincount)] = bincount

        if os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0] == 0:
            logger.info("scaler.pkl exists, loading...")
            scaler = pickle_load_data(os.path.join(feature_dir, "scaler.pkl"))
            logger.info("scaler.pkl is loaded.")
            # train_X = np.array([])
        elif not os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0] != 0:
            logger.info("scaler.pkl does not exist, training...")
            scaler = StandardScaler().fit(train_X)
            train_X = scaler.transform(train_X)
            pickle_save_data(os.path.join(feature_dir, "scaler.pkl"), scaler)
            logger.info("scaler is trained and saved in {}".format(feature_dir))
        else:
            logger.error("training examples do not exist, but are used to train a scaler.")
            exit(0)
        test_X = scaler.transform(test_X)
        print("bin finished: ", time() - t1)

        train_mat = {
            "features": [train_X, 0],
            "y": train_y,
            "filename": train_filename
        }
        test_mat = {
            "features": [test_X, 0],
            "y": test_y,
            "filename": test_filename
        }
        check_dir(feature_dir)
        pickle_save_data(os.path.join(feature_dir, "bias_train.pkl"), train_mat)
        pickle_save_data(os.path.join(feature_dir, "normal_test.pkl"), test_mat)
        logger.info("process finished")

    def HOG_features(self, method_name):
        feature_dir = os.path.join(self.feature_dir, str(method_name))
        n_cluster = int(method_name.split("-")[1])
        check_dir(feature_dir)
        dirs = [self.train_data_dir, self.test_data_dir]
        filename_list = []
        y_list = []
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
            y_list.append(y)
            filename = generator.filenames
            filename_list.append(filename)
        train_filename, test_filename = filename_list
        train_y, test_y = y_list
        logger.info("train instance: {}".format(len(train_filename)))


        cpu_kernel = 40
        train_des = []
        train_des_num = []
        test_des = []
        test_des_num = []
        logger.info("number of clusters is {}".format(n_cluster))
        if os.path.exists(os.path.join(feature_dir, "train_raw_feature.pkl")):
            logger.info("train_raw_sift_feature.pkl exists, loading...")
            train_des, train_des_num = pickle_load_data(
                os.path.join(feature_dir, "train_raw_feature.pkl")
            )
        else:
            t1 = time()
            step_size = math.ceil(len(train_filename) / cpu_kernel)
            print("step_size: ", step_size)
            start_ends = []
            train_des_res = [None for i in range(cpu_kernel)]
            train_des_num_res = [None for i in range(cpu_kernel)]
            for i in range(cpu_kernel):
                start_ends.append([i*step_size,
                                   min((i+1)*step_size, len(train_filename))])
            pool = Pool()
            res = [pool.apply_async(_hog_train,
                                    (start, end, train_filename, self.train_data_dir))
                   for start, end in start_ends]
            for idx, r in enumerate(res):
                train_des_res[idx], train_des_num_res[idx] = \
                    r.get()
            print("extraction finished: ", time() - t1)
            for l in train_des_res:
                train_des.extend(l)
            for l in train_des_num_res:
                train_des_num.extend(l)
            print("aggregation finished: ", time() - t1)
            pickle_save_data(os.path.join(feature_dir, "train_raw_feature.pkl"),
                            [train_des, train_des_num])
            print("save finished: ", time() - t1)

        if not os.path.exists(os.path.join(feature_dir, "km.pkl")):
            logger.info("kmeans model does not exist, training kmeans ......")
            km = MiniBatchKMeans(n_clusters=n_cluster, batch_size=5000,
                                 max_no_improvement=10, reassignment_ratio=1e-3, verbose=1, random_state=1223)
            km.fit(train_des)
            pickle_save_data(os.path.join(feature_dir, "km.pkl"), km)
        else:
            logger.info("kmeans model exists")
            km = pickle_load_data(os.path.join(os.path.join(feature_dir, "km.pkl")))

        km_res = []
        for filename, data_dir in zip([train_filename, test_filename], [self.train_data_dir, self.test_data_dir]):
            t1 = time()
            step_size = math.ceil(len(filename) / cpu_kernel)
            print("step_size: ", step_size)
            start_ends = []
            cluster_labels = []
            des_num = []
            cluster_labels_res = [None for i in range(cpu_kernel)]
            des_num_res = [None for i in range(cpu_kernel)]
            for i in range(cpu_kernel):
                start_ends.append([i * step_size,
                                   min((i + 1) * step_size, len(filename))])
            pool = Pool()
            res = [pool.apply_async(_hog_prediction,
                                    (start, end, filename, data_dir, km))
                   for start, end in start_ends]
            for idx, r in enumerate(res):
                cluster_labels_res[idx], des_num_res[idx] = \
                    r.get()
            print("extraction finished: ", time() - t1)
            for l in cluster_labels_res:
                cluster_labels.extend(l)
            for l in des_num_res:
                des_num.extend(l)
            print("aggregation finished: ", time() - t1)
            km_res.append([cluster_labels, des_num])

        train_labels_lists = list_of_groups(km_res[0][0], km_res[0][1])
        test_labels_lists = list_of_groups(km_res[1][0], km_res[1][1])

        train_X = np.zeros((len(train_labels_lists), n_cluster))
        for i in range(len(train_labels_lists)):
            bincount = np.bincount(np.array(train_labels_lists[i]))
            # bincount = bincount.astype(np.float32) / bincount.sum()
            train_X[i, :len(bincount)] = bincount
        test_X = np.zeros((len(test_labels_lists), n_cluster))
        for i in range(len(test_labels_lists)):
            bincount = np.bincount(np.array(test_labels_lists[i]))
            # bincount = bincount.astype(np.float32) / bincount.sum()
            test_X[i, :len(bincount)] = bincount

        if os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0] == 0:
            logger.info("scaler.pkl exists, loading...")
            scaler = pickle_load_data(os.path.join(feature_dir, "scaler.pkl"))
            logger.info("scaler.pkl is loaded.")
            # train_X = np.array([])
        elif not os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0] != 0:
            logger.info("scaler.pkl does not exist, training...")
            scaler = StandardScaler().fit(train_X)
            train_X = scaler.transform(train_X)
            pickle_save_data(os.path.join(feature_dir, "scaler.pkl"), scaler)
            logger.info("scaler is trained and saved in {}".format(feature_dir))
        else:
            logger.error("training examples do not exist, but are used to train a scaler.")
            exit(0)
        test_X = scaler.transform(test_X)
        print("bin finished: ", time() - t1)

        train_mat = {
            "features": [train_X, 0],
            "y": train_y,
            "filename": train_filename
        }
        test_mat = {
            "features": [test_X, 0],
            "y": test_y,
            "filename": test_filename
        }
        check_dir(feature_dir)
        pickle_save_data(os.path.join(feature_dir, "bias_train.pkl"), train_mat)
        pickle_save_data(os.path.join(feature_dir, "normal_test.pkl"), test_mat)
        logger.info("process finished")

    def LBP_features(self, method_name):
        feature_dir = os.path.join(self.feature_dir, str(method_name))
        check_dir(feature_dir)
        dirs = [self.train_data_dir, self.test_data_dir]
        filename_list = []
        y_list = []
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
            y_list.append(y)
            filename = generator.filenames
            filename_list.append(filename)
        train_filename, test_filename = filename_list
        train_y, test_y = y_list
        logger.info("train instance: {}".format(len(train_filename)))

        cpu_kernel = 40
        km_res = []
        for filename, data_dir in zip([train_filename, test_filename], [self.train_data_dir, self.test_data_dir]):
            t1 = time()
            step_size = math.ceil(len(filename) / cpu_kernel)
            print("step_size: ", step_size)
            start_ends = []
            X = []
            X_res = [None for i in range(cpu_kernel)]
            for i in range(cpu_kernel):
                start_ends.append([i * step_size,
                                   min((i + 1) * step_size, len(filename))])
            pool = Pool()
            res = [pool.apply_async(_lbp_prediction,
                                    (start, end, filename, data_dir))
                   for start, end in start_ends]
            for idx, r in enumerate(res):
                X_res[idx] = \
                    r.get()
            print("extraction finished: ", time() - t1)
            for l in X_res:
                X.extend(l)
            print("aggregation finished: ", time() - t1)
            km_res.append(X)

        train_X = km_res[0]
        tmp_test_X = km_res[1]
        test_X = []
        train_X = np.array(train_X)
        for x in tmp_test_X:
            if len(x) < 26:
                tmp_x = np.zeros(26)
                tmp_x[:len(x)] = x
                x = tmp_x
            test_X.append(x)
        test_X = np.array(test_X)
        # import IPython; IPython.embed()
        if os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0] == 0:
            logger.info("scaler.pkl exists, loading...")
            scaler = pickle_load_data(os.path.join(feature_dir, "scaler.pkl"))
            logger.info("scaler.pkl is loaded.")
            # train_X = np.array([])
        elif not os.path.exists(os.path.join(feature_dir, "scaler.pkl")) and train_X.shape[0] != 0:
            logger.info("scaler.pkl does not exist, training...")
            scaler = StandardScaler().fit(train_X)
            train_X = scaler.transform(train_X)
            pickle_save_data(os.path.join(feature_dir, "scaler.pkl"), scaler)
            logger.info("scaler is trained and saved in {}".format(feature_dir))
        else:
            logger.error("training examples do not exist, but are used to train a scaler.")
            exit(0)
        test_X = scaler.transform(test_X)
        print("bin finished: ", time() - t1)

        train_mat = {
            "features": [train_X, 0],
            "y": train_y,
            "filename": train_filename
        }
        test_mat = {
            "features": [test_X, 0],
            "y": test_y,
            "filename": test_filename
        }
        check_dir(feature_dir)
        pickle_save_data(os.path.join(feature_dir, "bias_train.pkl"), train_mat)
        pickle_save_data(os.path.join(feature_dir, "normal_test.pkl"), test_mat)
        logger.info("process finished")

    def pretrain_get_model(self, model_name="vgg", weight="imagenet"):
        input_shape = (self.img_width, self.img_height, 3)
        logger.warn("the model input is {}".format(input_shape))
        if model_name == "vgg":
            model = applications.VGG16(include_top=True, input_shape=input_shape, weights=weight)
            model = Model(inputs=model.input, outputs=[model.get_layer("fc2").output,
                                                       model.get_layer("predictions").output])
        elif model_name == "resnet50":
            model = applications.ResNet50(include_top=True, input_shape=input_shape, weights=weight)
            model = Model(inputs=model.input, outputs=[model.get_layer("avg_pool").output,
                                                       model.get_layer("fc1000").output])
        elif model_name == "xception":
            model = applications.Xception(include_top=True, input_shape=input_shape, weights=weight)
            model = Model(inputs=model.input, outputs=[model.get_layer("avg_pool").output,
                                                       model.get_layer("predictions").output])
        elif model_name == "inceptionv3":
            model = applications.InceptionV3(include_top=True, input_shape=input_shape, weights=weight)
            model = Model(inputs=model.input, outputs=[model.get_layer("avg_pool").output,
                                                       model.get_layer("predictions").output])
        elif model_name == "inceptionresnet":
            model = applications.InceptionResNetV2(include_top=True, input_shape=input_shape, weights=weight)
            model = Model(inputs=model.input, outputs=[model.get_layer("avg_pool").output,
                                                       model.get_layer("predictions").output])
        elif model_name == "mobilenet":
            model = applications.MobileNet(include_top=True, input_shape=input_shape, weights=weight)
            model = Model(inputs=model.input, outputs=[model.get_layer("global_average_pooling2d_1").output,
                                                       model.get_layer("reshape_2").output])
        else:
            raise ValueError("{} model is not support now".format(model_name))
        feature_model = model
        feature_dir = os.path.join(self.feature_dir, str(model_name) + "_" + str(weight))
        check_dir(feature_dir)
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        for data_dir in dirs:
            datagen = ImageDataGenerator(rescale=1. / 255)
            generator = datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                class_mode="categorical",
                shuffle=False
            )
            y = generator.classes[generator.index_array]
            filename = generator.filenames
            nb_samples = y.reshape(-1).shape[0]
            if nb_samples == 0:
                continue
            file_prefix = os.path.split(data_dir)[1].split(".")[0]
            logger.info("{} sampling number: {}".format(file_prefix, nb_samples))
            features = feature_model.predict_generator(
                generator, math.ceil(nb_samples / self.batch_size))
            logger.info("{} feature shape: {}".format(file_prefix, features[0].shape))
            mat = {
                "features": features,
                "y": y,
                "filename": filename
            }
            pickle_save_data(os.path.join(feature_dir, file_prefix + config.pkl_ext), mat)

    def pretrain_get_features(self):
        model_names = ["vgg", "resnet50", "xception", "inceptionv3",
                       "inceptionresnet", "mobilenet"]
        # model_names = ["mobilenet", "nasnet"]
        for model_name in model_names:
            super(DataMNIST, self).pretrain_get_model(model_name=model_name)

    def inplace_process_data(self):
        cnn_features_dir_name = [
            # # "weights.11-0.9437.h5",
            # "inceptionresnet_imagenet",
            # "inceptionv3_imagenet",
            # "mobilenet_imagenet",
            # "resnet50_imagenet",
            # "vgg_imagenet",
            # "xception_imagenet",
            # "sift-200",
            # # # "HOG",
            # "HOG-200",
            # # "LBP",
            # "LBP-hist",
            # # "superpixel-500",
            # # "sift-1000"
            "orb-200",
            "brief-200"

        ]
        for weight_name in cnn_features_dir_name:
            logger.info(weight_name)
            X = self.postprocess_data(weight_name, if_return=True, embedding=False)
            filename = os.path.join(self.feature_dir, weight_name, "X.pkl")
            pickle_save_data(filename, X)

    def postprocess_data(self, weight_name, if_return=False, embedding=False):
        feature_dir = os.path.join(self.feature_dir, weight_name)
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        # X = np.zeros((self.y.shape[0], 1024))
        X = None
        prediction = None
        for data_dir in dirs:
            file_prefix = os.path.split(data_dir)[1].split(".")[0]
            data_filename = os.path.join(feature_dir, file_prefix + config.pkl_ext)
            if not os.path.exists(data_filename):
                logger.warn("{} does not exist, skip!".format(data_filename))
                continue
            mat = pickle_load_data(data_filename)
            features = mat["features"][0]
            print(file_prefix, "features shape: ", features.shape)
            pred_y = mat["features"][1]
            filenames = mat["filename"]
            for idx, name in enumerate(filenames):
                name = name.replace("\\", "/")
                cls, img_name = name.split("/")
                img_id, _ = img_name.split(".")
                img_id = int(img_id)
                # if img_id >= 100000:
                #     continue
                if len(features.shape) > 2:
                    features = features.reshape(features.shape[0], -1)
                if X is None:
                    X = np.zeros((self.y.shape[0], features.shape[1]))
                    prediction = np.ones(self.y.shape[0]) * -1
                X[img_id, :] = features[idx, :]
                try:
                    prediction[img_id] = pred_y[idx,:].argmax()
                except:
                    prediction[img_id] = -1

        self.X = X
        self.add_info = {
            "prediction": prediction
        }
        if if_return:
            logger.info("'if return' flag is enabled. Returning immediately!")
            return X

        super(DataMNIST, self).postprocess_data(weight_name, embedding)


if __name__ == '__main__':
    d = DataMNIST()
    # d.save_raw_data()
    # d.preprocessing_data()
    d.load_data()
    # d.ft_train_top_model()
    # d.ft_train_conv_layer()
    # d.save_images()
    d.inplace_process_data()
    # d.postprocess_data("weights.11-0.9437.h5")

    # d.sift_features("sift-200")
    # d.HOG_features("HOG-200")
    # d.LBP_features("LBP-hist")
    # d.pretrain_get_features()

    # for idx, suffix in enumerate([
    #     "_repeat_1",
    #     "_repeat_2",
    #     "_repeat_3",
    #     "_repeat_4",
    #     "_repeat_5"]):
    #     weights_name = [
    #         "weights.20-0.9434.h5",
    #         "weights.13-0.9571.h5",
    #         "weights.17-0.9592.h5",
    #         "weights.19-0.9590.h5",
    #         "weights.11-0.9437.h5",]
    #     # weights_name = ["weights.20-0.8852.h5",
    #     #                 "weights.20-0.8526.h5",
    #     #                 "weights.20-0.8942.h5",
    #     #                 "weights.20-0.8934.h5"]
    #     d = DataMNIST(suffix)
    #     d.train_data_dir = os.path.join(config.data_root,
    #                                        d.dataname,
    #                                        "bias_train")
    # #
    # #     # # exit()
    # #     d.ft_get_model()
    # #     d.ft_train_conv_layer()
    #     d._save_features_and_results(weights_name[idx])
    # #     # d.load_data()
    # #     # d.postprocess_data(weights_name[idx])
    # #     # d.save_file(suffix)