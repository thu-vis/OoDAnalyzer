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
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Input
from keras import applications
from keras import optimizers
from keras import backend as K
from keras.utils import to_categorical
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

gradient = np.linspace(0, 1, 256)
gradient = np.vstack(gradient).reshape(-1)
color_gradient = plt.get_cmap("RdYlGn")(gradient)
color_gradient = (color_gradient[:,:3] * 255).astype(np.uint8)
# color_gradient = color_gradient[:,np.array([2,1,0])]
color_gradient = color_gradient.reshape(color_gradient.shape[0],1,-1)



class DataAnimals(DataBase):
    def __init__(self, dataname=None, suffix="", class_num=5):
        if dataname is None:
            dataname = config.animals
        self.class_num = class_num
        super(DataAnimals, self).__init__(dataname, suffix)

    def preprocessing_data(self):
        self.class_name = ["cat", "dog", "rabbit", "wolf", "tiger", "leopard"]
        self.class_name_encoding = {
            self.class_name[0]: 0,
            self.class_name[1]: 1,
            self.class_name[2]: 2,
            self.class_name[3]: 3,
            self.class_name[4]: 4,
            self.class_name[5]: 4,
        }
        images_dir = self.images_dir

        self.test_sub_y = []

        config.raw_data_root = r"H:\backup\RawData"
        test_sub_y = []
        # split train, valid, test
        all_test_dir = os.path.join(config.raw_data_root, self.dataname, "normal_test")
        self.test_idx = []
        for class_name in self.class_name[:2]:
            origin_dir = os.path.join(all_test_dir, class_name)
            origin_imgs = os.listdir(origin_dir)
            for img_name in origin_imgs:
                id = int(img_name.split(".")[0]) + 12500 * self.class_name_encoding[class_name]
                self.test_idx.append(id)
        logger.warn("info confirm: total test instances number: {}".format(len(self.test_idx)))

        all_bias_dir = os.path.join(config.raw_data_root, self.dataname, "all_bias")
        bias_idx = []
        for class_name in self.class_name[:2]:
            origin_dir = os.path.join(all_bias_dir, class_name)
            origin_imgs = os.listdir(origin_dir)
            for img_name in origin_imgs:
                id = int(img_name.split(".")[0]) + 12500 * self.class_name_encoding[class_name]
                bias_idx.append(id)
        logger.warn("info confirm: total bias instances number: {}".format(len(bias_idx)))

        for id in self.test_idx:
            if id < 12500:
                if bias_idx.count(id):
                    test_sub_y.append(1)
                else:
                    test_sub_y.append(0)
            else:
                if bias_idx.count(id):
                    test_sub_y.append(2)
                else:
                    test_sub_y.append(3)

        self.y = np.array([0] * 12500 + [1] * 12500).reshape(-1).astype(int)
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

        global_id = 25000

        y = []
        sub_y = []

        origin_data_dir = os.path.join(config.raw_data_root, self.dataname, "dog_cat_extension")
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
                target = os.path.join(self.images_dir, str(global_id) + ".jpg")
                global_id = global_id + 1
                y.append(cls)
                sub_y.append(idx+4)
                try:
                    shutil.copy(src, target)
                    pass
                except Exception as e:
                    print(e)
        assert len(y) == len(sub_y)



        self.y = self.y.tolist() + y
        test_sub_y = test_sub_y + sub_y
        self.test_idx = self.test_idx + list(range(25000, 25000 + len(y)))
        self.test_redundant_idx = self.test_redundant_idx + list(range(25000, 25000 + len(y)))

        print(len(self.test_sub_y), len(self.test_idx))

        y = []
        sub_y = []
        index = []
        for idx, cls_name in enumerate(self.class_name[2:5]):
            cls_dir = os.path.join(config.raw_data_root, self.dataname, cls_name)
            img_name_list = os.listdir(cls_dir)
            for img_name in img_name_list:
                src = os.path.join(cls_dir, img_name)
                target = os.path.join(self.images_dir, str(global_id) + ".jpg")
                index.append(global_id)
                global_id = global_id + 1
                y.append(idx+2)
                sub_y.append(idx+18)
                # shutil.copy(src, target)

        num_instance = len(y)
        index = np.array(index)
        order_idx = np.array(range(num_instance))
        np.random.seed(123)
        np.random.shuffle(index)
        train_num = int(num_instance * 0.25)
        self.train_idx = self.train_idx + index[order_idx[:train_num]].tolist()
        self.test_idx = self.test_idx + index[order_idx[train_num:]].tolist()
        self.y = self.y + y
        test_sub_y = test_sub_y + np.array(sub_y)[order_idx[train_num:]].tolist()

        print(len(self.test_sub_y), len(self.test_idx))

        y = []
        sub_y = []
        index = []
        for idx, cls_name in enumerate(["tiger-cat", "husky"]):
            cls_dir = os.path.join(config.raw_data_root, self.dataname, cls_name, "test")
            img_name_list = os.listdir(cls_dir)
            for img_name in img_name_list:
                src = os.path.join(cls_dir, img_name)
                target = os.path.join(self.images_dir, str(global_id) + ".jpg")
                index.append(global_id)
                global_id = global_id + 1
                y.append(idx)
                sub_y.append(idx+21)
                # shutil.copy(src, target)

        self.test_idx = self.test_idx + index
        self.test_redundant_idx = self.test_redundant_idx + index
        self.y = self.y + y
        self.test_sub_y = test_sub_y + sub_y

        print(len(self.test_sub_y), len(self.test_idx))

        y = []
        sub_y = []
        index = []
        for idx, cls_name in enumerate([self.class_name[5]]):
            cls_dir = os.path.join(config.raw_data_root, self.dataname, cls_name)
            img_name_list = os.listdir(cls_dir)
            for img_name in img_name_list:
                src = os.path.join(cls_dir, img_name)
                target = os.path.join(self.images_dir, str(global_id) + ".jpg")
                index.append(global_id)
                global_id = global_id + 1
                y.append(4)
                sub_y.append(idx+22)
                # shutil.copy(src, target)

        self.test_idx = self.test_idx + index
        self.test_redundant_idx = self.test_redundant_idx + index
        self.y = self.y + y
        self.test_sub_y = self.test_sub_y + sub_y

        print(len(self.test_sub_y), len(self.test_idx))

        print("test_idx len: {}, train_idx len: {}, train_redundant_idx len: {}, y len: {}, sub_test_y len:{}"
              .format(len(self.test_idx), len(self.train_idx),
                      len(self.train_redundant_idx), len(self.y), len(self.test_sub_y)))

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
            "test_sub_y": self.test_sub_y
        }
        self.save_cache()

    def load_data(self, loading_from_buffer=True):
        super(DataAnimals, self).load_data(loading_from_buffer)
        # self.class_name = self.all_data["class_name"]
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
        # self.sub_y = self.all_data["sub_y"]

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
                    # shutil.copy(src, target)

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
        model = applications.VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
        inputs = Input(shape=input_shape)
        tmp_x = inputs
        for layer in model.layers[:-1]:
            tmp_x = layer(tmp_x)
        bottleneck_output = tmp_x
        bottleneck_output_shape = bottleneck_output.shape.as_list()
        a = Input(shape=bottleneck_output_shape[1:])
        b = Conv2D(1024, (3, 3), activation="relu", padding="same")(a)
        last_but_one_output = GlobalAveragePooling2D()(b)
        top_outputs = Dense(self.class_num, activation="softmax")(last_but_one_output)

        ##############################################################
        self.top_model = Model(inputs=a, outputs=top_outputs)  #######
        ##############################################################

        outputs = self.top_model(bottleneck_output)

        ############################################################
        self.model = Model(inputs=inputs, outputs=outputs)  ########
        ############################################################

        tmp_model = Model(inputs=a, outputs=[last_but_one_output, b, top_outputs])
        last_but_one_output, b, outputs = tmp_model(bottleneck_output)


        #######################################################################################
        self.output_model = Model(inputs=inputs,  ##############################################
                                  outputs=[bottleneck_output, last_but_one_output, outputs])  ###
        #######################################################################################

        self.saliency_map_model = Model(inputs=inputs,
                                        outputs=[b])

    def ft_save_bottleneck(self):
        self.ft_get_model()
        model = self.output_model

        dirs = [self.train_data_dir, self.valid_data_dir]
        for data_dir in dirs:
            datagen = ImageDataGenerator(rescale=1. / 255)
            generator = datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                class_mode="categorical",
                shuffle=False
            )
            file_prefix = os.path.split(data_dir)[1].split(".")[0]
            y = generator.classes[generator.index_array]
            nb_samples = y.reshape(-1).shape[0]
            logger.info("{}, total instances: {}".format(file_prefix, nb_samples))
            features = model.predict_generator(generator, math.ceil(nb_samples / self.batch_size))
            bottleneck_output = features[0]
            mat = {
                "features": bottleneck_output,
                "y": y
            }
            pickle_save_data(os.path.join(self.feature_dir, file_prefix +
                                  "_bottleneck.pkl"), mat)
            logger.info("{} bottleneck file saved.".format(file_prefix))

    def ft_train_top_model(self):
        self.ft_get_model()
        dirs = [self.train_data_dir, self.valid_data_dir]

        # training top model
        model = self.top_model
        # get training data
        file_prefix = os.path.split(self.train_data_dir)[1].split(".")[0]
        data = pickle_load_data(os.path.join(self.feature_dir, file_prefix +
                                    "_bottleneck.pkl"))
        train_data = data["features"]
        train_labels = data["y"].reshape(-1)
        train_labels = to_categorical(train_labels)
        logger.info("top model training: input shape: {}".format(train_data.shape))
        file_prefix = os.path.split(self.valid_data_dir)[1].split(".")[0]
        data = pickle_load_data(os.path.join(self.feature_dir, file_prefix +
                                    "_bottleneck.pkl"))
        valid_data = data["features"]
        valid_labels = data["y"].reshape(-1)
        valid_labels = to_categorical(valid_labels)
        logger.info("top model training: input shape: {}".format(valid_data.shape))

        # shuffle
        np.random.seed(123)
        train_index = np.random.permutation(train_labels.shape[0])
        valid_index = np.random.permutation(valid_labels.shape[0])
        train_data = train_data[train_index, :]
        train_labels = train_labels[train_index, :]
        valid_data = valid_data[valid_index, :]
        valid_labels = valid_labels[valid_index, :]

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=2e-4, momentum=0.9),
                      # optimizer = "rmsprop",
                      metrics=['accuracy'])

        checkpointer = ModelCheckpoint(filepath=os.path.join(self.model_weight_dir,
                                                             "top_weights.{epoch:02d}-{val_acc:.4f}.h5"),
                                       verbose=1)
        model.fit(train_data, train_labels,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  validation_data=(valid_data, valid_labels),
                  callbacks=[checkpointer])
        model.save_weights(self.top_model_weights_path)

        # test: the following codes are designed for debugging
        # model = self.output_model
        # self.top_model.load_weights(self.top_model_weights_path)
        # datagen = ImageDataGenerator(rescale=1. / 255)
        # generator = datagen.flow_from_directory(
        #     self.valid_data_dir,
        #     target_size=(self.img_width, self.img_height),
        #     batch_size=self.batch_size,
        #     class_mode="binary",
        #     shuffle=False
        # )
        # y = generator.classes[generator.index_array]
        # filename = generator.filenames
        # nb_samples = y.reshape(-1).shape[0]
        # features = model.predict_generator(
        #     generator, math.ceil(nb_samples/self.batch_size))
        # pred_y = features[2].reshape(-1)
        # pred_y = (pred_y > 0.5).astype(int)
        # valid_acc = sum(y.reshape(-1) == pred_y) / nb_samples
        # logger.warn("test valid acc: {}".format(valid_acc))

    def ft_train_conv_layer(self):
        self.ft_get_model()
        model = self.model
        try:
            self.top_model.load_weights(self.top_model_weights_path)
        except Exception as e:
            logger.warn(e)
            logger.warn("training conv layers without pretrained top models")
        # model.summary()

        # freezing top layers of model
        for layer in model.layers[:16]:
            print(layer)
            layer.trainable = False

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=2e-4, momentum=0.9),
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
            epochs=50,
            validation_data=valid_generator,
            validation_steps=nb_valid_samples // self.batch_size,
            callbacks=[checkpointer])

    # def img_format_checking(self):
    #     img_name_list = os.listdir(self.images_dir)
    #     flag = 1
    #     for idx, img_name in enumerate(img_name_list):
    #         if idx % 1000 == 0:
    #             print(idx)
    #         if img_name == "45058.jpg":
    #             flag = 0
    #         if flag:
    #             continue
    #         src = os.path.join(config.data_root,
    #                            self.dataname,
    #                            "images", img_name)
    #         target = os.path.join(self.images_dir, img_name)
    #         # print(src)
    #         try:
    #             Image.open(src)
    #         except Exception as e:
    #             print(e)
    #             data = np.array(cv2.imread(src))
    #             img = Image.fromarray(data)
    #             img.save(src)
    #         # try:
    #         #     img_data = Image.open(src)
    #         #     img_data = img_data.resize((512, 512), Image.ANTIALIAS)
    #         #     img_data.save(src)
    #         # except Exception as e:
    #         #     print(e)
    #         #     print(src)
    #         try:
    #             img = Image.open(src)
    #             img = img.convert("RGB")
    #             img = img.resize((512, 512), Image.ANTIALIAS)
    #             img.save(target)
    #         except Exception as e:
    #             # print(e)
    #             print(src)
    #     exit()

    def img_format_checking(self):
        img_name_list = os.listdir(self.images_dir)
        flag = 1
        for idx, img_name in enumerate(img_name_list):
            src = os.path.join(config.data_root,
                               self.dataname,
                               "images", img_name)
            # try:
            #     Image.open(src)
            # except Exception as e:
            #     print(e)
            #     data = np.array(cv2.imread(src))
            #     img = Image.fromarray(data)
            #     img.save(src)
            # img = Image.open(src)
            # img_data = np.array(img)
            # if img_data.shape != (512, 512, 3):
            #     img = Image.open(src)
            #     img = img.convert("RGB")
            #     img = img.resize((512, 512), Image.ANTIALIAS)
            #     img.save(src)


    def postprocess_data(self, weight_name, if_return=False, embedding=True):
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
            pred_y = mat["features"][1]
            filenames = mat["filename"]
            for idx, name in enumerate(filenames):
                name = name.replace("\\", "/")
                cls, img_name = name.split("/")
                img_id, _ = img_name.split(".")
                try:
                    img_id = int(img_id)
                except:
                    img_id = 1000000
                if img_id >= 100000:
                    continue
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

        super(DataAnimals, self).postprocess_data(weight_name, embedding)

    def inplace_process_data(self):
        cnn_features_dir_name = [
                # "weights.20-0.8054.h5",
                # "weights.40-0.7456.h5",
                # "weights.40-0.7544.h5",
                # "inceptionresnet_imagenet",
                # "inceptionv3_imagenet",
                # "mobilenet_imagenet",
                # "resnet50_imagenet",
                # "vgg_imagenet",
                # "xception_imagenet",
                # "sift-200",
                "brief-200",
                # "orb-200"
                # "surf-200"
                # "superpixel-500",
                # "sift-1000"

        ]
        for weight_name in cnn_features_dir_name:
            logger.info(weight_name)
            X = self.postprocess_data(weight_name, if_return=True)
            filename = os.path.join(self.feature_dir, weight_name, "X.pkl")
            pickle_save_data(filename, X)

    def pretrain_get_features(self):
        model_names = ["vgg", "resnet50", "xception", "inceptionv3",
                       "inceptionresnet", "mobilenet"]
        # model_names = ["mobilenet", "nasnet"]
        for model_name in model_names:
            super(DataAnimals, self).pretrain_get_model(model_name=model_name)

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
            pred_y = pred_y.argmax(axis=1)
            for idx, name in enumerate(filenames):
                name = name.replace("\\", "/")
                cls, img_name = name.split("/")
                img_id, _ = img_name.split(".")
                img_id = int(img_id)
                img_path = os.path.join(saliency_map_dir, str(img_id) + ".jpg")
                if os.path.exists(img_path):
                    continue
                original_img_path = os.path.join(self.images_dir, str(img_id) + ".jpg")
                cam = saliency_map_data[idx, :,:,:]
                w,h = cam.shape[:2]
                cam = cam.reshape(-1, cam.shape[2])
                cam = np.dot(cam, weights[:,pred_y[idx]]).reshape(w,h)
                cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
                cam = cam.astype(np.uint8)
                cam_img = Image.fromarray(cam)
                cam_img = cam_img.resize((512, 512), Image.ANTIALIAS)
                cam_img = np.array(cam_img)
                mask = np.ones(cam_img.shape) * 0.7
                mask[cam_img>150] = 1.0
                img_data = Image.open(original_img_path)
                img_data = img_data.resize((512, 512), Image.ANTIALIAS)
                img_data = np.array(img_data).astype(float)
                if len(img_data.shape) == 2:
                    img_data = img_data.reshape(img_data.shape[0], img_data.shape[1], -1)
                    img_data = img_data.repeat(axis=2, repeats=3)
                    print(img_id)
                elif img_data.shape[2] == 4:
                    img_data = img_data[:,:,:3]
                img_data = (img_data * mask[:,:,None]) / 2.0
                img_data = img_data.astype(np.uint8)
                cam_img = Image.fromarray(img_data)
                cam_img.save(img_path)
            # exit()

    def process_superpixel(self):
        superpixel_feature_dir = os.path.join(self.feature_dir, "superpixel")
        X = np.zeros((len(self.y), 200))
        processed_list = []
        processed_name = []
        exclude_list = []
        unprocessed_list = []
        for file_name in ["binary_sp_features_train.pkl",
                          "binary_sp_features_test.pkl"]:
            mat = pickle_load_data(os.path.join(superpixel_feature_dir, file_name))
            a = 1
            for name in mat:
                feature = mat[name]
                img_id = name.split(".")[1]
                try:
                    img_id = int(img_id)
                except Exception as e:
                    print(e)
                    unprocessed_list.append(name)
                    continue
                if (img_id in self.train_idx) or (img_id in self.test_idx):
                    X[img_id] = feature
                    if img_id in processed_list:
                        print("dumplicated: ", name, processed_name[processed_list.index(img_id)])
                    else:
                        processed_list.append(img_id)
                        processed_name.append(name)
                else:
                    print("{} should not be in the training and test".format(name))
                    exclude_list.append(img_id)
        filename = os.path.join(self.feature_dir, "superpixel", "X.pkl")
        print(len(exclude_list), len(unprocessed_list))
        pickle_save_data(filename, X)

    def check_idx(self):
        for idx in self.train_idx:
            assert idx not in self.test_idx

if __name__ == '__main__':
    suffix = ""
    d = DataAnimals(suffix=suffix)
    d.load_data()
    d.postprocess_saliency_map("weights.30-0.7411.h5")
    # d.inplace_process_data()
    # d.preprocessing_data()
    # d.img_format_checking()
    # d.check_idx()
    # d.preprocessing_data(); exit()
    # for i in ["sift-200", "HOG-kmeans-200", "LBP-hist"]:
    # for i in ["sift-200"]:
    #     d.lowlevel_features(i)
    # d.sift_features("sift-200")
    # d.HOG_features("HOG-200")
    # d.LBP_features("LBP-hist")
    # # exit()
    # d.load_data()
    # # d.img_format_checking(); exit()
    # # d.feature_extraction()
    # # d.ft_save_bottleneck()
    # # d.ft_train_top_model()
    # # d.ft_train_conv_layer()
    # weights_name = "weights.40-0.7401.h5"
    # # # d.pretrain_get_features()
    # # # d.inplace_process_data()
    # # # d._save_features_and_results(weights_name)
    # d.postprocess_data(weights_name, embedding=False)
    # # # d.save_saliency_map_data(weights_name)
    # # # d.postprocess_saliency_map(weights_name)
    # d.save_file(suffix)
