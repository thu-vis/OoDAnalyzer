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
from scripts.database.animals import DataAnimals
from scripts.utils.log_utils import logger
from scripts.utils.embedder_utils import Embedder
import shutil

class DataAnimalsTest(DataAnimals):
    def __init__(self,  suffix="", class_num=5):
        dataname = config.animals
        self.class_num = class_num
        super(DataAnimalsTest, self).__init__(dataname, suffix)

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

    def test(self, weight_name):
        logger.info("begin save features and results.")
        feature_dir = os.path.join(self.feature_dir, weight_name)
        check_dir(feature_dir)
        print("feature_dir:", feature_dir)
        weight_path = os.path.join(self.model_weight_dir, weight_name)
        print("weight_name:", weight_path)
        self.ft_get_model()
        output_model = self.model
        output_model.load_weights(weight_path)

        data_dir = self.test_data_dir
        datagen = ImageDataGenerator(rescale = 1. / 255)
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        y = generator.classes[generator.index_array]
        filename = generator.filenames
        nb_samples = y.reshape(-1).shape[0]
        pred_y = output_model.predict_generator(
                generator, math.ceil(nb_samples / self.batch_size))

        self.metrics(pred_y)

    def metrics(self, pred_y):
        import IPython; IPython.embed()

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument("-s", "--suffix", default="", required=True)
    parser.add_argument("-w", "--weight_name", default=None, required=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    ##### config #####
    args = parse_args()
    print(args)
    d = DataAnimalsTest(args.suffix)
    d.test(args.weight_name)