import numpy as np
import os
import scipy.io as sio
from time import time
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_mldata
import tensorflow as tf
from tensorflow import keras
from scipy.interpolate import interp1d
from PIL import Image
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
from scripts.utils.helper_utils import check_dir
from scripts.database.database import DataBase
import shutil

pre_processing_func = lambda x: preprocess_input(x, mode="tf")

class DataDogCatNormalTrain(DataBase):
    def __init__(self):
        dataname = config.dog_cat_normal_train
        check_dir(os.path.join(config.data_root, dataname))
        super(DataDogCatNormalTrain, self).__init__(dataname)
        self.img_width = 224
        self.img_height = 224
        self.feature_dir = os.path.join(config.data_root,
                                   self.dataname,
                                   "feature")
        check_dir(self.feature_dir)
        self.model_weight_dir = os.path.join(config.data_root,
                                        self.dataname,
                                        "weights")
        check_dir(self.model_weight_dir)
        self.top_model_weights_path = os.path.join(self.model_weight_dir,
                                             "bottleneck_fc_model.h5")
        self.train_data_dir = os.path.join(config.data_root,
                                      self.dataname,
                                      "normal_train")
        self.train_cat_dir = os.path.join(self.train_data_dir,
                                          "cat")
        check_dir(self.train_cat_dir)
        self.train_dog_dir = os.path.join(self.train_data_dir,
                                          "dog")
        check_dir(self.train_dog_dir)
        self.validation_data_dir = os.path.join(config.data_root,
                                           self.dataname,
                                           "normal_test")
        self.validation_cat_dir = os.path.join(self.validation_data_dir,
                                          "cat")
        check_dir(self.validation_cat_dir)
        self.validation_dog_dir = os.path.join(self.validation_data_dir,
                                          "dog")
        check_dir(self.validation_dog_dir)
        self.test_data_dir = os.path.join(config.data_root,
                                          self.dataname,
                                          "bias_test")
        self.test_cat_dir = os.path.join(self.test_data_dir,
                                          "cat")
        check_dir(self.test_cat_dir)
        self.test_dog_dir = os.path.join(self.test_data_dir,
                                          "dog")
        check_dir(self.test_dog_dir)
        self.nb_train_samples = None
        self.nb_validation_samples = None
        self.epochs = 50
        self.batch_size = 16
        self.train_y = 0
        self.valid_y = 0

        warnings.warn("we use {} for training, {} and {} for validation information"
              .format(self.train_data_dir, self.validation_data_dir, self.test_data_dir), UserWarning)

    def _get_GAP_VGG_16_bottleneck(self, input_shape=None):
        model = applications.VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
        res_model = Sequential()
        for layer in model.layers[:-1]:
            res_model.add(layer)
        return res_model

    def _save_bottlebeck_features(self):
        train_y = self.train_y
        valid_y = self.valid_y
        nb_train_samples = self.nb_train_samples
        nb_validation_samples = self.nb_validation_samples
        train_data_dir = self.train_data_dir
        feature_dir = self.feature_dir
        validation_data_dir = self.validation_data_dir
        img_width = self.img_width
        img_height = self.img_height
        batch_size = self.batch_size

        datagen = ImageDataGenerator(rescale = 1. / 255)

        # build the VGG16 network
        model = self._get_GAP_VGG_16_bottleneck()

        generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode="binary",
            shuffle=False)

        train_y = generator.classes[generator.index_array]
        nb_train_samples = train_y.reshape(-1).shape[0] // batch_size * batch_size
        print("training sampling number: %s" % (nb_train_samples))

        bottleneck_features_train = model.predict_generator(
            generator, nb_train_samples // batch_size)
        print(bottleneck_features_train.shape)
        np.save(os.path.join(feature_dir, 'bottleneck_features_train.npy'),
                bottleneck_features_train)

        generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode="binary",
            shuffle=False)

        valid_y = generator.classes[generator.index_array]
        nb_validation_samples = valid_y.reshape(-1).shape[0]
        print("validation samples number: %s" % (nb_validation_samples))

        bottleneck_features_validation = model.predict_generator(
            generator, nb_validation_samples // batch_size)
        np.save(os.path.join(feature_dir, 'bottleneck_features_validation.npy'),
                bottleneck_features_validation)

        self.train_y = train_y
        self.valid_y = valid_y
        self.nb_train_samples = nb_train_samples
        self.nb_validation_samples = nb_validation_samples

        return

    def _train_top_model(self):
        train_y = self.train_y
        valid_y = self.valid_y
        nb_train_samples = self.nb_train_samples
        nb_validation_samples = self.nb_validation_samples
        train_data_dir = self.train_data_dir
        feature_dir = self.feature_dir
        validation_data_dir = self.validation_data_dir
        model_weight_dir = self.model_weight_dir
        epochs = self.epochs
        top_model_weights_path = self.top_model_weights_path
        img_width = self.img_width
        img_height = self.img_height
        batch_size = self.batch_size

        train_data = np.load(os.path.join(feature_dir, 'bottleneck_features_train.npy'))
        # train_labels = np.array(
        #     [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
        train_labels = train_y.reshape(-1,1)
        train_labels = train_labels[:train_data.shape[0],:]
        print(train_data.shape)
        print(train_labels.shape)

        validation_data = np.load(os.path.join(feature_dir, 'bottleneck_features_validation.npy'))
        # validation_labels = np.array(
        #     [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
        validation_labels = valid_y.reshape(-1,1)
        validation_labels = validation_labels[:validation_data.shape[0],:]
        print(validation_labels.shape)

        # shuffle
        np.random.seed(200)
        train_index = np.random.permutation(train_labels.shape[0])
        valid_index = np.random.permutation(validation_labels.shape[0])
        train_data = train_data[train_index, :]
        train_labels = train_labels[train_index, :]
        print(train_labels[:20])
        validation_data = validation_data[valid_index, :]
        validation_labels = validation_labels[valid_index, :]
        print(validation_labels[:20])

        a = Input(shape=train_data.shape[1:])
        b = Conv2D(1024, (3, 3), activation="relu", padding="same")(a)
        b = GlobalAveragePooling2D()(b)
        output = Dense(1, activation="sigmoid")(b)
        model = Model(inputs=a, outputs=output)

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=2e-4, momentum=0.9),
                      # optimizer = "rmsprop",
                      metrics=['accuracy'])
        # model.compile(optimizer='rmsprop',
        #               loss='binary_crossentropy', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath=os.path.join(model_weight_dir,
                                                             "top_weights.{epoch:02d}-{val_acc:.4f}.h5"),
                                       verbose=1)
        model.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(validation_data, validation_labels),
                    callbacks=[checkpointer])
        model.save_weights(top_model_weights_path)

        self.train_y = train_y
        self.valid_y = valid_y
        self.nb_train_samples = nb_train_samples
        self.nb_validation_samples = nb_validation_samples

    def _fine_tune(self):
        train_y = self.train_y
        valid_y = self.valid_y
        nb_train_samples = self.nb_train_samples
        nb_validation_samples = self.nb_validation_samples
        train_data_dir = self.train_data_dir
        feature_dir = self.feature_dir
        validation_data_dir = self.validation_data_dir
        model_weight_dir = self.model_weight_dir
        epochs = self.epochs
        top_model_weights_path = self.top_model_weights_path
        img_width = self.img_width
        img_height = self.img_height
        batch_size = self.batch_size

        input_shape = None
        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)
        base_model = self._get_GAP_VGG_16_bottleneck(input_shape=input_shape)
        print('Model loaded.')
        print(base_model.output_shape, len(base_model.layers))

        model = Sequential()
        for l in base_model.layers:
            model.add(l)
        print(base_model.output_shape)
        a = Input(shape=base_model.output_shape[1:])
        b = Conv2D(1024, (3, 3), activation="relu", padding="same")(a)
        b = GlobalAveragePooling2D()(b)
        output = Dense(1, activation="sigmoid")(b)
        top_model = Model(inputs=a, outputs=output)
        top_model.load_weights(top_model_weights_path)

        model.add(top_model)

        for layer in model.layers[:15]:
            print(layer)
            layer.trainable = False

        model.compile(loss='binary_crossentropy',
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

        test_datagen = ImageDataGenerator(rescale = 1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary')

        # fine-tune the model

        # err = model.evaluate_generator(validation_generator, steps=len(validation_generator))
        # print(err)

        checkpointer = ModelCheckpoint(filepath=os.path.join(model_weight_dir,
                                                             "weights.{epoch:02d}-{val_acc:.4f}.h5"),
                                       verbose=1)
        print("nb train sample: %s" % (nb_train_samples))
        model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=300,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size,
            callbacks=[checkpointer])

    def fine_tune(self):
        self._save_bottlebeck_features()
        self._train_top_model()
        self._fine_tune()

    def file_processing(self):

        '''
        # ############# image categorizing #################
        # origin_dir = os.path.join(config.data_root,
        #                           self.dataname,
        #                           "train_full")
        # target_dog_dir = os.path.join(config.data_root,
        #                               self.dataname,
        #                               "all_normal",
        #                               "dog")
        # check_dir(target_dog_dir)
        # target_cat_dir = os.path.join(config.data_root,
        #                               self.dataname,
        #                               "all_normal",
        #                               "cat")
        # check_dir(target_cat_dir)
        # origin_images_namelist = os.listdir(origin_dir)
        # for image_full_name in origin_images_namelist:
        #     cls, name, suffix = image_full_name.split(".")
        #     image_name = name + "." + suffix
        #     if cls == "cat":
        #         target_filename = os.path.join(target_cat_dir, image_name)
        #     elif cls == "dog":
        #         target_filename = os.path.join(target_dog_dir, image_name)
        #     origin_filename = os.path.join(origin_dir, image_full_name)
        #     shutil.copy(origin_filename, target_filename)
        # ############# image categorizing #################
        '''
        train_data_dir = os.path.join(config.data_root,
                                      self.dataname,
                                      "normal_train")
        test_data_dir = os.path.join(config.data_root,
                                     self.dataname,
                                     "normal_test")
        all_normal_filename = os.path.join(config.data_root, self.dataname, "all_normal")
        class_list = ["cat", "dog"]
        # bias training, testing
        for class_name in class_list:
            origin_dir = os.path.join(all_normal_filename, class_name)
            target_train_dir = os.path.join(train_data_dir, class_name)
            target_test_dir = os.path.join(test_data_dir, class_name)
            images_list = os.listdir(origin_dir)
            idx = np.array(range(len(images_list)))
            np.random.seed(123)
            np.random.shuffle(idx)
            split_point = int( len(idx) * 0.8 )
            training_images_list = images_list[:split_point]
            test_images_list = images_list[split_point:]
            for img_name in training_images_list:
                src_img_path = os.path.join(origin_dir, img_name)
                target_img_path = os.path.join(target_train_dir, img_name)
                shutil.copy(src_img_path, target_img_path)
            for img_name in test_images_list:
                src_img_path = os.path.join(origin_dir, img_name)
                target_img_path = os.path.join(target_test_dir, img_name)
                shutil.copy(src_img_path, target_img_path)

        all_bias_filename = os.path.join(config.data_root, self.dataname, "all_bias")


        bias_train_data_dir = os.path.join(config.data_root,
                                      self.dataname,
                                      "bias_train")
        bias_test_data_dir = os.path.join(config.data_root,
                                     self.dataname,
                                     "bias_test")
        check_dir(bias_train_data_dir)
        check_dir(bias_test_data_dir)
        class_list = ["cat", "dog"]
        # bias training, testing
        for class_name in class_list:
            all_bias_imagename = os.listdir(os.path.join(all_bias_filename, class_name))
            check_dir(os.path.join(all_bias_filename, class_name))
            normal_train_imagename = os.listdir(os.path.join(train_data_dir, class_name))
            for name in all_bias_imagename:
                if normal_train_imagename.count(name) > 0:
                    src_img_path = os.path.join(os.path.join(train_data_dir, class_name), name)
                    target_img_path = os.path.join(bias_train_data_dir, class_name, name)
                    shutil.copy(src_img_path, target_img_path)

            normal_test_imagename = os.listdir(os.path.join(test_data_dir, class_name))
            check_dir(os.path.join(test_data_dir, class_name))
            for name in all_bias_imagename:
                if normal_test_imagename.count(name) > 0:
                    src_img_path = os.path.join(os.path.join(test_data_dir, class_name), name)
                    target_img_path = os.path.join(bias_test_data_dir, class_name, name)
                    shutil.copy(src_img_path, target_img_path)

    def save_features(self, weight_name):
        train_y = self.train_y
        valid_y = self.valid_y
        nb_train_samples = self.nb_train_samples
        nb_validation_samples = self.nb_validation_samples
        train_data_dir = self.train_data_dir
        feature_dir = self.feature_dir
        validation_data_dir = self.validation_data_dir
        model_weight_dir = self.model_weight_dir
        epochs = self.epochs
        top_model_weights_path = self.top_model_weights_path
        img_width = self.img_width
        img_height = self.img_height
        batch_size = self.batch_size
        weight_path = os.path.join(self.model_weight_dir, weight_name)
        feature_result_dir = os.path.join(feature_dir,
                                          weight_name)
        check_dir(feature_result_dir)

        def load_weights_model(weights_path):
            input_shape = None
            if K.image_data_format() == 'channels_first':
                input_shape = (3, img_width, img_height)
            else:
                input_shape = (img_width, img_height, 3)
            base_model = self._get_GAP_VGG_16_bottleneck(input_shape=input_shape)
            print('Model loaded.')
            print(base_model.output_shape, len(base_model.layers))

            model = Sequential()
            for l in base_model.layers:
                model.add(l)
            print(base_model.output_shape)
            a = Input(shape=base_model.output_shape[1:])
            b = Conv2D(1024, (3, 3), activation="relu", padding="same")(a)
            b = GlobalAveragePooling2D()(b)
            output = Dense(1, activation="sigmoid")(b)
            top_model = Model(inputs=a, outputs=output)

            model.add(top_model)
            model.load_weights(weights_path)

            final_model = Sequential()
            for layer in model.layers[:-1]:
                final_model.add(layer)
            final_top_model = Sequential()
            for layer in model.layers[-1].layers[1:]:
                print(layer)
                final_model.add(layer)
            # final_model.add(final_top_model)

            # final_model.summary()
            return final_model

        final_model = load_weights_model(weight_path)
        model = Sequential()
        for layer in final_model.layers[:-1]:
            model.add(layer)


        datagen = ImageDataGenerator(rescale=1. / 255)

        generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode="binary",
            shuffle=False)

        train_y = generator.classes[generator.index_array]
        train_filename = generator.filenames
        nb_train_samples = train_y.reshape(-1).shape[0] // batch_size * batch_size
        print("training sampling number: %s" % (nb_train_samples))

        file_prefix = os.path.split(train_data_dir)[1].split(".")[0]
        train_features = model.predict_generator(
            generator, nb_train_samples // batch_size)
        print(train_features.shape)
        np.save(os.path.join(feature_result_dir, file_prefix + '.npy'),
                train_features)
        np.save(os.path.join(feature_result_dir, file_prefix + "_y.npy"),
                train_y)
        np.save(os.path.join(feature_result_dir, file_prefix + "_filename.npy"),
                train_filename)

        generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode="binary",
            shuffle=False)

        valid_y = generator.classes[generator.index_array]
        valid_filename = generator.filenames
        nb_validation_samples = valid_y.reshape(-1).shape[0]
        print("validation samples number: %s" % (nb_validation_samples))
        # from IPython import embed; embed()
        file_prefix = os.path.split(validation_data_dir)[1].split(".")[0]
        validation_features = model.predict_generator(
            generator, nb_validation_samples // batch_size)
        np.save(os.path.join(feature_result_dir, file_prefix + '.npy'),
                validation_features)
        np.save(os.path.join(feature_result_dir, file_prefix + "_y.npy"),
                valid_y)
        np.save(os.path.join(feature_result_dir, file_prefix + "_filename.npy"),
                valid_filename)

        generator = datagen.flow_from_directory(
            self.test_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode="binary",
            shuffle=False)

        test_y = generator.classes[generator.index_array]
        test_filename = generator.filenames
        nb_validation_samples = test_y.reshape(-1).shape[0]
        print("validation samples number: %s" % (nb_validation_samples))

        file_prefix = os.path.split(self.test_data_dir)[1].split(".")[0]
        test_features = model.predict_generator(
            generator, nb_validation_samples // batch_size)
        np.save(os.path.join(feature_result_dir, file_prefix + '.npy'),
                test_features)
        np.save(os.path.join(feature_result_dir, file_prefix + "_y.npy"),
                test_y)
        np.save(os.path.join(feature_result_dir, file_prefix + "_filename.npy"),
                test_filename)

    def imagenet_pretrained_model_feature(self):
        img_width = self.img_width
        img_height = self.img_height
        batch_size = self.batch_size
        feature_dir = self.feature_dir
        feature_result_dir = os.path.join(feature_dir,
                                          "imagenet_pretrained_model")
        check_dir(feature_result_dir)
        base_model = applications.VGG16(include_top=True, input_shape=None, weights='imagenet')
        model = Sequential()
        for i in base_model.layers[:-1]:
            model.add(i)
        datagen = ImageDataGenerator(rescale=1. / 255)

        data_dir = os.path.join(config.data_root,
                                          self.dataname,
                                          "all_normal")

        generator = datagen.flow_from_directory(
            data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode="binary",
            shuffle=False)

        test_y = generator.classes[generator.index_array]
        test_filename = generator.filenames
        nb_validation_samples = test_y.reshape(-1).shape[0]
        print("validation samples number: %s" % (nb_validation_samples))

        test_features = model.predict_generator(
            generator, nb_validation_samples // batch_size)
        np.save(os.path.join(feature_result_dir, 'feature.npy'),
                test_features)
        np.save(os.path.join(feature_result_dir, "y.npy"),
                test_y)
        np.save(os.path.join(feature_result_dir, "filename.npy"),
                test_filename)

        check_dir(feature_result_dir)

    def load_data(self):
        feature_dir = os.path.join(config.data_root,
                                   self.dataname,
                                   "feature",
                                   "weights.30-0.9766.h5")

        normal_train = np.load(os.path.join(feature_dir, "normal_train.npy"))
        normal_train_y = np.load(os.path.join(feature_dir, "normal_train_y.npy"))
        normal_train_y = np.array(normal_train_y).reshape(-1)[:normal_train.shape[0]]
        bias_test = np.load(os.path.join(feature_dir, "bias_test.npy"))
        bias_test_y = np.load(os.path.join(feature_dir, "bias_test_y.npy"))
        bias_test_y = np.array(bias_test_y).reshape(-1)[:bias_test.shape[0]]
        all_test = np.load(os.path.join(feature_dir, "normal_test.npy"))
        all_test_y = np.load(os.path.join(feature_dir, "normal_test_y.npy"))
        all_test_y = np.array(all_test_y).reshape(-1)[:all_test.shape[0]]

        self.X_train = normal_train
        self.y_train = normal_train_y
        # self.X_test = bias_test
        # self.y_test = bias_test_y
        self.X_test = all_test
        self.y_test = all_test_y

        print("data loaded!!")
        print("train data num: %s, test data num: %s" % (len(self.X_train), len(self.X_test)))

        self.save_cache()

    def normal_data_acc(self):
        feature_dir = os.path.join(config.data_root,
                                   self.dataname,
                                   "feature",
                                   "weights.20-0.9922.h5-1024")
        label_dir = os.path.join(config.data_root,
                                 self.dataname,
                                 "feature",
                                 "weights.20-0.9922.h5-1")
        bias_test_y = np.load(os.path.join(label_dir, "all_test_y.npy"))
        label_test_row = np.load(os.path.join(label_dir, "all_test.npy"))
        all_test_filename = np.load(os.path.join(label_dir, "all_test_filename.npy"))
        label_test = (label_test_row > 0.5).astype(int).reshape(-1)
        bias_test_y = np.array(bias_test_y).reshape(-1)[:len(label_test)]
        acc = sum(bias_test_y == label_test) / float(len(label_test))
        None

    def process_data(self):
        super(DataDogCatNormalTrain, self).process_data()

if __name__ == '__main__':
    d = DataDogCatNormalTrain()
    # d.file_processing()
    # d.fine_tune()

    d.load_data()
    d.process_data()
    d.save_file()

    # for weights_name in ["weights.01-0.9672.h5", "weights.05-0.9844.h5",
    #                      "weights.10-0.9875.h5", "weights.20-0.9922.h5"]:
    # for weights_name in ["weights.25-0.9859.h5", "weights.30-0.9844.h5"]:
    #     d.save_features(weights_name)
    # d.save_features("weights.30-0.9766.h5")
    # d.imagenet_pretrained_model_feature()
    # d.normal_data_acc()