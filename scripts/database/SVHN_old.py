import numpy as np
import os
import pandas as pd
import scipy.io as sio
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from PIL import Image
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Input, MaxPooling2D
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

class DataSVHN(DataBase):
    def __init__(self):
        dataname = config.svhn
        super(DataSVHN, self).__init__(dataname)
        print(config.data_root)
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
                                           "bias_train")
        self.train_cat_dir = os.path.join(self.train_data_dir,
                                          "3")
        check_dir(self.train_cat_dir)
        self.train_dog_dir = os.path.join(self.train_data_dir,
                                          "5")
        check_dir(self.train_dog_dir)
        self.validation_data_dir = os.path.join(config.data_root,
                                                self.dataname,
                                                "normal_test")
        self.validation_cat_dir = os.path.join(self.validation_data_dir,
                                               "3")
        check_dir(self.validation_cat_dir)
        self.validation_dog_dir = os.path.join(self.validation_data_dir,
                                               "5")
        check_dir(self.validation_dog_dir)
        self.test_data_dir = os.path.join(config.data_root,
                                          self.dataname,
                                          "normal_test")
        self.test_cat_dir = os.path.join(self.test_data_dir,
                                         "3")
        check_dir(self.test_cat_dir)
        self.test_dog_dir = os.path.join(self.test_data_dir,
                                         "5")

        check_dir(self.test_dog_dir)
        self.nb_train_samples = None
        self.nb_validation_samples = None
        self.epochs = 50
        self.batch_size = 16
        self.train_y = 0
        self.valid_y = 0

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

    def _load_weights_model(self, weights_path):
        img_width = self.img_width
        img_height = self.img_height
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

        final_model = self._load_weights_model(weight_path)
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

    def save_images(self):
        train_data_mat = sio.loadmat(self.raw_train_data_path)
        test_data_mat = sio.loadmat(self.raw_test_data_path)
        X_train = train_data_mat["X"].transpose(3, 0, 1, 2)
        y_train = train_data_mat["y"].reshape(-1).astype(int) - 1
        X_test = test_data_mat["X"].transpose(3, 0, 1, 2)
        y_test = test_data_mat["y"].reshape(-1).astype(int) - 1

        train_dir = os.path.join(config.data_root, self.dataname, "train")
        test_dir = os.path.join(config.data_root, self.dataname, "test")
        check_dir(train_dir)
        check_dir(test_dir)

        # save training data
        for i in range(X_train.shape[0]):
            x = X_train[i, :, :, :]
            target_dir = os.path.join(train_dir, str(y_train[i]))
            check_dir(target_dir)
            img = Image.fromarray(x)
            target_filename = os.path.join(target_dir, str(i) + ".jpg")
            img.save(target_filename)
            if i % 100 == 0:
                print("iter:", i, ", ", end="\t")

        # save testing data
        for i in range(X_test.shape[0]):
            x = X_test[i, :, :, :]
            target_dir = os.path.join(test_dir, str(y_test[i]))
            check_dir(target_dir)
            img = Image.fromarray(x)
            target_filename = os.path.join(target_dir, str(i) + ".jpg")
            img.save(target_filename)
            if i % 100 == 0:
                print("iter:", i, ", ", end="\t")

        None

    def _processing_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation="relu", input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3,3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(4096, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(10, activation="softmax"))

        adam = optimizers.Adam(lr=0.0001, epsilon=1e-6)
        model.compile(loss="categorical_crossentropy",
                      optimizer=adam,
                      metrics=["accuracy"])

        model.summary()
        return model

    def processing_fine_tune(self):

        model = self._processing_model()

        batch_size = 64

        train_datagen = ImageDataGenerator(rescale= 1. / 255)
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size = (32, 32),
            batch_size = batch_size,
            class_mode="categorical",
            shuffle=True
        )

        valid_generator = train_datagen.flow_from_directory(
            self.test_dir,
            target_size = (32, 32),
            batch_size = batch_size,
            class_mode="categorical",
            shuffle=True
        )

        nb_train_samples = len(train_generator.filenames)
        nb_valid_samples = len(train_generator.filenames)

        model_dir = os.path.join(config.data_root,
                                 self.dataname,
                                 "pre_weights")
        check_dir(model_dir)
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(model_dir,
                                  "weights.{epoch:02d}-{val_acc:.4f}.h5"),
            verbose=1
        )
        model.fit_generator(
            train_generator,
            steps_per_epoch = nb_train_samples // batch_size,
            epochs=300,
            validation_data = valid_generator,
            validation_steps= nb_valid_samples // batch_size,
            callbacks=[checkpointer]
        )

    def processing_save_feature(self):
        model_path = os.path.join(config.data_root,
                                  self.dataname,
                                  "pre_weights",
                                  "weights.50-0.9330.h5")
        model = self._processing_model()
        model.load_weights(model_path)
        batch_size = 64

        print("batch_size: {}".format(batch_size))

        train_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(32, 32),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True
        )

        valid_generator = train_datagen.flow_from_directory(
            self.test_dir,
            target_size=(32, 32),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True
        )

        nb_train_samples = len(train_generator.filenames)
        nb_valid_samples = len(valid_generator.filenames)

        model_dir = os.path.join(config.data_root,
                                 self.dataname,
                                 "pre_weights")
        check_dir(model_dir)

        feature_model = Sequential()
        for layer in model.layers[:-1]:
            feature_model.add(layer)

        feature_dir = os.path.join(config.data_root,
                                   self.dataname,
                                   "pre_features")
        check_dir(feature_dir)

        file_prefix = os.path.split(self.train_dir)[1].split(".")[0]
        train_features = feature_model.predict_generator(
            train_generator, nb_train_samples // batch_size)
        train_y = train_generator.classes[train_generator.index_array]
        train_filename = train_generator.filenames
        print("train feature shape:{}".format(train_features.shape))
        np.save(os.path.join(feature_dir, file_prefix + ".npy"), train_features)
        np.save(os.path.join(feature_dir, file_prefix + "_y.npy"), train_y)
        np.save(os.path.join(feature_dir, file_prefix + "_filename.npy"), train_filename)

        file_prefix = os.path.split(self.test_dir)[1].split(".")[0]
        valid_features = feature_model.predict_generator(
            valid_generator, nb_valid_samples // batch_size)
        valid_y = valid_generator.classes[valid_generator.index_array]
        valid_filename = valid_generator.filenames
        print("valid feature shape: {}".format(valid_features.shape))
        np.save(os.path.join(feature_dir, file_prefix + ".npy"), valid_features)
        np.save(os.path.join(feature_dir, file_prefix + "_y.npy"), valid_y)
        np.save(os.path.join(feature_dir, file_prefix + "_filename.npy"), valid_filename)

    def preocessing_bias(self):
        import matplotlib.pyplot as plt

        feature_dir = os.path.join(config.data_root, self.dataname, "pre_features")
        X_train = np.load(os.path.join(feature_dir, "train.npy"))
        y_train = np.load(os.path.join(feature_dir, "train_y.npy")).reshape(-1).astype(int)
        X_test = np.load(os.path.join(feature_dir, "test.npy"))
        y_test = np.load(os.path.join(config.data_root,self.dataname, "test_y.npy")).reshape(-1).astype(int)

        # data = np.load(os.path.join(feature_dir, "image_data.npz"))
        # X_train = data["X"]
        # y_train = data["Y"].reshape(-1).astype(int)

        X_test = np.load(os.path.join(config.raw_data_root, self.dataname, "test_activations.npy"))
        pred_y_test = np.load(os.path.join(config.raw_data_root, self.dataname, "test_probs.npy"))\
            .argmax(axis=1).astype(int)

        print((pred_y_test == y_test).sum() / float(len(y_test)))

        tsne = TSNE(n_components=2)
        X_embeding = tsne.fit_transform(X_test[:2000])
        color_map = plt.get_cmap("tab10")(pred_y_test[:2000])
        plt.scatter(X_embeding[:,0], X_embeding[:,1], s=8, c=color_map)
        plt.show()

    def test_bias_data(self):
        feature_dir = os.path.join(config.data_root, self.dataname, "test_feature")
        X_train = np.load(os.path.join(feature_dir, "test.npy"))
        y_train = np.load(os.path.join(feature_dir,"test_y.npy"))

        peak_dir = os.path.join(config.data_root, self.dataname, "two_peak_test")
        check_dir(peak_dir)

        Index = np.array(range(X_train.shape[0]))
        for i in range(10):
            selected_idx = (y_train == i)
            X = X_train[selected_idx,:]
            Y = Index[selected_idx]
            kmeans = KMeans(n_clusters=2, random_state=123)
            kmeans.fit(X)
            labels = kmeans.predict(X)
            dir_name = os.path.join(peak_dir, str(i))
            check_dir(dir_name)
            for j in range(2):
                idx = (labels==j)
                filename = os.path.join(dir_name, str(j) + ".npy")
                np.save(filename, Y[idx])


        X_train = np.load(os.path.join(feature_dir, "test_x.npy"))
        y_train = np.load(os.path.join(feature_dir, "test_y.npy")) \
            .reshape(-1).astype(int)
        for current_focus in range(10):
            dir_name = os.path.join(peak_dir, str(current_focus))
            for i in range(2):
                Index = np.load(os.path.join(dir_name, str(i) + ".npy"))
                sub_dir = os.path.join(dir_name, str(i))
                check_dir(sub_dir)
                for j in range(Index.shape[0]):
                    idx = Index[j]
                    assert y_train[idx] == current_focus
                    x = X_train[idx, :, :, :]
                    img = Image.fromarray(x)
                    img.save(os.path.join(sub_dir, str(idx) + ".jpg"))

    def bias_data(self):
        feature_dir = os.path.join(config.data_root, self.dataname, "pre_features")
        # data = np.load(os.path.join(feature_dir, "image_data.npz"))
        # X_train = data["X"]
        # y_train = data["Y"].reshape(-1).astype(int)
        # Index = np.array(range(X_train.shape[0]))
        # for i in range(10):
        #     selected_idx = (y_train == i)
        #     X = X_train[selected_idx,:]
        #     Y = Index[selected_idx]
        #     kmeans = KMeans(n_clusters=2, random_state=123)
        #     kmeans.fit(X)
        #     labels = kmeans.predict(X)
        #     dir_name = os.path.join(config.data_root, self.dataname, "two_peak", str(i))
        #     check_dir(dir_name)
        #     for j in range(2):
        #         idx = (labels==j)
        #         filename = os.path.join(dir_name, str(j) + ".npy")
        #         np.save(filename, Y[idx])

        peak_dir = os.path.join(config.data_root, self.dataname, "two_peak")
        check_dir(peak_dir)
        X_train = np.load(os.path.join(config.data_root, self.dataname, "train_x_original.npy"))
        y_train = np.load(os.path.join(config.data_root, self.dataname, "train_y_original.npy")) \
            .reshape(-1).astype(int)
        for current_focus in range(10):
            dir_name = os.path.join(peak_dir, str(current_focus))
            for i in range(2):
                Index = np.load(os.path.join(dir_name, str(i) + ".npy"))
                sub_dir = os.path.join(dir_name, str(i))
                check_dir(sub_dir)
                for j in range(Index.shape[0]):
                    idx = Index[j]
                    assert y_train[idx] == current_focus
                    x = X_train[idx, :, :, :]
                    img = Image.fromarray(x)
                    img.save(os.path.join(sub_dir, str(idx) + ".jpg"))

    def processing_labels(self):
        train_label_dir = os.path.join(config.raw_data_root,
                                       self.dataname,
                                       "two_peak")
        test_label_dir = os.path.join(config.raw_data_root,
                                      self.dataname,
                                      "two_peak_test")
        name_list = ["浅色底深色字", "深色底浅色字"]
        name_map = {
            name_list[0]: 0,
            name_list[1]: 1
        }
        for label_dir in [train_label_dir, test_label_dir]:
            for class_name in range(10):
                class_dir = os.path.join(label_dir, str(class_name))
                cleaned_0_dir = os.path.join(class_dir, name_list[0])
                cleaned_1_dir = os.path.join(class_dir, name_list[1])
                cleaned_0 = os.listdir(cleaned_0_dir)
                cleaned_0 = [int(s.split(".")[0]) for s in cleaned_0]
                cleaned_1 = os.listdir(cleaned_1_dir)
                cleaned_1 = [int(s.split(".")[0]) for s in cleaned_1]
                uncleaned_0 = np.load(os.path.join(class_dir, "0.npy"))
                uncleaned_1 = np.load(os.path.join(class_dir, "1.npy"))
                minus_0 = [i for i in uncleaned_0 if i not in cleaned_0]
                minus_1 = [i for i in uncleaned_1 if i not in cleaned_1]
                final_0 = cleaned_0 + minus_1
                final_1 = cleaned_1 + minus_0
                np.save(os.path.join(class_dir, "final_0.npy"), final_0)
                np.save(os.path.join(class_dir, "final_1.npy"), final_1)

    def load_data(self):
        feature_dir = os.path.join(config.data_root,
                                   self.dataname,
                                   "feature",
                                   "weights.100-0.5465.h5-1024")
        bias_train = np.load(os.path.join(feature_dir, "bias_train.npy"))
        bias_train_y = np.load(os.path.join(feature_dir, "bias_train_y.npy"))
        bias_train_y = np.array(bias_train_y).reshape(-1)[:bias_train.shape[0]]
        all_test = np.load(os.path.join(feature_dir, "normal_test.npy"))
        all_test_y = np.load(os.path.join(feature_dir, "normal_test_y.npy"))
        all_test_y = np.array(all_test_y).reshape(-1)[:all_test.shape[0]]

        self.X_train = bias_train
        self.y_train = bias_train_y
        # self.X_test = bias_test
        # self.y_test = bias_test_y
        self.X_test = all_test
        self.y_test = all_test_y
        X = np.concatenate((self.X_train, self.X_test), axis=0)
        y = np.array(self.y_train.tolist() + self.y_test.tolist())


        t0 = time()
        tsne = TSNE(n_components=2, random_state=123)
        X_embed = tsne.fit_transform(X)
        print("using sklean-tsne, running time: ", time() - t0)

        n_train_samples = self.X_train.shape[0]
        self.X_embed_train = X_embed[:n_train_samples, :]
        self.X_embed_test = X_embed[n_train_samples:, :]

        print("data loaded!!")
        print("train data num: %s, test data num: %s" % (len(self.X_train), len(self.X_test)))

    def _load_data(self):
        print("loading data from original data!!!")

        train_label_dir = os.path.join(config.raw_data_root,
                                       self.dataname,
                                       "two_peak")
        test_label_dir = os.path.join(config.raw_data_root,
                                      self.dataname,
                                      "two_peak_test")

        X_train = np.load(os.path.join(config.data_root, self.dataname, "train_x_original.npy"))
        y_train = np.load(os.path.join(config.data_root, self.dataname, "train_y_original.npy")) \
            .reshape(-1).astype(int)

        X_test = np.load(os.path.join(config.data_root, self.dataname, "test_x.npy"))
        y_test = np.load(os.path.join(config.data_root, self.dataname, "test_y.npy")) \
            .reshape(-1).astype(int)

        np.random.seed(123)

        bias_train_idx = []
        for idx, class_name in enumerate([3, 5]):
            class_dir = os.path.join(train_label_dir, str(class_name))
            class_labels = np.load(os.path.join(class_dir, "final_" + str(idx) + ".npy"))
            bias_train_idx = bias_train_idx + class_labels.reshape(-1).tolist()
        bias_train_idx = np.array(bias_train_idx)
        np.random.shuffle(bias_train_idx)

        test_idx = []
        for class_name in [3, 5]:
            class_dir = os.path.join(test_label_dir, str(class_name))
            class_label_0 = np.load(os.path.join(class_dir, "final_0.npy")).reshape(-1).tolist()
            class_label_1 = np.load(os.path.join(class_dir, "final_1.npy")).reshape(-1).tolist()
            test_idx = test_idx + class_label_0 + class_label_1
        test_idx = np.array(test_idx)
        np.random.shuffle(test_idx)

        self.X_train = X_train[bias_train_idx,:, :, :]
        self.y_train = y_train[bias_train_idx].astype(int)
        self.X_test = X_test[test_idx, :, :, :]
        self.y_test = y_test[test_idx].astype(int)

        train_data_dir = os.path.join(config.data_root, self.dataname, "bias_train")
        test_data_dir = os.path.join(config.data_root, self.dataname, "normal_test")
        check_dir(train_data_dir)
        check_dir(test_data_dir)

        for i in range(len(self.y_train)):
            label = self.y_train[i]
            x = self.X_train[i,:, :, :].astype(np.uint8)
            dir = os.path.join(train_data_dir, str(label))
            check_dir(dir)
            img = Image.fromarray(x)
            img.save(os.path.join(dir, str(bias_train_idx[i]) + ".jpg"))

        for i in range(len(self.y_test)):
            label = self.y_test[i]
            x = self.X_test[i, :, :, :].astype(np.uint8)
            dir = os.path.join(test_data_dir, str(label))
            check_dir(dir)
            img = Image.fromarray(x)
            img.save(os.path.join(dir, str(test_idx[i]) + ".jpg"))

        return True

    def process_data(self):
        super(DataSVHN, self).process_data()



if __name__ == '__main__':
    d = DataSVHN()

    d.load_data()
    d.process_data()
    d.save_file()

    # d._load_data()

    # d.fine_tune()
    # d.save_features("weights.100-0.5465.h5")
    # d.save_images()

    # d.processing_fine_tune()
    # d.processing_save_feature()

    # d.preocessing_bias()
    # d.processing_labels()
