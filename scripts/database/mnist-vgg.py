import numpy as np
import os
import scipy.io as sio
from time import time

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
from scripts.utils.helper_utils import check_dir, accuracy
from scripts.database.database import DataBase

def interpolate(a, x_extend, y_extend):
    x = np.array(range(a.shape[0]))
    xnew = np.linspace(x.min(), x.max(), x_extend)
    f = interp1d(x, a, axis=0)
    a = f(xnew)
    a = a.transpose()
    x = np.array(range(a.shape[0]))
    xnew = np.linspace(x.min(), x.max(), y_extend)
    f = interp1d(x, a, axis=0)
    a = f(xnew)
    a = a.transpose()
    return a


class DataMNIST_VGG(DataBase):
    def __init__(self):
        dataname = config.mnist_vgg
        super(DataMNIST_VGG, self).__init__(dataname)

    def fine_tune(self):
        img_width = 56
        img_height = 56
        feature_dir = os.path.join(config.data_root,
                                   self.dataname,
                                   "feature")
        check_dir(feature_dir)
        model_weight_dir = os.path.join(config.data_root,
                                        self.dataname,
                                        "weights")
        check_dir(model_weight_dir)
        top_model_weights_path = os.path.join(model_weight_dir,
                                             "bottleneck_fc_model.h5")
        train_data_dir = os.path.join(config.data_root,
                                      self.dataname,
                                      "train_all")
        validation_data_dir = os.path.join(config.data_root,
                                      self.dataname,
                                      "test")
        nb_train_samples = 60000
        nb_validation_samples = 10000
        pre_processing_func = lambda x: preprocess_input(x, mode="tf")
        epochs = 50
        batch_size = 32

        train_y = 0
        valid_y = 0

        def get_GAP_VGG_16_bottleneck(input_shape=None):
            model = applications.VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
            res_model = Sequential()
            for layer in model.layers[:-1]:
                res_model.add(layer)
            return res_model

        def save_bottlebeck_features():
            global train_y
            global valid_y
            global nb_train_samples
            global nb_validation_samples

            datagen = ImageDataGenerator(preprocessing_function=pre_processing_func)

            # build the VGG16 network
            model = get_GAP_VGG_16_bottleneck()

            generator = datagen.flow_from_directory(
                train_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=False)

            train_y = generator.classes[generator.index_array]
            nb_train_samples = train_y.reshape(-1).shape[0] // 16 * 16
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
                class_mode="categorical",
                shuffle=False)

            valid_y = generator.classes[generator.index_array]
            nb_validation_samples = valid_y.reshape(-1).shape[0]
            print("validation samples number: %s" % (nb_validation_samples))

            bottleneck_features_validation = model.predict_generator(
                generator, nb_validation_samples // batch_size)
            np.save(os.path.join(feature_dir, 'bottleneck_features_validation.npy'),
                    bottleneck_features_validation)

        def train_top_model():
            train_data = np.load(os.path.join(feature_dir, 'bottleneck_features_train.npy'))
            # train_labels = np.array(
            #     [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
            train_labels = keras.utils.to_categorical(train_y, num_classes=10)
            print(train_data.shape)
            print(train_labels.shape)

            validation_data = np.load(os.path.join(feature_dir, 'bottleneck_features_validation.npy'))
            # validation_labels = np.array(
            #     [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
            validation_labels = keras.utils.to_categorical(valid_y, num_classes=10)
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
            output = Dense(10, activation="softmax")(b)
            model = Model(inputs=a, outputs=output)
            model.compile(optimizer='rmsprop',
                          loss='categorical_crossentropy', metrics=['accuracy'])

            model.fit(train_data, train_labels,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(validation_data, validation_labels))
            model.save_weights(top_model_weights_path)

        def fine_tune():
            input_shape = None
            if K.image_data_format() == 'channels_first':
                input_shape = (3, img_width, img_height)
            else:
                input_shape = (img_width, img_height, 3)
            base_model = get_GAP_VGG_16_bottleneck(input_shape=input_shape)
            print('Model loaded.')
            print(base_model.output_shape, len(base_model.layers))

            model = Sequential()
            for l in base_model.layers:
                model.add(l)
            print(base_model.output_shape)
            a = Input(shape=base_model.output_shape[1:])
            b = Conv2D(1024, (3, 3), activation="relu", padding="same")(a)
            b = GlobalAveragePooling2D()(b)
            output = Dense(10, activation="softmax")(b)
            top_model = Model(inputs=a, outputs=output)
            top_model.load_weights(top_model_weights_path)

            model.add(top_model)

            for layer in model.layers[:15]:
                print(layer)
                layer.trainable = False

            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizers.SGD(lr=2e-4, momentum=0.9),
                          # optimizer = "rmsprop",
                          metrics=['accuracy'])

            # prepare data augmentation configuration
            train_datagen = ImageDataGenerator(
                preprocessing_function=pre_processing_func,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

            test_datagen = ImageDataGenerator(preprocessing_function=pre_processing_func)

            train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='categorical')

            validation_generator = test_datagen.flow_from_directory(
                validation_data_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='categorical')

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

        save_bottlebeck_features()
        train_top_model()
        fine_tune()

    def extract_feature(self, data):
        img_width = 56
        img_height = 56
        model_weight_dir = os.path.join(config.data_root,
                                        self.dataname,
                                        "weights")
        weights_path = os.path.join(model_weight_dir,
                                    "weights.29-0.9870.h5")
        feature_dir = os.path.join(config.data_root,
                                   self.dataname,
                                   "feature")

        def get_GAP_VGG_16_bottleneck(input_shape=None):
            model = applications.VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
            res_model = Sequential()
            for layer in model.layers[:-1]:
                res_model.add(layer)
            return res_model

        def load_weights_model(weights_path):
            input_shape = None
            if K.image_data_format() == 'channels_first':
                input_shape = (3, img_width, img_height)
            else:
                input_shape = (img_width, img_height, 3)
            base_model = get_GAP_VGG_16_bottleneck(input_shape=input_shape)
            print('Model loaded.')
            print(base_model.output_shape, len(base_model.layers))

            model = Sequential()
            for l in base_model.layers:
                model.add(l)
            print(base_model.output_shape)
            a = Input(shape=base_model.output_shape[1:])
            b = Conv2D(1024, (3, 3), activation="relu", padding="same")(a)
            b = GlobalAveragePooling2D()(b)
            output = Dense(10, activation="softmax")(b)
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

            final_model.summary()
            return final_model

        final_model = load_weights_model(weights_path)
        model = Sequential()
        for layer in final_model.layers[:-1]:
            model.add(layer)
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1).repeat(repeats=3, axis=3)
        pre_processing_func = lambda x: preprocess_input(x, mode="tf")
        data = pre_processing_func(data)
        feature = model.predict(data)
        print(feature.shape)
        feature_name = str(feature.shape) + "feature.npy"
        np.save(os.path.join(feature_dir, feature_name), feature)
        exit()

    def save_all_data(self, data, target):
        for i in range(70000):
            x = data[i,:, :]
            target_dir = os.path.join(config.data_root,
                                      self.dataname,
                                      "train_all_no_by_categories")
            check_dir(target_dir)
            target_filename = os.path.join(target_dir,
                                           str(i) + ".jpg")
            x = x.reshape(x.shape[0], x.shape[1], 1).repeat(repeats=3, axis=2)
            x = x.astype(np.uint8)
            img = Image.fromarray(x)
            img.save(target_filename)
            if i % 100 == 0:
                print("iter: ", i)
        exit()

    def save_data_by_category(self, data, target):
        for i in range(60000):
            x = data[i,:, :]
            target_dir = os.path.join(config.data_root,
                                      self.dataname,
                                      "train_all",
                                      str(int(target[i])))
            check_dir(target_dir)
            target_filename = os.path.join(target_dir,
                                           str(i) + ".jpg")
            x = x.reshape(x.shape[0], x.shape[1], 1).repeat(repeats=3, axis=2)
            x = x.astype(np.uint8)
            img = Image.fromarray(x)
            img.save(target_filename)
            if i % 100 == 0:
                print("iter: ", i)

        for i in range(60000,70000):
            x = data[i,:, :]
            target_dir = os.path.join(config.data_root,
                                      self.dataname,
                                      "test",
                                      str(int(target[i])))
            check_dir(target_dir)
            target_filename = os.path.join(target_dir,
                                           str(i) + ".jpg")
            x = x.reshape(x.shape[0], x.shape[1], 1).repeat(repeats=3, axis=2)
            x = x.astype(np.uint8)
            img = Image.fromarray(x)
            img.save(target_filename)
            if i % 100 == 0:
                print("iter: ", i)
        exit()

    def vgg_accuracy(self, target):
        feature_path = os.path.join(config.data_root,
                                    self.dataname,
                                    "feature",
                                    "(70000, 10)feature.npy")
        data = np.load(feature_path)
        pred = data.argmax(axis=1)
        # target = target[:60000]
        # pred = pred[:60000]
        target = target[60000:]
        pred = pred[60000:]
        classes = [3, 5]
        selected_idx = [idx for idx, c in enumerate(target) if c in classes]
        selected_idx = np.array(selected_idx)
        selected_target = target[selected_idx]
        selected_pred = pred[selected_idx]
        print(classes, "acc:", accuracy(selected_target, selected_pred))
        exit()

    def load_data(self):
        print("loading data from sklearn!")
        #mnist = fetch_mldata("MNIST original", data_home="./")
        mnist = sio.loadmat(os.path.join(config.scripts_root,
                                         "database",
                                         "mnist.mat"))
        target = mnist["target"].reshape(-1)
        data = mnist["data"]

        data = data
        index = np.array(range(len(target)))
        np.random.seed(123)
        np.random.shuffle(index)
        target = target[index]
        data = data[index, :]

        # self.vgg_accuracy(target)

        x_extend = 56
        y_extend = 56
        data_extend = np.zeros((data.shape[0],x_extend, y_extend))
        for i in range(data.shape[0]):
            x = data[i,:].reshape(28, 28)
            data_extend[i,:,:] = interpolate(x, x_extend, y_extend)
        data = data_extend

        self.save_all_data(data, target)
        # self.extract_feature(data)
        ################ test #################
        # x = data[2,:,:].astype(np.uint8)
        # x = x.reshape(x_extend, x_extend, 1).repeat(repeats=3, axis=2)
        # img = Image.fromarray(x)
        # img.show()
        ################ test #################

        # # test
        # x = data[0,:] * 255.0
        # x = x.reshape(28,28).astype(np.uint8)
        # # img_origin = x.reshape(28,28,1).repeat(repeats=3, axis=2)
        # # img_origin = Image.fromarray(img_origin)
        # # img_origin.show()
        # x_extend = interpolate(x, 56*10, 56*10).astype(np.uint8)
        # img_extend = x_extend.reshape(56*10, 56*10, 1).repeat(repeats=3, axis=2)
        # img_extend = Image.fromarray(img_extend)
        # img_extend.show()
        # exit()

        # data  = data.reshape(-1,x_extend,y_extend,1).repeat(repeats=3, axis=3)
        # print("data shape:{}".format(data.shape))
        # img_width, img_height = x_extend, y_extend
        # batch_size = 70
        # input_shape = (img_width, img_height, 3)
        # base_model = keras.applications.VGG16(include_top=False, input_shape=input_shape, weights="imagenet")
        # print("Model loaded.")
        # print("Model output shape:",base_model.output_shape)
        # print(base_model.summary())
        #
        # dataset = tf.data.Dataset.from_tensor_slices((data[:1000,:,:,:], target[:1000]))
        # dataset = dataset.batch(batch_size)
        # with tf.Session() as sess:
        #     a, b = sess.run(dataset.get_next())
        #     print(a.shape, b.shape)

        # image_placeholder = tf.placeholder(data.dtype, data.shape)
        # labels_placeholder = tf.placeholder(target.dtype, target.shape)
        # dataset = tf.data.Dataset.from_tensor_slices((image_placeholder, labels_placeholder))
        # dataset = dataset.batch(batch_size)
        # iterator = dataset.make_initializable_iterator()
        # with tf.Session() as sess:
        #     sess.run(iterator.initializer, feed_dict={
        #         image_placeholder: data,
        #         labels_placeholder: target
        #     })
        # data = base_model.predict(data)
        # print(data.shape)

        feature_path = os.path.join(config.data_root,
                                    self.dataname,
                                    "feature",
                                    "(70000, 1024)feature.npy")
        data = np.load(feature_path)
        data = data.reshape(70000, -1)
        self.X_train = data[:60000, :]
        self.X_test = data[60000:, :]
        self.y_train = target[:60000]
        self.y_test = target[60000:]

        print("data loaded!!")
        print("train data num: %s, test data num: %s" % (len(self.X_train), len(self.X_test)))

        self.save_cache()

    def process_data(self):
        super(DataMNIST_VGG, self).process_data()


if __name__ == '__main__':
    d = DataMNIST_VGG()
    d.load_data()
    d.process_data()
    d.save_file()
    # d.fine_tune()