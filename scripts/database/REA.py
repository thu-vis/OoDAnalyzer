import numpy as np
import os
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
from scripts.utils.helper_utils import check_dir, pickle_save_data, pickle_load_data
from scripts.database.database import DataBase
from scripts.utils.log_utils import logger
from scripts.utils.embedder_utils import Embedder
import shutil

class DataREA(DataBase):
    def __init__(self):
        dataname = config.rea
        super(DataREA, self).__init__(dataname)

    def preprocessing_data(self):
        self.class_name = ["rea-free", "rea"]
        self.class_name_encoding = {
            self.class_name[0]: 0,
            self.class_name[1]: 1
        }
        images_dir = self.images_dir
        images_pids = []
        # if len(os.listdir(images_dir)) == 20864:
        if 1:
            logger.warn("skip 'saving images to image_dir' process.")
        else:
            src_train_images_dir = \
            "G:\\BaiduNetdiskDownload\\AI Challenger眼底分割比赛数据集\\ai_challenger_fl2018_trainingset\\Edema_trainingset\\original_images"
            src_val_images_dir = \
            "G:\\BaiduNetdiskDownload\\AI Challenger眼底分割比赛数据集\\ai_challenger_fl2018_validationset\\Edema_validationset\\original_images"
            src_test_images_dir = \
                "G:\\BaiduNetdiskDownload\\AI Challenger眼底分割比赛数据集\\ai_challenger_fl2018_testset\\Edema_testset\\original_images"
            src_normal_images_dir = \
            "G:\\BaiduNetdiskDownload\\NORMAL 63\\NORMAL 63"
            id = 0
            for set_dir in [src_train_images_dir, src_val_images_dir,
                               src_test_images_dir]:
                               # src_test_images_dir, src_normal_images_dir]:
                logger.info(set_dir)
                pids = os.listdir(set_dir)
                images_pids.extend(pids)
                for pid in pids:
                    patient_dir = os.path.join(set_dir, pid)
                    for image_name in range(128):
                        src = os.path.join(patient_dir, str(image_name+1) + ".bmp")
                        # print(src)
                        target = os.path.join(images_dir, str(id) + ".jpg")
                        id = id + 1
                        # if not len(os.listdir(images_dir)) == 20864:
                        if 1:
                            shutil.copy(src, target)
            logger.info("total id: {}".format(id))

        label_dir = os.path.join(config.data_root,
                                 self.dataname,
                                 "labels")
        check_dir(label_dir)
        label_pids = []
        # if len(os.listdir(label_dir)) == 12800:
        if 1:
            logger.warn("skip 'saving images to image_dir' process.")
        else:
            src_train_label_dir = \
            r"G:\BaiduNetdiskDownload\AI Challenger眼底分割比赛数据集\ai_challenger_fl2018_trainingset\Edema_trainingset\label_images"
            src_val_label_dir = \
            r"G:\BaiduNetdiskDownload\AI Challenger眼底分割比赛数据集\ai_challenger_fl2018_validationset\Edema_validationset\label_images"
            src_test_label_dir = \
            r"G:\BaiduNetdiskDownload\AI Challenger眼底分割比赛数据集\ai_challenger_fl2018_testset\Edema_testset\label_images"
            id = 0
            for src_label_dir in [src_train_label_dir, src_val_label_dir, \
                              src_test_label_dir]:
                logger.info(src_label_dir)
                pids = os.listdir(src_label_dir)

                label_pids.extend(pids)
                for pid in pids:
                    patient_dir = os.path.join(src_label_dir, pid)
                    for image_name in range(128):
                        src = os.path.join(patient_dir, str(image_name+1) + ".bmp")
                        target = os.path.join(label_dir, str(id) + ".bmp")
                        id = id + 1
                        if not len(os.listdir(label_dir)) == 12800:
                            shutil.copy(src, target)
            logger.info("total id: {}".format(id))
            image_num = id

        # check
        # for i in range(len(label_pids)):
        #     assert images_pids[i].strip(".img")\
        #            == label_pids[i].strip("_labelMark")

        self.y = []
        # for i in range(12800):
        #     label_img = os.path.join(label_dir, str(i) + ".bmp")
        #     img = Image.open(label_img)
        #     label_data = np.array(img)
        #     gt = int((label_data > 0).sum() > 0)
        #     self.y.append(gt)
        self.y = pickle_load_data(os.path.join(config.data_root, self.dataname, "gt.pkl"))
        logger.info("y length info: {}".format(len(self.y)))

        self.train_idx = np.array(range(70*128))
        self.train_redundant_idx = np.array([])
        self.valid_idx = np.array([])
        self.valid_redundant_idx = np.array([])
        self.test_idx = np.array(range(70*128, (70+15)*128))
        bias = {
            2: [37, 38, 39, 40, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97],
            4: [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
            6: [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
                75, 76, 77, 78, 79, 80, 81, 82],
            8: [24, 25, 26, 27, 28, 29, 51, 52, 53, 54, 55, 56, 57],
            9: [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 92,
                93, 94, 95, 96, 97]
        }
        self.test_redundant_idx = []
        for val_id in bias:
            img_id_list = bias[val_id]
            for img_id in img_id_list:
                self.test_redundant_idx.append((70+val_id-1)*128 + img_id)

        logger.warn("info confirm: train num: {}, train_redundant num:{}, test_redundant num:{}"
                    .format(len(self.train_idx), len(self.train_redundant_idx), len(self.test_redundant_idx)))

        # save by categories
        # bias training
        # for idx in self.train_idx:
        #     src = os.path.join(images_dir, str(idx) + ".jpg")
        #     label = self.y[idx]
        #     img_dir = os.path.join(self.train_data_dir, self.class_name[label])
        #     check_dir(img_dir)
        #     target = os.path.join(img_dir, str(idx) + ".jpg")
        #     shutil.copy(src, target)
        #
        # # normal test
        # for idx in self.test_idx:
        #     src = os.path.join(images_dir, str(idx) + ".jpg")
        #     label = self.y[idx]
        #     img_dir = os.path.join(self.test_data_dir, self.class_name[label])
        #     check_dir(img_dir)
        #     target = os.path.join(img_dir, str(idx) + ".jpg")
        #     shutil.copy(src, target)



        self.all_data = {
            "class_name": self.class_name,
            "class_name_encoding": self.class_name_encoding,
            "X": None,
            "y": self.y,
            "train_idx": self.train_idx.tolist(),
            "train_redundant_idx": self.train_redundant_idx.tolist(),
            "valid_idx": self.valid_idx.tolist(),
            "valid_redundant_idx": self.valid_redundant_idx.tolist(),
            "test_idx": self.test_idx.tolist(),
            "test_redundant_idx": self.test_redundant_idx
        }
        self.save_cache()

    def load_data(self, loading_from_buffer=True):
        super(DataREA, self).load_data(loading_from_buffer)
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

    def inplace_process_data(self):
        features_dir_name = [
                            # "unet_nopre_mix_3_0.001_unet-wce-dice-bce",
                            #  "unet_nopre_mix_33_0.001_unet-wce-eldice-bce",
                            #  "unet_nested_nopre_mix_3_0.001_unet_nested-wce-dice-bce",
                            #  "unet_nested_nopre_mix_33_0.001_unet_nested-wce-eldice-bce",
                            #  "unet_nested_dilated_nopre_mix_3_0.001_unet_nested_dilated-wce-dice-bce",
                            #  "unet_nested_dilated_nopre_mix_33_0.001_unet_nested_dilated-wce-eldice-bce"
                            "unet_nested_dilated_nopre_mix_33_0.001_repeat_1",
                            "unet_nested_dilated_nopre_mix_33_0.001_repeat_2",
                            "unet_nested_dilated_nopre_mix_33_0.001_repeat_3",
                            "unet_nested_dilated_nopre_mix_33_0.001_repeat_4",
        ]
        for feature_dir_name in features_dir_name:
            feature_dir = os.path.join(self.feature_dir, feature_dir_name)
            train_info = pickle_load_data(os.path.join(feature_dir, "train_feature.npy"))
            val_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
            test_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
            X = np.concatenate((train_info["features"],
                                val_info["features"],
                                test_info["features"]), axis=0)
            pickle_save_data(os.path.join(feature_dir, "X.pkl"), X)

    def inplace_process_data_normal_features(self):

        cnn_features_dir_name = [
                # "brief-200",
                # "orb-200"
            "surf-200"

        ]
        for weight_name in cnn_features_dir_name:
            X = self.postprocess_data(weight_name, if_return=True)
            filename = os.path.join(self.feature_dir, weight_name, "X.pkl")
            pickle_save_data(filename, X)

    def fine_tune_inplace_process_data(self):
        feature_dir = os.path.join(self.feature_dir, "fine-tune")
        train_info = pickle_load_data(os.path.join(feature_dir, "train_feature.npy"))
        val_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
        test_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
        X = np.concatenate((train_info["features"],
                            val_info["features"],
                            test_info["features"]), axis=0)
        pickle_save_data(os.path.join(feature_dir, "X.pkl"), X)

    def pretrain_get_features(self):
        model_names = ["vgg", "resnet50", "xception", "inceptionv3",
                       "inceptionresnet", "mobilenet"]
        # model_names = ["mobilenet", "nasnet"]
        for model_name in model_names:
            super(DataREA, self).pretrain_get_model(model_name=model_name)

    def fine_tune_postprocess_data(self):
        feature_dir = os.path.join(self.feature_dir, "fine-tune")
        train_info = pickle_load_data(os.path.join(feature_dir, "train_feature.npy"))
        val_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
        test_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
        X = np.concatenate((train_info["features"],
                            val_info["features"],
                            test_info["features"]), axis=0)
        # X = pickle_load_data(os.path.join(config.data_root, self.dataname, "all_embed.pkl"))
        # X = X[:85*128, :]
        self.X = X

        super(DataREA, self).postprocess_data("useless-str")
        # self.all_embed_X = 1
        # self.embed_X = pickle_load_data(os.path.join(config.data_root, self.dataname, "patient_embed.pkl"))


    def postprocess_data(self, weight_name, if_return=False):
        """

        :return:
        """
        feature_dir = os.path.join(self.feature_dir, weight_name)
        dirs = [self.train_data_dir, self.valid_data_dir, self.test_data_dir]
        # X = np.zeros((self.y.shape[0], 1024))
        X = None

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
                img_id = int(img_id)
                if len(features.shape) > 2:
                    features = features.reshape(features.shape[0], -1)
                if X is None:
                    X = np.zeros((len(self.y), features.shape[1]))
                X[img_id,:] = features[idx,:]
            print("new train_idx len: ", len(self.train_idx))
        self.X = X
        if if_return:
            logger.info("'if return' flag is enabled. Returning immediately!")
            return X

        super(DataREA, self).postprocess_data(weight_name)

    def process_saliency_map(self, feature_name):
        feature_dir = os.path.join(self.feature_dir, feature_name)
        saliency_map_dir = os.path.join(self.feature_dir, feature_name, "saliency_map")
        check_dir(saliency_map_dir)
        train_cam = pickle_load_data(os.path.join(feature_dir, "train_cam.pkl"))
        train_info = pickle_load_data(os.path.join(feature_dir, "train_feature.npy"))
        val_info = pickle_load_data(os.path.join(feature_dir, "val_feature.npy"))
        pred_y = np.concatenate((train_info["pred_cls"],
                                 val_info["pred_cls"]), axis=0)[:,0]


        val_cam = pickle_load_data(os.path.join(feature_dir, "val_cam.pkl"))
        all_cams = train_cam + val_cam
        for idx, cam in enumerate(all_cams):
            original_img_path = os.path.join(config.data_root,
                                             self.dataname,
                                             "images_no_crop",
                                             str(idx) + ".jpg")
            img_path = os.path.join(saliency_map_dir,
                                    str(idx) + ".jpg")
            if idx % 100 == 0:
                print(idx)
            y = pred_y[idx]
            cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
            cam = cam.astype(np.uint8)
            cam_img = Image.fromarray(cam)
            cam_img = cam_img.resize((512, 1024), Image.ANTIALIAS)
            cam_img = np.array(cam_img)
            if y < 0.5:
                cam_img = 255 - cam_img
            mask = np.ones(cam_img.shape) * 0.7
            mask[cam_img>150] = 1.5
            original_img = Image.open(original_img_path)
            img_data = np.array(original_img).astype(float)
            img_data = (img_data * mask[:,:])
            saliency_img_data = img_data.astype(np.uint8)

            img_data = np.array(original_img).astype(float)
            max_value = -1
            max_idx = -1
            for i in range(512):
                value = img_data[i:i+512,:].sum()
                if value > max_value:
                    max_value = value
                    max_idx = i
            img_data = saliency_img_data[max_idx:max_idx+512]
            cam_img = Image.fromarray(img_data)
            cam_img.save(img_path)
            # exit()

if __name__ == '__main__':
    d = DataREA()
    # d.preprocessing_data();exit()
    # for i in ["sift-200", "HOG-means-200", "LBP-hist"]:
    # for i in ["HOG-kmeans-200", "LBP-hist"]:
    # # for i in ["LBP-hist"]:
    #     d.lowlevel_features(i)
    # d.pretrain_get_features()
    # d.load_data()
    # d.inplace_process_data_normal_features()
    d.inplace_process_data()
    # d.inplace_process_data()
    # d.fine_tune_postprocess_data()
    # d.fine_tune_inplace_process_data()
    # d.process_saliency_map()
    # d.save_file()
    # d.sift_features("orb-200")
    # d.sift_features("brief-200")
    # d.sift_features("surf-200")

    for feature_dir in [
        # "unet_nested_dilated_nopre_mix_33_0.001_repeat_1",
        # "unet_nested_dilated_nopre_mix_33_0.001_repeat_2",
        # "unet_nested_dilated_nopre_mix_33_0.001_repeat_3",
        # "unet_nested_dilated_nopre_mix_33_0.001_repeat_4",
        # "unet_nested_dilated_nopre_mix_33_0.001_unet_nested",
        "unet_nested_dilated_nopre_mix_33_0.001_unet_nested_dilated-wce-eldice-bce",
    ]:
        d.process_saliency_map(feature_dir)