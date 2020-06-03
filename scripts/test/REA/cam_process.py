import numpy as np
import os
import random
from PIL import Image
import shutil
from multiprocessing import Pool
import matplotlib.pyplot as plt
import cv2

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import *

gradient = np.linspace(0, 1, 256)
gradient = np.vstack(gradient).reshape(-1)
color_gradient = plt.get_cmap("RdYlGn")(gradient)
color_gradient = (color_gradient[:,:3] * 255).astype(np.uint8)
# color_gradient = color_gradient[:,np.array([2,1,0])]
color_gradient = color_gradient.reshape(color_gradient.shape[0],1,-1)


class cam_process(object):
    def __init__(self):
        self.all_name = []
        self.all_seg_confusion_matrix = []
        self.all_seg = []
        self.all_gt = []
        self.prediction = []
        self.normal_idx = []
        self.patient_idx = []
        self.train_idx = []
        self.root = os.path.join(config.raw_data_root, "REA")
        self.normal_root = os.path.join(self.root, "normal")
        self.patient_root = os.path.join(self.root, "val")
        self.train_root = os.path.join(self.root, "train")

    def handler(self,i, dir_name, target_dir, prob):
        print(i)
        for j in range(128):
            img_name = os.path.join(dir_name, "%02d" % j + ".png")
            pre_img_name = None
            if j == 0:
                pre_img_name = img_name
            else:
                pre_img_name = os.path.join(dir_name, "%02d" % (j-1) + ".png")
            img = Image.open(img_name)
            suc_img_name = None
            if j == 127:
                suc_img_name = img_name
            else:
                suc_img_name = os.path.join(dir_name, "%02d" % (j+1) + ".png")
            img_data = np.array(img)
            origin_data = img_data[:, :512, :]
            pred_data = img_data[:, 512:512 * 2, :]
            label_data = img_data[:, 512 * 2:512 * 3, :]
            cam, norm_cam = self.all_cam[(70 + i) * 128 + j]
            color_data = origin_data.copy()
            color_data[:, :512, 0] = np.array(Image.open(pre_img_name))[:, :512, 0]
            color_data[:, :512, 2] = np.array(Image.open(suc_img_name))[:, :512, 0]

            # cam = cam.reshape(cam.shape[0], cam.shape[1], -1) \
            #     .repeat(axis=2, repeats=3)
            img_cam = Image.fromarray(cam)
            img_cam = img_cam.resize((512, 1024), Image.ANTIALIAS)
            norm_cam = Image.fromarray(norm_cam)
            norm_cam = norm_cam.resize((512, 1024), Image.ANTIALIAS)
            stride = 4
            data = np.ones((1024, 6 * 512 + 5 * stride, 3)) * 255
            data[:, :512, :] = color_data
            data[:, stride + 512: stride + 512 * 2, :] = origin_data
            data[:, stride * 2 + 512 * 2: stride * 2 + 512 * 3, :] = pred_data
            data[:, stride * 3 + 512 * 3: stride * 3 + 512 * 4, :] = label_data
            data[:, stride * 4 + 512 * 4: stride * 4 + 512 * 5, :] = np.array(img_cam)
            data[:, stride * 5 + 512 * 5: stride * 5 + 512 * 6, :] = np.array(norm_cam)
            # data[:, stride*4 + 512 * 4: stride*4 + 512 * 5, :] = 255 - np.array(img_cam)
            img = Image.fromarray(data.astype(np.uint8))
            img.save(os.path.join(target_dir, "%02d" % j + "_" + str(prob[j][0]) + ".jpg"))
        print("finished:", i)
        return 0

    def process_data(self):
        train_names = open(os.path.join(self.train_root, "labels.txt")) \
            .read().strip("\n").split("\n")
        train_names = [os.path.join(self.train_root, "%04d" % (idx+1))
                       for idx in range(len(train_names))]
        self.train_idx = list(range(0, len(train_names)))
        train_info = pickle_load_data(os.path.join(self.train_root, "pred_feature.npy"))
        train_ref = train_info["label_cls"]
        train_pred = train_info["pred_cls"]
        train_feature = train_info["features"]
        train_cam = [None for i in range(70*128)]

        patient_names = open(os.path.join(self.patient_root, "labels.txt")) \
            .read().strip("\n").split("\n")
        patient_names = [os.path.join(self.patient_root, "%04d" % (idx+1))
                         for idx in range(len(patient_names))]
        self.patient_idx = list(range(len(train_names), len(train_names) + len(patient_names)))
        patient_info = pickle_load_data(os.path.join(self.patient_root, "pred_feature.npy"))
        patient_ref = patient_info["label_cls"]
        patient_pred = patient_info["pred_cls"]
        patient_feature = patient_info["features"]
        patient_cam = pickle_load_data(os.path.join(self.patient_root,
                                                    "saliency_folder", "cam.pkl"))

        normal_names = open(os.path.join(self.normal_root, "labels.txt")) \
            .read().strip("\n").split("\n")
        normal_names = [os.path.join(self.normal_root, "%04d" % (idx+1))
                        for idx in range(len(normal_names))]
        self.normal_idx = list(range(len(train_names) + len(patient_names),
                                 len(train_names) + len(patient_names) + len(normal_names)))
        normal_info = pickle_load_data(os.path.join(self.normal_root, "pred_feature.npy"))
        normal_ref = normal_info["label_cls"]
        normal_pred = normal_info["pred_cls"]
        normal_feature = normal_info["features"]
        normal_cam = [None for i in range(63*128)]


        def process_cam(patient_cam):
            original_patient_cam = patient_cam.copy()
            img_width, img_height = patient_cam[0].shape
            patient_cam = np.array(patient_cam).reshape(-1)
            hist, bins = np.histogram(patient_cam, 255, normed=True)
            cdf = hist.cumsum()
            cdf = cdf / cdf[-1]
            patient_cam = np.interp(patient_cam, bins[:-1], cdf)
            patient_cam = (patient_cam - patient_cam.min()) / \
                          (patient_cam.max() - patient_cam.min()) * 255
            patient_cam = patient_cam.astype(np.uint8)
            patient_cam = patient_cam.reshape(15 * 128, -1).tolist()
            for i in range(len(patient_cam)):
                cam = np.array(patient_cam[i]).reshape(img_width, img_height).astype(np.uint8)
                norm_cam = original_patient_cam[i]
                norm_cam = (norm_cam - norm_cam.min()) / (norm_cam.max() - norm_cam.min()) * 255
                norm_cam = cv2.applyColorMap(norm_cam.astype(np.uint8), color_gradient).astype(np.uint8)
                cam = cv2.applyColorMap(cam, color_gradient).astype(np.uint8)
                patient_cam[i] = [cam, norm_cam]
            return patient_cam

        patient_cam = process_cam(patient_cam)
        normal_cam = process_cam(normal_cam)

        self.all_name = train_names + patient_names + normal_names
        self.all_cam = train_cam + patient_cam + normal_cam

        # for i in range(len(train_names)):
        #     name_dir = train_names[i]
        #     for j in range(128):
        #         src = os.path.join(name_dir, "%02d"%j + ".png")
        #         gt = int(train_ref[i*128+j][0])
        #         pred_prob = train_pred[i*128+j][0]
        #         pred = int(pred_prob > 0.5)
        #         if gt == 1 and pred == 0:
        #             # FN
        #             target = os.path.join(self.train_root,
        #                                   "wrong", "FN",
        #                                   str(i+1) + "_" + str(j) + "_" + str(pred_prob) + ".png")
        #             shutil.copy(src, target)
        #         elif gt == 0 and pred == 1:
        #             # FP
        #             target = os.path.join(self.train_root,
        #                                   "wrong", "FP",
        #                                   str(i+1) + "_" + str(j) + "_" + str(pred_prob) + ".png")
        #             shutil.copy(src, target)
        #
        # exit()
        # # plt.hist(np.array(patient_cam).reshape(-1), bins= 255)
        # # plt.show()
        # # exit()
        # for i in range(len(self.all_cam)):
        #     if self.all_cam[i] is None:
        #         continue
        #     cam = self.all_cam[i]
        #     cam[cam<-4] = -4
        #     cam[cam>6] = 6
        #     cam = (cam + 4) / 10.0 * 255
        #     cam[cam >= 255] = 255
        #     cam = cam.astype(np.uint8)
        #     self.all_cam[i] = cam

        pool = Pool()
        res = []
        for i in range(70+15):
            dir_name = self.all_name[i]
            target_dir = os.path.join(self.patient_root, "saliency_folder", "%04d"%(i+1))
            prob = patient_pred[i*128: (i+1)*128]
            check_dir(target_dir)
            res.append(pool.apply_async(self.handler, (i, dir_name, target_dir, prob,)))
        for r in res:
            r.get()



if __name__ == '__main__':
    d = cam_process()
    d.process_data()