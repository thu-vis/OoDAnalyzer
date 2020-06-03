import numpy as np
import os
import random
from PIL import Image
import shutil
from multiprocessing import Pool
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import *


class all_data(object):
    def __init__(self):
        self.all_name = []
        self.all_seg_confusion_matrix = []
        self.all_seg = []
        self.all_gt = []
        self.prediction = []
        self.normal_idx = []
        self.patient_idx = []
        self.train_idx = []
        # self.root = os.path.join(config.raw_data_root, "REA")
        self.root = os.path.join(r"H:\REA")
        self.normal_root = os.path.join(self.root, "normal")
        self.patient_root = os.path.join(self.root, "val")
        # self.train_root = os.path.join(self.root, "train")
        self.train_root = os.path.join(self.root, "bias_train_result", "train")


    def _process_data(self):
        normal_names = open(os.path.join(self.normal_root, "labels.txt")) \
            .read().strip("\n").split("\n")
        normal_names = [name.split("/")[2] for name in normal_names]
        self.all_name = normal_names
        self.normal_idx = np.array(range(len(normal_names))).tolist()
        all_pred = []
        for i in range(4):
            detection_pred = np.load(os.path.join(self.root, "normal-" + str(i+1) + ".npy"))
            all_pred = all_pred + detection_pred.tolist()
        self.prediction = all_pred

        patient_pred = np.load(os.path.join(self.root, "val.npy"))
        self.prediction = self.prediction + patient_pred.tolist()
        for i in range(15):
            self.all_name.append("%04d" % i)
            self.patient_idx.append(len(normal_names) + i)


        train_names = open(os.path.join(self.train_root, "val_images.txt")) \
            .read().strip("\n").split("\n")
        train_names = [name.split("/")[2] for name in train_names]
        self.all_name = self.all_name + train_names
        self.train_idx = (np.array(range(len(train_names))) +
                          len(self.normal_idx) + len(self.patient_idx)).tolist()
        all_pred = []
        for i in range(4):
            detection_pred = np.load(os.path.join(self.train_root, str(i+1),  "pre.npy"))
            all_pred = all_pred + detection_pred.tolist()
        self.prediction = self.prediction + all_pred

        for idx, filename in enumerate(self.all_name):
            print(idx, ": ", filename)
            if idx in self.normal_idx:
                patient_dir = os.path.join(self.normal_root, filename)
            elif idx in self.train_idx:
                patient_dir = os.path.join(self.train_root, filename)
            else:
                patient_dir = os.path.join(self.patient_root, filename)
            gt_three = []
            confusion_matrix = np.zeros((3, 2, 2))
            for i in range(128):
                if i == 22:
                    a = 1
                id = "%02d" % i
                img_name = os.path.join(patient_dir, id + ".png")
                img = Image.open(img_name)
                img_data = np.array(img)
                origin_data = img_data[:, :512, :]
                pred_data = img_data[:, 512:512 * 2, :]
                label_data = img_data[:, 512 * 2:512 * 3, :]
                # pred_residual = pred_data - origin_data
                # label_residual = label_data - origin_data
                tmp_gt = [0,0,0]
                for j in range(3):
                    pred = None
                    gt = None
                    if (pred_data[:, :, j] == 255).sum() != 0:
                        pred = 1
                    else:
                        pred = 0
                    if (label_data[:, :, j] == 255).sum() != 0:
                        gt = 1
                    else:
                        gt = 0
                    confusion_matrix[2-j, pred, gt] += 1
                    tmp_gt[2-j] = gt
                gt_three.append(tmp_gt)
            self.all_seg_confusion_matrix.append(confusion_matrix)
            self.all_gt.append(gt_three)

    def process_single_people(self, idx):
        # if idx < 70:
        #     return None, None, None
        print("processing: ", idx)
        people_dir = self.all_name[idx]
        gt_three = []
        seg_three = []
        confusion_matrix = np.zeros((3, 2, 2))
        for i in range(128):
            id = "%02d" % i
            img_name = os.path.join(people_dir, id + ".png")
            img = Image.open(img_name)
            img_data = np.array(img)
            origin_data = img_data[:, :512, :]
            pred_data = img_data[:, 512:512 * 2, :]
            label_data = img_data[:, 512 * 2:512 * 3, :]
            # pred_residual = pred_data - origin_data
            # label_residual = label_data - origin_data
            tmp_gt = [0, 0, 0]
            tmp_seg = [0, 0, 0]
            for j in range(3):
                pred = (pred_data[:,:,j]==255).sum()
                gt = (label_data[:,:, j]==255).sum()

                confusion_matrix[2 - j, int(pred>0), int(gt>0)] += 1
                tmp_gt[2 - j] = gt
                tmp_seg[2 - j] = pred
            gt_three.append(tmp_gt)
            seg_three.append(tmp_seg)
        print("processing: ", idx, "finished")
        return confusion_matrix, gt_three, seg_three

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

        patient_names = open(os.path.join(self.patient_root, "labels.txt")) \
            .read().strip("\n").split("\n")
        patient_names = [os.path.join(self.patient_root, "%04d" % (idx+1))
                         for idx in range(len(patient_names))]
        self.patient_idx = list(range(len(train_names), len(train_names) + len(patient_names)))
        patient_info = pickle_load_data(os.path.join(self.patient_root, "pred_feature.npy"))
        patient_ref = patient_info["label_cls"]
        patient_pred = patient_info["pred_cls"]
        patient_feature = patient_info["features"]

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

        self.all_name = train_names + patient_names + normal_names
        self.all_seg_confusion_matrix = [0 for i in range(len(self.all_name))]
        self.all_gt = [0 for i in range(len(self.all_name))]
        self.all_seg = [0 for i in range(len(self.all_name))]
        self.prediction = np.vstack((train_pred, patient_pred, normal_pred))
        self.all_features = np.vstack((train_feature, patient_feature, normal_feature))

        pool = Pool()
        # for idx in range(len(self.all_name)):
        #     pool.apply_async(self.process_single_people, args=(idx,))
        # pool.map(self.process_single_people, range(len(self.all_name)))
        res = [pool.apply_async(self.process_single_people,
                                (idx,)) \
               for idx in range(len(self.all_name))]
        for idx, r in enumerate(res):
            self.all_seg_confusion_matrix[idx], \
            self.all_gt[idx], self.all_seg[idx] = r.get()

        # pool.close()
        # pool.join()



        # mat = {
        #     "all_name": self.all_name,
        #     "all_seg_confusion_matrix": self.all_seg_confusion_matrix,
        #     "all_gt": self.all_gt,
        #     "idx": [self.train_idx, self.patient_idx, self.normal_idx]
        # }
        # np.save("processed_data.npy", mat)

    def get_patient_info(self):
        confusion_matrix = np.zeros((3, 2, 2))
        gt = []
        seg = []
        pred = []
        pred_prob = []
        for idx in self.patient_idx:
            confusion_matrix = confusion_matrix + self.all_seg_confusion_matrix[idx]
            gt.append(self.all_gt[idx])
            seg.append(self.all_seg[idx])
            pred.append((np.array(self.prediction[idx * 128 : idx * 128 +128])>0.5).astype(int))
            pred_prob.append(np.array(self.prediction[idx * 128 : idx * 128 +128]))
            # pred.append((np.array(self.prediction[idx * 128 : idx * 128 +128])))


        for j in range(3):
            c = confusion_matrix[j, :, :]
            print(c)
            print("acc:", (c[0,0] + c[1,1]) / c.sum())
            c[:, 0] /= c[:, 0].sum()
            c[:, 1] /= c[:, 1].sum()
            print(c)
        # gt = ref
        gt = np.array(gt).reshape(-1)
        seg = np.array(seg).reshape(-1)
        pred = np.array(pred).reshape(-1)
        pred_prob = np.array(pred_prob).reshape(-1)
        res = gt == pred
        res = res.reshape(-1, 3).sum(axis=0) / len(self.patient_idx) / 128.0
        print(res)

        gt = gt.reshape(-1,3)
        seg = seg.reshape(-1,3)
        pred = pred.reshape(-1,3)
        pred_prob = pred_prob.reshape(-1, 3)
        for j in range(3):
            print(sk_confusion_matrix(gt[:,j], pred[:,j]))
            match_count = 0
            for i in range(len(self.patient_idx)):
                data_y = gt[i * 128 : (i+1) * 128,j]
                data_pred_y = pred[i * 128 : (i+1) * 128,j]
                match_count += (data_pred_y.sum()>0) == (data_y.sum()>0)
                # match_count += (data_y.sum()>0)
                # match_count += (data_pred_y.sum()>0)
            print(match_count / len(self.patient_idx))

        # for idx, i in enumerate(self.patient_idx):
        #     for j in range(128):
        #         if gt[idx*128 + j,0] != pred[idx*128 + j, 0]:
        #             if gt[idx*128+j,0] == 0:
        #                 target_subfolder_name = "FP"
        #             else:
        #                 target_subfolder_name = "FN"
        #             src = os.path.join(self.patient_root, self.all_name[i], ("%02d" % j) + ".png")
        #             target = os.path.join(self.root, "val_rea_wrong", target_subfolder_name,
        #                                   self.all_name[i] + "_" +("%02d" % j) + "_" + str(pred_prob[idx*128+j, 0])[:5] +".png")
        #             print(src, target)
        #             shutil.copy(src, target)

    def get_normal_info(self):
        normal_gt, normal_seg, normal_pred, normal_pred_prob, normal_feature = \
            self.get_info(list(range(70+15, 70+15+63)))

        for i in range(63):
            gt = normal_gt[i*128: (i+1)*128]
            seg = normal_seg[i*128: (i+1)*128]
            for j in range(128):
                if int(gt[j][0]>0) != int(seg[j][0]>0) and seg[j][0] > 10000:
                    dir_name = self.all_name[70 + 15 +i]
                    src = os.path.join(dir_name, "%02d"%j + ".png")
                    target = os.path.join(self.normal_root, "wrong" ,str(i+1) + "_" + str(j) + ".jpg")
                    shutil.copy(src, target)
                    # exit()

    def get_info(self, idxs):
        confusion_matrix = np.zeros((3, 2, 2))
        gt = []
        seg = []
        pred = []
        pred_prob = []
        feature = []
        for idx in idxs:
            confusion_matrix = confusion_matrix + self.all_seg_confusion_matrix[idx]
            gt.append(self.all_gt[idx])
            seg.append(self.all_seg[idx])
            pred.append((np.array(self.prediction[idx * 128: idx * 128 + 128]) > 0.5).astype(int))
            pred_prob.append(np.array(self.prediction[idx * 128: idx * 128 + 128]))
            feature.append(self.all_features[idx * 128: idx * 128 + 128])

        for j in range(3):
            c = confusion_matrix[j, :, :]
            print(c)
            c[:, 0] /= c[:, 0].sum()
            c[:, 1] /= c[:, 1].sum()
            print(c)

        gt = np.array(gt).reshape(-1)
        seg = np.array(seg).reshape(-1, 3)
        pred = np.array(pred).reshape(-1)
        pred_prob = np.array(pred_prob).reshape(-1)
        feature = np.array(feature)
        res = (gt>0).astype(int) == pred
        res = res.reshape(-1, 3).sum(axis=0) / len(idxs) / 128.0
        print("classification acc (slice level): ", res)

        print("classification acc (patient level)")
        gt = gt.reshape(-1,3)
        pred = pred.reshape(-1,3)
        pred_prob = pred_prob.reshape(-1, 3)
        feature = feature.reshape(-1, feature.shape[2])
        for j in range(3):
            match_count = 0
            for i in range(len(idxs)):
                data_y = gt[i * 128 : (i+1) * 128,j]
                data_pred_y = pred[i * 128 : (i+1) * 128,j]
                match_count += (data_pred_y.sum()>0) == (data_y.sum()>0)
                # match_count += (data_y.sum()>0)
                # match_count += (data_pred_y.sum()>0)
            print(match_count / len(idxs))

        print("AUC:")
        ret = [0.5, 0.5, 0.5]
        for j in range(3):
            fpr, tpr, thresholds = roc_curve((gt>0).astype(int)[:,j], pred_prob[:,j],pos_label=1)
            ret[j] = auc(fpr, tpr)
            print(ret[j])
        print(sum(ret) / 3.0)

        return gt, seg, pred, pred_prob, feature

    def save_embed(self, method="patient"):
        if method == "patient":
            idxs = list(range(70+15))
        else:
            idxs = list(range(70+15+63))

        gt, _, pred, pred_prob, feature = self.get_info(idxs)
        tsne = TSNE()
        embed_X = tsne.fit_transform(feature)
        pickle_save_data(os.path.join(self.root, method + "_embed.pkl"),embed_X)
        # ax = plt.subplot(111)
        # color_map = plt.get_cmap("tab10")(pred[:,0])
        # ax.scatter(embed_X[:, 0], embed_X[:, 1], c=color_map)
        # plt.show()

    def plot(self, method="patient"):
        train_gt, _, train_pred, train_pred_prob, train_feature = \
            self.get_info(list(range(70)))
        patient_gt, _, patient_pred, patient_pred_prob, patient_feature = \
            self.get_info(list(range(70, 70+15)))
        train_gt = (train_gt > 0).astype(int)
        patient_gt = (patient_gt > 0).astype(int)
        self.root = r"H:\REA"
        all_embed = pickle_load_data(os.path.join(self.root, method+"_embed.pkl"))
        train_embed = all_embed[:70*128]
        patient_embed = all_embed[70*128:(70+15)*128]

        def on_press(event):
            print("x,y:", event.xdata, event.ydata)
            try:
                title = event.inaxes.title._text
            except:
                print("out of axs")
                # import IPython; IPython.embed()
                return
            if title[:4] == "trai":
                distance = all_embed - np.array([event.xdata, event.ydata])
                distance = (distance**2).sum(axis=1)
                distance[np.array(range(70*128, (70+15)*128))] = 100000
            else:
                distance = all_embed - np.array([event.xdata, event.ydata])
                distance = (distance**2).sum(axis=1)
                distance[np.array(range(70*128))] = 100000

            arg_min = distance.argsort()[:5]
            for i in arg_min:
                print(self.all_name[i//128], "_", i % 128)
            if 1:
                img = Image.open(self.all_name[arg_min[0] // 128])

        color = ["#1f77b4", "#ff7f0e"]
        label = ["negative", "positive"]
        fig = plt.figure()
        ax = plt.subplot(221)
        for i in range(train_pred.max()+1):
            sub_idx = (train_pred[:,0]==i)
            ax.scatter(train_embed[sub_idx, 0], train_embed[sub_idx, 1],
                       c=color[i], label=label[i])
        ax.set_title("train_prediction")
        plt.xlim((-100,100))
        plt.ylim((-110,100))
        plt.legend()
        fig.canvas.mpl_connect("button_press_event", on_press)

        ax = plt.subplot(222)
        for i in range(train_gt.max()+1):
            sub_idx = (train_gt[:,0]==i)
            ax.scatter(train_embed[sub_idx, 0], train_embed[sub_idx, 1],
                       c=color[i], label=label[i])
        ax.set_title("train_ground_truth")
        plt.xlim((-100,100))
        plt.ylim((-110,100))

        ax = plt.subplot(223)
        for i in range(patient_pred.max()+1):
            sub_idx = (patient_pred[:,0]==i)
            ax.scatter(patient_embed[sub_idx, 0], patient_embed[sub_idx, 1],
                       c=color[i], label=label[i])
        ax.set_title("test_prediction")
        plt.xlim((-100,100))
        plt.ylim((-110,100))

        ax = plt.subplot(224)
        for i in range(patient_gt.max()+1):
            sub_idx = (patient_gt[:,0]==i)
            ax.scatter(patient_embed[sub_idx, 0], patient_embed[sub_idx, 1],
                       c=color[i], label=label[i])
        ax.set_title("test_ground_truth")
        plt.xlim((-100,100))
        plt.ylim((-110,100))
        plt.show()



if __name__ == '__main__':
    d = all_data()
    # d.process_data()
    # pickle_save_data(os.path.join(config.raw_data_root, "REA", "all_data.pkl"), d)


    d = pickle_load_data(os.path.join(d.root, "all_data.pkl"))
    #
    train_gt, _, train_pred, train_pred_prob, train_feature = \
        d.get_info(list(range(70)))
    patient_gt, _, patient_pred, patient_pred_prob, patient_feature = \
        d.get_info(list(range(70, 70 + 15)))
    train_gt = (train_gt > 0).astype(int)
    patient_gt = (patient_gt > 0).astype(int)
    y = train_gt[:,0].tolist() + patient_gt[:,0].tolist() + patient_gt[:,0].tolist()
    pickle_save_data(os.path.join(config.data_root, config.rea, "gt.pkl"), y)
    exit()

    d = pickle_load_data(os.path.join(d.root, "all_data.pkl"))
    d.plot()
    #
    # d.get_patient_info()
    # d.get_normal_info()
    # d.get_info(list(range(70)))
    # d.get_info(list(range(70, 70 + 15)))
    # print("normal");d.get_info(list(range(70, 70 + 15 + 63)))
    # d.get_info(list(range(78,78+70)))
    # d.save_embed(method="all")
    # d.get_normal_info()