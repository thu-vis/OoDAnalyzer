import numpy as np
import os
import random
from PIL import Image

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir

root = os.path.join(config.raw_data_root, "REA")

vis_root = os.path.join(root, "vis_val")
# vis_root = os.path.join(root, "vis")

patient_list = os.listdir(vis_root)

random.shuffle(patient_list)
patient_list = patient_list[:15]

confusion_matrix = np.zeros((3, 2, 2))


for patient in patient_list:
    patient_dir = os.path.join(vis_root, patient)
    print(patient)
    for i in range(128):
        id = "%02d" % i
        img_name = os.path.join(patient_dir, id + ".png")
        img = Image.open(img_name)
        img_data = np.array(img)
        origin_data = img_data[:, :512,:]
        pred_data = img_data[:, 512:512*2, :]
        label_data = img_data[:, 512*2:512*3, :]
        # pred_residual = pred_data - origin_data
        # label_residual = label_data - origin_data
        for j in range(3):
            pred = None
            gt = None
            if (pred_data[:,:,j]==255).sum() != 0 :
                pred = 1
            else:
                pred = 0
            if (label_data[:,:,j]==255).sum() != 0:
                gt = 1
            else:
                gt = 0
            confusion_matrix[j, pred, gt] += 1
        # print("{}, {}, {}".format(id, pred, gt))
    # exit()
# print(confusion_matrix)
for j in range(3):
    c = confusion_matrix[j,:,:]
    print(c)
    c[:,0] /= c[:,0].sum()
    c[:,1] /= c[:,1].sum()
    print(c)

# all_pred = []
# for i in range(4):
#     detection_pred = np.load(os.path.join(root, "normal-" + str(i+1) + ".npy"))
#     all_pred = all_pred + detection_pred[:,0].tolist()
#
# all_pred = np.array(all_pred)
# print(1 - sum(all_pred > 0.5) / len(all_pred))