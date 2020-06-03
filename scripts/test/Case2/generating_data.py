import numpy as np
import os

from PIL import Image
import shutil
from skimage import transform
from skimage.util import img_as_ubyte

from scripts.utils.helper_utils import check_dir

root = r"D:\Project\Project2019\DataBias2019\Project\data\REA\adding_data"

def get_edema_name_list():
    label_filename = r"H:\REA\train_data\val_labels.txt"
    image_filename = r"H:\REA\train_data\val_images.txt"
    label_list = open(label_filename, "r").read().strip("\n").split("\n")
    train_label_list = [os.path.join(r"H:\REA\train_data\label_images", s.split("/")[2]) for s in label_list]
    img_list = open(image_filename, "r").read().strip("\n").split("\n")
    train_img_list = [os.path.join(r"H:\REA\train_data\trans3channel_images", s.split("/")[2]) for s in img_list]


    label_filename = r"H:\REA\val\labels.txt"
    image_filename = r"H:\REA\val\images.txt"
    label_list = open(label_filename, "r").read().strip("\n").split("\n")
    val_label_list = [os.path.join(r"H:\REA\val_data\label_images", s.split("/")[2]) for s in label_list]
    img_list = open(image_filename, "r").read().strip("\n").split("\n")
    val_img_list = [os.path.join(r"H:\REA\val_data\trans3channel_images", s.split("/")[2]) for s in img_list]

    label_filename = r"H:\REA\test\test_labels.txt"
    image_filename = r"H:\REA\test\test_images.txt"
    label_list = open(label_filename, "r").read().strip("\n").split("\n")
    test_label_list = [os.path.join(r"H:\REA\test\label_images", s.split("/")[2]) for s in label_list]
    img_list = open(image_filename, "r").read().strip("\n").split("\n")
    test_img_list = [os.path.join(r"H:\REA\test\trans3channel_images", s.split("/")[2]) for s in img_list]


    return train_img_list + val_img_list + test_img_list, \
           train_label_list + val_label_list + test_label_list

def data_augmentation(img, label, type=0):
    if type == 0:
        degree = int(np.random.rand()*10)
        img = transform.rotate(np.array(img), degree)
        img = img_as_ubyte(img)
        label = transform.rotate(np.array(label), degree)
        label = img_as_ubyte(label)
        mask0 = label >= 0
        mask1 = label > 118
        mask2 = label > 181
        mask3 = label > 245
        label[mask0] = 0
        label[mask1] = 128
        label[mask2] = 191
        label[mask3] = 255
    elif type == 1:
        scale = np.random.rand() * 0.1 + 1.0
        _img = transform.rescale(np.array(img), scale)
        img = img_as_ubyte(_img)[:1024,:512,:]
        _label = transform.rescale(np.array(label), scale)
        label = img_as_ubyte(_label)[:1024, :512]
        mask0 = label >= 0
        mask1 = label > 118
        mask2 = label > 181
        mask3 = label > 245
        label[mask0] = 0
        label[mask1] = 128
        label[mask2] = 191
        label[mask3] = 255
    elif type == 2:
        img = np.array(img)
        img = img[:,::-1,:]
        label = np.array(label)
        label = label[:, ::-1]
    else:
        raise ValueError("type {} is not supported currently".format(type))

    array_np = np.unique(label)
    if len(array_np)> 5:
        print(1111)
        exit()

    img = Image.fromarray(img)
    label = Image.fromarray(label)
    return img, label

def process(dir_name, adding_num = 4):
    global root
    root = os.path.join(root, dir_name)
    # read data
    # list_name = [3,4,5,6,7,14,23,71,73,74]
    list_name = os.listdir(os.path.join(root, "images"))
    print(list_name)
    # list_name = [str(s) for s in list_name]

    img_list = []
    label_list = []
    for l in list_name:
        sub_dir = os.path.join(root, "images", l)
        img_list = img_list + [os.path.join(l, s) for s in os.listdir(sub_dir)]
        sub_dir = os.path.join(root, "labels", l)
        label_list = label_list + [os.path.join(l, s) for s in os.listdir(sub_dir)]

    print(img_list, label_list)
    global_id = 0
    name_list = [dir_name + str(i) for i in range(adding_num)]
    for l in name_list:
        check_dir(os.path.join(root, "images", l))
        check_dir(os.path.join(root, "labels", l))
        for id in range(128):
            img_name = img_list[global_id]
            label_name = label_list[global_id]
            global_id = global_id + 1
            global_id = global_id % len(img_list)
            img_src = os.path.join(root, "images", img_name)
            label_src = os.path.join(root, "labels", label_name)
            img_target = os.path.join(root, "images", l, "%03d"%id +".png")
            label_target = os.path.join(root, "labels", l, str(id+1) + ".bmp")
            print(img_target, label_target)
            img = Image.open(img_src)
            label = Image.open(label_src)
            img, label = data_augmentation(img, label, int(np.random.rand()*3))
            img.save(img_target)
            label.save(label_target)

            # no data
            # shutil.copy(img_src, img_target)
            # shutil.copy(label_src, label_target)

def preprocess_dir(dir_name):
    img_patient_name_list, label_patient_name_list = get_edema_name_list()
    dir = os.path.join(root, dir_name)
    img_dir = os.path.join(dir, "images")
    label_dir = os.path.join(dir, "labels")
    sub_dir_list = os.listdir(img_dir)
    for sub_dir in sub_dir_list:
        patient_id = int(sub_dir) - 1
        img_sub_dir = os.path.join(img_dir, sub_dir)
        label_sub_dir = os.path.join(label_dir, sub_dir)
        img_name_list = os.listdir(img_sub_dir)
        for img_name in img_name_list:
            id = int(img_name.split(".")[0])
            label_name = str(id+1) + ".bmp"
            src = os.path.join(label_patient_name_list[patient_id], label_name)
            target = os.path.join(label_sub_dir, label_name)
            shutil.copy(src, target)

def generating_dir(id_list, dir_name):
    img_patient_name_list, label_patient_name_list = get_edema_name_list()
    dir = os.path.join(root, dir_name)
    check_dir(dir)
    img_dir = os.path.join(dir, "images")
    check_dir(img_dir)
    label_dir = os.path.join(dir, "labels")
    check_dir(label_dir)
    for global_id in id_list:
        patient_id = global_id // 128
        img_sub_dir = os.path.join(img_dir, str(patient_id))
        check_dir(img_sub_dir)
        label_sub_dir = os.path.join(label_dir, str(patient_id))
        check_dir(label_sub_dir)
        id = global_id % 128
        img_name = "%03d.png" % id
        label_name = str(id+1) + ".bmp"
        img_src = os.path.join(img_patient_name_list[patient_id], img_name)
        label_src = os.path.join(label_patient_name_list[patient_id], label_name)
        img_target = os.path.join(img_sub_dir, img_name)
        label_target = os.path.join(label_sub_dir, label_name)
        shutil.copy(img_src, img_target)
        shutil.copy(label_src, label_target)

if __name__ == '__main__':
    # get_edema_name_list()
    process("fovea", 6)
    # process("step1-correct")
    # a = [497, 499, 501, 597, 607, 610, 623, 626, 634, 635, 637, 638, 642, 643, 645, 728,
    #     730, 734, 738, 739, 740, 744, 747, 750, 754, 655, 762, 763, 878, 879, 880, 897,
    #     898, 899, 905, 911, 916, 920, 924, 1795, 1796, 1816, 1817, 1818, 2955, 2970, 2973]
    # generating_dir(a + [9376, 9350, 9485, 9380, 9170, 9359, 9399, 9356, 9378, 9168,
    #                 9481, 9496, 9348, 9349, 9352,9479], "HRF")

#     a = [9782, 9783, 9784, 9785, 9786, 9787, 9788, 9789, 9790, 9791, 9792, 9793, 9794, 9795,
# 9796, 9797, 9798, 9799, 9800, 9801,
# 10686, 10687, 10688, 10689, 10690, 10691, 10692, 10693, 10694, 10695, 10696,
# 10697, 10698, 10699, 10700, 10701, 10702, 10703, 10704,
# 10814, 10815, 10816, 10817, 10818, 10819, 10820, 10821, 10822, 10823, 10824,
# 10825, 10826, 10827, 10828, 10829,
# 9028, 9029, 9030, 9031, 9032, 9033, 9034, 9035, 9036, 9037, 9038,
# 9019, 9020, 9021, 9022, 9023, 9024, 9025, 9026, 9027,]
#     b = [11445, 11446, 11447, 11448, 11449, 11450, 11451,
#          11453, 11454, 11455, 11456, 11457, 11458, 11459, 11460, 11461, 11462, 11463, 11464, 11465,
#     12084, 12085, 12086, 12087, 12088, 12089, 12090, 12091, 12092, 12093, 12094,
#     12095, 12096, 12097, 12098, 12099, 12100, 12101,
#     12342, 12343, 12344, 12345, 12346, 12347, 12348, 12349, 12350, 12351, 12352,
#     12353, 12354, 12355, 12356, 12357, 12358,]
#
#     generating_dir(a+b, "fovea")