import numpy as np
import os
import shutil

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir

root = os.path.join(config.raw_data_root, "REA")

vis_root = os.path.join(root, "vis_train")

images_filename = open(os.path.join(root, "vis_train", "val_images.txt"),"r").read()
images_filename = images_filename.strip("\n").split("\n")
print("num of image dir: {}".format(len(images_filename)))

def dir_copy(src_root, target_root):
    file_list = os.listdir(src_root)
    for filename in file_list:
        shutil.copy(os.path.join(src_root, filename),
                    os.path.join(target_root, filename))

# for n in range(3):
#     src_root = os.path.join(root, "vis_train", str(n+1))
#     for i in range(20):
#         id = "%04d" % i
#         target_name = images_filename[n * 20 + i]
#         target_name = target_name.split("/")[2]
#         target = os.path.join(vis_root, target_name)
#         check_dir(target)
#         src = os.path.join(src_root, id)
#         dir_copy(src, target)
#         # exit()
for i in range(10):
    src_root = os.path.join(root, "vis_train", "4")
    id = "%04d" % i
    target_name = images_filename[60 + i]
    target_name = target_name.split("/")[2]
    target = os.path.join(vis_root, target_name)
    check_dir(target)
    src = os.path.join(src_root, id)
    dir_copy(src, target)
