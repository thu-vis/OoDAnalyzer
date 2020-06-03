import os
import shutil

dir_list = os.listdir("./original_images")

for dir in dir_list:
    target_dir = os.path.join("label_images", dir)
    os.makedirs(target_dir)
    for i in range(128):
        id = i + 1
        arc = "../../7.bmp"
        target = os.path.join(target_dir, str(id) + ".bmp")
        shutil.copy(arc, target)


import os
label_file = open("val_labels.txt", "w")
img_file = open("val_images.txt", "w")

label_dir = "label_images"
img_dir = "trans3channel_images"

for dir in os.listdir(label_dir):
    s = os.path.join("normal", label_dir, dir)
    label_file.writelines(s + "\n")

for dir in os.listdir(img_dir):
    s = os.path.join("normal", img_dir, dir)
    img_file.writelines(s + "\n")