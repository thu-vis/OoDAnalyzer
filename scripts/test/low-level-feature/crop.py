from time import time
import numpy as np
import os
from PIL import Image
import cv2

from scripts.utils.config_utils import config

def crop_image(img):
    t = time()
    img_data = np.array(img)
    max_value = -1
    max_idx = -1
    for i in range(512):
        value = img_data[i:i + 512, :].sum()
        if value > max_value:
            max_value = value
            max_idx = i
    print(time() - t)
    return img_data[max_idx:max_idx + 512]


def crop_img_dir(src_img_dir, target_img_dir):
    img_list = os.listdir(src_img_dir)
    for img_name in img_list:
        img = Image.open(os.path.join(src_img_dir, img_name))
        img_data = crop_image(img)
        img = Image.fromarray(img_data)
        img.save(os.path.join(target_img_dir, img_name))


if __name__ == "__main__":
    src_img_dir = os.path.join(config.data_root, config.rea, "images_no_crop")
    target_img_dir = os.path.join(config.data_root, config.rea, "images")
    crop_img_dir(src_img_dir, target_img_dir)