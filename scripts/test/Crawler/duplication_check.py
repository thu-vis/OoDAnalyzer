import numpy as np
import os
import cv2
from skimage.measure import compare_ssim as ssim
from multiprocessing import Pool
from PIL import Image

from scripts.utils.config_utils import config

def process_single(new_img_name, new_dir, root_img_list, root_dir):
    print("process  ")
    new_img = Image.open(os.path.join(new_dir, new_img_name))
    new_img = new_img.resize((512, 512), Image.ANTIALIAS)
    max_score = 0
    max_idx = 0
    for root_img_name in root_img_list:
        try:
            root_img = Image.open(os.path.join(root_dir, root_img_name))
        except Exception as e:
            print(e)
            data = np.array(cv2.imread(os.path.join(root_dir, root_img_name)))
            img = Image.fromarray(data)
            img.save(os.path.join(root_dir, root_img_name))
            root_img = Image.open(os.path.join(root_dir, root_img_name))
        root_img = root_img.resize((512, 512), Image.ANTIALIAS)
        try:
            ssim_score = ssim(np.array(new_img)[:, :, :3], np.array(root_img)[:, :, :3],
                              multichannel=True)
            if ssim_score > max_score:
                max_score = ssim_score
                max_idx = root_img_name
        except Exception as e:
            print(e)
            print(new_img_name, root_img_name)
            import IPython;
            IPython.embed()
    if max_score > 0.90:
        print(new_img_name, max_score, max_idx)

def comparison(new_dir, root_dir):
    root_img_list = os.listdir(root_dir)
    new_img_list = os.listdir(new_dir)
    pool = Pool()
    res = [pool.apply_async(process_single,
                            (new_img_name, new_dir, root_img_list, root_dir))
           for new_img_name in new_img_list]
    for idx, r in enumerate(res):
        r.get()

def _comparison(new_dir, root_dir):
    root_img_list = os.listdir(root_dir)
    new_img_list = os.listdir(new_dir)
    for new_img_name in new_img_list:
        new_img = Image.open(os.path.join(new_dir, new_img_name))
        new_img = new_img.resize((512, 512), Image.ANTIALIAS)
        max_score = 0
        max_idx = 0
        for root_img_name in root_img_list:
            if new_img_name == root_img_name:
                continue
            try:
                root_img = Image.open(os.path.join(root_dir, root_img_name))
            except Exception as e:
                print(e)
                data = np.array(cv2.imread(os.path.join(root_dir, root_img_name)))
                img = Image.fromarray(data)
                img.save(os.path.join(root_dir, root_img_name))
                root_img = Image.open(os.path.join(root_dir, root_img_name))
            root_img = root_img.resize((512, 512), Image.ANTIALIAS)
            try:
                ssim_score = ssim(np.array(new_img)[:,:,:3], np.array(root_img)[:,:,:3],
                              multichannel=True)
                if ssim_score > max_score:
                    max_score = ssim_score
                    max_idx = root_img_name
            except Exception as e:
                print(e)
                import IPython; IPython.embed()
        if max_score > 0.90:
            print(new_img_name, max_score, max_idx)

def img_format_checking(images_dir):
    img_name_list = os.listdir(images_dir)
    # for img_name in img_name_list:
    #     src = os.path.join(images_dir, img_name)
    #     try:
    #         Image.open(src)
    #     except Exception as e:
    #         print(e)
    #         data = np.array(cv2.imread(src))
    #         img = Image.fromarray(data)
    #         img.save(src)
    for img_name in img_name_list:
        src = os.path.join(images_dir, img_name)
        img = Image.open(src)
        img_data = np.array(img)
        try:
            if len(img_data.shape) >= 3 and img_data.shape[2] > 3:
                print("4 dim", src, img_data.shape)
                img_data = img_data[:, :, :3]
                img = Image.fromarray(img_data)
                img.save(src)
            elif len(img_data.shape) < 3:
                print("gray", src, img_data.shape)
                img_data = img_data.reshape(img_data.shape[0], img_data.shape[1], 1)
                img_data = img_data.repeat(axis=2, repeats=3)
                img = Image.fromarray(img_data)
                img.save(src)

        except:
            print(src, img_data.shape)
    exit()
if __name__ == '__main__':
    # img_format_checking(r"D:\Project\Project2019\DataBias2019\RawData\Animals\tiger")

    root = os.path.join(config.raw_data_root, config.animals)
    new_dir = os.path.join(root, "new-rabbit")
    root_dir = os.path.join(root, "rabbit")
    print(new_dir)
    comparison(new_dir, root_dir)