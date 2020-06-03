import numpy as np
import shutil
import os
import random

wrong_data = {
"P0077_MacularCube512x128_1-15-2014_8-43-12_OD_sn16128_cube_z.img": [[104, 105],[37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]],
"P1830_MacularCube512x128_3-24-2016_9-32-54_OD_sn79873_cube_z.img": [[34, 72, 73, 74], [66, 67, 68, 69, 70, 71]],
"P1830_MacularCube512x128_3-24-2016_9-33-19_OD_sn79874_cube_z.img": [[32, 33, 34, 72, 73], [65, 66, 67, 68, 69, 70, 71, 72]],
"P1830_MacularCube512x128_4-26-2016_14-19-10_OD_sn82890_cube_z.img": [[36,],[62, 63, 64, 65, 66, 67, 68, 69, 70]],
"P1830_MacularCube512x128_5-26-2016_14-26-14_OD_sn86211_cube_z.img": [[36, 37], [62, 63, 64, 65, 66, 67, 68, 69, 70]],
"P1830_MacularCube512x128_6-30-2016_9-22-9_OD_sn89751_cube_z.img": [[36, 37],[65, 66, 67, 68, 69, 70, 71]],
"P8066_MacularCube512x128_11-24-2015_8-24-13_OS_sn69774_cube_z.img": [[65, 66, 67, 78, 79, 80], [68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84]],
"P8066_MacularCube512x128_12-1-2015_9-25-7_OS_sn70487_cube_z.img": [[60, 61,74, 75],[62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]],
"P8317_MacularCube512x128_1-9-2017_14-33-47_OS_sn108878_cube_z.img": [[57, 73, 74, 75],[60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]],
"PC007_MacularCube512x128_7-15-2014_12-25-40_OS_sn26577_cube_z.img": [[45, 46, 47, 48, 81, 82, 83, 84, 85, 86, 87],[49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]],
"PC009_MacularCube512x128_9-17-2013_10-30-4_OD_sn9821_cube_z.img": [[63, 64, 65, 66, 67, 68, 69, 70, 83, 84],[85, 86, 87, 88]],
"PC010_MacularCube512x128_4-10-2014_9-54-34_OS_sn20125_cube_z.img": [[62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],[91, 92, 93, 94, 95, 96, 97]],
"PC010_MacularCube512x128_9-3-2013_8-18-29_OS_sn9009_cube_z.img": [[26, 54, 64, 108, 109],[27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107]],
"PC010_MacularCube512x128_10-29-2013_9-43-12_OS_sn11798_cube_z.img": [[61, 62, 63, 64, 65, 66, 67, 68, 69, 70],[22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]],
"PC010_MacularCube512x128_12-10-2013_8-48-33_OS_sn14189_cube_z.img": [[63, 64, 65, 66, 67, 68, 69, 70, 71],[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]],
}

root = r"H:\REA\train_data"

for patient_name in wrong_data:
    wrong = wrong_data[patient_name][0]
    correct = wrong_data[patient_name][1]
    img_dir = os.path.join(root, "trans3channel_images", patient_name)
    label_dir = os.path.join(root, "label_images", patient_name.replace(".img", "_labelMark"))
    for wi in wrong:
        ci = random.choice(correct)
        img_src = os.path.join(img_dir, "%03d.png"%ci)
        print(img_src)
        img_target = os.path.join(img_dir, "%03d.png"%wi)
        print(img_target)
        label_src = os.path.join(label_dir, str(ci+1) + ".bmp")
        label_target = os.path.join(label_dir,  str(wi+1) + ".bmp")
        shutil.copy(img_src, img_target)
        shutil.copy(label_src, label_target)
        exit()
