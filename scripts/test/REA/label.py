import numpy as np
import os
from PIL import Image

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir

root = os.path.join(config.raw_data_root, "REA")

data = np.load(os.path.join(root, "val.npy"))

gt = np.load(os.path.join(root, "val_ref.npy"))

s = 0
for i in range(3):
    p = data[:,i]
    p = (p > 0.5).astype(int)
    t = gt[:,i]
    print((p==t).sum() / float(len(p)))
    s += (p==t).sum() / float(len(p))

print(s / 3)

all_pred = []
for i in range(4):
    detection_pred = np.load(os.path.join(root, "normal-" + str(i+1) + ".npy"))
    all_pred = all_pred + detection_pred[:,0].tolist()

all_pred = np.array(all_pred)
print(1 - sum(all_pred > 0.5) / len(all_pred))



for i in range(3):
    p = all_pred
    s = 0
    total = 0
    for j in range(len(p) // 128):
        q = p[j * 128: (j+1) * 128]
        if (q>0.5).sum() > 0:
            s += 1
        total += 1
    print(1 - s / total, s, total)