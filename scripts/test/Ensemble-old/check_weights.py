import numpy as np
import os

from scripts.utils.helper_utils import pickle_load_data

root = r"H:\backup\Lipsticks\weights"

a = pickle_load_data(os.path.join(root, "weights.pkl"))
b = pickle_load_data(os.path.join(root, "weights_1.pkl"))

print((a**2).sum())
print((b**2).sum())
print(((a-b)**2).sum())

cv = 1