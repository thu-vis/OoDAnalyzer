import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class test(Dataset):
    def __init__(self):
        self.len = 1000

    def __getitem__(self, item):
        return item

    def __len__(self):
        return self.len


t = test()
print(len(t))

loader = DataLoader(t, batch_size=8, shuffle=True)

for i in range(5):

    for j, idx in enumerate(loader):
        if j > 20:
            break
        print(j, idx, end=", ")
    print(" ")