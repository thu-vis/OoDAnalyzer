import numpy as np
import os

from scripts.utils.config_utils import config
from scripts.database.database import DataBase

class DataTwoDim(DataBase):
    def __init__(self):
        dataname = config.two_dim
        super(DataTwoDim, self).__init__(dataname)

    def preprocessing_data(self):
        self.class_name = ["cat", "dog"]
        self.class_name_encoding = {
            self.class_name[0]: 0,
            self.class_name[1]: 1
        }
        print("generating data!!!")
        anchor_points = [[0,0],[3,0]]
        num_per_class = 500
        data = None
        y = None
        for idx, point in enumerate(anchor_points):
            noise = np.random.normal(0,0.5,(num_per_class, 2))
            local_data = np.array(point).reshape(1,-1) + noise
            local_y = np.array([idx for _ in range(num_per_class)]).reshape(-1,1)
            if data is None:
                data = local_data.copy()
                y = local_y
            else:
                data = np.concatenate((data, local_data), axis=0)
                y = np.concatenate((y, local_y), axis=0)
        # shuffle and split training set and test set
        idx = np.array(range(len(y)))
        np.random.seed(0)
        np.random.shuffle(idx)
        np.random.seed()

        X = data[idx,:]
        y = y[idx]
        split_num = int(len(y) * 0.8)
        self.train_idx = np.array(range(split_num))
        self.test_idx = np.array(range(split_num, len(y)))
        self.all_data = {
            "class_name": self.class_name,
            "class_name_encoding": self.class_name_encoding,
            "X": X,
            "y": y,
            "train_idx": self.train_idx,
            "train_redundant_idx": [],
            "valid_idx": [],
            "valid_redundant_idx": [],
            "test_idx": self.test_idx,
            "test_redundant_idx": []
        }
        self.save_cache()


    def load_data(self, loading_from_buffer=True):
        super(DataTwoDim, self).load_data(loading_from_buffer)
        self.class_name = self.all_data["class_name"]
        self.class_name_encoding = self.all_data["class_name_encoding"]
        self.X = self.all_data["X"]
        self.y = self.all_data["y"]
        self.train_idx = self.all_data["train_idx"]
        self.train_redundant_idx = self.all_data["train_redundant_idx"]
        self.valid_idx = self.all_data["valid_idx"]
        self.valid_redundant_idx = self.all_data["valid_redundant_idx"]
        self.test_idx = self.all_data["test_idx"]
        self.test_redundant_idx = self.all_data["test_redundant_idx"]

    def process_data(self):
        super(DataTwoDim, self).process_data()

    def postprocess_data(self, weight_name):
        projected_method = ["tsne", "pca", "mds"]
        defalut_method = "tsne"
        self.all_embed_X = {}
        self.embed_X = {}
        for pm in projected_method:
            embed_X = self.X.copy()
            self.all_embed_X[pm] = embed_X
            if pm == defalut_method:
                self.embed_X = embed_X

if __name__ == '__main__':
    d = DataTwoDim()
    d.preprocessing_data()
    d.load_data()
    # d.process_data()
    d.postprocess_data("none")
    d.save_file()