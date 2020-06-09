import numpy as np
import os
from time import time

from scripts.Sampler import Sampler
from scripts.utils.data_utils import Data
from scripts.utils.config_utils import config


class ExchangePort(object):
    def __init__(self, dataname=None):
        self.dataname = dataname
        if dataname is None:
            self.data = None
        else:
            self.data = Data(dataname)
            self.sampler = Sampler(dataname)

    def reset_dataname(self, dataname):
        self.dataname = dataname
        t = time()
        self.data = Data(dataname)
        print("data time:", time() - t)
        self.sampler = Sampler(dataname)
        print("sampler time:", time() - t)

    def get_manifest(self):
        return self.data.get_manifest()

    def get_embed_data(self, embed_method="tsne"):
        embed_X_train, embed_X_valid, embed_X_test = self.data.get_embed_X("all", embed_method)
        mat = {
            "embed_X_train": embed_X_train.tolist(),
            "embed_X_valid": embed_X_valid.tolist(),
            "embed_X_test": embed_X_test.tolist()
        }
        return mat

    def get_data(self):
        # TODO: BUG
        X_train, y_train, X_valid, y_valid, X_test, y_test = self.data.get_data("all")
        embed_X_train, embed_X_valid, embed_X_test = self.data.get_embed_X("all")
        mat = {
            "X_train": X_train.tolist(),
            "y_train": y_train.tolist(),
            "X_valid": X_valid.tolist(),
            "y_valid": y_valid.tolist(),
            "X_test": X_test.tolist(),
            "y_test": y_test.tolist(),
            "embed_X_train": embed_X_train.tolist(),
            "embed_X_valid": embed_X_valid.tolist(),
            "embed_X_test": embed_X_test.tolist()
        }
        return mat

    def get_idx(self):
        mat = {
            "train_idx": self.data.train_idx,
            "valid_idx": [], # TODO
            "test_idx": self.data.test_idx
        }
        return mat

    def get_original_samples(self):
        train_data = self.sampler.get_sampler("tsne", "train", 0.0, 0.0, 1, 1)
        test_data = self.sampler.get_sampler("tsne", "test", 0.0, 0.0, 1, 1)
        all_data = self.sampler.get_sampler("tsne", "all", 0.0, 0.0, 1, 1)
        # train_samples = []
        # test_samples = []
        # all_samples = []
        # for id in train_data["layout"]:
        #     train_samples.append(id["id"])
        # for id in test_data["layout"]:
        #     test_samples.append(id["id"])
        # for id in all_data["layout"]:
        #     all_samples.append(id["id"])
        # mat = {
        #     "train": train_samples,
        #     "test": test_samples,
        #     "all": all_samples
        # }
        mat = {
            "train": train_data["layout"],
            "test": test_data["layout"],
            "all": all_data["layout"]
        }
        return mat

    def get_feature(self):
        X_train, y_train, X_valid, y_valid, X_test, y_test = self.data.get_data("all")
        mat = {
            "X_train": X_train.tolist(),
            "X_valid": X_valid.tolist(),
            "X_test": X_test.tolist()
        }
        return mat

    def get_label(self):
        X_train, y_train, X_valid, y_valid, X_test, y_test = self.data.get_data("all")
        pred_train, pred_valid, pred_test = self.data.get_prediction()
        mat = {
            "y_train": y_train.tolist(),
            "y_valid": y_valid.tolist(),
            "y_test": y_test.tolist(),
            "pred_train": pred_train.tolist(),
            "pred_valid": pred_valid.tolist(),
            "pred_test": pred_test.tolist()
        }
        return mat

    def get_image_path(self, id):
        dataname = self.dataname.split("-")[0]
        img_dir = os.path.join(config.data_root, dataname, "images")
        return os.path.join(img_dir, str(id) + ".jpg")

    def get_thumbnail_path(self, id):
        dataname = self.dataname.split("-")[0]
        img_dir = os.path.join(config.data_root, dataname, "thumbnail")
        img_path = os.path.join(img_dir, str(id) + ".jpg")
        if not os.path.exists(img_path):
            img_path = self.get_image_path(id)
        return img_path

    def get_grid_layout(self, embed_method="tsne"):
        print("this method is disabled.")
        # grid_X_train, grid_X_test, train_row_asses, test_row_asses, train_min_width, test_min_width = self.grid_layout.get_grid_layout_native_lap_knn(embed_method)
        # mat = {
        #     config.grid_X_train_name: grid_X_train.tolist(),
        #     config.grid_X_test_name: grid_X_test.tolist(),
        #     "train_grid_attachments": [int(idx) for idx in train_row_asses],
        #     "test_grid_attachments": [int(idx) for idx in test_row_asses],
        #     "train_width": train_min_width.tolist(),
        #     "test_width": test_min_width.tolist()
        # }
        # return mat


    def get_decision_boundary(self, data_type):
        decision_boundary = self.grid_layout.get_decision_boundary(data_type)
        mat = {
            "boundary": decision_boundary
        }
        return mat

    def get_prediction(self):
        # train_pred_y, test_pred_y = self.grid_layout.get_prediction_results()
        train_pred_y = self.data.pred_train
        test_pred_y = self.data.pred_test
        mat = {
            "train_pred_y": train_pred_y.tolist(),
            "test_pred_y": test_pred_y.tolist()
        }
        return mat

    def get_entropy(self):
        train_entropy, test_entropy = self.data.get_entropy()
        train_entropy = train_entropy * 0
        mat = {
            "train_entropy": train_entropy.tolist(),
            "test_entropy": test_entropy.tolist()
        }
        return mat

    def get_confidence(self):
        # train_confidence, test_confidence = self.grid_layout.get_predict_probability()
        train_confidence, test_confidence = self.data.get_confidence()
        mat = {
            "train_confidence": train_confidence.tolist(),
            "test_confidence": test_confidence.tolist()
        }
        return mat

    def get_focus(self, id, k):
        focus_instance_list = self.data.get_similar(id, k)
        mat = {
            "similar_instances": focus_instance_list
        }
        return mat

    def get_saliency_map_path(self, id):
        dataname = self.dataname.split("-")[0]
        saliency_map_dir = os.path.join(config.data_root, dataname, "saliency-map")
        return os.path.join(saliency_map_dir, str(id) + ".jpg")

    def get_individual_info(self, id):
        ent = self.data.entropy[id]
        gt = float(self.data.y[id])
        print(ent, gt)
        info = {
            "id": id,
            "ent": ent,
            "gt": gt
        }
        return info

    def get_grid_layout_of_sampled_instances(self, embed_method, datatype, left_x,
                                             top_y, width, height, class_selection,
                                             node_id):
        return self.sampler.get_sampler_and_set_class(embed_method, datatype,
                                                      left_x, top_y, width, height,
                                                      class_selection, node_id)

    def get_grid_layout_query(self, embed_method, datatype, left_x, top_y, range_size):
        return self.sampler.get_query(embed_method, datatype, left_x, top_y, range_size)

    def get_decision_boundary_of_sampled_instances(self):
        mat = {
            "boundary": self.sampler.get_boundary()
        }
        return mat

    def change_class(self):
        return None

exchange_port = ExchangePort()


# functions
def set_dataname(dataname):
    exchange_port.reset_dataname(dataname)


def get_manifest():
    return exchange_port.get_manifest()


def get_embed_data(embed_method="tsne"):
    return exchange_port.get_embed_data(embed_method)


def get_idx():
    return exchange_port.get_idx()

def get_original_samples():
    return exchange_port.get_original_samples()

def get_feature():
    return exchange_port.get_feature()

def get_label():
    return exchange_port.get_label()

def get_image_path(id):
    return exchange_port.get_image_path(id)

def get_thumbnail_path(id):
    return exchange_port.get_thumbnail_path(id)

def get_grid_layout(embed_method="tsne"):
    return exchange_port.get_grid_layout(embed_method)


def get_decision_boundary(data_type):
    return exchange_port.get_decision_boundary(data_type)

def get_entropy():
    return exchange_port.get_entropy()

def get_prediction():
    return exchange_port.get_prediction()

def get_confidence():
    return exchange_port.get_confidence()

def get_focus(id, k):
    return exchange_port.get_focus(id, k)

def get_saliency_map_path(id):
    return exchange_port.get_saliency_map_path(id)

def get_individual_info(id):
    return exchange_port.get_individual_info(id)

def get_grid_layout_of_sampled_instances(embed_method, datatype, left_x, top_y, width, height, class_selection, node_id):
    return exchange_port.get_grid_layout_of_sampled_instances(embed_method, datatype,
                                                              left_x, top_y, width, height, class_selection, node_id)

def get_decision_boundary_of_sampled_instances():
    return exchange_port.get_decision_boundary_of_sampled_instances()

def get_grid_layout_query(embed_method, datatype, left_x, top_y, range_size):
    return exchange_port.get_grid_layout_query(embed_method, datatype,
                                                left_x, top_y, range_size)

def change_class():
    return exchange_port.change_class()