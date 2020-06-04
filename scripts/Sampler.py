import numpy as np
import os
import sys
import ctypes
import math
from time import time

from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from scipy.spatial.distance import cdist
from PIL import Image
from fastlapjv import fastlapjv
import multiprocessing
from anytree import Node


from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.utils.data_utils import Data
from scripts.utils.log_utils import logger
from scripts.utils.sampling_utils import DensityBasedSampler
from scripts.utils.config_utils import config



def grid_layout(idx, X, selected_list, selected_pos, ent,
                k=100, constraint_matrix = None):
    X -= X.min(axis=0)
    X /= X.max(axis=0)
    num = X.shape[0]
    square_len = math.ceil(np.sqrt(num))
    N = square_len * square_len
    grids = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / square_len, square_len),
            np.linspace(0, 1 - 1.0 / square_len, square_len))) \
            .reshape(-1, 2)

    original_cost_matrix = cdist(grids, X, "euclidean")
    # knn process
    dummy_points = np.ones((N - original_cost_matrix.shape[1], 2)) * 0.5
    # dummy at [0.5, 0.5]
    dummy_vertices = (1 - cdist(grids, dummy_points, "euclidean")) * 100
    cost_matrix = np.concatenate((original_cost_matrix, dummy_vertices), axis=1)
    cost_matrix[cost_matrix==0]  = 1000000


    if len(selected_list) > 0:
        print("*****************************constraint******************************")
        pointer = 0
        selected_ent = ent[np.array(selected_list)]
        for i, id in enumerate(selected_list):
            if pointer > len(idx):
                break
            while pointer < len(idx) and idx[pointer] < id:
                pointer += 1
            if idx[pointer] == id:
                pointer += 1
                # print(i, id, selected_pos[i])
                if selected_ent[i] > 0.4:
                    dis = ((grids - selected_pos[i])**2).sum(axis=1)
                    sorted_idx = dis.argsort()
                    nearest_grids = grids[sorted_idx[:25]]
                    this_position = X[pointer-1,:]
                    dis = ((nearest_grids - this_position)**2).sum(axis=1)
                    nearest_idx = sorted_idx[:25][dis.argsort()[:5]]
                    cost_matrix[nearest_idx, pointer-1] = 0.001
                    # print(nearest_idx, pointer-1, id)
                # X[pointer-1,:] = selected_pos[i]
    else:
        print("root used")


    # begin LAP-JV
    logger.info("begin LAP JV")
    t = time()
    # row_asses, col_asses, info = lapjv(cost_matrix)
    row_asses, col_asses, info = fastlapjv(cost_matrix, k_value=50)
    col_asses = col_asses[:num]
    grid_X = grids[col_asses]
    logger.info("train cost: {}, time cost: {}"
                .format(info[0], time() - t))

    return grid_X, row_asses, col_asses

class Sampler(object):
    def __init__(self, dataname, sampling_square_len=45):
        t = time()
        sampling_num = sampling_square_len * sampling_square_len
        self.sampling_square_len = sampling_square_len
        self.sampling_num = sampling_num
        self.dataname = dataname
        self.data = Data(self.dataname)
        self.class_selection = "1" * len(self.data.mat['class_name'])
        self.entropy = self.data.entropy
        self.confidence = np.max(self.data.pred_prob, axis=1)
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = self.data.get_data("all")
        self.train_idx = self.data.train_idx
        self.test_idx = self.data.test_idx
        self.pred_y_train = self.data.prediction[np.array(self.train_idx)]
        self.pred_y_test = self.data.prediction[np.array(self.test_idx)]
        self.embed_X_train, self.embed_X_valid, self.embed_X_test = self.data.get_embed_X("all")

        # self.sampler = DensityBasedSampler(n_samples=sampling_num)
        self.min_adding_len = 0.05
        self.current_count = 1
        # self._OoD_norm_by_prediction()
        logger.info("time cost before self._init {}".format( time() - t))
        self._init()


    def _init(self):
        self.train_tree = None
        self.train_tree_data = {}
        self.train_focus_node = None
        self.test_tree = None
        self.test_tree_data = {}
        self.test_focus_node = None
        self.all_tree = None
        self.all_tree_data = {}
        self.all_focus_node = None
        self.state = "train"
        self.current_tree = self.train_tree
        self.current_tree_data = self.train_tree_data
        # self.current_count = 1
        self._preprocess()

        self.current_sampled_idx = None
        self.current_sampled_X = None
        self.current_grid_layout = None

    def get_class_selection(self):
        return self.class_selection

    def set_class_selection(self, class_selection):
        if (class_selection == self.class_selection):
            return
        self.class_selection = class_selection
        class_indicator = []
        for i in class_selection:
            class_indicator.append(int(i))
        train_idx_indicator = []
        test_idx_indicator = []
        for y in self.pred_y_train:
            if class_indicator[y] == 1:
                train_idx_indicator.append(True)
            else:
                train_idx_indicator.append(False)
        for y in self.pred_y_test:
            if class_indicator[y] == 1:
                test_idx_indicator.append(True)
            else:
                test_idx_indicator.append(False)
        print("test")
        self.train_idx = np.array(self.data.train_idx)[np.array(train_idx_indicator)].tolist()
        self.test_idx = np.array(self.data.test_idx)[np.array(test_idx_indicator)].tolist()
        print(len(self.train_idx), len(self.test_idx))
        self._init()


    def process_idx(self, idx):
        # print("process_idx", idx)
        t = time()
        X = self.data.embed_X[np.array(idx)]
        entropy = self.entropy[np.array(idx)]
        confidence = self.confidence[np.array(idx)]
        if len(idx) < self.sampling_num:
            sampled_idx = idx
            unsampled_idx = []
            sampled_X = X
        else:
            # sampler = DensityBasedSampler(n_samples=self.sampling_num)
            # res = sampler.fit_sample(X, confidence=confidence, entropy=entropy)
            res = [0]
            res[0] = np.zeros(len(idx)).astype(bool)
            res[0][:self.sampling_num] = True
            sampled_idx = np.array(idx)[res[0]]
            sampled_X = X[res[0]]
            unsampled_idx = np.array(idx)[np.array(1-res[0]).astype(bool)]
        # grid_x = grid_layout(sampled_X)[0]
        print("process_idx. total idx: {}, sampled idx: {}, unsampled idx: {}".format(len(idx),
                                                                         len(sampled_idx),
                                                                         len(unsampled_idx)))
        grid_x = np.ones(sampled_X.shape) * 0.5
        hiera = {}
        for idx, id in enumerate(sampled_idx):
            try:
                hiera[id] = {}
                hiera[id]["g"] = grid_x[idx]
                hiera[id]["c"] = []
            except Exception as e:
                print(e)


        for idx, id in enumerate(unsampled_idx):
            # x = self.data.embed_X[id]
            # dis = ((sampled_X - x) ** 2).sum(axis=1)
            # parent_id = sampled_idx[dis.argmin()]
            hiera[sampled_idx[0]]["c"].append(id)

        # check hiera
        count = 0
        for i in hiera:
            count = count + len(hiera[i]["c"]) + 1
        print("tree_process. hiera total num {}, time cost: {}".format( count, time() - t))

        return idx, sampled_idx, grid_x, hiera

    def tree_process(self, idx):
        idx, sampled_idx, grid_x, hiera = self.process_idx(idx)

        tree = Node("root")
        focus_node = tree
        tree_data = {}
        tree_data[focus_node.name] = {
            "instance_idx": idx,
            "sampled_idx": sampled_idx,
            "grid_layout": grid_x,
            "hiera": hiera,
            "node": tree,
            "class_selection": self.get_class_selection()
        }
        return tree, focus_node, tree_data

    def _preprocess(self):
        # training hierarchical process
        self.train_tree, self.train_focus_node, self.train_tree_data = \
            self.tree_process(self.train_idx)
        self.test_tree, self.test_focus_node, self.test_tree_data = \
            self.tree_process(self.test_idx)
        self.all_tree, self.all_focus_node, self.all_tree_data = \
            self.tree_process(self.train_idx + self.test_idx)


    def get_sampler_and_set_class(self, embed_method, datatype, left_x, top_y,
                                  width, height, class_selection, node_id):
        if class_selection is None:
            class_selection = self.get_class_selection()
        else:
            self.set_class_selection(class_selection)
        return self.get_sampler(embed_method, datatype, left_x, \
                                top_y, width, height, class_selection, node_id)

    def get_sampler(self, embed_method, datatype, left_x, top_y, width, height,
                    class_selection, node_id):
        # change stage
        t = time()
        if datatype == "train":
            self.current_tree = self.train_tree
            self.current_focus_node = self.train_focus_node
            self.current_tree_data = self.train_tree_data
        elif datatype == "test":
            self.current_tree = self.test_tree
            self.current_focus_node = self.test_focus_node
            self.current_tree_data = self.test_tree_data
        elif datatype == "all":
            self.current_tree = self.all_tree
            self.current_focus_node = self.all_focus_node
            self.current_tree_data = self.all_tree_data
        else:
            raise ValueError("Sampler.py: unsupported datatype: {}".format(datatype))



        if node_id < 0:
            node_id = "root"
            left_x = 0
            top_y = 0
            width = height = 1
            print("using root")
        else:
            node_id = "id-" + str(node_id)
        self.current_focus_node = self.current_tree_data[node_id]["node"]
        node_data = self.current_tree_data[self.current_focus_node.name]
        current_hiera = node_data["hiera"]

        # check hiera
        count = 0
        for i in current_hiera:
            count = count + len(current_hiera[i]["c"]) + 1
        print("hiera total num",count)

        selected_list = []
        selected_pos = []
        for id in current_hiera:
            x = current_hiera[id]["g"][0]
            y = current_hiera[id]["g"][1]
            if (x > left_x and x < (left_x + width)) and \
                    (y > top_y and y < (top_y + height)):
                selected_list.append(id)
                selected_pos.append([(x - left_x) / width, (y - top_y)/height])
        idx, sampled_idx, grid_x, hiera, sampling_res = \
            self._get_sampler(selected_list, selected_pos, current_hiera, class_selection, node_id, datatype)
        new_node = Node("id-" + str(self.current_count), parent=self.current_focus_node)
        self.current_count = self.current_count + 1
        self.current_focus_node = new_node
        self.current_tree_data[self.current_focus_node.name] = {
            "instance_idx": idx,
            "sampled_idx": sampled_idx,
            "grid_layout": grid_x,
            "hiera": hiera,
            "node": new_node,
            "class_selection": class_selection
        }
        print("self.get_sampler grid layout and sampler time cost:", time() - t)
        boundary_res = self._get_boundary_points()
        print("self.get_sampler decision boundary time cost:", time() - t)
        return {
            "id": self.current_count - 1,
            "layout": sampling_res,
            "boundary": boundary_res
        }

    def _get_sampler(self, selected_list, selected_pos, current_hiera, class_selection, old_node_id, datatype):
        idx = []
        print("selected idx len", len(selected_list))
        for id in selected_list:
            idx.extend(current_hiera[id]["c"] + [id])

        print("all idx len:", len(idx))
        # class_indicator = []
        # for i in class_selection:
        #     class_indicator.append(int(i))
        # idx_indicator = []
        # all_y = np.array(self.data.y)[np.array(idx)]
        # for y in all_y:
        #     if class_indicator[y] == 1:
        #         idx_indicator.append(True)
        #     else:
        #         idx_indicator.append(False)
        # idx = np.array(idx)[np.array(idx_indicator)]
        print("selected list", selected_list)


        # 特殊处理 for the first time
        if old_node_id == "root":
            selected_list = []
            selected_pos = []
        cls_str = "".join(self.class_selection)
        buffer_path = os.path.join(config.data_root, self.dataname, "heira_" + datatype + cls_str + ".pkl")

        if os.path.exists(buffer_path) and old_node_id == "root":
            logger.info("using hiera buffer!!!")
            idx, sampled_idx, self.current_grid_layout, hiera, res = pickle_load_data(buffer_path)
            self.current_sampled_idx = sampled_idx
            grid_x = self.current_grid_layout[0]

        else:
            X = self.data.embed_X[np.array(idx)]
            entropy = self.entropy[np.array(idx)]
            confidence = self.confidence[np.array(idx)]
            if len(idx) < self.sampling_num:
                sampled_idx = np.array(idx)
                sampled_X = X
                unsampled_idx = []
            else:
                print("total instances needed to sampled: ", len(idx))
                intersection_idx = list(set(idx).intersection(set(selected_list)))
                selection = np.zeros(len(idx), dtype=bool)
                for id in intersection_idx:
                    selection[idx.index(id)] = True
                sampler = DensityBasedSampler(n_samples=self.sampling_num)
                res = sampler.fit_sample(X, entropy=entropy, confidence=confidence, selection=np.array(selection))
                sampled_idx = np.array(idx)[res[0]]
                print("sampled idx",sampled_idx)
                sampled_X = X[res[0], :]
                unsampled_idx = np.array(idx)[np.array(1-res[0]).astype(bool)]
            self.current_sampled_idx = sampled_idx
            self.current_history_idx = sampled_idx.copy()
            self.current_sampled_x = sampled_X
            test_sampled_X = sampled_X.copy()
            self.current_grid_layout = grid_layout(sampled_idx, sampled_X,
                selected_list, selected_pos, self.entropy)
            res = []
            grid_x = self.current_grid_layout[0]

            hiera = {}
            for idx, id in enumerate(sampled_idx):
                hiera[id] = {}
                hiera[id]["g"] = grid_x[idx]
                hiera[id]["c"] = []

            for idx, id in enumerate(unsampled_idx):
                x = self.data.embed_X[id]
                dis = ((sampled_X - x) ** 2).sum(axis=1)
                parent_id = sampled_idx[dis.argmin()]
                hiera[parent_id]["c"].append(id)

            for idx, id in enumerate(sampled_idx):
                res.append({
                    'id': int(id),
                    'pos': grid_x[idx].tolist()
                })
            if old_node_id == "root":
                pickle_save_data(buffer_path, [idx, sampled_idx, self.current_grid_layout, hiera, res])

        return idx, sampled_idx, grid_x, hiera, res

    def _get_boundary(self):
        def _meanSqaureResidue(x, y, regr):
            if regr == None:
                return 0
            else:
                x = x.reshape(-1, 1)
                y = y
                pred_y = regr.predict(x)
                return np.mean((y - pred_y) ** 2)
        def _getIntersection(lr1, lr2):
            if lr1[2] == None and lr2[2] == None:
                return None
            if lr1[2] == None:
                m = lr2[2]
                x = _points[lr1[0]][0]
                y = m.coef_[0] * x + m.intercept_
                return (x, y)
            if lr2[2] == None:
                m = lr1[2]
                x = _points[lr2[0]][0]
                y = m.coef_[0] * x + m.intercept_
                return (x, y)
            m1 = lr1[2]
            m2 = lr2[2]
            if m1.coef_[0] == m2.coef_[0]:
                return None
            else:
                x = (m2.intercept_ - m1.intercept_) / (m1.coef_[0] - m2.coef_[0])
                y = m1.coef_[0] * x + m1.intercept_
                return (x, y)
        def _recursivelyFit(points, l, r):
            regr, s = _regress(points, l, r)
            if s < 1e-3:
                return [[l, r, regr]]
            else:
                mid1 = int((2 * l + r) / 3)
                mid2 = int((l + 2 * r) / 3)
                part1 = _recursivelyFit(points, l, mid1)
                part2 = _recursivelyFit(points, mid1 + 1, mid2)
                part3 = _recursivelyFit(points, mid2 + 1, r)
                part2.extend(part3)
                part1.extend(part2)
                return part1
                # mid = int((l + r) / 2)
                # part1 = _recursivelyFit(points, l, mid)
                # part2 = _recursivelyFit(points, mid + 1, r)
                # part1.extend(part2)
                # return part1
        def _mergeFit(points, segments):
            flag = -1
            new_regr = None
            min_loss = 1
            for i in range(len(segments) - 1):
                l = segments[i][0]
                r = segments[i + 1][1]
                regr, s = _regress(points, l, r)
                if s < 1e-3 and s < min_loss:
                    new_regr = [[l, r, regr]]
                    flag = i
                    min_loss = s

            if flag == -1:
                return segments
            else:
                new_segments = segments[:flag]
                new_segments.extend(new_regr)
                new_segments.extend(segments[flag + 2:])
                return _mergeFit(points, new_segments)
        def _adjustFit(points, segments):
            for i in range(len(segments) - 1):
                # Nearly parallel
                intersection = _getIntersection(segments[i], segments[i + 1])
                l = segments[i][0]
                r = segments[i + 1][1]
                p = points[l:r + 1]
                x = p[:, 0]
                y = p[:, 1]
                if intersection == None or \
                        (intersection[0] - np.max(x) > 0.5 * (np.max(x) - np.min(x))) or \
                        (np.min(x) - intersection[0] > 0.5 * (np.max(x) - np.min(x))) or \
                        (intersection[1] - np.max(y) > 0.5 * (np.max(y) - np.min(y))) or \
                        (np.min(y) - intersection[1] > 0.5 * (np.max(y) - np.min(y))):
                    regr, s = _regress(points, l, r)
                    _adjust_result = [[l, r, regr]]
                    new_segments = segments[:i]
                    new_segments.extend(_adjust_result)
                    new_segments.extend(segments[i + 2:])
                    return _adjustFit(points, new_segments)
            return segments
        def _fit(points, l, r):
            result = _recursivelyFit(points, l, r)
            result = _mergeFit(points, result)
            result = _adjustFit(points, result)
            return np.array(result)
        def _getBoundaryPoints(classes_num = 2):
            N = math.ceil(math.sqrt(self.current_sampled_idx.shape[0]))
            boundary_points_idx = []
            grids = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / N, N),
                                               np.linspace(0, 1 - 1.0 / N, N))) \
                .reshape(-1, 2)
            predictions = self.data.prediction[self.current_sampled_idx]
            row_asses = self.current_grid_layout[1]
            # Define valid indices
            valid_idx = [-1 for _ in range(N**2)]
            f_neighbour = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            next_to_visit = 0
            cluster_count = 0
            cluster_pred = []
            cluster_size = []
            while True:
                while next_to_visit < N**2 and  \
                        (valid_idx[next_to_visit] != -1 or row_asses[next_to_visit] >= predictions.shape[0]):
                    next_to_visit += 1
                if next_to_visit >= N**2:
                    break
                head = 0
                tail = 0
                queue = [next_to_visit]
                valid_idx[next_to_visit] = cluster_count
                this_label = predictions[row_asses[next_to_visit]]
                while head <= tail:
                    cur = queue[head]
                    i = cur // N
                    j = cur - i * N
                    for t in range(4):
                        new_i = i + f_neighbour[t][0]
                        new_j = j + f_neighbour[t][1]
                        if new_i < 0 or new_i >= N or new_j < 0 or new_j >= N:
                            continue
                        new_pos = new_i * N + new_j
                        if row_asses[new_pos] >= predictions.shape[0] or \
                                this_label != predictions[row_asses[new_pos]] or \
                                valid_idx[new_pos] != -1:
                            continue
                        queue.append(new_pos)
                        valid_idx[new_pos] = cluster_count
                        tail += 1
                    head += 1
                # if tail > 0.01 * (N**2):
                #     for idx in queue:
                #         valid_idx[idx] = 1
                cluster_pred.append(this_label)
                cluster_size.append(tail + 1)
                cluster_count += 1
            # validate clusters
            classes_max_size = [-1 for _ in range(classes_num)]
            for i in range(cluster_count):
                if classes_max_size[cluster_pred[i]] < cluster_size[i]:
                    classes_max_size[cluster_pred[i]] = cluster_size[i]
            for i in range(cluster_count):
                if cluster_size[i] < 0.2 * classes_max_size[cluster_pred[i]]:
                    cluster_pred[i] = -1

            # print(N)
            # print(cluster_pred)
            # print(cluster_size)

            for i in range(int(N)):
                for j in range(int(N)):
                    idx = i * N + j
                    if valid_idx[idx] < 0 or cluster_pred[valid_idx[idx]] < 0:
                        continue
                    this_label = predictions[row_asses[idx]]
                    if i < N - 1:
                        bottom_neighbour = idx + N
                        if row_asses[bottom_neighbour] < predictions.shape[0] and \
                                predictions[row_asses[bottom_neighbour]] != this_label and \
                                cluster_pred[valid_idx[bottom_neighbour]] >= 0:
                            boundary_points_idx.append((idx, bottom_neighbour))
                    if j < N - 1:
                        right_neighbour = idx + 1
                        if row_asses[right_neighbour] < predictions.shape[0] and \
                                predictions[row_asses[right_neighbour]] != this_label and \
                                cluster_pred[valid_idx[right_neighbour]] >= 0:
                            boundary_points_idx.append((idx, right_neighbour))
            boundary_points = [(grids[i] + grids[j]) / 2 for (i, j) in boundary_points_idx]
            return np.array(boundary_points)
        def _sortPoints(points):
            cnt = points.shape[0]
            inf = 1e8
            dist = [inf for _ in range(cnt)]
            visited = [False for _ in range(cnt)]
            # Select seed point
            corner_dist = [min(p[0], 1 - p[0]) + min(p[1], 1 - p[1]) for p in points]
            seed_idx = corner_dist.index(min(corner_dist))
            dist[seed_idx] = 0
            visited[seed_idx] = True
            idx_list = [seed_idx]
            # Add nearest point to idx_list
            for i in range(1, cnt):
                current = idx_list[-1]
                min_dist = inf
                next_idx = -1
                for j in range(cnt):
                    if not visited[j]:
                        dist[j] = min(dist[j], np.sqrt(np.sum(np.square(points[current] - points[j]))))
                        if dist[j] <= min_dist:
                            next_idx = j
                            min_dist = dist[j]
                idx_list.append(next_idx)
                visited[next_idx] = True
            sorted_points = points[idx_list]
            return sorted_points
        def _regress(points, l, r):
            p = points[l:r + 1]
            x = p[:, 0]
            y = p[:, 1]
            flag = False
            for i in range(x.shape[0]):
                if x[i] != x[0]:
                    flag = True
                    break
            if not flag:
                regr = None
            else:
                regr = linear_model.LinearRegression()
                regr.fit(x.reshape(-1, 1), y)
            return regr, _meanSqaureResidue(x, y, regr)
        def _createKeypoints(segments):
            lines_to_plot = []
            for i in range(len(segments)):
                l = lr[i][0]
                r = lr[i][1]
                p = _points[l:r + 1]
                x_coords = p[:, 0]
                x = [np.min(x_coords), np.max(x_coords)]
                y_coords = p[:, 1]
                y = [np.min(y_coords), np.max(y_coords)]
                lines_to_plot.append((lr[i][2], np.array(x), np.array(y)))
            lines_orignal = []
            for i, line in enumerate(lines_to_plot):
                regr = line[0]
                x = line[1]
                if regr == None:
                    # yl = lines_to_plot[i - 1][1].predict(lines_to_plot[i - 1][0])[1] if i > 0 else 0
                    # yr = lines_to_plot[i + 1][1].predict(lines_to_plot[i + 1][0])[0] if i < len(lines_to_plot)-1 else 1
                    y = line[2]
                else:
                    y = regr.predict(x.reshape(-1, 1))
                start_point = [x[0], y[0]]
                end_point = [x[1], y[1]]
                if i == 0:
                    start_point_offset = min(start_point[0], 1 - start_point[0], \
                                             abs(start_point[1]), abs(1 - start_point[1]))
                    end_point_offset = min(end_point[0], 1 - end_point[0], \
                                           abs(end_point[1]), abs(1 - end_point[1]))
                else:
                    prev_end_point = lines_orignal[-1][1]
                    start_point_offset = np.sum(np.square(np.array(prev_end_point) - np.array(start_point)))
                    end_point_offset = np.sum(np.square(np.array(prev_end_point) - np.array(end_point)))
                if start_point_offset > end_point_offset:
                    start_point = [x[1], y[1]]
                    end_point = [x[0], y[0]]
                # print(i, start_point, end_point)
                k = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0]) \
                    if start_point[0] != end_point[0] else None
                if i == 0:
                    if start_point[0] < end_point[0]:
                        start_point[0] = 0
                        start_point[1] = end_point[1] - k * end_point[0]
                    elif start_point[0] > end_point[0]:
                        start_point[0] = 1
                        start_point[1] = end_point[1] + k - k * end_point[0]
                if i == len(lines_to_plot) - 1:
                    if end_point[0] < start_point[0]:
                        end_point[0] = 0
                        end_point[1] = start_point[1] - k * start_point[0]
                    elif end_point[0] > start_point[0]:
                        end_point[0] = 1
                        end_point[1] = start_point[1] + k - k * start_point[0]
                if k == None:
                    if start_point[1] > 1:
                        start_point[1] = 1
                    elif start_point[1] < 0:
                        start_point[1] = 0
                    if end_point[1] > 1:
                        end_point[1] = 1
                    elif end_point[1] < 0:
                        end_point[1] = 0
                else:
                    if start_point[1] > 1:
                        start_point[1] = 1
                        start_point[0] = (1 - end_point[1]) / k + end_point[0]
                    elif start_point[1] < 0:
                        start_point[1] = 0
                        start_point[0] = (0 - end_point[1]) / k + end_point[0]
                    if end_point[1] > 1:
                        end_point[1] = 1
                        end_point[0] = (1 - start_point[1]) / k + start_point[0]
                    elif end_point[1] < 0:
                        end_point[1] = 0
                        end_point[0] = (0 - start_point[1]) / k + start_point[0]
                lines_orignal.append((start_point, end_point))
            kp = []
            for i in range(len(lines_orignal)):
                if i == 0:
                    kp.append(lines_orignal[i][0])
                if i == len(lines_orignal) - 1:
                    kp.append(lines_orignal[i][1])
                else:
                    kp.append([(lines_orignal[i][1][0] + lines_orignal[i + 1][0][0]) / 2, \
                               (lines_orignal[i][1][1] + lines_orignal[i + 1][0][1]) / 2])
            return kp
        classes_num = len(self.data.mat[config.class_name])
        points = _getBoundaryPoints(classes_num)
        if points.shape[0] == 0:
            return []
        _points = _sortPoints(points)
        color_map = plt.get_cmap('tab20')(np.array([math.floor(i / 5) for i in range(_points.shape[0])]))
        plt.scatter(_points[:, 0], 0 - _points[:, 1], color=color_map, s=10, marker='+')
        lr = _fit(_points, 0, _points.shape[0] - 1)
        keypoints = _createKeypoints(lr)
        return keypoints

    def _get_boundary_points(self):
        classes_num = len(self.data.mat[config.class_name])
        N = math.ceil(math.sqrt(self.current_sampled_idx.shape[0]))
        boundary_points_idx = []
        grids = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / N, N),
                                      np.linspace(0, 1 - 1.0 / N, N))) \
            .reshape(-1, 2)
        predictions = self.data.prediction[self.current_sampled_idx]
        row_asses = self.current_grid_layout[1]
        # Define valid indices
        valid_idx = [-1 for _ in range(N ** 2)]
        f_neighbour = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        next_to_visit = 0
        cluster_count = 0
        cluster_pred = []
        cluster_size = []
        while True:
            while next_to_visit < N ** 2 and \
                    (valid_idx[next_to_visit] != -1 or row_asses[next_to_visit] >= predictions.shape[0]):
                next_to_visit += 1
            if next_to_visit >= N ** 2:
                break
            head = 0
            tail = 0
            queue = [next_to_visit]
            valid_idx[next_to_visit] = cluster_count
            this_label = predictions[row_asses[next_to_visit]]
            while head <= tail:
                cur = queue[head]
                i = cur // N
                j = cur - i * N
                for t in range(4):
                    new_i = i + f_neighbour[t][0]
                    new_j = j + f_neighbour[t][1]
                    if new_i < 0 or new_i >= N or new_j < 0 or new_j >= N:
                        continue
                    new_pos = new_i * N + new_j
                    if row_asses[new_pos] >= predictions.shape[0] or \
                            this_label != predictions[row_asses[new_pos]] or \
                            valid_idx[new_pos] != -1:
                        continue
                    queue.append(new_pos)
                    valid_idx[new_pos] = cluster_count
                    tail += 1
                head += 1
            cluster_pred.append(this_label)
            cluster_size.append(tail + 1)
            cluster_count += 1

        # validate clusters
        classes_max_size = [-1 for _ in range(classes_num)]
        for i in range(cluster_count):
            if classes_max_size[cluster_pred[i]] < cluster_size[i]:
                classes_max_size[cluster_pred[i]] = cluster_size[i]
        for i in range(cluster_count):
            if cluster_size[i] < 0.2 * classes_max_size[cluster_pred[i]]:
                cluster_pred[i] = -1

        for i in range(int(N)):
            for j in range(int(N)):
                idx = i * N + j
                if valid_idx[idx] < 0 or cluster_pred[valid_idx[idx]] < 0:
                    continue
                this_label = predictions[row_asses[idx]]
                if i < N - 1:
                    bottom_neighbour = idx + N
                    if row_asses[bottom_neighbour] < predictions.shape[0] and \
                            predictions[row_asses[bottom_neighbour]] != this_label and \
                            cluster_pred[valid_idx[bottom_neighbour]] >= 0:
                        boundary_points_idx.append((idx, bottom_neighbour))
                if j < N - 1:
                    right_neighbour = idx + 1
                    if row_asses[right_neighbour] < predictions.shape[0] and \
                            predictions[row_asses[right_neighbour]] != this_label and \
                            cluster_pred[valid_idx[right_neighbour]] >= 0:
                        boundary_points_idx.append((idx, right_neighbour))
        boundary_points = [((grids[i] + grids[j]) / 2).tolist() for (i, j) in boundary_points_idx]
        # print(boundary_points)
        return boundary_points

    def _OoD_norm_by_prediction(self):
        conf_threshold = 0.97
        test_ent = self.entropy[np.array(self.test_idx)]
        test_conf = self.confidence[np.array(self.test_idx)]
        for label in range(len(self.data.mat['class_name'])):
            ent = test_ent[self.y_test == label]
            conf = test_conf[self.y_test == label]
            ent_high_conf = ent[conf > conf_threshold]
            ent_high_conf = (ent_high_conf - ent_high_conf.min()) / (ent_high_conf.max() - ent_high_conf.min())
            ent_low_conf = ent[conf <= conf_threshold]
            ent_low_conf = (ent_low_conf - ent_low_conf.min()) / (ent_low_conf.max() - ent_low_conf.min())
            ent[conf > conf_threshold] = ent_high_conf
            ent[conf <= conf_threshold] = ent_low_conf
            test_ent[self.y_test == label] = ent

        self.entropy[np.array(self.test_idx)] = test_ent


if __name__ == '__main__':
    s = Sampler(config.animals)