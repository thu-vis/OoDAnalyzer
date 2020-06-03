import numpy as np
import os
import math
import tensorflow as tf
from time import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from scipy.spatial.distance import cdist
from PIL import Image
from lapjv import lapjv
from scipy import interpolate
from sklearn import linear_model

from scripts.utils.config_utils import config
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.utils.data_utils import Data
from scripts.Grid import GridLayout

class GridTest(object):
    def __init__(self, dataname):
        self.dataname = dataname
        self.data = Data(self.dataname)
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = self.data.get_data("all")
        self.train_idx = self.data.train_idx
        self.valid_idx = self.data.valid_idx
        self.test_idx = self.data.test_idx
        self.embed_X_train, self.embed_X_valid, self.embed_X_test = self.data.get_embed_X("all")
        self.train_num, self.feature_num = self.X_train.shape
        self.num_class = self.y_train.max() + 1
        self.clf = None
        self.kernel = None

        self.grid_layout = GridLayout(self.dataname)

    def training(self, kernel="linear", C=1, gamma="auto"):
        self.kernel = kernel
        self.clf = SVC(kernel=kernel,
                       C=C, gamma=gamma,
                       verbose=1, max_iter=5000)
        print("parameter:", self.clf.get_params())
        print("training data shape:{}, test data shape: {}".format(self.X_train.shape, self.X_test.shape))
        self.clf.fit(self.X_train, self.y_train)
        train_score = self.clf.score(self.X_train, self.y_train)
        test_score = self.clf.score(self.X_test, self.y_test)
        if kernel == "linear":
            weights = self.clf.coef_
            margin = 1.0 / ((weights**2).sum())**0.5
        else:            margin = "not defined"
        print("\n training acc: {}, test acc: {}, margin value: {}."
              .format(train_score, test_score, margin))

    def segmentedLinearRegression(self, points):
        def _meanSqaureResidue(x, y, regr):
            if regr == None:
                return 0
            else:
                x = x.reshape(-1, 1)
                y = y
                pred_y = regr.predict(x)
                return np.mean((y - pred_y) ** 2)
        def _getintersection(lr1, lr2):
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
                intersection = _getintersection(segments[i], segments[i + 1])
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
        def _createKeypoints(points, segments):
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
                print(i, start_point, end_point)
                k = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0]) \
                    if start_point[0] != end_point[0] else None
                if i == 0:
                    if start_point[0] < end_point[0]:
                        start_point[0] = 0
                        start_point[1] = end_point[1] - k * end_point[0]
                    else:
                        start_point[0] = 1
                        start_point[1] = end_point[1] + k - k * end_point[0]
                if i == len(lines_to_plot) - 1:
                    if end_point[0] < start_point[0]:
                        end_point[0] = 0
                        end_point[1] = start_point[1] - k * start_point[0]
                    else:
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

        _points = _sortPoints(points)[:]
        lr = _fit(_points, 0, _points.shape[0] - 1)
        keypoints = _createKeypoints(points, lr)

        # Plot
        color_map = plt.get_cmap('tab20')(np.array([math.floor(i / 10) for i in range(_points.shape[0])]))
        plt.scatter(_points[:, 0], 0 - _points[:, 1], color=color_map, s=10, marker='+')
        for i in range(len(keypoints) - 1):
            x = np.array([keypoints[i][0], keypoints[i + 1][0]])
            y = np.array([keypoints[i][1], keypoints[i + 1][1]])
            plt.plot(x, 0 - y, color='red')

    def native_lap(self):
        # if self.clf is None:
        #     raise ValueError("you need train the svm if you want to get the support vector.")
        X_train_embeddings = self.data.embed_X_train
        X_test_embeddings = self.data.embed_X_test
        print("X_train_embeddings shape:{}, X_test_embeddings shape:{}"
                  .format(X_train_embeddings.shape, X_test_embeddings.shape))
        # grid_X_train, grid_X_test, _, _ = self.grid_layout.get_grid_layout()
        grid_X_train, grid_X_test, train_row_asses, test_row_asses ,_ ,_ \
            = self.grid_layout.get_grid_layout_native_lap_knn()
        # grid_X_train, grid_X_test, _, _ = self.grid_layout.get_grid_layout_native_lap_inverse_knn()
        # support_vectors_index = np.array(self.clf.support_)
        # no_support_vectors_index = np.array([i for i in range(self.train_num) if i not in self.clf.support_])
        # y_train = self.y_train.astype(int) * 2 + 1
        # y_test = self.y_test.astype(int) * 2 + 1
        # y_train[support_vectors_index] = y_train[support_vectors_index] - 1

        # y_train_pred = self.grid_layout.clf.predict(self.X_train) * 2 + 1
        # y_test_pred = self.grid_layout.clf.predict(self.X_test) * 2 + 1

        # color_map = plt.get_cmap("tab20")(np.array(y_train.tolist() + y_test.tolist()))
        # color_map = plt.get_cmap("tab20")(np.array(y_train_pred.tolist() + y_test_pred.tolist()))
        # color_map_train = color_map[:self.X_train.shape[0], :]
        # color_map_test = color_map[self.X_train.shape[0]:, :]
        #
        # ax = plt.subplot(221)
        # ax.scatter(X_train_embeddings[:,0], 0 - X_train_embeddings[:, 1],
        #            s=8,
        #            marker="o",
        #            c=color_map_train,
        #            alpha=0.7)
        #
        # ax = plt.subplot(222)
        # ax.scatter(X_test_embeddings[:,0], 0 - X_test_embeddings[:, 1],
        #            s=8,
        #            marker="o",
        #            c=color_map_test,
        #            alpha=0.7)
        #
        #
        # ax_train = plt.subplot(221)
        # ax_train.scatter(X_train_embeddings[no_support_vectors_index, 0],
        #                  0 - X_train_embeddings[no_support_vectors_index, 1],
        #                s=8,
        #                marker="o",
        #                c=color_map_train[no_support_vectors_index,:],
        #                alpha=0.7)
        # ax_train.scatter(X_train_embeddings[support_vectors_index, 0],
        #                  0 - X_train_embeddings[support_vectors_index, 1],
        #                s=20,
        #                marker="x",
        #                c=color_map_train[support_vectors_index,:])
        #
        # ax_train = plt.subplot(222)
        #
        # ax_train.scatter(grid_X_train[no_support_vectors_index, 0],
        #                  0 - grid_X_train[no_support_vectors_index, 1],
        #                s=8,
        #                marker="o",
        #                c=color_map_train[no_support_vectors_index,:],
        #                alpha=0.7)
        # ax_train.scatter(grid_X_train[support_vectors_index, 0],
        #                  0 - grid_X_train[support_vectors_index, 1],
        #                s=20,
        #                marker="x",
        #                c=color_map_train[support_vectors_index,:])






        # ax = plt.subplot(223)
        # ax.scatter(X_test_embeddings[:,0], 0 - X_test_embeddings[:, 1],
        #            s=8,
        #            marker="o",
        #            c=color_map_test,
        #            alpha=0.7)

        # ax = plt.subplot(224)


        # ax = plt.subplot(111)
        # color_map_test = plt.get_cmap("tab10")(self.clf.predict(self.X_test).astype(int))
        # ax.scatter(grid_X_test[:, 0], 0 - grid_X_test[:, 1],
        #            s=8,
        #            marker="o",
        #            c=color_map_test,
        #            alpha=0.7)
        #
        # keypoints = self.grid_layout.get_decision_boundary("test")
        # for i in range(len(keypoints) - 1):
        #     x = np.array([keypoints[i][0], keypoints[i + 1][0]])
        #     y = np.array([keypoints[i][1], keypoints[i + 1][1]])
        #     plt.plot(x, 0 - y, color='red')

        plt.show()

if __name__ == '__main__':
    g = GridTest(config.svhn)


    # g.training(kernel='linear')
    g.native_lap()
    # g._sampling_and_assignment_test()