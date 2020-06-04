# --coding:utf-8 --
import numpy as np
import sys
import os
import cffi
import threading
import time
from sklearn.neighbors import BallTree
import math

from scripts.utils.config_utils import config
from scripts.utils.data_utils import Data

SO_PATH = os.path.join(config.scripts_root)

class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        return self._target(*self._args)


def Knn(X1, N, D, n_neighbors, forest_size, subdivide_variance_size, leaf_number):
    ffi = cffi.FFI()
    ffi.cdef(
        """void knn(double* X, int N, int D, int n_neighbors, int* neighbors_nn, double* distances_nn, int forest_size,
    		int subdivide_variance_size, int leaf_number);
         """)
    import os
    try:
        t1 = time.time()
        dllPath = os.path.join(SO_PATH, "utils", 'knnDll.dll')
        C = ffi.dlopen(dllPath)
        cffi_X1 = ffi.cast('double*', X1.ctypes.data)
        neighbors_nn = np.zeros((N, n_neighbors), dtype=np.int32)
        distances_nn = np.zeros((N, n_neighbors), dtype=np.float64)
        cffi_neighbors_nn = ffi.cast('int*', neighbors_nn.ctypes.data)
        cffi_distances_nn = ffi.cast('double*', distances_nn.ctypes.data)
        t = FuncThread(C.knn, cffi_X1, N, D, n_neighbors, cffi_neighbors_nn, cffi_distances_nn, forest_size, subdivide_variance_size, leaf_number)
        t.daemon = True
        t.start()
        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()
        print("knn runtime = %f"%(time.time() - t1))
        return neighbors_nn, distances_nn
    except Exception as ex:
        print(ex)
    return [[], []]


class DensityBasedSampler(object):
    """
    exact density biased sampling:
    under-sample dense regions and over-sample light regions.

    Ref: Palmer et al., Density Biased Sampling: An Improved Method for Data Mining and Clustering ,SIGMOD 2000
    """
    random_state = 42
    n_samples = -1
    N = -1
    tree = "None"
    estimated_density = "None"
    prob = "None"
    alpha = 1

    def __init__(self, n_samples, annFileName="none.ann", alpha=1, beta=1, random_state=0, use_pca=False,
                 pca_dim=50):
        self.n_samples = n_samples
        self.random_state = random_state
        self.alpha = alpha
        self.beta = beta
        assert beta >= 0, 'beta should be non-negative'
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        self.annFileName = annFileName

    def fit_sample(self, data: np.ndarray, label=None, return_others=True, selection=None, entropy=None, confidence=None):
        if type(data) == list:
            data = np.array(data)

        if self.use_pca:
            data = data - np.mean(data, axis=0)
            cov_x = np.dot(np.transpose(data), data)
            [eig_val, eig_vec] = np.linalg.eig(cov_x)

            # sorting the eigen-values in the descending order
            eig_vec = eig_vec[:, eig_val.argsort()[::-1]]
            initial_dims = self.pca_dim
            if initial_dims > len(eig_vec):
                initial_dims = len(eig_vec)

            # truncating the eigen-vectors matrix to keep the most important vectors
            eig_vec = np.real(eig_vec[:, :initial_dims])
            data = np.dot(data, eig_vec)

        self.N = len(data)
        if self.N <= self.n_samples:
            return [True] * self.N
        # np.random.seed(42)
        selection = self._fit_sample(data, label=label, selection=selection, entropy=entropy, confidence=confidence)
        if return_others:
            return selection, self.estimated_density, self.prob
        else:
            return selection

    def _fit_sample(self, data: np.ndarray, label=None, selection=None, entropy=None, confidence=None):
        if selection is not None and selection.sum() >= self.n_samples:
            return selection
        # self.tree = BallTree(data, leaf_size=2)
        knn = 50

        # guang 8-30
        X = np.array(data.tolist(), dtype=np.float64)
        N, D = X.shape
        if knn + 1 > N:
            knn = int((N - 1) / 2)
        # dist, neighbor = self.tree.query(X, k=knn + 1, return_distance=True)
        neighbor, dist = Knn(X, N, D, knn + 1, 1, 1, int(N))
        # ==================== shouxing 9-15

        # r = math.sqrt(np.mean(dist[:, -1]))    # 之前的距离没有开方，所以密度采样的计算有误
        # print("r = %f"%(r))

        # # old knn
        # # dist, _ = self.tree.query(data, k=knn + 1, return_distance=True)
        # r = np.mean(dist[:, -1])

        # # guang 9/5
        # r = np.mean(self._kDist(data, knn + 1))

        # guang 9-6
        # import IPython; IPython.embed(); exit()
        self.radius_of_k_neighbor = dist[:, -1]
        for i in range(len(self.radius_of_k_neighbor)):
            self.radius_of_k_neighbor[i] = math.sqrt(self.radius_of_k_neighbor[i])
        maxD = np.max(self.radius_of_k_neighbor)
        minD = np.min(self.radius_of_k_neighbor)
        for i in range(len(self.radius_of_k_neighbor)):
            self.radius_of_k_neighbor[i] = ((self.radius_of_k_neighbor[i] - minD) * 1.0 / (maxD - minD)) * 0.5 + 0.5
            self.radius_of_k_neighbor[i] = 1
        # for i in range(len(self.estimated_density)):
        #     self.estimated_density[i] = self.estimated_density[i]  #采样概率与r成正比

        # hist, bins = np.histogram(entropy, 100, normed=False)
        # print(entropy.shape)
        # cdf = hist.cumsum()
        # print(cdf)
        # cdf = cdf / cdf[-1]
        # entropy = np.interp(entropy, bins[:-1], cdf)
        #
        # ent_max = np.max(entropy)
        # if ent_max < 1e-6:
        #     ent_max = 1
        # self.prob = np.ones_like(self.radius_of_k_neighbor) * 0.001
        self.prob = self.radius_of_k_neighbor + self.beta * (entropy**2) * confidence # 采样概率与r和类标混杂度成正比

        ## old estimated_density
        # self.estimated_density = self.tree.query_radius(data, r=r, count_only=True)
        # if self.alpha > 0:
        #     self.prob = 1 / ((self.estimated_density + 1) ** self.alpha)
        # else:
        #     self.prob = (self.estimated_density + 1) ** abs(self.alpha)

        self.prob = self.prob / self.prob.sum()
        if selection is None:
            selection = np.zeros(self.N, dtype=bool)
        np.random.seed(self.random_state)
        selected_index = np.random.choice(self.N, self.n_samples,
                                          replace=False, p=self.prob)
        np.random.seed()
        count = selection.sum()
        for i in range(self.N):
            if count >= self.n_samples:
                break
            if not selection[selected_index[i]]:
                count += 1
                selection[selected_index[i]] = True
        return selection


if __name__ == '__main__':
    d = Data(config.dog_cat)
    X_train, y_train, X_valid, y_valid, X_test, y_test = d.get_data("all")
    embed_X_train, embed_X_valid, embed_X_test = d.get_embed_X("all")
    s = DensityBasedSampler(n_samples=20)
    sub = s.fit_sample(X_train)

    print(sub)
