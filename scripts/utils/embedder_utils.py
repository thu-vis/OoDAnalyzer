import numpy as np
from time import time

from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from .log_utils import logger

class Embedder(object):
    def __init__(self, method_name, *args, **kwargs):
        self.projector = None
        self.method_name = method_name
        if method_name == "tsne":
            self.projector = TSNE(*args, **kwargs)
        elif method_name == "pca":
            self.projector = PCA(*args, **kwargs)
        elif method_name == "mds":
            self.projector = MDS(n_jobs=-1, *args, **kwargs)
        else:
            logger.error("the projection method is not supported now!!")

    def fit(self, X, y):
        t = time()
        self.projector.fit(X, y)
        logger.info("{} fit function time cost: {}".format(self.method_name, time()-t))

    def transform(self, X, y):
        t = time()
        self.projector.transform(X, y)
        logger.info("{} transform function time cost: {}".format(self.method_name, time()-t))

    def fit_transform(self, X, y):
        t = time()
        res = self.projector.fit_transform(X, y)
        logger.info("{} fit_transform function time cost: {}".format(self.method_name, time()-t))
        return res
