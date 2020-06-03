import numpy as np
from sklearn import datasets, linear_model, tree, ensemble
from sklearn import svm, datasets

from scripts.DecisionBoundary.pylib import plot_decision_boundary
from scripts.DecisionBoundary.two_dim_decision_boundary import two_dim_decision_boundary


def pylib_decision_boundary():
    X, y = datasets.make_blobs(centers=6, random_state=11)

    model = linear_model.LogisticRegression()
    model.fit(X, y)
    a = plot_decision_boundary(model, X=X, Y=y)

def sklearn_decision_boundary():
    # import some data to play with
    iris = datasets.load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    model = svm.SVC(kernel='linear', C=C)
    model = model.fit(X, y)
    two_dim_decision_boundary(model, X, y)

if __name__ == '__main__':
    sklearn_decision_boundary()
    # import sys
    # import os
    #
    # SERVER_ROOT = os.path.dirname(sys.modules[__name__].__file__)
    # SERVER_ROOT = os.path.join(SERVER_ROOT, "..")
    # print(SERVER_ROOT)