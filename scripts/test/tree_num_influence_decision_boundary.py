import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from scripts.DecisionBoundary.two_dim_decision_boundary import two_dim_decision_boundary

def decision_tree(X, y):
    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X, y)
    two_dim_decision_boundary(clf, X, y)

def random_forest(X, y):
    clf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=14)
    # clf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=5)
    clf.fit(X, y)
    two_dim_decision_boundary(clf, X, y)
    exit()
    for m in clf.estimators_:
        two_dim_decision_boundary(m, X, y)

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:, [0,2]]
    y = iris.target
    # decision_tree(X, y)
    random_forest(X, y)