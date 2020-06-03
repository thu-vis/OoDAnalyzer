from __future__ import absolute_import
# import numpy as np
# import os
# import xgboost as xgb
# import lightgbm as lgbm
# from keras.datasets import mnist
# from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from scripts.utils.log_utils import get_logger
from scripts.gcforest.datasets import get_dataset

Logger = get_logger("test.xgboost-randomforest")

def get_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0],-1)
    X_test = X_test.reshape(X_test.shape[0],-1)
    return (X_train, y_train), (X_test, y_test)

def get_semg_data():
    data_train = get_dataset({"type": "uci_semg", "data_set": "train", "layout_x": "tensor"})
    data_test = get_dataset({"type": "uci_semg", "data_set": "test", "layout_x": "tensor"})
    X_train = data_train.X
    X_train = X_train.reshape(X_train.shape[0],-1)
    y_train = data_train.y
    X_test = data_test.X
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_test = data_test.y
    return (X_train, y_train), (X_test, y_test)

def get_gtzan_data():
    data_train = get_dataset({"type": "gtzan", "data_set": "genre.train", "layout_x": "tensor", "cache": "mfcc"})
    data_test = get_dataset({"type": "gtzan", "data_set": "genre.val", "layout_x": "tensor", "cache": "mfcc"})
    X_train = data_train.X
    X_train = X_train.reshape(X_train.shape[0],-1)
    y_train = data_train.y
    X_test = data_test.X
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_test = data_test.y
    return (X_train, y_train), (X_test, y_test)

def random_forest_sklearn(get_data_method=get_mnist_data):
    (X_train, y_train), (X_test, y_test) = get_data_method()
    Logger.info("begin training of random forest (sklearn).")
    random_config = {
        "n_estimators": 500,
        "max_depth": 100,
        "min_samples_split": 2,
        "min_samples_leaf":1,
        "n_jobs":-1
    }
    rf = RandomForestClassifier(**random_config)
    rf.fit(X_train, y_train)
    pred_label = rf.predict(X_test)
    score = sum(pred_label==y_test) / len(y_test)
    Logger.info("end of training of random forest (sklearn), final test accuracy is {}.".format(score))

def gbdt_lightgbm(n_round = 50, max_depth=5, get_data_method=get_mnist_data):
    (X_train, y_train), (X_test, y_test) = get_data_method()
    Logger.info("begin training of gbdt (xgboost).")
    dtrain = lgbm.Dataset(X_train, label=y_train)
    dtest = lgbm.Dataset(X_test)
    params = {
        "max_depth": -1,
        "objective": "multiclass",
        # "boosting":"rf",
        "num_class": 10,
        "verbose":-1,
        "num_leaves": 1000,
        # "grow_policy":"lossguide",
        # "max_leaves": 1,
        # "min_samples_leaf": 1,
        "learning_rate": 0.1,
        # "feature_fraction": int(np.sqrt(X_train.shape[1])) / float(X_train.shape[1]),
        "feature_fraction": 0.5,
        "device": "cpu"
        # "bagging_fraction": 0.4,
    }
    bst = lgbm.train(params, dtrain, num_boost_round = n_round)
    pred_label = bst.predict(X_test)
    pred_label = pred_label.argmax(axis=1)
    score = sum(np.array(pred_label)==y_test) / len(y_test)
    Logger.info("end of training of gbdt (xgboost), final test accuracy is {}.".format(score))

def gbdt_xgboost(n_round = 50, max_depth=5, get_data_method=get_semg_data):
    (X_train, y_train), (X_test, y_test) = get_data_method()
    Logger.info("begin training of gbdt (xgboost).")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_test, label=y_test)
    dtest = xgb.DMatrix(X_test)
    params = {
        "max_depth": 5,
        "objective": "multi:softmax",
        "num_class": 10,
        "silent":1,
        "grow_policy":"lossguide",
        # "max_leaves": 8,
        # "min_samples_leaf": 1,
        "learning_rate": 0.1,
        # "colsample_bytree": int(np.sqrt(X_train.shape[1])) / float(X_train.shape[1]),
        "colsample_bytree": 0.5
    }
    evallist = [(dval, "eval"),(dtrain, "train")]
    bst = xgb.train(params, dtrain, num_boost_round=100, evals=evallist)
    pred_label = bst.predict(dtest)
    score = sum(pred_label==y_test) / len(y_test)
    Logger.info("end of training of gbdt (xgboost), final test accuracy is {}.".format(score))


def random_forest_xgboost(n_round = 50, max_depth=5, get_data_method=get_mnist_data):
    (X_train, y_train), (X_test, y_test) = get_data_method()
    Logger.info("begin training of gbdt (xgboost).")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    params = {
        "n_round":1,
        "max_depth":100,
        "num_parallel_tree":500,
        "silent":1,
        "subsample": 0.632,
        "colsample_bytree": int(np.sqrt(X_train.shape[1])) / float(X_train.shape[1])
    }
    bst = xgb.train(params, dtrain, num_boost_round = n_round)
    pred_label = bst.predict(dtest)
    score = sum(pred_label==y_test) / len(y_test)
    Logger.info("end of training of gbdt (xgboost), final test accuracy is {}.".format(score))


if __name__ == '__main__':
    # random_forest_sklearn(get_data_method=get_semg_data)
    # for n_round in [50]:
    #     for i in [5]:
    #         gbdt_xgboost(n_round=n_round, max_depth=i, get_data_method=get_semg_data)
    # random_forest_xgboost()
    None
