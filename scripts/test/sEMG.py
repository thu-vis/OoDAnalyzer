from scripts.gcforest.datasets.uci_semg import UCISEMG
import argparse
import numpy as np
import sys
from sklearn.datasets import fetch_mldata
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


from scripts.gcforest.gcforest import GCForest
from scripts.gcforest.utils.config_utils import load_json

config = {
"net":{
"outputs": ["pool1/7x7/ets", "pool1/7x7/rf", "pool1/10x10/ets", "pool1/10x10/rf", "pool1/14x14/ets", "pool1/14x14/rf"],
"layers":[
    {
        "type":"FGWinLayer",
        "name":"win1/7x7",
        "bottoms": ["X","y"],
        "tops":["win1/7x7/ets", "win1/7x7/rf"],
        "n_classes": 6,
        "estimators": [
            {"n_folds":3,"type":"ExtraTreesClassifier","n_estimators":20,"max_depth":10,"n_jobs":-1,"min_samples_leaf":10},
            {"n_folds":3,"type":"RandomForestClassifier","n_estimators":20,"max_depth":10,"n_jobs":-1,"min_samples_leaf":10}
        ],
        "stride_x": 1,
        "stride_y": 1,
        "win_x":1,
        "win_y":7
    },
    {
        "type":"FGWinLayer",
        "name":"win1/10x10",
        "bottoms": ["X","y"],
        "tops":["win1/10x10/ets", "win1/10x10/rf"],
        "n_classes": 6,
        "estimators": [
            {"n_folds":3,"type":"ExtraTreesClassifier","n_estimators":20,"max_depth":10,"n_jobs":-1,"min_samples_leaf":10},
            {"n_folds":3,"type":"RandomForestClassifier","n_estimators":20,"max_depth":10,"n_jobs":-1,"min_samples_leaf":10}
        ],
        "stride_x": 1,
        "stride_y": 1,
        "win_x":1,
        "win_y":10
    },
    {
        "type":"FGWinLayer",
        "name":"win1/14x14",
        "bottoms": ["X","y"],
        "tops":["win1/14x14/ets", "win1/14x14/rf"],
        "n_classes": 6,
        "estimators": [
            {"n_folds":3,"type":"ExtraTreesClassifier","n_estimators":20,"max_depth":10,"n_jobs":-1,"min_samples_leaf":10},
            {"n_folds":3,"type":"RandomForestClassifier","n_estimators":20,"max_depth":10,"n_jobs":-1,"min_samples_leaf":10}
        ],
        "stride_x": 1,
        "stride_y": 1,
        "win_x":1,
        "win_y":14
    },
    {
        "type":"FGPoolLayer",
        "name":"pool1",
        "bottoms": ["win1/7x7/ets", "win1/7x7/rf", "win1/10x10/ets", "win1/10x10/rf", "win1/14x14/ets", "win1/14x14/rf"],
        "tops": ["pool1/7x7/ets", "pool1/7x7/rf", "pool1/10x10/ets", "pool1/10x10/rf", "pool1/14x14/ets", "pool1/14x14/rf"],
        "pool_method": "avg",
        "win_x":1,
        "win_y":2
    }
]

},

"cascade": {
    "random_state": 0,
    "max_layers": 100,
    "early_stopping_rounds": 3,
    "look_indexs_cycle": [
        [0, 1],
        [2, 3],
        [4, 5]
    ],
    "n_classes": 6,
    "estimators": [
        {"n_folds":3,"type":"ExtraTreesClassifier","n_estimators":500,"max_depth":None,"n_jobs":-1},
        {"n_folds":3,"type":"RandomForestClassifier","n_estimators":500,"max_depth":None,"n_jobs":-1},
        {"n_folds":3,"type":"ExtraTreesClassifier","n_estimators":500,"max_depth":None,"n_jobs":-1},
        {"n_folds":3,"type":"RandomForestClassifier","n_estimators":500,"max_depth":None,"n_jobs":-1},
        {"n_folds":3,"type":"ExtraTreesClassifier","n_estimators":500,"max_depth":None,"n_jobs":-1},
        {"n_folds":3,"type":"RandomForestClassifier","n_estimators":500,"max_depth":None,"n_jobs":-1},
        {"n_folds":3,"type":"ExtraTreesClassifier","n_estimators":500,"max_depth":None,"n_jobs":-1},
        {"n_folds":3,"type":"RandomForestClassifier","n_estimators":500,"max_depth":None,"n_jobs":-1}
    ]
}
}

if __name__ == "__main__":
    gc = GCForest(config)

    u = UCISEMG(data_set="all")
    data = u.X
    target = u.y
    print(data.shape, target.shape)
    split_point = int(len(target) * 0.8)
    index = np.array(range(len(target)))
    np.random.shuffle(index)
    target = target[index]
    data = data[index,:]
    X_train = data[:split_point,:]
    X_test = data[split_point:,:]
    y_train = target[:split_point]
    y_test = target[split_point:]

    X_train_enc = gc.fit_transform(X_train, y_train)

    y_pred = gc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))