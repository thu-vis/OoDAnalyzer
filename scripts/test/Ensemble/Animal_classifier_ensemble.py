import numpy as np
import os
from time import time
from scipy.stats import entropy
from scripts.utils.config_utils import config
from scripts.utils.log_utils import logger
from scripts.utils.helper_utils import check_dir, pickle_load_data, pickle_save_data
from scripts.database.database import DataBase
from scripts.test.Ensemble.src_class.Ensemble_classifier import EnsembleClassifier


class EnsembleClassifierAnimals(EnsembleClassifier):
    def __init__(self, feature_dir_name = None):
        dataname = config.animals
        if feature_dir_name is None:
            feature_dir_name = [
                    "weights.30-0.7411.h5",
                    "weights.40-0.7544.h5",
                    "weights.40-0.7456.h5",
                    "inceptionresnet_imagenet",
                    "inceptionv3_imagenet",
                    "mobilenet_imagenet",
                    "resnet50_imagenet",
                    "vgg_imagenet",
                    "xception_imagenet",
                    "sift-200",
                    "brief-200",
                    # "orb-200",
                    "surf-200",
                    "superpixel"
            ]
        super(EnsembleClassifierAnimals, self).__init__(dataname,
                                               feature_dir_name)

        all_idx = np.zeros(max(self.test_idx) + 100)
        all_idx[np.array(self.test_idx)] = np.array(range(len(self.test_idx)))
        bias_idx = all_idx[np.array(self.test_redundant_idx)].astype(int)
        normal_idx = [i for i in range(len(self.test_idx)) if i not in bias_idx]

        self.bias_idx = bias_idx
        self.normal_idx = normal_idx
        prediction = self.data.prediction

        a = 1



    # def get_all_entropy(self, subset, M3V=False):
    #     entropy_filename = os.path.join(self.data_dir, "all_entropystep1.pkl")
    #     return pickle_load_data(entropy_filename)

    def merge_result(self, test_predys, _test_predy):
        # if _test_predy.max() < 0.60:
        #     _test_predy = _test_predy * 0
        # _test_predy_confidence = _test_predy.max(axis=1)
        # _test_predy = np.zeros((_test_predy_confidence.shape[0], 2))
        # _test_predy[:,0] = _test_predy_confidence
        # _test_predy[:,1] = 1 - _test_predy_confidence
        # if _test_predy_confidence.mean() < 0.5:
        #     _test_predy = _test_predy * 0
        # if _test_predy_confidence < 0.6:
        #     _test_predy = _test_predy * 0
        # _test_predy[_test_predy.max(axis=1) < 0.6] = _test_predy[_test_predy.max(axis=1) < 0.6] * 0
        if test_predys is None:
            test_predys = _test_predy
        else:
            test_predys = test_predys + _test_predy
        return test_predys

    def get_all_pred(self, subset, M3V=False):
        t = time()
        print(subset)
        test_predys = []
        count = 0
        for weight_name in self.feature_dir_name:
            feature_dir = os.path.join(config.data_root, self.dataname,
                                           "feature", weight_name)
            X = pickle_load_data(os.path.join(feature_dir, "X.pkl"))
            print(X.shape)
            # clf = pickle_load_data(os.path.join(feature_dir, model_name + "_model.pkl"))
            y = self.data.y
            train_X = X[np.array(self.train_idx), :]
            train_y = y[np.array(self.train_idx)]
            test_X = X[np.array(self.test_idx), :]
            test_y = y[np.array(self.test_idx)]
            grid_search_dir = os.path.join(feature_dir, "logistic")
            for C in subset:
                # print("C:", C, "using prob")
                count = count + 1
                model_path = os.path.join(os.path.join(grid_search_dir), str(C))
                clf = pickle_load_data(model_path)
                _test_predy = clf.predict_proba(test_X)
                score = clf.score(test_X, test_y)
                # test_predys = test_predys + _test_predy
                # test_predys = self.merge_result(test_predys, _test_predy)
                test_predys.append(_test_predy)
                print(len(test_predys))
                # print("C:", C, "using prob", "score:", score)
        print("total count:", count)

        # test_entropy = entropy(test_predys.T)
        return test_predys


    def confidence_analysis(self, subset):
        pred_y = self.get_all_pred(subset)
        pred_y = np.array(pred_y).transpose(1,0,2)
        print(pred_y.shape)
        all_pred_y = np.zeros((len(self.data.y), pred_y.shape[1], pred_y.shape[2]))
        all_pred_y[np.array(self.test_idx)] = pred_y
        # to get all_sub_y
        tmp_data = pickle_load_data(os.path.join(config.data_root, config.animals_step1
                                                       , "all_data_cache.pkl"))
        all_sub_y = tmp_data["all_sub_y"].astype(int)

        test_categories_idx = [[] for i in range(22)]
        for idx in self.test_idx:
            try:
                test_categories_idx[all_sub_y[idx]].append(idx)
            except:
                # print(idx)
                None

        pred_0 = all_pred_y[np.array(test_categories_idx[0])]
        pred_0_max = pred_0.max(axis=2)
        pred_0_hard_max = pred_0.argmax(axis=2)
        pred_6 = all_pred_y[np.array(test_categories_idx[6])]
        pred_6_max = pred_6.max(axis=2)
        pred_6_hard_max = pred_6.max(axis=2)
        for i in np.arange(0,1,0.01):
            pred_0_tmp = pred_0.copy()
            pred_0_tmp[pred_0_max<i] = 0
            pred_6_tmp = pred_6.copy()
            pred_6_tmp[pred_6_max<i] = 0
            en_0 = entropy(pred_0_tmp.mean(axis=1).T)
            en_6 = entropy(pred_6_tmp.mean(axis=1).T)
            print(en_0.mean(), en_6.mean())

        for idxs in test_categories_idx:
            if len(idxs) ==0:
                print(0)
            else:
                # print(len(idxs),ent[np.array(idxs)].mean())
                pred = all_pred_y[np.array(idxs)]
                hard_pred = pred.argmax(axis=2)
                pred = pred.max(axis=2)
                print(pred.mean())
                a = 1
                None
        a = 1

if __name__ == '__main__':
    d = EnsembleClassifierAnimals()
    # d.grid_search_logistic([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    # d.ensemble_result_save([1e-3, 1e0], suffix="step0_4_feature_8")
    d.confidence_analysis([1e-3, 1e0])
    # d.ensemble_result_logistic([1e0])
    # d.ensemble_result_logistic([1e-7, 1e-5, 1e-3, 1e0], hist=False)
    # d.ensemble_result_logistic([1e-7, 1e-5, 1e-3, 1e0], hist=False)

    import itertools

    # a = [1e-7, 1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5, 1e7]
    #
    # for number in range(3,7):
    #     for s in itertools.combinations(range(len(a)), number):
    #         subset = [a[i] for i in s] + [1e0]
    #         d = EnsembleClassifierAnimals()
    #         d.ensemble_result_logistic(subset,hist=False)

    # feature_dir_name_list = [
    #
    #     "weights.30-0.7411.h5",
    #     "weights.40-0.7544.h5",
    #     "weights.40-0.7456.h5",
    #     "inceptionresnet_imagenet",
    #     "inceptionv3_imagenet",
    #     "mobilenet_imagenet",
    #     "resnet50_imagenet",
    #     "vgg_imagenet",
    #     "xception_imagenet",
    #     "sift-200",
    #     "brief-200",
    #     "orb-200",
    #     "surf-200",
    #     "superpixel"
    # ]
    #
    # for number in range(4, len(feature_dir_name_list)):
    #     for s in itertools.combinations(range(len(feature_dir_name_list)), number):
    #         subset = [feature_dir_name_list[i] for i in s]
    #         print(subset)
    #         d = EnsembleClassifierAnimals(subset)
    #         d.ensemble_result_logistic([1e-7, 1e-5, 1e-3, 1e0],hist=False)

