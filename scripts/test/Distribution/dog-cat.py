import numpy as np
import os


import matplotlib.pyplot as plt

from scripts.utils.config_utils import config

def highlight_bias_test():
    bias_test_filename = np.load(os.path.join(config.data_root,
                                              config.dog_cat_normal_train,
                                              "feature",
                                              "bias_test_filename.npy")).tolist()
    all_test_filename = np.load(os.path.join(config.data_root,
                                              config.dog_cat_normal_train,
                                              "feature",
                                              "normal_test_filename.npy")).tolist()
    # tsne_bias_test = np.load(os.path.join(config.data_root,
    #                                           config.dog_cat,
    #                                           "tsne_bias_test_sv.npy"))
    tsne_all_test = np.load(os.path.join(config.data_root,
                                              config.dog_cat_normal_train,
                                              "tsne_normal_test_sv.npy"))
    # label_bias_test = np.load(os.path.join(config.data_root,
    #                                           config.dog_cat,
    #                                           "label_bias_test_sv.npy"))
    label_all_test = np.load(os.path.join(config.data_root,
                                              config.dog_cat_combine,
                                              "label_normal_test_sv.npy"))
    test_idx = []
    for fn in bias_test_filename:
        # try:
        idx = all_test_filename.index(fn)
        label_all_test[idx] = label_all_test[idx] - 1
        test_idx.append(idx)
        # except:
        #     print(fn)
    test_idx = np.array(test_idx)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color_map = plt.get_cmap("tab20")(label_all_test.astype(int))
    ax.scatter(tsne_all_test[:,0], tsne_all_test[:, 1], s=8, marker="o", c=color_map, alpha=0.4)
    ax.scatter(tsne_all_test[test_idx,0], tsne_all_test[test_idx, 1], s=20, marker="o", c=color_map[test_idx,:])
    plt.show()

if __name__ == '__main__':
    highlight_bias_test()