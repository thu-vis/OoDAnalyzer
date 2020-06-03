import numpy as np
import os
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.utils.data_utils import Data
from scripts.test.Projection.projection_base import ProjectionBase
from scripts.utils.config_utils import config

DATANAME = config.pidd

def plot_tsne():
    dataname = DATANAME
    d = ProjectionBase(dataname)
    d.t_sne(show=True)

def plot_pca():
    d = ProjectionBase(DATANAME)
    d.pca(show=True)

def plot_mds():
    d = ProjectionBase(DATANAME)
    d.mds(show=True)

def plot_lda():
    d = ProjectionBase(DATANAME)
    d.lda(show=True)

def exploration_3d():
    d = Data(dataname=DATANAME)
    X_train = d.X_train
    plt_X = d.X_train[:,[1,5,7]]
    y_train = d.y_train
    color_map = plt.get_cmap("tab10")(y_train.astype(int))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(plt_X[:,0], plt_X[:,1], plt_X[:,2], s=3, c=color_map)
    plt.show()

def SPM():
    sns.set(style="ticks")
    data_path = os.path.join(config.raw_data_root,
                             DATANAME)
    file_path = os.path.join(data_path, "data.csv")
    df = pd.read_csv(file_path)
    col_name = df.columns
    data = pd.DataFrame(df, columns=col_name)
    sns.pairplot(data, hue="Outcome")
    plt.show()


if __name__ == '__main__':
    plot_tsne()
    # plot_pca()
    # plot_mds()
    # plot_lda()
    # exploration_3d()
    # SPM()