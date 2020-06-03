'''
    Develop a function to visualize decision boundary
        for any classification models in 2D
    source: https://github.com/PanWu/pylib
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def calculate_rgb(shift, radius, theta, psi):
    '''
    In the spherical coordinate, calculate RGB value
    '''
    d2r = np.pi / 180.  # degree to radian
    r = radius * np.cos(theta * d2r) * np.sin(psi * d2r)
    g = radius * np.sin(theta * d2r) * np.sin(psi * d2r)
    b = radius * np.cos(psi * d2r)
    result = np.array([r, g, b]) + shift
    return tuple([round(result[0], 3),
                  round(result[1], 3),
                  round(result[2], 3)])


# define an infinite class color retriever
def retrieve_n_class_color_sphere(N):
    '''
    Retrieve class color, over the sphere object inside RGB 3D cubic box.
    (intend to keep class color strict convex)
    Input: class number
    Output: list of RGB color code
    '''
    color_list = []
    center = np.array([0.5, 0.5, 0.5])
    radius = 0.5

    interval = 90
    np.random.seed(1)  # pre-define the seed for consistency
    while len(color_list) < N:
        the_list = []
        for psi in np.arange(0, 360 + 0.1 * interval, interval):
            for theta in np.arange(0, 360 + 0.1 * interval, interval):
                new_color = calculate_rgb(center, radius, theta, psi)
                if new_color not in color_list:
                    the_list.append(new_color)
        the_list = list(set(the_list))
        np.random.shuffle(the_list)
        color_list.extend(the_list)
        interval = interval / 2.

    return color_list[:N]


def retrieve_n_class_color_cubic(N):
    '''
    retrive color code for N given classes
    Input: class number
    Output: list of RGB color code
    '''

    # manualy encode the top 8 colors
    # the order is intuitive to be used
    color_list = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (0, 0, 0),
        (1, 1, 1)
    ]

    # if N is larger than 8 iteratively generate more random colors
    np.random.seed(1)  # pre-define the seed for consistency

    interval = 0.5
    while len(color_list) < N:
        the_list = []
        iterator = np.arange(0, 1.0001, interval)
        for i in iterator:
            for j in iterator:
                for k in iterator:
                    if (i, j, k) not in color_list:
                        the_list.append((i, j, k))
        the_list = list(set(the_list))
        np.random.shuffle(the_list)
        color_list.extend(the_list)
        interval = interval / 2.0

    return color_list[:N]


def plot_decision_boundary(model, dim_red_method='pca',
                           X=None, Y=None,
                           xrg=None, yrg=None,
                           Nx=300, Ny=300,
                           scatter_sample=None,
                           figsize=[6, 6], alpha=0.7,
                           random_state=111):
    '''
    Plot decision boundary for any two dimension classification models
        in sklearn.
    Input:
        model: sklearn classification model class - already fitted
                (with "predict" and "predict_proba" method)
        dim_red_method: sklearn dimension reduction model
                (with "fit_transform" and "inverse_transform" method)
        xrg (list/tuple): xrange
        yrg (list/tuple): yrange
        Nx (int): x axis grid size
        Ny (int): y axis grid size
        X (nparray): dataset to project over decision boundary (X)
        Y (nparray): dataset to project over decision boundary (Y)
        figsize, alpha are parameters in matplotlib
    Output:
        matplotlib figure object
    '''

    # check model is legit to use
    try:
        getattr(model, 'predict')
    except:
        print("model do not have method predict 'predict' ")
        return None

    use_prob = True
    try:
        getattr(model, 'predict_proba')
    except:
        print("model do not have method predict 'predict_proba' ")
        use_prob = False

    # convert X into 2D data
    ss, dr_model = None, None
    if X is not None:
        if X.shape[1] == 2:
            X2D = X
        elif X.shape[1] > 2:
            # leverage PCA to dimension reduction to 2D if not already
            ss = StandardScaler()
            if dim_red_method == 'pca':
                dr_model = PCA(n_components=2)
            elif dim_red_method == 'kernal_pca':
                dr_model = KernelPCA(n_components=2,
                                     fit_inverse_transform=True)
            else:
                print('dim_red_method {0} is not supported'.format(
                    dim_red_method))

            X2D = dr_model.fit_transform(ss.fit_transform(X))
        else:
            print('X dimension is strange: {0}'.format(X.shape))
            return None

        # extract two dimension info.
        x1 = X2D[:, 0].min() - 0.1 * (X2D[:, 0].max() - X2D[:, 0].min())
        x2 = X2D[:, 0].max() + 0.1 * (X2D[:, 0].max() - X2D[:, 0].min())
        y1 = X2D[:, 1].min() - 0.1 * (X2D[:, 1].max() - X2D[:, 1].min())
        y2 = X2D[:, 1].max() + 0.1 * (X2D[:, 1].max() - X2D[:, 1].min())
    else:
        raise ValueError("X is None")

    # inti xrg and yrg based on given value
    if xrg is None:
        if X is None:
            xrg = [-10, 10]
        else:
            xrg = [x1, x2]

    if yrg is None:
        if X is None:
            yrg = [-10, 10]
        else:
            yrg = [y1, y2]

    # generate grid, mesh, and X for model prediction
    xgrid = np.arange(xrg[0], xrg[1], 1. * (xrg[1] - xrg[0]) / Nx)
    ygrid = np.arange(yrg[0], yrg[1], 1. * (yrg[1] - yrg[0]) / Ny)

    xx, yy = np.meshgrid(xgrid, ygrid)
    X_full_grid = np.array(list(zip(np.ravel(xx), np.ravel(yy))))

    # initialize figure & axes object
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    # get data from model predictions
    if dr_model is None:
        Yp = model.predict(X_full_grid)
        if use_prob:
            Ypp = model.predict_proba(X_full_grid)
        else:
            Ypp = pd.get_dummies(Yp).values
    else:
        X_full_grid_inverse = ss.inverse_transform(
            dr_model.inverse_transform(X_full_grid))

        Yp = model.predict(X_full_grid_inverse)
        if use_prob:
            Ypp = model.predict_proba(X_full_grid_inverse)
        else:
            Ypp = pd.get_dummies(Yp).values

    # retrieve n class from util function
    nclass = Ypp.shape[1]
    colors = np.array(retrieve_n_class_color_cubic(N=nclass))

    # get decision boundary line
    Yp = Yp.reshape(xx.shape)
    Yb = np.zeros(xx.shape)

    Yb[:-1, :] = np.maximum((Yp[:-1, :] != Yp[1:, :]), Yb[:-1, :])
    Yb[1:, :] = np.maximum((Yp[:-1, :] != Yp[1:, :]), Yb[1:, :])
    Yb[:, :-1] = np.maximum((Yp[:, :-1] != Yp[:, 1:]), Yb[:, :-1])
    Yb[:, 1:] = np.maximum((Yp[:, :-1] != Yp[:, 1:]), Yb[:, 1:])

    # plot decision boundary first
    ax.imshow(Yb, origin='lower', interpolation=None, cmap='Greys',
              extent=[xrg[0], xrg[1], yrg[0], yrg[1]],
              alpha=1.0)

    # plot probability surface
    zz = np.dot(Ypp, colors[:nclass, :])
    zz_r = zz.reshape(xx.shape[0], xx.shape[1], 3)
    ax.imshow(zz_r, origin='lower', interpolation=None,
              extent=[xrg[0], xrg[1], yrg[0], yrg[1]],
              alpha=alpha)

    # add scatter plot for X & Y if given
    if X is not None:
        # down sample point if needed
        if Y is not None:
            if scatter_sample is not None:
                X2DS, _, YS, _ = train_test_split(X2D, Y, stratify=Y,
                                                  train_size=scatter_sample,
                                                  random_state=random_state)
            else:
                X2DS = X2D
                YS = Y
        else:
            if scatter_sample is not None:
                X2DS, _ = train_test_split(X2D, train_size=scatter_sample,
                                           random_state=random_state)
            else:
                X2DS = X2D

        # convert Y into point color
        if Y is not None:
            # presume Y is labeled from 0 to N-1
            cYS = [colors[i] for i in YS]

        if Y is not None:
            ax.scatter(X2DS[:, 0], X2DS[:, 1], c=cYS)
        else:
            ax.scatter(X2DS[:, 0], X2DS[:, 1])

    # add legend on each class
    colors_bar = []
    for v1 in colors[:nclass, :]:
        v1 = list(v1)
        v1.append(alpha)
        colors_bar.append(v1)

    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors_bar[i],
                              label="Class {k}".format(k=i))
               for i in range(nclass)]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1),
               loc=2, borderaxespad=0., framealpha=0.5)

    # make the figure nicer
    ax.set_title('Classification decision boundary')
    if dr_model is None:
        ax.set_xlabel('Raw axis X')
        ax.set_ylabel('Raw axis Y')
    else:
        ax.set_xlabel('Dimension reduced axis 1')
        ax.set_ylabel('Dimension reduced axis 2')
    ax.set_xlim(xrg)
    ax.set_ylim(yrg)
    ax.set_xticks(np.arange(xrg[0], xrg[1], (xrg[1] - xrg[0])/5.))
    ax.set_yticks(np.arange(yrg[0], yrg[1], (yrg[1] - yrg[0])/5.))
    ax.grid(True)

    return fig, ss, dr_model