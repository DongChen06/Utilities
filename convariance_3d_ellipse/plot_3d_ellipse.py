from __future__ import absolute_import, division, unicode_literals

import numpy as np
import math
import os
from numpy.linalg import inv
import matplotlib.pyplot as plt
from math import cos, sin
import scipy.linalg as linalg
from scipy.stats import norm
from os import walk


def covariance_ellipse(P, deviations=1):
    """
    Returns a tuple defining the ellipse representing the 2 dimensional
    covariance matrix P.
    Parameters
    ----------
    P : nd.array shape (2,2)
       covariance matrix
    deviations : int (optional, default = 1)
       # of standard deviations. Default is 1.
    Returns (angle_radians, width_radius, height_radius)
    """

    U, s, _ = linalg.svd(P)
    orientation = math.atan2(U[1, 0], U[0, 0])
    width = deviations * math.sqrt(s[0])
    height = deviations * math.sqrt(s[1])

    if height > width:
        raise ValueError('width must be greater than height')

    return (orientation, width, height)


def _std_tuple_of(var=None, std=None, interval=None):
    """
    Convienence function for plotting. Given one of var, standard
    deviation, or interval, return the std. Any of the three can be an
    iterable list.
    Examples
    --------
    >>>_std_tuple_of(var=[1, 3, 9])
    (1, 2, 3)
    """

    if std is not None:
        if np.isscalar(std):
            std = (std,)
        return std

    if interval is not None:
        if np.isscalar(interval):
            interval = (interval,)

        return norm.interval(interval)[1]

    if var is None:
        raise ValueError("no inputs were provided")

    if np.isscalar(var):
        var = (var,)
    return np.sqrt(var)


def _eigsorted(cov, asc=True):
    """
    Computes eigenvalues and eigenvectors of a covariance matrix and returns
    them sorted by eigenvalue.
    Parameters
    ----------
    cov : ndarray
        covariance matrix
    asc : bool, default=True
        determines whether we are sorted smallest to largest (asc=True),
        or largest to smallest (asc=False)
    Returns
    -------
    eigval : 1D ndarray
        eigenvalues of covariance ordered largest to smallest
    eigvec : 2D ndarray
        eigenvectors of covariance matrix ordered to match `eigval` ordering.
        I.e eigvec[:, 0] is the rotation vector for eigval[0]
    """
    eigval, eigvec = np.linalg.eigh(cov)
    order = eigval.argsort()
    if not asc:
        # sort largest to smallest
        order = order[::-1]

    return eigval[order], eigvec[:, order]


def plot_covariance(
        mean, cov=None, variance=1.0, std=None, enlarge_factor=5, interval=None,
        ellipse=None, title=None, axis_equal=True,
        show_semiaxis=False, show_center=False,
        facecolor=None, edgecolor=None,
        fc='none', ec='#004080',
        alpha=1.0, xlim=None, ylim=None,
        ls='solid'):
    """
    Plots the covariance ellipse for the 2D normal defined by (mean, cov)
    `variance` is the normal sigma^2 that we want to plot. If list-like,
    ellipses for all ellipses will be ploted. E.g. [1,2] will plot the
    sigma^2 = 1 and sigma^2 = 2 ellipses. Alternatively, use std for the
    standard deviation, in which case `variance` will be ignored.
    ellipse is a (angle,width,height) tuple containing the angle in radians,
    and width and height radii.
    You may provide either cov or ellipse, but not both.
    Parameters
    ----------
    mean : row vector like (2x1)
        The mean of the normal
    cov : ndarray-like
        2x2 covariance matrix
    variance : float, default 1, or iterable float, optional
        Variance of the plotted ellipse. May specify std or interval instead.
        If iterable, such as (1, 2**2, 3**2), then ellipses will be drawn
        for all in the list.
    std : float, or iterable float, optional
        Standard deviation of the plotted ellipse. If specified, variance
        is ignored, and interval must be `None`.
        If iterable, such as (1, 2, 3), then ellipses will be drawn
        for all in the list.
    interval : float range [0,1), or iterable float, optional
        Confidence interval for the plotted ellipse. For example, .68 (for
        68%) gives roughly 1 standand deviation. If specified, variance
        is ignored and `std` must be `None`
        If iterable, such as (.68, .95), then ellipses will be drawn
        for all in the list.
    ellipse: (float, float, float)
        Instead of a covariance, plots an ellipse described by (angle, width,
        height), where angle is in radians, and the width and height are the
        minor and major sub-axis radii. `cov` must be `None`.
    title: str, optional
        title for the plot
    axis_equal: bool, default=True
        Use the same scale for the x-axis and y-axis to ensure the aspect
        ratio is correct.
    show_semiaxis: bool, default=False
        Draw the semiaxis of the ellipse
    show_center: bool, default=True
        Mark the center of the ellipse with a cross
    facecolor, fc: color, default=None
        If specified, fills the ellipse with the specified color. `fc` is an
        allowed abbreviation
    edgecolor, ec: color, default=None
        If specified, overrides the default color sequence for the edge color
        of the ellipse. `ec` is an allowed abbreviation
    alpha: float range [0,1], default=1.
        alpha value for the ellipse
    xlim: float or (float,float), default=None
       specifies the limits for the x-axis
    ylim: float or (float,float), default=None
       specifies the limits for the y-axis
    ls: str, default='solid':
        line style for the edge of the ellipse
    """

    from matplotlib.patches import Ellipse

    if cov is not None and ellipse is not None:
        raise ValueError('You cannot specify both cov and ellipse')

    if cov is None and ellipse is None:
        raise ValueError('Specify one of cov or ellipse')

    if facecolor is None:
        facecolor = fc

    if edgecolor is None:
        edgecolor = ec

    if cov is not None:
        ellipse = covariance_ellipse(cov)

    if axis_equal:
        plt.axis('equal')

    if title is not None:
        plt.title(title)

    ax = plt.gca()

    angle = np.degrees(ellipse[0])
    width = ellipse[1] * 2. * enlarge_factor
    height = ellipse[2] * 2. * enlarge_factor

    std = _std_tuple_of(variance, std, interval)
    for sd in std:
        e = Ellipse(xy=mean, width=sd * width, height=sd * height, angle=angle,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                    lw=2, ls=ls)
        ax.add_patch(e)
    x, y = mean
    if show_center:
        plt.scatter(x, y, marker='+', color=edgecolor)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if show_semiaxis:
        a = ellipse[0]
        h, w = height / 4, width / 4
        plt.plot([x, x + h * cos(a + np.pi / 2)],
                 [y, y + h * sin(a + np.pi / 2)])
        plt.plot([x, x + w * cos(a)], [y, y + w * sin(a)])


def plot_3d_covariance(mean, cov, std=1.,
                       ax=None, enlarge_factor=5, title=None,
                       color=None, alpha=1.,
                       label_xyz=True,
                       N=60,
                       shade=True,
                       limit_xyz=True,
                       **kwargs):
    """
    Plots a covariance matrix `cov` as a 3D ellipsoid centered around
    the `mean`.
    Parameters
    ----------
    mean : 3-vector
        mean in x, y, z. Can be any type convertable to a row vector.
    cov : ndarray 3x3
        covariance matrix
    std : double, default=1
        standard deviation of ellipsoid
    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        Axis to draw on. If not provided, a new 3d axis will be generated
        for the current figure
    title : str, optional
        If provided, specifies the title for the plot
    color : any value convertible to a color
        if specified, color of the ellipsoid.
    alpha : float, default 1.
        Alpha value of the ellipsoid. <1 makes is semi-transparent.
    label_xyz: bool, default True
        Gives labels 'X', 'Y', and 'Z' to the axis.
    N : int, default=60
        Number of segments to compute ellipsoid in u,v space. Large numbers
        can take a very long time to plot. Default looks nice.
    shade : bool, default=True
        Use shading to draw the ellipse
    limit_xyz : bool, default=True
        Limit the axis range to fit the ellipse
    **kwargs : optional
        keyword arguments to supply to the call to plot_surface()
    """

    # force mean to be a 1d vector no matter its shape when passed in
    mean = np.atleast_2d(mean)
    if mean.shape[1] == 1:
        mean = mean.T

    if not (mean.shape[0] == 1 and mean.shape[1] == 3):
        raise ValueError('mean must be convertible to a 1x3 row vector')
    mean = mean[0]

    # force covariance to be 3x3 np.array
    cov = np.asarray(cov)
    if cov.shape[0] != 3 or cov.shape[1] != 3:
        raise ValueError("covariance must be 3x3")

    # The idea is simple - find the 3 axis of the covariance matrix
    # by finding the eigenvalues and vectors. The eigenvalues are the
    # radii (squared, since covariance has squared terms), and the
    # eigenvectors give the rotation. So we make an ellipse with the
    # given radii and then rotate it to the proper orientation.

    eigval, eigvec = _eigsorted(cov, asc=True)
    # [0.00474644 0.00571878 0.06257601]
    radii = std * np.sqrt(np.real(eigval)) * enlarge_factor

    if eigval[0] < 0:
        raise ValueError("covariance matrix must be positive definite")

    # calculate cartesian coordinates for the ellipsoid surface
    u = np.linspace(0.0, 2.0 * np.pi, N)
    v = np.linspace(0.0, np.pi, N)
    x = np.outer(np.cos(u), np.sin(v)) * radii[0]
    y = np.outer(np.sin(u), np.sin(v)) * radii[1]
    z = np.outer(np.ones_like(u), np.cos(v)) * radii[2]

    # rotate data with eigenvector and center on mu
    a = np.kron(eigvec[:, 0], x)
    b = np.kron(eigvec[:, 1], y)
    c = np.kron(eigvec[:, 2], z)

    data = a + b + c
    N = data.shape[0]
    x = data[:, 0:N] + mean[0]
    y = data[:, N:N * 2] + mean[1]
    z = data[:, N * 2:] + mean[2]

    return x, y, z, radii


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


if __name__ == "__main__":
    plot_single_ellipse = not True
    # True, False
    plot_camera_3d = True
    plot_lidar_3d = False
    plot_camera_2d = False
    plot_lidar_2d = False

    is_colorful = True
    colors = []
    indexes = []

    if plot_single_ellipse:
        # reference: https://github.com/rlabbe/filterpy/blob/master/filterpy/stats/tests/test_stats.py
        mu = [-1.2586303712416274, -0.025740651401615102, 4.322911357758138]
        Cov = np.array([[3.27432355e-04, 4.68901366e-06, -1.04597828e-03],
                        [4.68901366e-06, 3.27774687e-05, -1.61867587e-05],
                        [-1.04597828e-03, -1.61867587e-05, 3.61078026e-03]])

        mu1 = [2.08847725, 0.02990916, 5.14069407]
        Cov1 = np.array([[7.79609828e-04, 1.06958847e-05, 1.83837576e-03],
                         [1.06958847e-05, 4.76082027e-05, 2.64231410e-05],
                         [1.83837576e-03, 2.64231410e-05, 4.54152817e-03]])

        x, y, z, radii = plot_3d_covariance(
            mu, Cov, alpha=.4, std=1, limit_xyz=True)
        x1, y1, z1, radii1 = plot_3d_covariance(
            mu1, Cov1, alpha=.4, std=1, limit_xyz=True)

        fig = plt.gcf()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x, y, z,
                        rstride=3, cstride=3, linewidth=0.1, alpha=0.4,
                        shade=True, color=None)

        ax.plot_surface(x1, y1, z1,
                        rstride=3, cstride=3, linewidth=0.1, alpha=0.4,
                        shade=True, color=None)

        label_xyz = True
        if label_xyz:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        title = None
        if title is not None:
            plt.title(title)
        plt.show()

    elif plot_camera_3d:
        # reference: https://github.com/rlabbe/filterpy/blob/master/filterpy/stats/tests/test_stats.py

        input_dir = 'cov_results_camera/'
        cov_files = []

        for (dirpath, dirnames, filenames) in walk(input_dir):
            for filename in filenames:
                if filename.endswith('cov.npy'):
                    cov_files.append(os.path.join(dirpath, filename))

        fig = plt.gcf()
        ax = fig.add_subplot(111, projection='3d')

        for cov_file in cov_files:
            Cov = np.load(cov_file)
            filename = cov_file.split('/')[-1]
            index = filename.split('_')[0]
            mu = np.load(input_dir + index + '{}'.format('_xyz') + '.npy')
            x, y, z, radii = plot_3d_covariance(
                mu, Cov, alpha=.4, std=1, enlarge_factor=8, limit_xyz=True)
            if is_colorful:
                color = next(ax._get_lines.prop_cycler)['color']
            else:
                color = 'b'
            colors.append(color)
            indexes.append(index)
            ax.plot_surface(x, y, z,
                            rstride=3, cstride=3, linewidth=0.2, alpha=0.4,
                            shade=True, color=color)

        label_xyz = True
        if label_xyz:
            ax.set_xlabel('X/m')
            ax.set_ylabel('Y/m')
            ax.set_zlabel('Z/m')

        set_axes_equal(ax)
        np.save('colors.npy', colors)
        np.save('indexes.npy', indexes)
        title = None
        if title is not None:
            plt.title(title)
        plt.grid(True)
        plt.show()

    elif plot_lidar_3d:
        # reference: https://github.com/rlabbe/filterpy/blob/master/filterpy/stats/tests/test_stats.py

        input_dir = 'cov_results_lidar/'
        cov_files = []

        for (dirpath, dirnames, filenames) in walk(input_dir):
            for filename in filenames:
                if filename.endswith('cov.npy'):
                    cov_files.append(os.path.join(dirpath, filename))

        fig = plt.gcf()
        ax = fig.add_subplot(111, projection='3d')

        for cov_file in cov_files:
            Cov = np.load(cov_file)
            filename = cov_file.split('/')[-1]
            index = filename.split('_')[0]
            colors = np.load('colors' + '.npy')
            indexes = (np.load('indexes' + '.npy')).tolist()
            if is_colorful:
                color = colors[indexes.index(index)]
            else:
                color = 'b'
            mu = np.load(input_dir + index + '{}'.format('_xyz') + '.npy')
            x, y, z, radii = plot_3d_covariance(
                mu, Cov, alpha=.4, std=1, enlarge_factor=60, limit_xyz=True)
            ax.plot_surface(x, y, z,
                            rstride=3, cstride=3, linewidth=0.2, alpha=0.4,
                            shade=True, color=color)

        label_xyz = True
        if label_xyz:
            ax.set_xlabel('X/m')
            ax.set_ylabel('Y/m')
            ax.set_zlabel('Z/m')

        set_axes_equal(ax)
        title = None
        if title is not None:
            plt.title(title)
        plt.show()

    elif plot_camera_2d:
        # reference: https://github.com/rlabbe/filterpy/blob/master/filterpy/stats/tests/test_stats.py

        input_dir = 'cov_results_camera/'
        cov_files = []

        for (dirpath, dirnames, filenames) in walk(input_dir):
            for filename in filenames:
                if filename.endswith('cov.npy'):
                    cov_files.append(os.path.join(dirpath, filename))

        fig = plt.gcf()
        ax = fig.add_subplot(1, 1, 1)

        for cov_file in cov_files:
            Cov = np.load(cov_file)
            Cov = np.delete(Cov, 1, 0)
            Cov = np.delete(Cov, 1, 1)
            filename = cov_file.split('/')[-1]
            index = filename.split('_')[0]
            mu = np.load(input_dir + index + '{}'.format('_xyz') + '.npy')
            mu = np.delete(mu, 1, 0)

            if is_colorful:
                color = next(ax._get_lines.prop_cycler)['color']
            else:
                color = 'b'
            colors.append(color)
            indexes.append(index)
            plot_covariance(mu, Cov, alpha=.7, std=1,
                            enlarge_factor=5, edgecolor=color)

        label_xy = True
        if label_xy:
            ax.set_xlabel('X/m')
            ax.set_ylabel('Z/m')

        np.save('colors.npy', colors)
        np.save('indexes.npy', indexes)
        title = None
        if title is not None:
            plt.title(title)
        plt.grid(True)
        plt.show()

    elif plot_lidar_2d:
        # reference: https://github.com/rlabbe/filterpy/blob/master/filterpy/stats/tests/test_stats.py

        input_dir = 'cov_results_lidar/'
        cov_files = []

        for (dirpath, dirnames, filenames) in walk(input_dir):
            for filename in filenames:
                if filename.endswith('cov.npy'):
                    cov_files.append(os.path.join(dirpath, filename))

        fig = plt.gcf()
        ax = fig.add_subplot(1, 1, 1)

        for cov_file in cov_files:
            Cov = np.load(cov_file)
            Cov = np.delete(Cov, 1, 0)
            Cov = np.delete(Cov, 1, 1)
            filename = cov_file.split('/')[-1]
            index = filename.split('_')[0]
            colors = np.load('colors' + '.npy')
            indexes = (np.load('indexes' + '.npy')).tolist()
            if is_colorful:
                color = colors[indexes.index(index)]
            else:
                color = 'b'
            mu = np.load(input_dir + index + '{}'.format('_xyz') + '.npy')
            mu = np.delete(mu, 1, 0)
            plot_covariance(mu, Cov, alpha=.7, std=1,
                            enlarge_factor=30, edgecolor=color)

        label_xy = True
        if label_xy:
            ax.set_xlabel('X/m')
            ax.set_ylabel('Z/m')

        title = None
        if title is not None:
            plt.title(title)
        plt.grid(True)
        plt.show()

    else:
        # plot 2d
        # reference: https://github.com/rlabbe/filterpy/blob/master/filterpy/stats/tests/test_stats.py
        mu = [-1.2586303712416274, 4.322911357758138]
        Cov = np.array([[3.27432355e-04, -1.04597828e-03],
                        [-1.04597828e-03, 3.61078026e-03]])
        mu1 = [2.08847725, 5.14069407]
        Cov1 = np.array([[7.79609828e-04, 1.83837576e-03],
                         [1.83837576e-03, 4.54152817e-03]])
        fig = plt.gcf()
        plot_covariance(mu, Cov, alpha=.4, std=1)
        plot_covariance(mu1, Cov, alpha=.4, std=1)
        plt.show()
