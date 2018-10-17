from numpy import arange, meshgrid, ravel, array
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


def evaluate_surface(f, x_range, y_range):
    """
    (Float -> Float -> Float) -> (Float, Float, Float)
        -> (Float, Float, Float) -> (ndarray, ndarray, ndarray)
    Evaluate the given function at every 2D coordinate that can be
    made by combining the values passed to the x and y range parameters.
    The input generator should take the coordinate pair and generate
    the input that will be passed to the function.
    """
    x = arange(x_range[0], x_range[1], x_range[2])
    y = arange(y_range[0], y_range[1], y_range[2])
    X, Y = meshgrid(x, y)
    z = array([f(x, y) for x, y in zip(ravel(X), ravel(Y))])
    Z = z.reshape(X.shape)
    return X, Y, Z


def make_3d_axes():
    """
    () -> Axes3D
    Create a default Axes3D object.
    """
    figure = plt.figure()
    return figure.add_subplot(111, projection='3d')


def plot_surface(xyz_ndarrays, x_label='x', y_label='y', z_label='z'):
    """
    (ndarray, ndarray, ndarray) -> String? -> String? -> String? -> ()
    Plot the surface, described by the z-values at a series of evenly-
    spaced points in the x-y plane, in three dimensions.
    """
    axes = make_3d_axes()
    axes.plot_surface(xyz_ndarrays[0], xyz_ndarrays[1], xyz_ndarrays[2])

    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_zlabel(z_label)

    plt.show()
