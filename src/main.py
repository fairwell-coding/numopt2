#!/usr/bin/env python

""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
- Prior to your submission, check that the pdf showing your plots is generated.
"""
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages
import numpy.polynomial as pol


def task1():

    """ Lagrange Multiplier Problem

        Requirements for the plots:
            - ax[0] Contour plot for a)
            - ax[1] Contour plot for b)
            - ax[2] Contour plot for c)
    """

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Task 1 - Contour plots + Constraints', fontsize=16)

    ax[0].set_title('a)')
    ax[0].set_xlabel('$x_1$')
    ax[0].set_ylabel('$x_2$')
    ax[0].set_aspect('equal')

    ax[1].set_title('b)')
    ax[1].set_xlabel('$x_1$')
    ax[1].set_ylabel('$x_2$')
    ax[1].set_aspect('equal')

    ax[2].set_title('c)')
    ax[2].set_xlabel('$x_1$')
    ax[2].set_ylabel('$x_2$')
    ax[2].set_aspect('equal')


    """ Start of your code
    """

    __plot_1a(ax)
    __plot_1b(ax)
    __plot_1c(ax)

    """ End of your code
    """
    return fig


def __plot_1c(ax):
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3
    x1, x2 = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
    ax[2].contourf(x1, x2, (x1 - 1)**2 + x1 * x2**2 - 2, 100, cmap='gist_rainbow')
    ax[2].add_patch(Circle((0, 0), radius=2, color='yellow', fill=False))  # inequality constraint: x1**2 + x2**2 <= 4

    ax[2].plot(-0.55, 1.92, color='green', marker="x", markersize=8)
    ax[2].annotate("S1(-0.55|1.92)", (-0.55 - 0.8, 1.92 + 0.3), color='green')

    ax[2].plot(-0.55, -1.92, color='green', marker="x", markersize=8)
    ax[2].annotate("S2(-0.55|-1.92)", (-0.55 - 0.8, -1.92 - 0.4), color='green')

    ax[2].plot(0, sqrt(2), color='green', marker="x", markersize=8)
    ax[2].annotate("S3(0|sqrt(2))", (0 - 0.6, sqrt(2) - 0.4), color='green')

    ax[2].plot(0, -sqrt(2), color='green', marker="x", markersize=8)
    ax[2].annotate("S4(0|-sqrt(2))", (0 - 0.6, -sqrt(2) + 0.3), color='green')

    ax[2].plot(1, 0, color='blue', marker="x", markersize=8)
    ax[2].annotate("S5(1|0)", (1 - 0.3, 0 + 0.2), color='blue')


def __plot_1b(ax):
    x_min = -2
    x_max = 10
    y_min = -7
    y_max = 7
    x1, x2 = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
    ax[1].contourf(x1, x2, x1**2 + x2**2, 100, cmap='gist_rainbow')
    ax[1].plot(np.linspace(x_min, x_max), 3 - np.linspace(x_min, x_max), color='yellow')  # inequality constraint
    ax[1].plot(np.linspace(x_min, x_max), np.linspace(2, 2), color='yellow')  # inequality constraint

    # optimal solution: S1(1|2)
    ax[1].plot(1, 2, color='blue', marker="x", markersize=8)
    ax[1].annotate("S1(1|2)", (1 + 0.2, 2 + 0.4), color='blue')


def __plot_1a(ax):
    x_min = -2
    x_max = 10
    y_min = -5
    y_max = 7
    x1, x2 = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
    ax[0].contourf(x1, x2, x2 - x1, 100, cmap='gist_rainbow')
    ax[0].plot(np.linspace(x_min, x_max), 1 / 10 * np.linspace(x_min, x_max) ** 2 - 3, color='red')  # equality constraint
    ax[0].plot(np.linspace(x_min, x_max), 1 / 4 * np.linspace(x_min, x_max), color='yellow')  # inequality constraint

    # optimal solution: S1(5|-1/2)
    ax[0].plot(5, -1/2, color='blue', marker="x", markersize=8)
    ax[0].annotate("S1(5, -1/2)", (5 + 0.8, -1/2 + 0), color='blue')

    # TODO: add labels for constraints


def task2():

    """ Lagrange Augmentation
        ax Filled contour plot including the constraints and the iterates of x_k

    """
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    fig.suptitle('Task 2 - Contour plots + Constraints + Iterations over k', fontsize=16)
    """ Start of your code
    """

    __plot_task2(ax)
    xks = __calculate_augmented_lagrangian_iteratively()
    __plot_iteratively_approximated_augmented_lagrangian(ax, xks)

    """ End of your code
    """
    return fig


def __plot_iteratively_approximated_augmented_lagrangian(ax, xks):
    for xk in xks:  # iteratively approximated points xk = (x1, x2)
        ax.plot(xk[0], xk[1], color='green', marker="o", markersize=4)
    ax.annotate("Iteratively calculated solutions", (3 / 2 - 1, 5 / 2 + 1.3), color='green')


def __calculate_augmented_lagrangian_iteratively():
    lambda_k = 0  # initialize lambda_k value with reasonable value
    alpha = 0.6  # choose alpha > 0.5

    k = 0
    xks = []
    lambdas = []

    while k < 20:
        x1 = (12 * alpha - 2 * lambda_k + 6 * alpha**2 - alpha * lambda_k) / ((alpha + 2) * (4 * alpha - 1))
        x2 = (3 * lambda_k - 10 * alpha - 2) / (1 - 4 * alpha)
        lambda_k = alpha * (x1 + x2 - 4) + lambda_k
        xks.append((x1, x2))
        lambdas.append(lambda_k)
        k += 1

    return xks


def __plot_task2(ax):
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 7
    x1, x2 = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
    ax.contourf(x1, x2, (x1 - 1) ** 2 - x1 * x2, 100, cmap='gist_rainbow')
    ax.plot(np.linspace(x_min, x_max), -np.linspace(x_min, x_max) + 4, color='red')  # equality constraint

    ax.plot(3 / 2, 5 / 2, color='blue', marker="x", markersize=15)
    ax.plot(3 / 2, 5 / 2, color='blue', marker="o", markersize=8)
    ax.annotate("S1(3/2|5/2)", (3 / 2 + 0.2, 5 / 2 + 0.2), color='blue')  # optimal solution


def task3():

    """ Least Squares Fitting
        ax 3D scatter plot and wireframe of the computed solution
    """
    fig = plt.figure(figsize=(8,24))
    fig.suptitle('Task 3 - Data points vs. LS solution', fontsize=16)

    with np.load('data.npz') as fc:
        x = fc['data'][:,0]
        y = fc['data'][:,1]
        z = fc['data'][:,2]

    """ Start of your codey
    """

    ax = __plot_data_3dscatter(fig, x, y, z)
    A, approximated_coefficients, z_approximated = __calculate_and_plot_bivariate_nonlinear_polynomial_function_using_leastsquares(ax, x, y, z)
    __plot_xz_projection(fig, x, z, z_approximated)
    __plot_yz_projection(fig, y, z, z_approximated)

    """ End of your code
    """
    return fig, A, approximated_coefficients


def __plot_yz_projection(fig, y, z, z_approximated):
    """ Plot (y,z)-projection: allows us to evaluate the relation between y and z, i.e. to estimate the polynomial dimension of x needed to approximate the corresponding z value based on y
    """

    ax = fig.add_subplot(313)
    ax.plot(y, z, 'bo')  # original data
    ax.plot(y, z_approximated, 'go')  # approximated data by least squares
    ax.title.set_text('(y,z) data projection')
    ax.set_xlabel('y coordinates')
    ax.set_ylabel('z coordinates')
    return fig


def __plot_xz_projection(fig, x, z, z_approximated):
    """ Plot (x,z)-projection: allows us to evaluate the relation between x and z, i.e. to estimate the polynomial dimension of x needed to approximate the corresponding z value based on x
    """

    ax = fig.add_subplot(312)
    ax.plot(x, z, 'bo')  # original data
    ax.plot(x, z_approximated, 'go')  # approximated data by least squares
    ax.title.set_text('(x,z) data projection')
    ax.set_xlabel('x coordinates')
    ax.set_ylabel('z coordinates')


def __calculate_and_plot_bivariate_nonlinear_polynomial_function_using_leastsquares(ax, x, y, z):
    x_dim = 3
    y_dim = 2
    A = pol.polynomial.polyvander2d(x, y, [x_dim, y_dim])
    approximated_coefficients = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.transpose(), A)), A.transpose()), z)  # Calculate approximated least squares coefficients using the
    # Moore-Penrose-Inverse, i.e. Ax = b --> x = (X^t * X)^-1 * X^t * y

    V = approximated_coefficients.reshape((x_dim + 1, y_dim + 1))  # reshape the flattened structure of the flattened Vandermonde matrix into the two different dimensions for x and y
    z_approximated = np.matmul(A, approximated_coefficients)  # calculate bivariate polnoymially estimated z values based on x, y
    mse = np.square(np.subtract(z, z_approximated)).mean()  # manually calculate MSE (mean squared error) in order to numerically evaluate in addition to plotting what dimensions should be used

    xs = np.linspace(-4, 4)
    ys = np.linspace(-4, 4)
    X, Y = np.meshgrid(xs, ys)
    Z = pol.polynomial.polygrid2d(xs, ys, V)
    ax.plot_wireframe(X, Y, Z, color='green')  # add wireframe of approximated data to existing 3d scatter plot of original data

    return A, approximated_coefficients, z_approximated


def __plot_data_3dscatter(fig, x, y, z):
    ax = fig.add_subplot(311, projection='3d')

    ax.scatter(x, y, z)
    ax.title.set_text('3d data points (blue) vs their NLS approximated bivariate polynomial data points')
    ax.set_xlabel('Xn')
    ax.set_ylabel('Yn')
    ax.set_zlabel('Zn')

    return ax


if __name__ == '__main__':
    tasks = [task1, task2, task3]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        retval = task()
        fig = retval[0] if type(retval) is tuple else retval
        pdf.savefig(fig)
    pdf.close()
