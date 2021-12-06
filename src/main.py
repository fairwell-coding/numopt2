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
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import inv
from matplotlib.backends.backend_pdf import PdfPages
from typing import Callable


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

    plot_1a(ax)
    plot_1b(ax)
    plot_1c(ax)

    """ End of your code
    """
    return fig


def plot_1c(ax):
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3
    x1, x2 = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
    ax[2].contourf(x1, x2, (x1 - 1)**2 + x1 * x2**2 - 2, 100, cmap='gist_rainbow')
    ax[2].add_patch(Circle((0, 0), radius=2, color='purple', fill=False))  # inequality constraint: x1**2 + x2**2 <= 4

    ax[2].plot(-0.55, 1.92, color='green', marker="x", markersize=8)
    ax[2].annotate("S1(-0.55|1.92)", (-0.55 + 0.2, 1.92 + 0.2), color='green')

    ax[2].plot(-0.55, -1.92, color='green', marker="x", markersize=8)
    ax[2].annotate("S2(-0.55|-1.92)", (-0.55 - 0.3, -1.92 - 0.3), color='green')

    ax[2].plot(0, sqrt(2), color='green', marker="x", markersize=8)
    ax[2].annotate("S3(0|sqrt(2))", (0 + 0.2, sqrt(2) + 0.2), color='green')

    ax[2].plot(0, -sqrt(2), color='green', marker="x", markersize=8)
    ax[2].annotate("S4(0|-sqrt(2))", (0 + 0.2, -sqrt(2) + 0.2), color='green')

    ax[2].plot(1, 0, color='blue', marker="x", markersize=8)
    ax[2].annotate("S5(1|0)", (1 + 0.2, 0 + 0.2), color='blue')

    # TODO: add labels for constraints


def plot_1b(ax):
    x_min = -2
    x_max = 10
    y_min = -7
    y_max = 7
    x1, x2 = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
    ax[1].contourf(x1, x2, x1**2 + x2**2, 100, cmap='gist_rainbow')
    ax[1].plot(np.linspace(x_min, x_max), 3 - np.linspace(x_min, x_max), color='purple')  # inequality constraint
    ax[1].plot(np.linspace(x_min, x_max), np.linspace(2, 2), color='purple')  # inequality constraint

    # optimal solution: S1(1|2)
    ax[1].plot(1, 2, color='blue', marker="x", markersize=8)
    ax[1].annotate("S1(1|2)", (1 + 0.2, 2 + 0.2), color='blue')

    # TODO: add labels for constraints


def plot_1a(ax):
    x_min = -2
    x_max = 10
    y_min = -5
    y_max = 7
    x1, x2 = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
    ax[0].contourf(x1, x2, x2 - x1, 100, cmap='gist_rainbow')
    ax[0].plot(np.linspace(x_min, x_max), 1 / 10 * np.linspace(x_min, x_max) ** 2 - 3, color='blue')  # equality constraint
    ax[0].plot(np.linspace(x_min, x_max), 1 / 4 * np.linspace(x_min, x_max), color='purple')  # inequality constraint

    # optimal solution: S1(5|-1/2)
    ax[0].plot(5, -1/2, color='blue', marker="x", markersize=8)
    ax[0].annotate("S1(5, -1/2)", (5 + 0.2, -1/2 + 0.2), color='blue')

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
        ax.plot(xk[0], xk[1], color='red', marker="o", markersize=8)
        ax.annotate("S1({0}|{1})".format(xk[0], xk[1]), (xk[0] + 0.2, xk[1] + 0.2), color='red')


def __calculate_augmented_lagrangian_iteratively():
    lambda_k = 1.5  # initialize lambda_k value with reasonable value
    alpha = 0.51  # choose alpha > 0.5

    k = 0
    xks = []
    lambdas = []

    while k < 20:
        x1 = lambda_k
        x2 = 3 * lambda_k - 2
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
    ax.plot(np.linspace(x_min, x_max), -np.linspace(x_min, x_max) + 4, color='blue')  # equality constraint

    ax.plot(3 / 2, 5 / 2, color='blue', marker="x", markersize=8)
    ax.annotate("S1(3/2|5/2)", (3 / 2 + 0.2, 5 / 2 + 0.2), color='blue')  # optimal solution


def task3():

    """ Least Squares Fitting
        ax 3D scatter plot and wireframe of the computed solution
    """
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle('Task 3 - Data points vs. LS solution', fontsize=16)

    with np.load('data.npz') as fc:
        x = fc['data'][:,0]
        y = fc['data'][:,1]
        z = fc['data'][:,2]
        print('x')

    N = len(x)
    A = None
    # x_solutions = None
    """ Start of your codey
    """

    ax.scatter(x, y, z)
    ax.set_xlabel('Xn')
    ax.set_ylabel('Yn')
    ax.set_zlabel('Zn')


    """ End of your code
    """
    return fig, A, x


if __name__ == '__main__':
    # tasks = [task1, task2, task3]
    tasks = [task1, task2]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        retval = task()
        fig = retval[0] if type(retval) is tuple else retval
        pdf.savefig(fig)
    pdf.close()
