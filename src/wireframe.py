import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # X, Y, Z = axes3d.get_test_data(0.05)
    # ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    xs = np.linspace(-4, 4, 21)
    ys = np.linspace(-4, 4, 21)
    V = v.reshape(2, 2)
    Z = pol.polynomial.polygrid2d(xs, ys, V)
    X, Y = np.meshgrid(xs, ys)
    fig = plt.figure()
    axs = Axes3D(fig)
    axs.plot_wireframe(X, Y, Z)
    axs.scatter(x, y, z)

    plt.show()
