# import numpy
import numpy as np
# import numpy’s linear algebra module
import numpy.linalg as la
# import numpy’s random module
import numpy.random as rnd
# import numpy’s polynomial module
import numpy.polynomial as pol
# import matplotlib (pyplot)
import matplotlib.pyplot as plt
# import Axes3D for 3D plotting
from mpl_toolkits.mplot3d import Axes3D


def __create_data_3D(n, v, p, q, xmin, xmax, ymin, ymax):
    x = rnd.random(n) * (xmax-xmin) + xmin
    y = rnd.random(n) * (ymax-ymin) + ymin
    Y = pol.polynomial.polyvander2d(x, y, [p, q])
    z = np.dot(Y, v) + rnd.randn(n) * 0.5
    return x, y, z

if __name__ == '__main__':
    v = np.array([1., 0.5, 0.5, -2.])
    # v = np.array([1., 0.5, 6.0, 0.5, -2., 3.0, 3, 1, 2])
    x, y, z = __create_data_3D(100,v,1,1,-2,2,-2,2)

    Y = pol.polynomial.polyvander2d(x, y, [1, 1])
    V_approximated = la.lstsq(Y, z)[0]

    approximated_coefficients = np.matmul(np.matmul(np.linalg.inv(np.matmul(Y.transpose(), Y)), Y.transpose()), z)  # Calculate approximated least squares coefficients using the
    # Moore-Penrose-Inverse, i.e. Ax = b --> x = (X^t * X)^-1 * X^t * y


    print(V_approximated)





































