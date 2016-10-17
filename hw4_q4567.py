"""learning from data - HW4"""
from math import sin
from math import pi

from random import uniform

import numpy as np  # for pseudo-inverse of matrix
import numpy.matlib as ml


class Point(object):
    "Points class"
    x = 0.0
    "attribute Point.x x coordinate (0.0)"
    y = 0.0
    "attribute Point.y y coordinate (0.0)"

    def __init__(self, x=None, y=None):
        if x is None:
            self.x = uniform(-1.0, 1.0)
        else:
            self.x = x
        if y is None:
            self.y = sin(pi*self.x)
        else:
            self.y = y

    def print_object(self):
        print(self.x, self.y)


def generate_point():
    return Point()


def gen_points(number_of_points):
    return [generate_point() for i in range(number_of_points)]


def gen_xmatrix(points, method='xy'):
    count = 0
    if method == 'xy':
        x = np.empty(shape=(len(points), 3))
        for point in points:
            x[count] = [1, point.x, point.y]
            count += 1
    elif method == 'ax':
        x = np.empty(shape=(len(points), 1))
        for point in points:
            x[count] = [point.x]
            count += 1
    elif method == 'ax2':
        x = np.empty(shape=(len(points), 1))
        for point in points:
            x[count] = [(point.x)**2]
            count += 1
    elif method == 'ax+b':
        x = np.empty(shape=(len(points), 2))
        for point in points:
            x[count] = [1, point.x]
            count += 1
    elif method == 'ax2+b':
        x = np.empty(shape=(len(points), 2))
        for point in points:
            x[count] = [1, (point.x)**2]
            count += 1
    elif method == 'b':
        x = np.empty(shape=(len(points), 1))
        for point in points:
            x[count] = [1]
            count += 1
    return x


def gen_yvector(points):
    return [point.y for point in points]


def lr(training_points, method='ax'):
    '''
    http://en.wikipedia.org/wiki/Ordinary_least_squares
    w = ((t(X)X)^-1)t(X)Y
    '''
    x = gen_xmatrix(training_points, method)
    xdagger = np.dot(np.linalg.pinv(np.dot(np.transpose(x), x)),
                     np.transpose(x))
    y = gen_yvector(training_points)  # target_vector, aka y
    w = np.dot(xdagger, y)
    return w


def q4():
    num_train_sets = 1000
    num_test_sets = 1000
    method = 'ax2+b'
    if method in {'ax', 'ax2', 'b'}:
        numDim = 1
    elif method in {'ax+b', 'ax2+b'}:
        numDim = 2
    w_all = np.empty(shape=(numDim, num_train_sets))
    for i in range(num_train_sets):
        pairOfPoints = gen_points(2)
        w = lr(pairOfPoints, method)
        w_all[:, i] = w
    w_mean = w_all.mean(axis=1)
    testPoints = gen_points(num_test_sets)
    x = gen_xmatrix(testPoints, method)
    y = gen_yvector(testPoints)  # target_vector, aka y
    g_bar_x = x.dot(w_mean)
    bias = ((g_bar_x-y)**2).mean()
    var = (np.array(x.dot(w_all)-ml.repmat(
        np.asmatrix(g_bar_x.reshape(num_test_sets, 1)),
        1, num_train_sets))**2).mean()
    print(method)
    print(('bias: %s' % bias))
    print(('var: %s' % var))
    print(('Eout: %s' % (bias+var)))


q4()
