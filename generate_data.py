import numpy as np
from math import *
import numpy.linalg as li
import math


def generate_centroids(num, dimension, variance=1, min_dist=0.0):
    '''
    A function to generate centroids according to normal distribution.
    :param dimension: The dimension we want.                             [ 1 ] integer
    :param num: The number of centroids.                                 [ 1 ] integer
    :param variance: The variance of matrix.                             [ 1 ] real
    :param min_dist: The minimal distance between centroids.             [ 1 ] real
    :return:
           centroids: The coordinate of centroids.                       [m,n] matrix
    '''
    centroids = np.random.randn(num, dimension) * sqrt(variance)

    if min_dist > 0:
        min_ = np.inf

        for i in range(1, num):

            for j in range(i):
                min_ = min(min_, li.norm(centroids[i] - centroids[j], 2))
        centroids = min_dist / min_ * centroids

    return centroids

def generate_2centroids(size,num):
    '''
    A fucntion to generate 2 dimensional coordinates for centroids.
    :param size: The size of grids.                                        [ 1 ] integer
    :param num:  The num of coordinates.                                   [ 1 ] integer
    :return:
           centroids: 2 dimensional coordinates.                           [n,2] array
    '''
    centroids_2 = np.random.randn(num,2)
    centroids_2 = centroids_2 - np.min(centroids_2)
    centroids_2 = centroids_2*(size-1)/np.max(centroids_2)

    return centroids_2

def generate_data(centroids, num, variances=1, merge=False):
    '''
    A function to generate data according to normal distributions.
    :param centroids: The mean of normal distributions.                       [n,m] array
    :param num: The num of each centroids.                                    [ n ] array
    :param variances: The variance of each normal distribution.               [ 1 ] real
    :param merge: A Boolean to decide if we get the mean of data.             [ 1 ] Boolean
    :return:
           data: The data we want.                                            [s,m] array  or [ m ] array
           label: The label of data.                                          [s,n] array
    '''
    dimension = np.size(centroids[0])
    num_centroids = np.size(centroids, 0)
    data = np.zeros([np.sum(num), dimension])
    labels = np.zeros([np.sum(num), num_centroids])
    accu_sum = 0
    for i in range(num_centroids):

        if np.size(variances) != 1:
            variance = variances[i]
        else:
            variance = variances
        noise = np.random.randn(num[i], dimension) * (variance) ** 0.5
        # noise                      = 0
        data[accu_sum: accu_sum + num[i], :] = noise + centroids[i]
        labels[accu_sum:accu_sum + num[i], i] = 1
        accu_sum += num[i]

    if merge == True:
        data = np.mean(data, 0)
        labels = np.mean(labels, 0)
        return data
    else:
        return data, labels


def generate_weight(size, centroids_2):
    '''
    A function to generate weight for grids data.
    :param size: The size of grid.                                                   [ 1 ] integer
    :param centroids_2: The 2 dimensional coordinates of centroids.                  [n,2] array
    :return:
           weight: The weight of centroids for each pixel.                           [m,n] array
    '''
    num_cent = np.size(centroids_2, 0)
    weight = np.zeros([size * size, num_cent])

    cordin = lambda x: np.array([x % size, x // size])

    for i in range(size * size):
        for j in range(num_cent):
            cord = cordin(i)
            # print(cord)
            weight[i, j] = exp(-li.norm(cord - centroids_2[j], 2))

        weight[i] = weight[i] / np.sum(weight[i])

    return weight


def generate_grid_samples(centroids, centroids_2, num1, num2, size, variance=1):
    '''
    A function to generate grids and samples.
    :param centroids: The coordinates of centroids.                           [n,m] array
    :param centroids_2: The 2 dimensional coordinates of centroids.           [n,2] array
    :param num1: The num of cells in each pixel.                              [ 1 ] integer
    :param num2: The num of samples.                                          [ 1 ] integer
    :param size: The size of grids.                                           [ 1 ] integer
    :param variance: The variance of normal distribution.                     [ 1 ] real
    :return:
         grids: The grids data.                                               [k,m] array
         label: The labels for grids data.                                    [k,n] array
         samples: The sample data.                                            [t,m] array
         labels: The labels for sample data.                                  [t,n] array
         distribution_x: The distribution of x.                               [ n ] array
    '''
    dimension = np.size(centroids, 1)
    label = generate_weight(size, centroids_2)
    grids = np.zeros([size * size, dimension])
    weight = (label * num1).astype(int)
    label = weight / np.sum(weight, 1).reshape([size * size, 1])

    for i in range(size * size):
        grids[i] = generate_data(centroids, weight[i], variance, merge=True)

    total_weight = np.sum(weight, 0)
    total_weight2 = (total_weight * num2 / np.sum(total_weight)).astype(int)
    samples, labels = generate_data(centroids, total_weight2, variance)
    distribution_x = total_weight+total_weight2
    distribution_x = distribution_x/np.sum(distribution_x)

    return grids, label, samples, labels,distribution_x

# Test
# centroids =np.array([[1,0,0,0],[0,0,0,1]])
# centroids_2 = np.array([[0,0],[2,2]])
# print(generate_grid_samples(centroids,centroids_2,10,1000,10,variance=0))