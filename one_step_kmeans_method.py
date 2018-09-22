import numpy as np
from sklearn.cluster import KMeans
from sinkhorn import *
from generate_data import *
import time

def one_step_kmeans(y,z,num_centroids,opts):
    '''
    Solving kmeans and mirror descent separately.
    :param y: The grid data.                                    [n,m] array
    :param z: The sample data.                                  [t,m] array
    :param num_centroids: The num of centroids.                 [ 1 ] integer
    :return:
           A dictionary. The keys are x,r,t,method, and time.   [ 5 ] list
    '''
    start = time.time()
    x,r = get_x(z,num_centroids)
    t = get_t(y,x,r,opts)

    return {'x':x,"r":r,"t":t,'method':'Ones step kmeans','time':time.time()-start}

def get_x(data, num_centroids):
    '''
    Using kmeans to get x.
    :param data: The sample data.                              [t,m] array
    :param num_centroids: The num of centroids.                [ 1 ] integer
    :return:
           The coordinates of centrids.                        [k,m] array
           The label of sample data.                           [t,k] array
    '''
    num_data = np.size(data, 0)
    estimator = KMeans(n_clusters=num_centroids)
    estimator.fit(data)
    x = estimator.cluster_centers_
    y_indices = estimator.labels_
    r = np.zeros([num_data, num_centroids])
    x_indices = np.linspace(0, num_data - 1, num_data).astype(int)
    r[x_indices, y_indices] = 1
    return x, r


def mirror_T(t, y, x, r, c, opts):
    '''
    A function to update T using mirror descent and Sinkhorn.
    :param y: The data of grids.                                                             [n,d] matrix
    :param x: The data of centroids.                                                         [m,d] matrix
    :param r: The row sum.                                                                   [ n ] matrix
    :param c: The col sum.                                                                   [ m ] matrix
    :param opts: The parameter including max iteration, step size, warm_start parameter.     [ 3 ] list
    :return:
           T: The new T.                                                                     [n,m] matrix
    '''

    yxt = y.dot(x.T)
    xxt = x.dot(x.T)

    gradient = lambda t: 2 * (t.dot(xxt) - yxt)
    print(gradient(t))
    max_iter = opts[0]
    stepsize = opts[1]
    lv = opts[2]

    for i in range(max_iter):
        z = gradient(t)
        # print(np.mean(z))
        stepsize = stepsize * lv
        K = np.multiply(np.exp(-z*stepsize),np.power(t,stepsize))
        t = approx_sinkhorn(r,c,K)

    return t


def get_t(y, x, r,opts):
    '''
    Using mirror descent to get t.
    :param y: The grid data.                              [n,m] array
    :param x: The coordinates of centroids.               [k,m] array
    :param r: The label of sample data.                   [t,k] array
    :return:
           t: The linear transform between y and x.       [n,k] array
    '''
    num_grids = np.size(y, 0)
    num_centroids = np.size(x, 0)
    t = np.ones([num_grids, num_centroids])
    r = np.sum(r, 0) / np.sum(r)*num_grids
    c = np.ones(num_grids)
    print(r,c)
    opts1 = [50,opts[0],(1/1000)*(1/1000)]
    t = mirror_T(t,y,x,c,r,opts1)

    return t



# num_x = 10
# num_y = 100
# num_z = 1000
# num_pixel = 100
# size_grid = 10
# dimension = 10
#
# min_dist_x = 0
# variance_x = 100
# variance_yz = 100
#
# x = generate_centroids(num_x,dimension,variance_x,min_dist_x)
# x_2 = generate_2centroids(size_grid,num_x)
# y,t,z,r,distribution_x = generate_grid_samples(x,x_2,num_pixel,num_z,size_grid,variance_yz)
#
# opts = [1]
# result_o = one_step_kmeans(y,z,num_x,opts)
# x_o = result_o['x']
# t_o = result_o['t']
# r_o = result_o['r']