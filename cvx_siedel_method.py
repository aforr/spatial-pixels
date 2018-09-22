import cvxpy as cvx
import numpy as np
from one_step_kmeans_method import *

def get_val(x,y,z,t,r):
    '''
    A function to get the objective value.
    :param x: The centroids.                           [n,m] array
    :param y: The grids.                               [t,m] array
    :param z: The samples.                             [k,m] array
    :param t: The formal t.                            [t,n] array
    :param r: The formal r.                            [k,n] array
    :return: The value of object function.
    '''
    num_centers = np.size(x,0)
    num_samples = np.size(z,0)

    yyt = y.dot(y.T)
    xxt = x.dot(x.T)
    xyt = x.dot(y.T)

    m   = np.kron(np.ones(num_centers), np.sum(np.mat(np.power(z, 2)), 1)) + \
    np.kron(np.ones([num_samples, 1]), np.sum(np.mat(np.power(x, 2)), 1).T) - 2 * z.dot(x.T)

    return np.trace(yyt) - 2 * np.trace(t .dot(xyt))+np.trace((t.T.dot(t)).dot(xxt))+np.trace(r.dot(m.T))

def cvx_solve(x,y,z):
    '''
    Using mosek to update t,r.
    :param x: The centroids.                           [n,m] array
    :param y: The grids.                               [t,m] array
    :param z: The samples.                             [k,m] array
    :return: !) t: The new t.                          [t,n] array
             2) r: The new r.                          [k,n] array
    '''
    num_centers = np.size(x,0)
    num_grids   = np.size(y,0)
    size = int(sqrt(num_grids))
    num_samples = np.size(z,0)

    yyt = y.dot(y.T)
    xxt = x.dot(x.T)
    xyt = x.dot(y.T)
    m   = np.kron(np.ones(num_centers), np.sum(np.mat(np.power(z, 2)), 1)) + \
    np.kron(np.ones([num_samples, 1]), np.sum(np.mat(np.power(x, 2)), 1).T) - 2 * z.dot(x.T)

    t = cvx.Variable((num_grids,num_centers))
    r = cvx.Variable((num_samples,num_centers))

    constraints = [t*np.ones([num_centers,1])==np.ones([num_grids,1]),
                   r*np.ones([num_centers,1])==np.ones([num_samples,1]),
                   t.T*np.ones([num_grids,1])/num_grids == r.T*np.ones([num_samples,1])/num_samples,
                   t>=0,r>=0]
    err = np.trace(yyt)-2*cvx.trace(t*xyt)
    for i in range(num_grids):
        err += cvx.quad_form(t[i],xxt)
    # for i in range(num_centers):
    #     err += cvx.tv(t[:,i].reshape([size,size]))
    err += cvx.trace(r*m.T)
    prob = cvx.Problem(cvx.Minimize(err), constraints)
    prob.solve(solver=cvx.MOSEK)

    return t.value,r.value

def siedel_x(x, y, z, t, r, opts):
    '''
    Using siedel to update x.
    :param x: The centroids.                          [n,m] matrix
    :param y: The grids.                              [t,m] matrix
    :param z: The samples.                            [k,m] matrix
    :param t: The weights of grids.                   [t,n] matrix
    :param r: The weights of samples.                 [k,n] matrix
    :param opts: A list including max iterations.     [ 1 ] list
    :return:
           x: The centroids.                          [n,m] matrix
    '''
    max_iter = opts[0]

    gradient_c = lambda x: np.dot(t.T, t).dot(x) - np.dot(t.T, y)
    gradient_r = r.T.dot(z)

    iter = 0

    while iter < max_iter:
        p = np.mat(np.sum(r, 0))
        x = (gradient_r - gradient_c(x))/p.T
        iter = iter+1

    return x.getA()

def Alter_cvx_siedel(y,z,n_centers,opts,verbose = False):
    '''
    Using cvx and siedel to find t,r,x.
    :param y: The grids.                                              [t,m] matrix
    :param z: The samples.                                            [k,m] matrix
    :param n_centers: The number of centroids.                        [ 1 ] integer
    :param opts: Including max iteration and parameter for siedel.    [ 2 ] list
    :return: #todo
    '''
    n_grids = np.size(y,0)
    n_sample = np.size(z,0)
    dimension = np.size(y,1)
    result = one_step_kmeans(y,z,10,opts = [10**(-6)])
    t = result['t']
    x = result['x']
    r = result['r']
    iter = 0
    max_iter = opts[0]
    opts2=opts[1]
    verbose_var = np.zeros(opts[0])

    while iter < max_iter:

        t,r = cvx_solve(x,y,z)

        x = siedel_x(x,y,z,t,r,opts2)

        if verbose == True:
            verbose_var[iter] = get_val(x,y,z,t,r)

        iter +=1
        print(iter)

    if verbose == True:
        return verbose_var,r,t,x
    else:
        return r,t,x
