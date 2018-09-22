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


def update_t_r(x,y,z,t,r,opts):
    '''
    Update t and r by admm.
    :param x: The centroids.                           [n,m] array
    :param y: The grids.                               [t,m] array
    :param z: The samples.                             [k,m] array
    :param t: The formal t.                            [t,n] array
    :param r: The formal r.                            [k,n] array
    :param opts: Opts including max iteration.         [ 1 ] list
    :return: 1) t: The new t.                          [t,n] array
             2) r: The new r.                          [k,n] array
    '''
    n_grids = np.size(y,0)
    n_sample = np.size(z,0)
    n_center = np.size(x,0)

    mu = opts[0]
    lam = np.ones([n_center,1])
    yxt = y.dot(x.T)
    xxt = x.dot(x.T)
    s   = np.kron(np.ones(n_center), np.sum(np.mat(np.power(z, 2)), 1)) + \
    np.kron(np.ones([n_sample, 1]), np.sum(np.mat(np.power(x, 2)), 1).T) - 2 * z.dot(x.T)

    # The gradient of t
    gradient_t = lambda t:2*(-yxt+t.dot(xxt))+\
                          mu*np.ones([n_grids,1]).dot(n_sample*np.ones([1,n_grids]).dot(t)-
                          n_grids*np.ones([1,n_sample]).dot(r))+n_sample*np.ones([n_grids,1]).dot(lam.T)

    # The gradient of r
    gradient_r = lambda r:s+mu*np.ones([n_sample,1]).dot(-n_sample*np.ones([1,n_grids]).dot(t)+
                          n_grids*np.ones([1,n_sample]).dot(r))-n_grids*np.ones([n_sample,1]).dot(lam.T)

    max_iter = opts[1]
    alpha1 = opts[2]
    alpha2 = opts[3]
    lv = opts[4]
    iter = 0

    while iter < max_iter:

        # Update t
        t = np.multiply(t,np.exp(-alpha1*gradient_t(t)))
        t = t/np.sum(t,1).reshape([n_grids,1])

        # Update r
        r = np.multiply(r,np.exp(-alpha2*gradient_r(r)))
        r = r/np.sum(r,1).reshape([n_sample,1])

        # Update dual variable lam
        lam = lam + 1.618*mu*(t.T.dot(np.ones([n_grids,1]))*n_sample-r.T.dot(np.ones([n_sample,1]))*n_grids)

        # The warm start
        iter +=1
        alpha1 = alpha1*lv
        alpha2 = alpha2*lv

    return t,r



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

def Alter_ADMM_siedel(y,z,n_centers,opts,verbose = False):
    '''
    Using ADMM and siedel to update x,r,t.
    :param y: The grids.                                          [t,m] array
    :param z: The samples.                                        [k,m] array
    :param n_centers: The number of centers.                      [ 1 ] integer
    :param opts: Opts including max iteration, opts1,opts2.       [ 3 ] list
    :return: #todo
    '''
    num_grids   = np.size(y,0)
    num_samples = np.size(z,0)
    dimension   = np.size(y,1)

    # Get the initial value from kmeans
    result = one_step_kmeans(y,z,n_centers,opts = [10**(-6)])
    t = result['t']
    x = result['x']
    r = result['r']

    iter = 0
    max_iter = opts[0]
    opts1=opts[1]
    opts2=opts[2]
    verbose_var = np.zeros(opts[0])

    while iter < max_iter:

        # Update t,r
        t,r = update_t_r(x,y,z,t,r,opts1)

        # Update x
        x = siedel_x(x,y,z,t,r,opts2)

        if verbose == True:
            verbose_var[iter] = get_val(x,y,z,t,r)

        iter +=1
        print(iter)

    if verbose == True:
        return verbose_var,r,t,x
    else:
        return r,t,x

