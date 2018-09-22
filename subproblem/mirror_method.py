import numpy as np


def update_t_r(x,y,z,t,r,opts):

    n_grids = np.size(y,0)
    n_sample = np.size(z,0)
    n_center = np.size(x,0)

    mu = opts[0]
    lam = np.ones([n_center,1])
    yxt = y.dot(x.T)
    xxt = x.dot(x.T)
    s   = np.kron(np.ones(n_center), np.sum(np.mat(np.power(z, 2)), 1)) + \
    np.kron(np.ones([n_sample, 1]), np.sum(np.mat(np.power(x, 2)), 1).T) - 2 * z.dot(x.T)

    gradient_t = lambda t:2*(-yxt+t.dot(xxt))+\
                          mu*np.ones([n_grids,1]).dot(n_sample*np.ones([1,n_grids]).dot(t)-
                          n_grids*np.ones([1,n_sample]).dot(r))+n_sample*np.ones([n_grids,1]).dot(lam.T)

    gradient_r = lambda r:s+mu*np.ones([n_sample,1]).dot(-n_sample*np.ones([1,n_grids]).dot(t)+
                          n_grids*np.ones([1,n_sample]).dot(r))-n_grids*np.ones([n_sample,1]).dot(lam.T)

    max_iter = opts[1]
    alpha1 = opts[2]
    alpha2 = opts[3]
    lv = opts[4]
    iter = 0

    while iter < max_iter:

        t = np.multiply(t,np.exp(-alpha1*gradient_t(t)))
        t = t/np.sum(t,1).reshape([n_grids,1])

        r = np.multiply(r,np.exp(-alpha2*gradient_r(r)))
        r = r/np.sum(r,1).reshape([n_sample,1])

        lam = lam + 1.618*mu*(t.T.dot(np.ones([n_grids,1]))*n_sample-r.T.dot(np.ones([n_sample,1]))*n_grids)

        iter +=1
        alpha1 = alpha1*lv
        alpha2 = alpha2*lv

    return t,r