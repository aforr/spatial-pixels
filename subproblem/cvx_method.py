import cvxpy as cvx
import numpy as np

def cvx_solve(x,y,z):

    num_centers = np.size(x,0)
    num_grids   = np.size(y,0)
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

    err = np.trace(yyt)-2*cvx.trace(t*xyt)+cvx.tv(t)
    for i in range(num_grids):
        err += cvx.quad_form(t[i],xxt)
    err += cvx.trace(r*m.T)

    prob = cvx.Problem(cvx.Minimize(err),constraints)
    prob.solve(solver=cvx.MOSEK)

    return t.value,r.value


