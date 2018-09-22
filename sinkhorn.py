#%%
import numpy as np
import numpy.linalg as li
import ot

def approx_sinkhorn(row,col,K):
    '''
    A function to run sinkhorn.
    :param K: The matrix of exp(-c/epsilon)               [m,n] matrix
    :param row: The row sum.                              [ m ] array
    :param col: The col sum.                              [ n ] array
    :return:
           t: The transport map.                          [m,n] matrix
    '''
    t = sinkhorn(K,row,col,[100])
    # print(t)
    t = round(t,row,col)
    return t

def round(t, row, col, epsilon=10**(-8)):
    '''
    A function to adjust t's row sum and col sum.
    :param t: The transport map.                               [m,n] matrix
    :param row: Row sum.                                       [ m ] array
    :param col: Col sum.                                       [ n ] array
    :param epsilon: A parameter to control the distance.       [ 1 ] positive
    :return:
           t: The adjusted transport map.                      [m,n] matrix
    '''

    n = np.size(t,0)
    m = np.size(t,1)

    row_t = np.sum(t,1)
    x = row/row_t
    x = np.minimum(x,1)
    t = np.dot(np.diag(x), t)
    col_t = np.sum(t,0)
    y = col/(col_t+epsilon)
    y = np.minimum(y,1)
    t = np.dot(t, np.diag(y))
    row_t = np.sum(t,1)
    col_t = np.sum(t,0)
    err_r = row - row_t
    err_c = col - col_t

    return t + np.dot(err_r.reshape([n,1]),err_c.reshape([1,m])) / li.norm(err_r, 1)


def sinkhorn(K,row,col,opts):
    '''
    A function to do sinkhorn iterations.
    :param K: The matrix of exp(-c/epsilon)               [m,n] matrix
    :param row: The row sum.                              [ m ] array
    :param col: The col sum.                              [ n ] array
    :param opts: A list including max iterations.         [ 1 ] list
    :return:
           t: The transport map we want.                  [m,n] matrix
    '''
    max_iter = opts[0]
    iter = 0
    b = np.ones(len(col))
    a = np.ones(len(row))

    while iter < max_iter:
        a = row/K.dot(b)
        b = col/(K.T).dot(a)
        iter+=1
    # print((K.T*a).T*b)
    #     print(np.min(a))
    #     print(np.min(b))

    return (K.T*a).T*b



