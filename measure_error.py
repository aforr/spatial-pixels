import numpy as np
import ot
import numpy.linalg as li

def measure_X(X,X_real,source_distribution,target_distribution):
    '''
    A function to measure the distance bwtween distributions of X and X_real.
    :param X: The estimated X.                                       [n,m] array
    :param X_real: The real X.                                       [t,m] array
    :param source_distribution: The distribution of estimated X.     [ n ] array
    :param target_distribution: The distribution of real X.          [ t ] array
    :return:
           The Wasserstein distance between two empirical distributions.
    '''
    num_X = np.size(X,0)
    num_X_real = np.size(X_real,0)
    M = np.zeros([num_X,num_X_real])
    for i in range(num_X):
        for j in range(num_X_real):
            M[i,j]=li.norm(X[i]-X_real[j],2)**2

    return ot.emd2(source_distribution,target_distribution,M)


def measure_T(X,T,X_real,T_real,source_distribution,target_disribution):
    '''
    A function to measure the distance between T*OT and T_real.
    :param X: The estimated X.                                      [n,m] matrix
    :param:T: The estimated T.                                      [k,n] matrix
    :param X_real: The real X.                                      [s,m] matrix
    :param T_real: The real T.                                      [k,s] matrix
    :param source_distribution: The dstribution of estimated X.     [ n ] array
    :param target_disribution: The distribution of real X.          [ s ] array
    :return:
         The fro norm between T and real_T.
    '''
    num_X = np.size(X,0)
    num_X_real = np.size(X_real,0)
    M = np.zeros([num_X,num_X_real])
    for i in range(num_X):
        for j in range(num_X_real):
            M[i,j]=li.norm(X[i]-X_real[j],2)**2
    T_ot = ot.emd(source_distribution,target_disribution,M)
    T_ot = T_ot/np.sum(T_ot,1).reshape([num_X,1])
    T_T_ot = T.dot(T_ot)

    return li.norm(T_T_ot-T_real,'fro')

def measure_T2(Y,X,T,X_real,T_real):
    '''
    Another function to measure the distance between T and T_real.
    :param Y: The grids Y.                                          [t,m] array
    :param X: The estimated X.                                      [n,m] matrix
    :param T: The estimated T.                                      [k,n] matrix
    :param X_real: The real X.                                      [s,m] matrix
    :param T_real: The real T.                                      [k,s] matrix
    :return:
         The Wasserstein distance between T and real_T.
    '''
    num_T = np.size(T)
    num_T_real = np.size(T_real)
    M = np.zeros(num_T,num_T_real)
    num_Y = np.size(Y,0)
    num_X = np.size(X,0)
    num_x_real = np.size(X_real,0)
    for i in range(num_Y):
        for j in range(num_X):
            for k in range(num_Y):
                for l in range(num_x_real):
                    M[i*num_X+j,k*num_x_real+l]=li.norm(Y[i]-Y[k],'fro')**2+li.norm(X[j]-\
                                                X_real[num_x_real],'fro')**2
    return ot.emd(T.reshape([num_T]),T_real.reshape([num_T_real]),M)






