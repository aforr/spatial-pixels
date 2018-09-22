import cvxpy as cvx
import numpy as np
import ot

def warm_sinkhorn(row,col,C,epsilon,scaling_iter=3000,Inner_iter=100,epsilon0=0.1,extra_iter=1000,tau=10000):
    '''
    Using Sinkhorn with warm start.
    :param C: The distance matrix.                                        [n,m] array
    :param row: The source distribution.                                  [ n ] array
    :param col: The target distribution.                                  [ m ] array
    :param epsilon: The parameter for entropy regularization.             [ 1 ] real
    :param scaling_iter: The total iteration.                             [ 1 ] integer
    :param Inner_iter: The small iterations for changing epsilon.         [ 1 ] integer
    :param epsilon0: The initial epsilon.                                 [ 1 ] real
    :param extra_iter: The extra iterations for final epsilon.            [ 1 ] integer
    :param tau: The upper bound for absorbing.                            [ 1 ] real
    :return:
    1) K: The transport maps.                                             [n,m] array
    2) u: The dual variable.                                              [ n ] array
    3) v: The other dual variable.                                        [ m ] array
    '''

    times = scaling_iter//Inner_iter
    def get_reg(n):
        lv = epsilon/epsilon0
        return epsilon0*lv**(n/(times-1))

    epsilon_i = epsilon0
    K = np.exp(-C/epsilon_i)
    b = np.ones(len(col))
    u = np.ones(len(row))
    v = np.ones(len(col))
    epsilon_index = 0
    iterations_since_epsilon_adjusted = 0

    for iter in range(scaling_iter):
        a = row/K.dot(b)
        b = col/(K.T).dot(a)
        # print(col)
        # print((K.T).dot(a))
        iterations_since_epsilon_adjusted+=1
        # print(a)
        # print(b)
        # print(iter)

        if max(max(abs(a)), max(abs(b))) > tau:
            u = u + epsilon_i * np.log(a)
            v = v + epsilon_i * np.log(b)  # absorb
            K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
            a = np.ones(len(row))
            b = np.ones(len(col))


        if  iterations_since_epsilon_adjusted == Inner_iter:
            epsilon_index += 1
            iterations_since_epsilon_adjusted = 0
            u = u + epsilon_i * np.log(a)
            v = v + epsilon_i * np.log(b)  # absorb
            epsilon_i = get_reg(epsilon_index)
            K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
            a = np.ones(len(row))
            b = np.ones(len(col))

    for i in range(extra_iter):
        a = row / K.dot(b)
        b = col / (K.T).dot(a)

    u = u + epsilon_i * np.log(a)
    v = v + epsilon_i * np.log(b)

    return (K.T*a).T*b,u,v


a = np.array([1,2,3])
b = np.array([3,2,1])
a = a/ np.sum(a)
b = b/ np.sum(b)
C = np.array([[0,1,2],
              [1,0,1],
              [2,1,0]])


pi2 = warm_sinkhorn(a,b,C,10**(-3))