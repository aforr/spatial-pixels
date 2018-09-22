'''
A test to find good parameter for lam.
'''


#%% import different modules

import matplotlib
import matplotlib.pyplot as plt
from one_step_kmeans_method import *
import numpy as np
import numpy.linalg as li
from generate_data import *
from measure_error import *
from ADMM_sidel_method import *
from cvx_siedel_method import *
from ADMM_siedel_tv_method import *
from cvx_siedel_tv_method import *

#%% parameter figuration

num_x = 10
num_y = 100
num_z = 1000
num_pixel = 100
size_grid = 10
dimension = 10
min_dist_x = 0
variance_x = 100
variance_yz = 100

#%% Generate data

x = generate_centroids(num_x,dimension,variance_x,min_dist_x)
x_2 = generate_2centroids(size_grid,num_x)
y,t,z,r,distribution_x = generate_grid_samples(x,x_2,num_pixel,num_z,size_grid,variance_yz)

#%%lambda=1
opts=[50,[20]]
verbose_cv,r_cv,t_cv,x_cv = Alter_cvx_siedel_tv(y,z,10,opts,lam=0.1,verbose=True)
verbose_cv1,r_cv1,t_cv1,x_cv1 = Alter_cvx_siedel_tv(y,z,10,opts,lam=1,verbose=True)
verbose_cv2,r_cv2,t_cv2,x_cv2 = Alter_cvx_siedel_tv(y,z,10,opts,lam=10,verbose=True)
verbose_cv3,r_cv3,t_cv3,x_cv3 = Alter_cvx_siedel_tv(y,z,10,opts,lam=100,verbose=True)



#%%
opts=[30,[20]]
verbose_c,r_c,t_c,x_c = Alter_cvx_siedel(y,z,10,opts,verbose=True)

#%%
# distribution_x_cv = np.mean(np.vstack((r_cv,t_cv)),0)
# distribution_x_cv1 = np.mean(np.vstack((r_cv1,t_cv1)),0)
distribution_x_cv2 = np.mean(np.vstack((r_cv2,t_cv2)),0)
# distribution_x_cv3 = np.mean(np.vstack((r_cv3,t_cv3)),0)
distribution_x_c = np.mean(np.vstack((r_c,t_c)),0)
# err_cv = measure_X(x_cv,x,distribution_x_cv,distribution_x)
# err_cv1 = measure_X(x_cv1,x,distribution_x_cv1,distribution_x)
err_cv2 = measure_X(x_cv2,x,distribution_x_cv2,distribution_x)
# err_cv3 = measure_X(x_cv3,x,distribution_x_cv3,distribution_x)
err_c = measure_X(x_c,x,distribution_x_c,distribution_x)
# print(err_cv)
# print(err_cv1)
print(err_cv2)
# print(err_cv3)
print(err_c)

#%%
def obj1(t1,x1,r1):
    global y
    global z
    m=np.kron(np.ones(num_x), np.sum(np.mat(np.power(z, 2)), 1)) + \
    np.kron(np.ones([np.size(z, 0), 1]), np.sum(np.mat(np.power(x1, 2)), 1).T) - 2 * z.dot(x1.T)
    return li.norm(y-t1.dot(x1),'fro')**2+ np.sum(np.multiply(r1,m))

#%%
print(obj1(t_c,x_c,r_c))
print(obj1(t_cv2,x_cv2,r_cv2))