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

#%% Two model

opts = [10**(-6)]
result_o = one_step_kmeans(y,z,num_x,opts)
x_o = result_o['x']
t_o = result_o['t']
r_o = result_o['r']

#%% admm

opts=[100,[10**(-5),2000,4*10**(-3),10**(-2),(1/1000)**(1/10000)],[20]]
verbose_a,r_a,t_a,x_a= Alter_ADMM_siedel(y,z,num_x,opts,verbose=True)

#%% admm_tv

opts=[100,[10**(-5),2000,4*10**(-3),10**(-2),(1/1000)**(1/10000)],[20]]
verbose_av,r_av,t_av,x_av= Alter_ADMM_siedel_tv(y,z,num_x,opts,verbose=True)

#%% cvx

opts=[100,[20]]
verbose_c,r_c,t_c,x_c = Alter_cvx_siedel(y,z,10,opts,verbose=True)

#%% cvx_tv

opts=[100,[20]]
verbose_cv,r_cv,t_cv,x_cv = Alter_cvx_siedel_tv(y,z,10,opts,verbose=True)


#%% measure1

distribution_x_a = np.mean(np.vstack((r_a,t_a)),0).getA1()
distribution_x_av = np.mean(np.vstack((r_av,t_av)),0).getA1()
distribution_x_c = np.mean(np.vstack((r_c,t_c)),0).getA1()
distribution_x_o = np.mean(np.vstack((r_o,t_o)),0).flatten()
distribution_x_cv = np.mean(np.vstack((r_cv,t_cv)),0).getA1()


err_o = measure_X(x_o,x,distribution_x_o,distribution_x)
err_c = measure_X(x_c,x,distribution_x_c,distribution_x)
err_a = measure_X(x_a,x,distribution_x_a,distribution_x)
err_av = measure_X(x_av,x,distribution_x_av,distribution_x)
err_cv = measure_X(x_cv,x,distribution_x_cv,distribution_x)

print(err_o)
print(err_c)
print(err_a)
print(err_cv)
print(err_av)



#%% Value

def obj1(t1,x1,r1):
    global y
    global z
    m=np.kron(np.ones(num_x), np.sum(np.mat(np.power(z, 2)), 1)) + \
       np.kron(np.ones([np.size(z, 0), 1]), np.sum(np.mat(np.power(x1, 2)), 1).T) - 2 * z.dot(x1.T)
    return li.norm(y-t1.dot(x1),'fro')**2+ np.sum(np.multiply(r1,m))



#%% Object value

print(obj1(t_a,x_a,r_a))
print(obj1(t_o,x_o,r_o))
print(obj1(t_c,x_c,r_c))
print(obj1(t_cv,x_cv,r_cv))
print(obj1(t_av,x_av,r_av))
print(obj1(t,x,r))


#%% measure2

err_T_av = measure_T(x_av,t_av,x,t,distribution_x_av,distribution_x)
err_T_cv = measure_T(x_cv,t_cv,x,t,distribution_x_cv,distribution_x)

err_T_a = measure_T(x_a,t_a,x,t,distribution_x_a,distribution_x)
err_T_c = measure_T(x_c,t_c,x,t,distribution_x_c,distribution_x)

print(err_T_av)
print(err_T_cv)
print(err_T_a)
print(err_T_c)


#%% measure 3
err_T_2_av = measure_T2(y,x_av,t_av,x,t)
err_T_2_cv = measure_T2(y,x_cv,t_cv,x,t)
err_T_2_a = measure_T2(y,x_a,t_a,x,t)
err_T_2_c = measure_T2(y,x_c,t_c,x,t)

print(err_T_2_av)
print(err_T_2_cv)
print(err_T_2_a)
print(err_T_2_c)



#%% giv e a plot

size = np.size(verbose_a)
iter = np.linspace(0,size-1,size)

fig,ax = plt.subplots()
ax.line1 = plt.plot(iter,verbose_a,label='admm')
ax.line2 = plt.plot(iter,verbose_c,label='cvx')
ax.line3 = plt.plot(iter,obj1(t,x,r)*np.ones(size),label='real')
ax.line4 = plt.plot(iter,obj1(t_o,x_o,r_o)*np.ones(size),label='Kmeans')
plt.legend()
plt.show()
