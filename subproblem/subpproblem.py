from generate_data import *
from subproblem.mirror_method import *
from subproblem.cvx_method import *
#%% parameter figuration

num_x = 10
num_y = 100
num_z = 1000
num_pixel = 100
size_grid = 10
dimension = 10

min_dist_x = 0
variance_x = 10
variance_yz = 100

#%% Generate data

x = generate_centroids(num_x,dimension,variance_x,min_dist_x)
x_2 = generate_2centroids(size_grid,num_x)
y,t,z,r,distribution_x = generate_grid_samples(x,x_2,num_pixel,num_z,size_grid,variance_yz)

#%%
x = np.ones([10,10])
t_c,r_c = cvx_solve(x,y,z)

#%%

t_= np.ones([num_y,num_x])/num_x
r_= np.ones([np.size(z,0),num_x])/num_x
t_,r_ = update_t_r(x,y,z,t_,r_,opts=[10**(-5),2000,4*10**(-3),10**(-1),(1/1000)**(1/10000)])


print(li.norm(t_-t_c,'fro')/(1+li.norm(t_c,'fro')))
print(li.norm(r_-r_c,'fro')/(1+li.norm(r_c,'fro')))
print(li.norm(y-t_c.dot(x),'fro')**2)
print(li.norm(y-t_.dot(x),'fro')**2)
