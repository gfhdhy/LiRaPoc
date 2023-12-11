import numpy as np
import scipy.optimize as opt
import os
import struct
from draw_fig import *
from cost import *
from bev_utils import *

def residuals(p,lidar_car,radar_occupancy_map,radar_size):
    print("matrix")
    x_t, y_t, z_t, x_r, y_r, z_r = p

    print(x_t, y_t, z_t, x_r, y_r, z_r)
    
    T = [x_t, y_t, z_t]
    R = [x_r, y_r, z_r]
    _,lidar_polar = transform_lidar(lidar_car,T,R)
    loss = count_points(lidar_polar,radar_occupancy_map,radar_size)
    
    dirs = 'loss_txt'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    path = 'loss_txt/loss.txt'
    
    file = open(path,'a')
    #print(loss.max())
    print("loss")
    print(loss)
    cost = (15000-loss).sum()
    #.sum()
    print("cost")
    print(cost)
    print("--")
    file.write(str(loss.sum())+'\n')

    return cost
 

def draw_cost(iter,cost_list):
    #labels = [r"${\Theta}_{x}$", r"${\Theta}_{y}$", r"${\Theta}_{z}$"]
    plt.plot(iter, cost_list,'-o', label="x")
    plt.savefig("loss_fig/"+"loss" + ".jpg", bbox_inches='tight', pad_inches=0)
    plt.close()
    return 0


           
def loss(radar_occupancy_map,radar_car,lidar_car,radar_size,stride):

    #p0 = np.zeros(3)
    #p0[2] = 4
    p0 = np.zeros(6)
    #p0 = np.ones(6)
    lower_bounds = [-2, -2,-2,-5,-5,-5]
    upper_bounds = [2, 2,2,5,5,5]
    # stride = [0.1,0.1,0.1,0.2,0.2,0.2]

    #p0[2] = -180
    #p0 = np.ones(3)
    #p0 = -p0
    #loss = cal_loss(p0,lidar_car,radar_occupancy_map)
    #loss = opt.minimize(cal_loss,p0,args=(lidar_car,radar_occupancy_map),method='Newton-CG',jac=None)
    #loss = opt.least_squares(cal_loss,p0,args=(lidar_car,radar_occupancy_map),method="trf")
    #loss = opt.least_squares(cal_loss,p0,args=(lidar_car,radar_occupancy_map),method="trf", ftol=1e-15, max_nfev=100000)
    #stride = [0.1,0.1,1,1,1,1]
    

    # p0 = np.zeros(1)
    # lower_bounds = [-2]
    # upper_bounds = [2]
    # stride = [0.1]

    
    ftol_ = len(radar_occupancy_map) * 0.01
    #print(ftol_)
    loss = residuals(p0,lidar_car,radar_occupancy_map,radar_size)
    
    #print(loss)
    #loss = opt.minimize(cal_loss,p0,args=(lidar_car,radar_occupancy_map),method='Newton-CG',jac=None)f_scale=0.1,loss='soft_l1',
    #loss = opt.least_squares(residuals,p0,bounds=(lower_bounds, upper_bounds),diff_step = stride,args=(lidar_car,radar_occupancy_map,radar_size),method="trf",gtol=1e-9,ftol=1e-9,xtol=1e-9,verbose=1)
    loss = opt.least_squares(residuals,p0,bounds=(lower_bounds, upper_bounds),diff_step = stride,args=(lidar_car,radar_occupancy_map,radar_size),method="trf",gtol=None,ftol=ftol_,xtol=None,verbose=2)
    #loss = opt.minimize(residuals, p0, method='Powell', args=(lidar_car,radar_occupancy_map,radar_size),tol=1e-8)
    #loss = opt.least_squares(residuals,p0,args=(lidar_car,radar_occupancy_map,radar_size),diff_step = stride,method="lm",gtol=1e-8,ftol=1e-8,xtol=1e-8,verbose=1)
    p = loss.x
    print(p)
    c = loss.cost
    print(c)
    

    #iter = np.arange(len(cost_list)) 
    #draw_cost(iter,cost_list)

    #p0 = np.zeros(6)
        #loss = residuals(p0,ndt1,ndt2,1)
        #loss = opt.minimize(residuals,p0,args=(ndt1,ndt2,1),method='Newton-CG',jac=None)
        #loss = opt.minimize(residuals,p0,args=(ndt1,ndt2,4))
        #print(loss)
    path = 'vis_loss/loss.txt' 
    file = open(path,'a')
    file.write('\n')    
    T = p[:3]
    R = p[3:]
    print(p)
    return p
    #T = np.zeros(3)
    #R = np.zeros(3)
    #T[0:2] = p[0:2]
    #R[2] = p[2]
    #vis_transform(lidar_car,radar_car,T,R)
