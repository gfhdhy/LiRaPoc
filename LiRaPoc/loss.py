import numpy as np
import scipy.optimize as opt
import os
from cost import count_points


def car2polar(lidar_data):
    lidar_polar = np.zeros_like(lidar_data)
    lidar_polar[:, 1] = np.arctan2(lidar_data[:, 1], lidar_data[:, 0]) / np.pi * 180
    lidar_polar[lidar_polar[:, 1] >= 360] = lidar_polar[lidar_polar[:, 1] >= 360] - 360
    lidar_polar[lidar_polar[:, 1] < 0] = lidar_polar[lidar_polar[:, 1] < 0] + 360
    lidar_polar[lidar_polar[:, 1] >= 360] = 360 - 1
    lidar_polar[:, 0] = np.sqrt(lidar_data[:, 0] * lidar_data[:, 0] + lidar_data[:, 1] * lidar_data[:, 1])
    lidar_polar[:, 2:5] = lidar_data[:, 2:5]
    lidar_polar = np.array(lidar_polar)
    return lidar_polar

def calibration(x_t, y_t, z_t, x_r, y_r, z_r):
    # RX
    thetaX = np.deg2rad(x_r)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(thetaX), -np.sin(thetaX)],
                   [0, np.sin(thetaX), np.cos(thetaX)]]).astype(np.float64)
    # RY
    thetaY = np.deg2rad(y_r)
    Ry = np.array([[np.cos(thetaY), 0, np.sin(thetaY)],
                   [0, 1, 0],
                   [-np.sin(thetaY), 0, np.cos(thetaY)]])
    # RZ
    thetaZ = np.deg2rad(z_r)
    Rz = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0],
                   [np.sin(thetaZ), np.cos(thetaZ), 0],
                   [0, 0, 1]]).astype(np.float64)

    # transfomer
    R = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]).astype(np.float64)
    R = np.matmul(R, np.matmul(Rx, np.matmul(Ry, Rz)))

    LidarToRadar = np.array([[R[0, 0], R[0, 1], R[0, 2], 0.0],
                             [R[1, 0], R[1, 1], R[1, 2], 0.0],
                             [R[2, 0], R[2, 1], R[2, 2], 0.0],
                             [x_t, y_t, z_t, 1.0]]).T
    return LidarToRadar	

def transform_lidar(lidars,T,R):
    x_t = T[0]
    y_t = T[1]
    z_t = T[2]
    x_r = R[0]
    y_r = R[1]
    z_r = R[2]
    LidarToRadar = calibration(x_t, y_t, z_t, x_r, y_r, z_r)
    lidar_transformed_polars = []
    lidar_transformed = []
    frames = len(lidars)
    for n in range(frames):
        lidar_data = lidars[n]
        lidar_points = np.zeros_like(lidar_data)
        x = lidar_data[:, 0]
        y = lidar_data[:, 1]
        z = lidar_data[:, 2]
        lidar_points[:, 0] = LidarToRadar[0][0] * x + LidarToRadar[0][1] * y + LidarToRadar[0][2] * z + LidarToRadar[0][3]
        lidar_points[:, 1] = LidarToRadar[1][0] * x + LidarToRadar[1][1] * y + LidarToRadar[1][2] * z + LidarToRadar[1][3]
        lidar_points[:, 2] = LidarToRadar[2][0] * x + LidarToRadar[2][1] * y + LidarToRadar[2][2] * z + LidarToRadar[2][3]
        lidar_points[:, 3:] = lidar_data[:, 3:]
        
        lidar_polar = car2polar(lidar_points)
        lidar_transformed.append(lidar_points)
        lidar_transformed_polars.append(lidar_polar)
    
    return lidar_transformed,lidar_transformed_polars

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
    # print("loss", loss)
    cost = (20000-loss).sum()
    file.write(str(loss.sum())+'\n')

    return cost
           
def run_lirapoc(radar_occupancy_map,radar_car,lidar_car,radar_size):

    p0 = np.zeros(6)
    lower_bounds = [-2, -2,-2,-10,-10,-10]
    upper_bounds = [2, 2,2,10,10,10]
    stride = [0.1,0.1,0.1,0.2,0.2,0.1]

    ftol_ = len(radar_occupancy_map) * 0.001
    loss = residuals(p0,lidar_car,radar_occupancy_map,radar_size)
    
    
    loss = opt.least_squares(residuals,p0,bounds=(lower_bounds, upper_bounds),diff_step = stride,args=(lidar_car,radar_occupancy_map,radar_size),method="trf",gtol=None,ftol=ftol_,xtol=None,verbose=2)
    p = loss.x
    print("lirapoc result: ",p)
    # c = loss.cost
    # print("cost",c)
    

    # path = 'vis_loss/loss.txt' 
    # file = open(path,'a')
    # file.write('\n')    
    T = p[:3]
    R = p[3:]

    return p
