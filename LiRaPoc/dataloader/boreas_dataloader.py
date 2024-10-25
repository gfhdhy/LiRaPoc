import numpy as np
import open3d as o3d
import os
import struct
import cv2
import os
from bev_utils import *

def read_bin_lidar(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2],point[3]])
    return np.asarray(pc_list,dtype=np.float32)

def read_radar(radar_name,threshhold=[45,50]):
    
    radar = cv2.imread(radar_name)
    w,h,_ = radar.shape
    t = int(w/2)
    # radar_[:t,:,:] = radar[t:,:,:]
    # radar_[t :,:,:] = radar[0:t,:,:]
    points = []
    occupancy_num = [0,0,0]
    for i in range(w):
        for j in range(h):
            intensity = radar[i,j,0]
            occupancy = 0
            if intensity > threshhold[0]:
                occupancy = 1
                occupancy_num[1] = occupancy_num[1]+1
                if intensity > threshhold[1]:
                    occupancy = 2
                    occupancy_num[2] = occupancy_num[2] + 1
            
            theta = i / w * 360
            if theta >= 360:
                theta = theta-360
            r = j /h * 200.256
            z = np.tan(0.9 / 180 * np.pi) * r
            #z = np.tan(0.65 / 180 * np.pi) * r
            
            ## z is here
            #z = 0
            points.append(np.array([r, theta, z,intensity,occupancy]))

                
    radar_points = np.array(points)
    #radar_points N*5: r, theta, z,intensity,occupancy

    #radar_grid = radar_occupancy_grid(radar_points)
    return radar_points

def transform_matrix(x_t, y_t, z_t, x_r, y_r, z_r):
    
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
    # R
    R = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]).astype(np.float64)
    R = np.matmul(R, np.matmul(Rx, np.matmul(Ry, Rz)))

    Transform_Matrix = np.array([[R[0, 0], R[0, 1], R[0, 2], 0.0],
                             [R[1, 0], R[1, 1], R[1, 2], 0.0],
                             [R[2, 0], R[2, 1], R[2, 2], 0.0],
                             [x_t, y_t, z_t, 1.0]]).T
    return Transform_Matrix	

def read_lidar(lidar_name):
    lidar_data = read_bin_lidar(lidar_name)
    TR = np.zeros(6)
    x_t = 0
    y_t = 0
    z_t = 0
    x_r = 180
    y_r = 0
    z_r = 0
    LidarToRadar = transform_matrix(x_t, y_t, z_t, x_r, y_r, z_r)
    lidar_points = np.zeros_like(lidar_data)
    x = lidar_data[:, 0]
    y = lidar_data[:, 1]
    z = lidar_data[:, 2]
    lidar_points[:, 0] = LidarToRadar[0][0] * x + LidarToRadar[0][1] * y + LidarToRadar[0][2] * z + LidarToRadar[0][3]
    lidar_points[:, 1] = LidarToRadar[1][0] * x + LidarToRadar[1][1] * y + LidarToRadar[1][2] * z + LidarToRadar[1][3]
    lidar_points[:, 2] = LidarToRadar[2][0] * x + LidarToRadar[2][1] * y + LidarToRadar[2][2] * z + LidarToRadar[2][3]

    return lidar_points

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

def polar2car(radar_data):
    radar_car = np.zeros_like(radar_data)
    #radar_car[:, 0] = radar_data[:,0]*np.sin(radar_data[:,1]/180*np.pi)
    #radar_car[:, 1] = radar_data[:,0]*np.cos(radar_data[:,1]/180*np.pi)
    radar_car[:, 1] = radar_data[:,0]*np.sin(radar_data[:,1]/180*np.pi)
    radar_car[:, 0] = radar_data[:,0]*np.cos(radar_data[:,1]/180*np.pi)
    # radar_car[:, 2] = -radar_car[:, 2]
    radar_car[:, 3:5] = radar_data[:, 3:5]
    return radar_car

def set_radar_boundary(points,range):
    points = points[points[:,0] <= range, :]   
    return points

def set_lidar_boundary(points,range):
    r = np.sqrt(points[:,0]*points[:,0] + points[:,1]*points[:,1])
    points = points[r < range, :]   
    return points

def boreas_dataloader(lidar_root_dir,radar_root_dir):
    """ load boreas data
    Args: 
        lidar_root_dir: Boreas Dataset Lidar data dir  '/data/boreas/sequense/lidar'
        radar_root_dir: Boreas Dataset Radar data dir  '/data/boreas/sequense/radar'
    Returns:
        lidar_polars (list)(np.ndarray): lidar points in polar 
        radar_points (list)(np.ndarray): radar points in polar 
        lidar_point (list)(np.ndarray):  lidar points in Cartesian 
        radar_cars (list)(np.ndarray):  radar points in Cartesian  
    """
    # k varies 
    
    filenames = os.listdir(radar_root_dir)
    file_number = len(filenames)

    maxRange = 95
    lidar_polars = []
    radar_points = []
    lidar_points = []
    radar_cars = []
    for n in range(file_number):
        if filenames[n][0] == '.':
            continue
        lidar_path = os.path.join(lidar_root_dir, filenames[n][0:16]+".bin")
        radar_path = os.path.join(radar_root_dir, filenames[n][0:16]+".jpg")
        print("---load data----",lidar_path,radar_path)
        
        ## read radar ##
        radar_threshhold=[50,50]
        radar_point = read_radar(radar_path,radar_threshhold) # read radar polar point from radar polar image
        radar_point = radar_point[radar_point[:,4] == 2]
        radar_point = set_radar_boundary(radar_point, maxRange) # remove points out of polar_range
        radar_car = polar2car(radar_point) # transfrom radar to car form
        #radar_car = removal_out_points(radar_car,maxCoord,minCoord)
        radar_points.append(radar_point)
        radar_cars.append(radar_car)
        # radar_points N*5: r, theta, z,intensity,occupancy
        
        ## read lidar ##
        # read lidar s
        lidar_point = read_lidar(lidar_path)
        lidar_point = set_lidar_boundary(lidar_point, maxRange)
        lidar_polar = car2polar(lidar_point)
        lidar_polars.append(lidar_polar)
        lidar_points.append(lidar_point)
        
        
        ## show bev ##
        #radar_bev = bev(radar_car,str(filenames[n][0:16])+'radar.jpg')
        #radar_bev_polar = bev(radar_point,str(filenames[n][0:16])+'radar_polar.jpg','polar')
        #lidar_bev = bev(lidar_point,str(filenames[n][0:16])+'lidar.jpg')
        #lidar_bev_polar = bev(lidar_polar,str(filenames[n][0:16])+'lidar_polar.jpg','polar')
        
        #lidar_points N*5 r,theta,z,i,id
    
    return lidar_polars,radar_points,lidar_points,radar_cars
