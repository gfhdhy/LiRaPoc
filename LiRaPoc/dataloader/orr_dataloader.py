import numpy as np
import open3d as o3d
import os
import struct
import cv2
import os
from bev_utils import *
from joblib  import Parallel, delayed

def read_bin_lidar(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        # 几个f几维
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],-point[2],point[3]])
    return np.asarray(pc_list,dtype=np.float32)

def removal_ground(lidar,segment_x = 20):
    d = pow(lidar[:, 0] ** 2 + lidar[:, 1] ** 2, 0.5)
    lidar_data_in = lidar[d <= segment_x, :]
    lidar_data_out = lidar[d > segment_x, :]
    indices,_ = my_ransac_v2(lidar_data_in[:, :3])
    lidar_data_in[indices, :] = 0
    idx = lidar_data_in[:, 2] > 0
    lidar_data_in = lidar_data_in[idx, :]
    lidar = np.vstack((lidar_data_in, lidar_data_out))
    return lidar

def read_radar(radar_name,threshhold=[50,55]):
    radar = cv2.imread(radar_name)
    w,h,num = radar.shape
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
            
            #r = i/w*100*10
            theta = (i+0.5) / w * 360
            r = (j+0.5)/h*165
            #z = np.tan(0.9 / 180 * np.pi) * r
            z = np.tan(0.9 / 180 * np.pi) * r
            #z = np.tan(0.65 / 180 * np.pi) * r
            
            ## z is here
            #z = 0
            points.append(np.array([r, theta, z,intensity,occupancy]))

                
    radar_points = np.array(points)
    #radar_points N*5: r, theta, z,intensity,occupancy

    #radar_grid = radar_occupancy_grid(radar_points)
    return radar_points

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


def read_lidar(lidar_name):
    lidar_data = read_bin_lidar(lidar_name)
    TR = np.zeros(6)
    x_t = 0
    y_t = 0
    z_t = 0
    x_r = 0
    y_r = 0
    z_r = 180
    LidarToRadar = calibration(x_t, y_t, z_t, x_r, y_r, z_r)
    Calibration = True
    lidar_points = np.zeros_like(lidar_data)
    lidar_points[:, 2:] = lidar_data[:, 2:]
    if Calibration == True:
        x = lidar_data[:, 0]
        y = lidar_data[:, 1]
        z = lidar_data[:, 2]
        lidar_points[:, 0] = LidarToRadar[0][0] * x + LidarToRadar[0][1] * y + LidarToRadar[0][2] * z + LidarToRadar[0][3]
        lidar_points[:, 1] = LidarToRadar[1][0] * x + LidarToRadar[1][1] * y + LidarToRadar[1][2] * z + LidarToRadar[1][3]
        lidar_points[:, 2] = LidarToRadar[2][0] * x + LidarToRadar[2][1] * y + LidarToRadar[2][2] * z + LidarToRadar[2][3]
    else:
        lidar_points =  lidar_data
    return lidar_points

def car2polar(lidar_data):
    lidar_polar = np.zeros_like(lidar_data)
    lidar_polar[:, 1] = np.arctan2(lidar_data[:, 0], lidar_data[:, 1]) / np.pi * 180 + 90
    lidar_polar[lidar_polar[:, 1] < 0] = lidar_polar[lidar_polar[:, 1] < 0] + 360
    lidar_polar[lidar_polar[:, 1] >= 360] = 360-1
    #lidar_polar[:, 0] = np.sqrt(lidar_data[:, 0] * lidar_data[:, 0] + lidar_data[:, 1] * lidar_data[:, 1]) * 10
    lidar_polar[:, 0] = np.sqrt(lidar_data[:, 0] * lidar_data[:, 0] + lidar_data[:, 1] * lidar_data[:, 1])
    lidar_polar[:, 2:5] = lidar_data[:, 2:5]
    lidar_polar = np.array(lidar_polar)
    return lidar_polar

def polar2car(radar_data):
    radar_car = np.zeros_like(radar_data)
    radar_car[:, 1] = radar_data[:,0]*np.sin(radar_data[:,1]/180*np.pi)
    radar_car[:, 0] = radar_data[:,0]*np.cos(radar_data[:,1]/180*np.pi)
    radar_car[:, 2:5] = radar_data[:, 2:5]
    return radar_car

def set_radar_boundary(points,range):
    points = points[points[:,0] <= range, :]   
    return points

def set_lidar_boundary(points,range):
    r = np.sqrt(points[:,0]*points[:,0]  + points[:,1]*points[:,1])
    points = points[r <= range, :]   
    return points

def read_orr_data_one_frame(n,filenames,lidar_root_dir,radar_root_dir,maxRange):
    lidar_path = os.path.join(lidar_root_dir, filenames[n][0:16]+".bin")
    radar_path = os.path.join(radar_root_dir, filenames[n][0:16]+".jpg")
    print("---load data----",lidar_path,radar_path)
        
    ## read radar ##
    radar_threshhold=[50,55]
    radar_point = read_radar(radar_path,radar_threshhold)
    #radar_grid = radar_occupancy_grid(radar_points)
    radar_point = radar_point[radar_point[:,4] == 2]
    radar_point = set_radar_boundary(radar_point, maxRange)
    radar_car = polar2car(radar_point)
    #radar_points N*5: r, theta, z,intensity,occupancy
       
    ## read lidar ##
    lidar_point = read_lidar(lidar_path)
    lidar_point = set_lidar_boundary(lidar_point, maxRange)
    #lidar_point = removal_ground(lidar_point,20)
    lidar_polar = car2polar(lidar_point)
    #lidar_polar = removal_out_lidar(lidar_polar,radar_grid)
    #lidar_points N*5 r,theta,z,i,id

    return lidar_polar,radar_point,lidar_point,radar_car
    
def orr_dataloader_v1(lidar_root_dir,radar_root_dir):
    # Dof 
    # k varies 
    
    filenames = os.listdir(radar_root_dir)
    file_number = len(filenames)
    polar_maxCoord = [100,360,5]
    polar_minCoord = [0,0,-3]
    maxRange = 95
    lidar_polars = []
    radar_points = []
    lidar_points = []
    radar_cars = []
    for n in range(file_number):
        lidar_path = os.path.join(lidar_root_dir, filenames[n][0:16]+".bin")
        radar_path = os.path.join(radar_root_dir, filenames[n][0:16]+".jpg")
        print("---load data----",lidar_path,radar_path)
        
        ## read radar ##
        radar_threshhold=[50,55]
        radar_point = read_radar(radar_path,radar_threshhold)
        #radar_grid = radar_occupancy_grid(radar_points)
        radar_point = radar_point[radar_point[:,4] == 2]
        radar_point = set_radar_boundary(radar_point, maxRange)
        radar_car = polar2car(radar_point)
        #radar_points N*5: r, theta, z,intensity,occupancy
        radar_points.append(radar_point)
        radar_cars.append(radar_car)
        
        ## read lidar ##
        lidar_point = read_lidar(lidar_path)
        lidar_point = set_lidar_boundary(lidar_point, maxRange)
        #lidar_point = removal_ground(lidar_point,20)
        lidar_polar = car2polar(lidar_point)
        #lidar_polar = removal_out_lidar(lidar_polar,radar_grid)
        #lidar_points N*5 r,theta,z,i,id
        lidar_polars.append(lidar_polar)
        lidar_points.append(lidar_point)
        
        
        ## show bev ##
        #radar_bev = bev(radar_car,str(filenames[n][0:16])+'radar.jpg')
        #radar_bev_polar = bev(radar_point,str(filenames[n][0:16])+'radar_polar.jpg','polar')
        #lidar_bev = bev(lidar_point,str(filenames[n][0:16])+'lidar.jpg')
        #lidar_bev_polar = bev(lidar_polar,str(filenames[n][0:16])+'lidar_polar.jpg','polar')
    
    return lidar_polars,radar_points,lidar_points,radar_cars

def read_orr_data_one_frame(n,filenames,lidar_root_dir,radar_root_dir,maxRange):
    lidar_path = os.path.join(lidar_root_dir, filenames[n][0:16]+".bin")
    radar_path = os.path.join(radar_root_dir, filenames[n][0:16]+".jpg")
    print("---load data----",lidar_path,radar_path)
        
    ## read radar ##
    radar_threshhold=[50,55]
    radar_point = read_radar(radar_path,radar_threshhold)
    #radar_grid = radar_occupancy_grid(radar_points)
    radar_point = radar_point[radar_point[:,4] == 2]
    radar_point = set_radar_boundary(radar_point, maxRange)
    radar_car = polar2car(radar_point)
    #radar_points N*5: r, theta, z,intensity,occupancy
       
    ## read lidar ##
    lidar_point = read_lidar(lidar_path)
    lidar_point = set_lidar_boundary(lidar_point, maxRange)
    #lidar_point = removal_ground(lidar_point,20)
    lidar_polar = car2polar(lidar_point)
    #lidar_polar = removal_out_lidar(lidar_polar,radar_grid)
    #lidar_points N*5 r,theta,z,i,id

    return lidar_polar,radar_point,lidar_point,radar_car
    

def orr_dataloader(lidar_root_dir,radar_root_dir):
    
    filenames = os.listdir(radar_root_dir)
    file_number = len(filenames)
    maxRange = 95
    
    data = Parallel(n_jobs = 1)(delayed(read_orr_data_one_frame)(n,filenames,lidar_root_dir,radar_root_dir,maxRange) for n in range(file_number))
    lidar_polars = [item[0] for item in data]
    radar_points=[item[1] for item in data]
    lidar_points=[item[2] for item in data]
    radar_cars=[item[3] for item in data]
    
    return lidar_polars,radar_points,lidar_points,radar_cars