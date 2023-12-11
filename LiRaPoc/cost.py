import numpy as np
import scipy.optimize as opt
import os
import struct
import math
from draw_fig import draw_t_loss, draw_r_loss
import numba
from numba import jit
from joblib  import Parallel, delayed

def car2polar_boreas(lidar_data):
    lidar_polar = np.zeros_like(lidar_data)
    lidar_polar[:, 1] = np.arctan2(lidar_data[:, 1], lidar_data[:, 0]) / np.pi * 180
    lidar_polar[lidar_polar[:, 1] >= 360] = lidar_polar[lidar_polar[:, 1] >= 360] - 360
    lidar_polar[lidar_polar[:, 1] < 0] = lidar_polar[lidar_polar[:, 1] < 0] + 360
    lidar_polar[lidar_polar[:, 1] >= 360] = 360 - 1
    lidar_polar[:, 0] = np.sqrt(lidar_data[:, 0] * lidar_data[:, 0] + lidar_data[:, 1] * lidar_data[:, 1])
    lidar_polar[:, 2:5] = lidar_data[:, 2:5]
    lidar_polar = np.array(lidar_polar)
    return lidar_polar

# def car2polar_orr(lidar_data):
#     lidar_polar = np.zeros_like(lidar_data)
#     lidar_polar[:, 1] = np.arctan2(lidar_data[:, 1], lidar_data[:, 0]) / np.pi * 180 + 90
#     lidar_polar[lidar_polar[:, 1] >= 360] = lidar_polar[lidar_polar[:, 1] >= 360] - 360
#     lidar_polar[lidar_polar[:, 1] < 0] = lidar_polar[lidar_polar[:, 1] < 0] + 360
#     lidar_polar[lidar_polar[:, 1] >= 360] = 360 - 1
#     lidar_polar[:, 0] = np.sqrt(lidar_data[:, 0] * lidar_data[:, 0] + lidar_data[:, 1] * lidar_data[:, 1])
#     lidar_polar[:, 2:5] = lidar_data[:, 2:5]
#     lidar_polar = np.array(lidar_polar)
#     return lidar_polar
def car2polar_orr(lidar_data):
    lidar_polar = np.zeros_like(lidar_data)
    lidar_polar[:, 1] = np.arctan2(lidar_data[:, 0], lidar_data[:, 1]) / np.pi * 180 + 90
    lidar_polar[lidar_polar[:, 1] < 0] = lidar_polar[lidar_polar[:, 1] < 0] + 360
    lidar_polar[lidar_polar[:, 1] >= 360] = 360-1
    #lidar_polar[:, 0] = np.sqrt(lidar_data[:, 0] * lidar_data[:, 0] + lidar_data[:, 1] * lidar_data[:, 1]) * 10
    lidar_polar[:, 0] = np.sqrt(lidar_data[:, 0] * lidar_data[:, 0] + lidar_data[:, 1] * lidar_data[:, 1])
    lidar_polar[:, 2:5] = lidar_data[:, 2:5]
    lidar_polar = np.array(lidar_polar)
    return lidar_polar

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
        
        lidar_polar = car2polar_boreas(lidar_points)
        lidar_transformed.append(lidar_points)
        lidar_transformed_polars.append(lidar_polar)
    
    return lidar_transformed,lidar_transformed_polars
    

def build_radar_occupancy_map(radars,image_size):
    ## generation radar_occupancy_map from radar points ##
    # radars: radar points in polar form
    # image_size: radar image resolutions 
    
    frames = len(radars)
    radar_occupancys = []
    w = image_size[0]
    h = image_size[1]
    for n in range(frames):
        radar = radars[n]
        radar_voxel_idx = np.zeros([len(radar), 2])
        # radar_voxel_idx (n x 2) 记录每个点在格子中的(i,j)位置
        radar_voxel_idx[:, 0] = np.floor(radar[:, 0]/100*w)
        radar_voxel_idx[:, 1] = np.floor(radar[:, 1]/360*h)
        radar_voxel_height = radar[:,2]
        radar_intensity = radar[:,3]
        #radar_voxel_idx[radar_voxel_idx >= 2*boundary] = 2*boundary - 2
        radar_voxel_idx = radar_voxel_idx.astype(np.int)
        radar_occupancy = np.zeros([w, h, 4])
        for i in range(len(radar_voxel_idx)):
            radar_occupancy[radar_voxel_idx[i, 0], radar_voxel_idx[i, 1], 0] = radar_voxel_height[i]
            radar_occupancy[radar_voxel_idx[i, 0], radar_voxel_idx[i, 1], 1] = -radar_voxel_height[i]
            if radar_intensity[i] > radar_occupancy[radar_voxel_idx[i, 0], radar_voxel_idx[i, 1], 2]:
                radar_occupancy[radar_voxel_idx[i, 0], radar_voxel_idx[i, 1], 2] = radar_intensity[i]
            radar_occupancy[radar_voxel_idx[i, 0], radar_voxel_idx[i, 1], 3] = 1
        # radar_occupancy W x H x 3(height_up, height_bottom, intensity, mask)
        radar_occupancys.append(radar_occupancy)
        
    return radar_occupancys

def count_points_v1(lidars,radar_occupancy_map,radar_size):
    frames = len(lidars)
    cost = np.zeros(frames)
    w = radar_size[0]
    h = radar_size[1]
    for n in range(frames):
        radar_occupancy = radar_occupancy_map[n]
        radar_occupancy_num = np.zeros_like(radar_occupancy[:,:,3])
        radar_occupancy_num = radar_occupancy[:,:,3]
        lidar = lidars[n]
        lidar_num = lidar.shape[0]
        for i in range(lidar_num):
            x = np.floor(lidar[i,0]/100*w).astype(np.int64)
            y = np.floor(lidar[i,1]/360*h).astype(np.int64)
            z = lidar[i,2]
            intensity = lidar[i,3]
            if(x < w and radar_occupancy[x,y,1] != 0):
                upper = radar_occupancy[x,y,0]
                bottom = radar_occupancy[x,y,1]
                #theta = (lidar[i,4]-23)*1.33
                #print(lidar[i,4])
                if(z< upper and z > bottom ):
                    radar_occupancy_num[x,y] = radar_occupancy_num[x,y]+1
                    # if(radar_occupancy_num[x,y] == 5):
                    #      cost[n] = cost[n] + 3
                    upper = radar_occupancy[x,y,0]
                    bottom = radar_occupancy[x,y,1]
                    d = upper - bottom
                    du = upper - z
                    db = z - bottom
                    weight = d*d /(2*(du*du+db*db))
                    if radar_occupancy[x,y,2]>80:
                        weight = weight*1.5
                    #if intensity < 0.05:
                    #    weight = 0
                        
                    #print(radar_occupancy[x,y,0])
                    #print(radar_occupancy[x,y,0])
                    #print(radar_occupancy[x,y,1])
                    #index[i] = 1
                    cost[n] = cost[n]+weight
                    #if(radar_occupancy[x,y,2]>70):
                    #    cost[n] = cost[n]+0.3
                    #    if(radar_occupancy[x,y,2]>75):
                    #        cost[n] = cost[n]+0.2
        #lidar = lidar[index == 1]
    #print(cost) 
    #print(np.sum(cost))      
    return cost

#@jit(nopython=True)
def count_points_normal(lidars,radar_occupancy_maps,radar_size):
    '''
    count Lidar Points fall in Radar Occupancy_voxel Map
    lidars: (Frams x (N x 4) )
    radar_occupancy_maps: ( Frams x (W x H x 4) )
    radar_size: (W x H) Radar Map size 
    '''
    max_point_number = 100
    frames = len(lidars)
    cost = np.zeros(frames)
    w = radar_size[0]
    h = radar_size[1]
    
    for n in range(frames):
        radar_occupancy_map = radar_occupancy_maps[n]
        lidar = lidars[n]

        lidar_voxel_idx = np.zeros([len(lidar), 2])
        lidar_voxel_idx[:, 0] = np.floor(lidar[:, 0]/100*w).astype(np.int)
        lidar_voxel_idx[:, 1] = np.floor(lidar[:, 1]/360*h).astype(np.int)
        lidar_voxel_idx = lidar_voxel_idx.astype(np.int)  # N x 2
        
        ''' remove Lidar out of Radar-Map '''
        ## Height: Lidars in -0.9d ~ 0.9d
        angle = np.arctan2(lidar[:, 2],lidar[:, 0])/np.pi*180
        lidar_angle_mask_top = angle < 0.9
        lidar_angle_mask_bottom = angle > -0.9
        lidar_angle_mask = np.logical_and(lidar_angle_mask_top,lidar_angle_mask_bottom)
        ## r-theta: Lidars fall in Radar-Map 
        radar_occupancy_mask = radar_occupancy_map[:,:,3] > 0  # W x H
        lidar_horizon_mask = radar_occupancy_mask[lidar_voxel_idx[:, 0],lidar_voxel_idx[:, 1]] > 0
        lidar_in_radar_map_mask = np.logical_and(lidar_angle_mask,lidar_horizon_mask)
        lidar_voxel_idx = lidar_voxel_idx[lidar_in_radar_map_mask]
        lidar = lidar[lidar_in_radar_map_mask]
        
        '''  Spacial Constaint '''
        # weight = d*d /(2*(du*du+db*db))
        voxel_height = lidar[:,0] * np.tan(0.9/180*np.pi)
        z = lidar[:,2]
        # upper = voxel_height
        # bottom = -voxel_height
        # d = upper - bottom
        # du = upper - z
        # db = z - bottom
        # weight = d*d /(2*(du*du+db*db))
        weight = (2*voxel_height)**2 /(2*((voxel_height - z)**2+(z + voxel_height)**2))
        
        
        ''' Intensity Constraint '''
        ## Radar intensity mask 
        # if radar_occupancy_intenstiy > 80:   weight = weight*1.5
        radar_intensity_mask = radar_occupancy_map[lidar_voxel_idx[:, 0],lidar_voxel_idx[:, 1],2] > 80
        lidar_intenty_weight = np.zeros_like(weight)
        lidar_intenty_weight[radar_intensity_mask] = weight[radar_intensity_mask]
        weight = weight + 0.5 * lidar_intenty_weight
        weight_one_frame = np.sum(weight)
        

        cost[n] = weight_one_frame
        
         
         
         
        # for i in range(len(lidar_voxel_idx)):
        #     lidar_voxel[radar_voxel_idx[i, 0], radar_voxel_idx[i, 1], 0] = radar_voxel_height[i]
        #     lidar_voxel[radar_voxel_idx[i, 0], radar_voxel_idx[i, 1], 1] = -radar_voxel_height[i]
        # # lidar_in_radar_map = (radar_occupancy_mask[lidar_voxel_idx] == True)
        # # print(lidar_in_radar_map.shape)
        # # lidar_voxel_idx = lidar_voxel_idx[lidar_in_radar_map]
        
        # # print(lidar_voxel_idx.shape)
        # # lidar_voxel_idx = lidar_voxel_idx[radar_occupancy_mask[lidar_voxel_idx] > 0 ] 
        # # print(lidar_voxel_idx.shape)
        
        # #number_buffer = 
        # # coordinate_counts 某一voxel内的lidar点数
        # coordinate_buffer,coordinate_index,coordinate_counts = np.unique(lidar_voxel_idx, axis=0, return_index = True, return_counts = True)
        # coodinate_height = lidar[coordinate_index,3]
        # print(coodinate_height.shape)
        # coordinate_buffer =  coordinate_buffer.astype(np.int64) - 1
        # K = len(coordinate_buffer)
        # mask = np.zeros_like(radar_occupancy)
        # print(radar_occupancy.shape)
        # counts = np.zeros_like(radar_occupancy)
        # #height = 
        # # for i in range(K):
            
        # #     mask[coordinate_buffer[i]] = 1
        # #     counts[coordinate_buffer[i]] = coordinate_counts[i]
            
        
        
        
        # # #print(coordinate_buffer.shape)
        # # #print(coordinate_index.shape)
        # # #print(coordinate_counts.shape)
        # # # print(coordinate_buffer[0])
        # # # print(coordinate_counts[0])
        # # # print(coordinate_buffer[1])
        # # # print(coordinate_counts[1])
        # # K = len(coordinate_buffer)
        
        # # T = max_point_number
        
        # # # [K, 1] store number of points in each voxel grid
        # # number_buffer = np.zeros(shape=(K), dtype=np.int64)

        # # # [K, T, 4] feature buffer as described in the paper
        # # # height
        # # feature_buffer = np.zeros(shape=(K, T, 4), dtype=np.float32)
        
        # # cost_buffer = np.zeros(shape=(K), dtype=np.int64)
        
        # # radar_occupancy_num = np.zeros_like(radar_occupancy[:,:,3])
        # # radar_occupancy_num = radar_occupancy[:,:,3]
        
        
        # # # build a reverse index for coordinate buffer
        # # index_buffer = {}
        # # for i in range(K):
        # #     index_buffer[tuple(coordinate_buffer[i])] = i

        # # for voxel, point in zip(lidar_voxel_idx, lidar):
        # #     index = index_buffer[tuple(voxel)]
        # #     number = number_buffer[index]
        # #     if number < T:
        # #         feature_buffer[index, number, :4] = point
        # #         number_buffer[index] += 1
        # #     # if 
        # #     # cost_buffer =          
        

        # # #feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - feature_buffer[:, :, :3].sum(axis=1, keepdims=True)/number_buffer.reshape(K, 1, 1)
    return cost
# _parallel
def count_points(lidars,radar_occupancy_maps,radar_size):
    '''
    count Lidar Points fall in Radar Occupancy_voxel Map
    lidars: (Frams x (N x 4) )
    radar_occupancy_maps: ( Frams x (W x H x 4) )
    radar_size: (W x H) Radar Map size 
    '''
    frames = len(lidars)
    #cost = np.zeros(frames)
    w = radar_size[0]
    h = radar_size[1]
    
    # for n in range(frames):
    #     radar_occupancy_map = radar_occupancy_maps[n]
    #     lidar = lidars[n]
    #     cost_one_framse = count_points_one_framse(lidar,radar_occupancy_map,radar_size)
        
    results = Parallel(n_jobs=4)(delayed(count_points_one_frame)(lidar,radar_occupancy_map,radar_size) for lidar,radar_occupancy_map in zip(lidars,radar_occupancy_maps))
    # backend='multiprocessing'
    #print(results)
    cost = np.array(results)
        

    return cost

def count_points_one_frame(lidar,radar_occupancy_map,radar_size):
    '''
    count Lidar Points fall in Radar Occupancy_voxel Map
    '''
    '''
    lidar: N x 4
    radar_occupancy_map: W x H x 4 
    radar_size: (W x H) Radar Map size 
    '''
    w = radar_size[0]
    h = radar_size[1]
    
    lidar_voxel_idx = np.zeros([len(lidar), 2])
    lidar_voxel_idx[:, 0] = np.floor(lidar[:, 0]/100*w).astype(np.int)
    lidar_voxel_idx[:, 1] = np.floor(lidar[:, 1]/360*h).astype(np.int)
    lidar_voxel_idx = lidar_voxel_idx.astype(np.int)  # N x 2
        
    ''' remove Lidar out of Radar-Map '''
    ## Height: Lidars in -1.8d ~ 1.8d
    angle = np.arctan2(lidar[:, 2],lidar[:, 0])/np.pi*180
    lidar_angle_mask_top = angle < 0.9
    lidar_angle_mask_bottom = angle > -0.9
    lidar_angle_mask = np.logical_and(lidar_angle_mask_top,lidar_angle_mask_bottom)
    ## r-theta: Lidars fall in Radar-Map 
    radar_occupancy_mask = radar_occupancy_map[:,:,3] > 0  # W x H
    lidar_horizon_mask = radar_occupancy_mask[lidar_voxel_idx[:, 0],lidar_voxel_idx[:, 1]] > 0
    lidar_in_radar_map_mask = np.logical_and(lidar_angle_mask,lidar_horizon_mask)
    lidar_voxel_idx = lidar_voxel_idx[lidar_in_radar_map_mask]
    lidar = lidar[lidar_in_radar_map_mask]
        
    '''  Spacial Constaint '''
    # weight = d*d /(2*(du*du+db*db))
    voxel_height = lidar[:,0] * np.tan(0.9/180*np.pi)
    z = lidar[:,2]
    # upper = voxel_height
    # bottom = -voxel_height
    # d = upper - bottom
    # du = upper - z
    # db = z - bottom
    # weight = d*d /(2*(du*du+db*db))
    weight = (2*voxel_height)**2 /(2*((voxel_height - z)**2+(z + voxel_height)**2))
        
        
    ''' Intensity Constraint '''
    ## Radar intensity mask 
    # if radar_occupancy_intenstiy > 80:   weight = weight * 1.5
    radar_intensity_mask = radar_occupancy_map[lidar_voxel_idx[:, 0],lidar_voxel_idx[:, 1],2] > 80
    lidar_intenty_weight = np.zeros_like(weight)
    lidar_intenty_weight[radar_intensity_mask] = weight[radar_intensity_mask]
    weight = weight + 0.5 * lidar_intenty_weight
    weight_one_frame = np.sum(weight)
        
    return weight_one_frame

def count_grids(lidars,radar_occupancy_map,radar_size):
    frames = len(lidars)
    #w,h = 100,360
    cost = np.zeros(frames)
    w,h = 576,400
    w = radar_size[0]
    h = radar_size[1]
    for n in range(frames):
        lidar = lidars[n]
        radar_occupancy = radar_occupancy_map[n]
        grid_count = np.zeros([w, h])
        lidar_num = lidar.shape[0]
        index = np.zeros(lidar_num)
        for i in range(lidar_num):
            x = np.floor(lidar[i,0]/100*w).astype(np.int64)
            y = np.floor(lidar[i,1]/360*h).astype(np.int64)
            z = lidar[i,2]
            intensity = lidar[i,3]
            print(intensity)
            if(x < w and radar_occupancy[x,y,1] != 0):
                
                #theta = (lidar[i,4]-23)*1.33
                #print(lidar[i,4])
                if(z< radar_occupancy[x,y,0] and z > radar_occupancy[x,y,1]):
                    grid_count[x,y] = grid_count[x,y]+1
                    #print(radar_occupancy[x,y,0])
                    #print(radar_occupancy[x,y,1])
                    index[i] = 1
        lidar = lidar[index == 1]
        for i in range(w):
            for j in range(h):
                if grid_count[i,j]>4:
                    cost[n]= cost[n]+1
    #print(cost)
            
    return cost
    

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

def count_T(lidar_car,radar_occupancy_map,radar_size, t = 5,res = 0.1,is_save = False,method = 'point' ):
    cost_tx = []
    cost_ty = []
    cost_tz = []
    R = np.zeros(3)
    T = np.zeros(3)
    cost_e = 1000
    num = len(lidar_car)
    for i in range (int(t/res)+1):
        T[0] = i*res-t/2
        print(T,R)
        _,lidar_polar = transform_lidar(lidar_car,T,R)
        if method == 'point':
            cost = count_points(lidar_polar,radar_occupancy_map,radar_size)
            cost = np.sum(cost)/(num*cost_e)
        elif method == 'grid':
            cost = count_grids(lidar_polar,radar_occupancy_map,radar_size)
            cost = np.sum(cost)/(num*cost_e)
        cost_tx.append(cost)     
    T = np.zeros(3)
    for i in range (int(t/res)+1):
        T[1] = i*res-t/2
        print(T,R)
        _,lidar_polar = transform_lidar(lidar_car,T,R)
        if method == 'point':
            cost = count_points(lidar_polar,radar_occupancy_map,radar_size)
            cost = np.sum(cost)/(num*cost_e)
        elif method == 'grid':
            cost = count_grids(lidar_polar,radar_occupancy_map,radar_size)
            cost = np.sum(cost)/(num*cost_e)
        cost_ty.append(cost)     
    T = np.zeros(3)
    for i in range (int(t/res)+1):
        T[2] = i*res-t/2
        print(T,R)
        _,lidar_polar = transform_lidar(lidar_car,T,R)
        if method == 'point':
            cost = count_points(lidar_polar,radar_occupancy_map,radar_size)
            cost = np.sum(cost)/(num*cost_e)
        elif method == 'grid':
            cost = count_grids(lidar_polar,radar_occupancy_map,radar_size)
            cost = np.sum(cost)/(num*cost_e)
        cost_tz.append(cost)     
    
    if is_save == True:
        f=open("cost_tx.txt","w")
        for line in cost_tx:
            f.write(str(line)+'\n')
        f.close()
        f=open("cost_ty.txt","w")
        for line in cost_ty:
            f.write(str(line)+'\n')
        f.close()
        f=open("cost_tz.txt","w")
        for line in cost_tz:
            f.write(str(line)+'\n')
        f.close()
    
    return cost_tx,cost_ty,cost_tz

def count_R(lidar_car,radar_occupancy_map,radar_size, r = 5,res = 0.1,is_save = False,method= 'point'):
    cost_rx = []
    cost_ry = []
    cost_rz = []
    R = np.zeros(3)
    T = np.zeros(3)
    cost_e = 1000
    num = len(lidar_car)
    for i in range (int(r/res)+1):
        R[0] = i*res-r/2
        print(T,R )
        _,lidar_polar = transform_lidar(lidar_car,T,R)
        if method == 'point':
            cost = count_points(lidar_polar,radar_occupancy_map,radar_size)
            cost = np.sum(cost)/(num*cost_e)
        elif method == 'grid':
            cost = count_grids(lidar_polar,radar_occupancy_map,radar_size)
            cost = np.sum(cost)/(num*cost_e)
        cost_rx.append(cost)     
    R = np.zeros(3)
    for i in range (int(r/res)+1):
        R[1] = i*res-r/2
        print(T,R)
        _,lidar_polar = transform_lidar(lidar_car,T,R)
        if method == 'point':
            cost = count_points(lidar_polar,radar_occupancy_map,radar_size)
            cost = np.sum(cost)/(num*cost_e)
        elif method == 'grid':
            cost = count_grids(lidar_polar,radar_occupancy_map,radar_size)
            cost = np.sum(cost)/(num*cost_e)
        cost_ry.append(cost)     
    R = np.zeros(3)
    for i in range (int(r/res)+1):
        R[2] = i*res-r/2
        print(T,R)
        _,lidar_polar = transform_lidar(lidar_car,T,R)
        if method == 'point':
            cost = count_points(lidar_polar,radar_occupancy_map,radar_size)
            cost = np.sum(cost)/(num*cost_e)
        elif method == 'grid':
            cost = count_grids(lidar_polar,radar_occupancy_map,radar_size)
            cost = np.sum(cost)/(num*cost_e)
        cost_rz.append(cost)     
    
    if is_save == True:
        f=open("cost_rx.txt","w")
        for line in cost_rx:
            f.write(str(line)+'\n')
        f.close()
        f=open("cost_ry.txt","w")
        for line in cost_ry:
            f.write(str(line)+'\n')
        f.close()
        f=open("cost_rz.txt","w")
        for line in cost_rz:
            f.write(str(line)+'\n')
        f.close()
    
    return cost_rx,cost_ry,cost_rz
    
def cost_fig(lidar_car,radar_car,radar_occupancy_map,radar_size,draw_T = True,t = 4,res_t = 0.1,draw_R = True,r = 20,res_r = 1,file_name = 'a',save_txt = True):
    
    #method = 'grid'
    method = 'point'
      
    if draw_T == True:
        print("cost_T")
        cost_tx,cost_ty,cost_tz = count_T(lidar_car,radar_occupancy_map,radar_size,t,res_t,save_txt,method)
        t=np.arange(-t/2.0,(t+res_t)/2.0,res_t) 
        draw_t_loss(t,cost_tx,cost_ty,cost_tz,file_name)

    if draw_R == True:
        print("cost_R")
        cost_rx,cost_ry,cost_rz = count_R(lidar_car,radar_occupancy_map,radar_size,r,res_r,save_txt,method)
        r=np.arange(-r/2.0,(r+res_r)/2.0,res_r) 
        draw_r_loss(r,cost_rx,cost_ry,cost_rz,file_name)
    
    