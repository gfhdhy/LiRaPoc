import numpy as np
import scipy.optimize as opt
from joblib  import Parallel, delayed


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
        # radar_voxel_idx (n x 2) 
        radar_voxel_idx[:, 0] = np.floor(radar[:, 0]/100*w)
        radar_voxel_idx[:, 1] = np.floor(radar[:, 1]/360*h)
        radar_voxel_height = radar[:,2]
        radar_intensity = radar[:,3]
        radar_voxel_idx = radar_voxel_idx.astype(np.int64)
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

def count_points(lidars,radar_occupancy_maps,radar_size):
    '''
    count Lidar Points fall in Radar Occupancy_voxel Map
    lidars: (Frams x (N x 4) )
    radar_occupancy_maps: ( Frams x (W x H x 4) )
    radar_size: (W x H) Radar Map size 
    '''

    # for n in range(len(lidars)):
    #     radar_occupancy_map = radar_occupancy_maps[n]
    #     lidar = lidars[n]
    #     cost_one_framse = count_points_one_framse(lidar,radar_occupancy_map,radar_size)
        
    results = Parallel(n_jobs=8)(delayed(count_points_one_frame)(lidar,radar_occupancy_map,radar_size) for lidar,radar_occupancy_map in zip(lidars,radar_occupancy_maps))
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
    lidar_voxel_idx[:, 0] = np.floor(lidar[:, 0]/100*w).astype(np.int64)
    lidar_voxel_idx[:, 1] = np.floor(lidar[:, 1]/360*h).astype(np.int64)
    lidar_voxel_idx = lidar_voxel_idx.astype(np.int64)  # N x 2
        
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
    # if radar_occupancy_intenstiy > 80:   weight = weight * 1.5
    radar_intensity_mask = radar_occupancy_map[lidar_voxel_idx[:, 0],lidar_voxel_idx[:, 1],2] > 80
    lidar_intenty_weight = np.zeros_like(weight)
    lidar_intenty_weight[radar_intensity_mask] = weight[radar_intensity_mask]
    weight = weight + 0.5 * lidar_intenty_weight
    weight_one_frame = np.sum(weight)
        
    return weight_one_frame

    

