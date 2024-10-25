import numpy as np
import cv2


def bev(points,save_name ='bev.jpg',mode = 'car', delta_l=0.4, pixel_l=500):
    
    if mode == 'car':
        bev_image = point2bev(points,delta_l, pixel_l).astype(np.float32) 
        bev_image = (bev_image * 255).astype(np.uint8)
        bev_image = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2RGB)
    return bev_image


def point2bev(lidar_data, delta_l=0.2, pixel_l=500, h1=-10, h2=10, delta_h=0.1):
    l1 = (-pixel_l/2) * delta_l
    l2 = (pixel_l/2) * delta_l

    idx_x = np.logical_and(lidar_data[:,0] >= l1, lidar_data[:,0] < l2)
    idx_y = np.logical_and(lidar_data[:,1] >= l1, lidar_data[:,1] < l2)
    idx_z = np.logical_and(lidar_data[:,2] >= h1, lidar_data[:,2] < h2)
    idx_valid = np.logical_and(idx_z, np.logical_and(idx_y, idx_x))
    lidar_data = lidar_data[idx_valid, :]


    lidar_bev_idx = np.zeros([len(lidar_data), 2])
    lidar_bev_idx[:,0] = np.floor((-lidar_data[:,0] - l1) / delta_l)
    lidar_bev_idx[:,1] = np.floor((-lidar_data[:,1] - l1) / delta_l)
    lidar_bev_idx[lidar_bev_idx == pixel_l] = pixel_l - 1
    lidar_bev_idx = lidar_bev_idx.astype(np.int64)

    lidar_intensity = np.zeros([pixel_l, pixel_l])

    for i in range(len(lidar_bev_idx)):
        lidar_intensity[lidar_bev_idx[i,0], lidar_bev_idx[i,1]] = 1
    return lidar_intensity