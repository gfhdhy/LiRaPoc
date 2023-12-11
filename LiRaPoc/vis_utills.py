import numpy as np
import scipy.optimize as opt
import os
import struct
from draw_fig import *
from bev_utils import bev, fusion_bev
from cost import transform_lidar
import vedo

def vis_transform(lidar_cars,radar_cars,T,R):
    frames = len(lidar_cars)
    lidar_transforms,_ = transform_lidar(lidar_cars,T,R)
    root = "vis/"
    for n in range(frames):
        radar_car = radar_cars[n]
        lidar_transform = lidar_transforms[n]
        lidar_bev = bev(lidar_transform,root+str(n)+str(R[0])+'lidar.jpg')
        radar_bev = bev(radar_car,root+str(n)+str(R[0])+'radar.jpg')
        
def vis_data(lidar_polars,radar_polars,lidar_cars,radar_cars):
    frames = len(lidar_cars)
    root = "vis/"
    for n in range(frames):
        radar_car = radar_cars[n]
        lidar_car = lidar_cars[n]
        radar_polar = radar_polars[n]
        lidar_polar = lidar_polars[n]
        lidar_bev = bev(lidar_car,root+str(n)+'lidar.jpg',mode = 'car')
        radar_bev = bev(radar_car,root+str(n)+'radar.jpg',mode = 'car')
        lidar_polar = bev(lidar_polar,root+str(n)+'lidar_polar.jpg',mode = 'polar')
        radar_polar = bev(radar_polar,root+str(n)+'radar_polar.jpg',mode = 'polar')
    
def vis_transform_fusion(lidar_cars,radar_cars,T,R,fig_file):
    frames = len(lidar_cars)
    lidar_transforms,_ = transform_lidar(lidar_cars,T,R)
    root = "vis_fusion/"
    if not os.path.exists(root):
        os.makedirs(root)
    for n in range(frames):
        radar_car = radar_cars[n]
        lidar_transform = lidar_transforms[n]
        fusion = fusion_bev(lidar_transform,radar_car,root+fig_file+str(n)+'fusion.jpg')

def vis_3d(lidar_polar,radar_occupancy_map,radar_size):
        for n in range(len(lidar_polar)):
            radar_occupancy = radar_occupancy_map[0]
            lidar = lidar_polar[n]
            boxes = []
            for i in range(radar_size[0]):
                for j in range(radar_size[1]):
                    if radar_occupancy[i][j][0]!= 0:
                        box = vedo.Box((i, j, 0), length = 0.9, width = 0.9, height=2*radar_occupancy[i,j,0], c=(251,132,2),
                                alpha=0.5)

                        box.lc((251, 132, 2))
                        boxes.append(box)
            w = radar_size[0]
            h = radar_size[1]
            lidar[:, 0] = lidar[:, 0] / 100 * w
            lidar[:, 1] = lidar[:, 1] / 360 * h
            lidar[:, 2] = lidar[:, 2]
            lidar_point = vedo.Points(lidar[:, :3],c = (39,158,188))
            vedo.show(lidar_point, boxes,  __doc__,axes=4).close()

def vis_radar_box(radar_occupancy_map,radar_size):
        for n in range(len(radar_occupancy_map)):
            radar_occupancy = radar_occupancy_map[0]
            boxes = []
            for i in range(radar_size[0]):
                for j in range(radar_size[1]):
                    if radar_occupancy[i][j][0]!= 0:
                        box = vedo.Box((i, j, 0), length = 0.9, width = 0.9, height=10*radar_occupancy[i,j,0], c=(251,132,2),
                                alpha=0.5)

                        box.lc((251, 132, 2))
                        boxes.append(box)
            w = radar_size[0]
            h = radar_size[1]
            vedo.show(boxes,  __doc__,axes=4).close()     
    
