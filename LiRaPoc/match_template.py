import numpy as np
import scipy.optimize as opt
import os
import struct
from draw_fig import *
from cost import *
import cv2
    
def match_template(lidars,radars):
    frames = len(lidars)
    # matrixs = np.zeros((4,4))
    # T = np.zeros(3)
    # R = np.zeros(3)
    trans = np.zeros(2)
    # R[2] = 180
    # lidar_transforms,_ = transform_lidar(lidars,T,R)
    for i in range(frames):
        lidar = lidars[i]
        radar = radars[i]
        lidar_bev = bev(lidar,'vis_mt/'+str(i)+'lidar.jpg',mode = 'car',delta_l=0.4, pixel_l=490)
        radar_bev = bev(radar,'vis_mt/'+str(i)+'radar.jpg')
        result = cv2.matchTemplate(lidar_bev,radar_bev,cv2.TM_CCOEFF)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        maxLoc = np.array(maxLoc)
        maxLoc = maxLoc - 5
        print(i)
        print(maxLoc)
        trans = trans+maxLoc
    trans = trans/frames
    print("all")
    print(trans)
    return trans
        
    