import numpy as np
from bev_utils import bev
import cv2
    
def match_template(lidars,radars):
    frames = len(lidars)
    trans = np.zeros(2)
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
    trans = trans * 0.4
    print("match template")
    print("x",trans[1] )
    print("y",trans[0])
    return trans
        
    