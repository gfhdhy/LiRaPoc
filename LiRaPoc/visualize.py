import numpy as np
import os
import struct
from dataloader.orr_dataloader import orr_dataloader
from dataloader.boreas_dataloader import boreas_dataloader
from cost import build_radar_occupancy_map
from vis_utills import vis_data,vis_transform,vis_transform_fusion,vis_3d,vis_radar_box
import argparse

        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='LiRaPoc')
    parser.add_argument('--dataset', default='boreas', type=str, help='dataset')
    parser.add_argument('--datapath', default='/home/dy/2dar-ndt/data/boreas/', type=str, help='dataset path')
    parser.add_argument('--datasequence', default='boreas_example', help='dataset sequence')
    parser.add_argument('--vismode', default = '2d', help='method')
    args = parser.parse_args()

    dataset = args.dataset
    sequence = args.datasequence
    datapath = args.datapath 
    lidar_root_dir = datapath + sequence + '/lidar/'
    radar_root_dir = datapath + sequence + '/radar/'
    
    root_dir = datapath + sequence + '/radar/'
    filenames = os.listdir(root_dir)
    file_numbers = len(filenames)
    print("read ",file_numbers," file") 
     
    if dataset == 'orr':
        lidar_polar,radar_polar,lidar_car,radar_car = orr_dataloader(lidar_root_dir,radar_root_dir)
        radar_size = [3768,400] 
    elif dataset == 'boreas':
        lidar_polar,radar_polar,lidar_car,radar_car = boreas_dataloader(lidar_root_dir,radar_root_dir)
        radar_size = [3360,400]
    else:
        print("dataset error")
    
    radar_occupancy_map = build_radar_occupancy_map(radar_polar,radar_size )
    print("--suceess buliding radar map--")
    

    
    if args.vismode == "2d":
        T = [0,0,0]
        R = [0,0,0]
        # # orr
        # # gt
        # T = [0.09,0.44,0.28]
        # R = [-0.17,-0.46,0.34]  
        # # mt
        # T = [0.14,0.45,0]
        # R = [0,0,0]
        # # icp
        # T = [-0.28,0.39,0.96]
        # R = [-2.27,0.68,0.16]
        # ours
        # T = [0.08,0.31,0.14]
        # R = [0.02,-0.01,0.09]
        
        
        # #  boreas
        # T = [0.0,0.0,0.21]
        # R = [0.0,0.0,-2.2517]
        # #  boreas_mt
        # T = [-0.3,0.2,0.0]
        # R = [0.0,0.0,0.0]
        # # #  boreas_icp
        # T = [-0.13,0.11,0.36]
        # R = [0.07,0.09,-2.57]
        # # #  boreas_ours
        # T = [0.0,0.0,0.29]
        # R = [0.09,0.51,-2.61]
        #vis_transform(lidar_car,radar_car,T,R)
        method = 'gt' # 0 gt mt icp ours 
        fig_file = str(sequence[:-1]) + '_' + str(method)
        vis_transform_fusion(lidar_car,radar_car,T,R,fig_file)
    elif args.vismode == '3d':
        vis_3d(lidar_polar,radar_occupancy_map,radar_size)
    elif args.vismode == 'radar_box':
        vis_radar_box(radar_occupancy_map,radar_size)
    else:
        print("vismode error")
        
        

    