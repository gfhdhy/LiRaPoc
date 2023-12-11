import numpy as np
import os
import struct
from dataloader.orr_dataloader import orr_dataloader
from dataloader.boreas_dataloader import boreas_dataloader
from cost import build_radar_occupancy_map
from loss import loss
from icp import icp
from match_template import match_template
import argparse

        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='LiRaPoc')
    parser.add_argument('--dataset', default='boreas', type=str, help='dataset')
    parser.add_argument('--datapath', default='/home/dy/2dar-ndt/data/boreas/', type=str, help='dataset path')
    parser.add_argument('--datasequence', default='boreas_example', help='dataset sequence')
    parser.add_argument('--method', default = 'LiRaPoc', help='method')
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
    
    if args.method == "icp":
        icp(lidar_car,radar_car)
    elif args.method == 'mt':
        match_template(lidar_car,radar_car)
    elif args.method == 'LiRaPoc':
        stride = [0.1,0.1,0.1,0.2,0.2,0.2]
        loss(radar_occupancy_map,radar_car,lidar_car,radar_size,stride)
    else:
        print("method error")
        
        

    