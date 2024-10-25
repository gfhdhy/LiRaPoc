import numpy as np
import os
import argparse
from dataloader.orr_dataloader import orr_dataloader
from dataloader.boreas_dataloader import boreas_dataloader
from cost import build_radar_occupancy_map
from loss import run_lirapoc,transform_lidar
from icp import icp
from match_template import match_template


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

    initial_T = [1,1,1]  
    initial_R = [2,2,2]  

     
    if dataset == "orr":
        lidar_polar,radar_polar,lidar_car,radar_car = orr_dataloader(lidar_root_dir,radar_root_dir)
        radar_size = [3768,400] # original range 165m
        radar_size = [2283,400] # cropped to 100m
        # orr extrinsic
        T = [0.09,0.44,0.28]  
        R = [-0.17,-0.46,0.34]
        lidar_car,_ = transform_lidar(lidar_car,T,R)
    elif dataset == "boreas":
        lidar_polar,radar_polar,lidar_car,radar_car = boreas_dataloader(lidar_root_dir,radar_root_dir)
        radar_size = [3360,400] # original range 200m
        radar_size = [1680,400] # cropped to 100m
        # boreas extrinsic
        T = [0.0,0.0,0.21]
        R = [0.00,0.00,-2.25]
        lidar_car,_ = transform_lidar(lidar_car,T,R)
    else:
        print("dataset error")

    ## generate radar occupancy gird map
    radar_occupancy_map = build_radar_occupancy_map(radar_polar,radar_size )
    print("--suceess buliding radar map--")


    lidar_car,_ = transform_lidar(lidar_car,initial_T,initial_R)


    if args.method == "icp":
        icp(lidar_car,radar_car)
    elif args.method == 'mt':
        match_template(lidar_car,radar_car)
    elif args.method == 'LiRaPoc':
        run_lirapoc(radar_occupancy_map,radar_car,lidar_car,radar_size)
    else:
        print("method error")
 
    