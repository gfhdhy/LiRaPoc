import numpy as np
import os
import struct
from draw_fig import *
from cost import *
import random
import matplotlib.pyplot as plt
from loss import loss,residuals
import pandas as pd
    
def loss_random(radar_occupancy_map,radar_car,lidar_car,radar_size,stride,n = 3):
    initial_p0 = []
    optimized_p = []
    for i in range(n):
        p0 = np.zeros(6)
        p0[0] = random.uniform(-1, 1)
        p0[1] = random.uniform(-1, 1)
        p0[2] = random.uniform(-1, 1)
        p0[3] = random.uniform(-5, 5)
        p0[4] = random.uniform(-5, 5)
        p0[5] = random.uniform(-5, 5)
        p0 = np.around(p0, 4)
        initial_p0.append(p0)
        p = loss_p0(radar_occupancy_map,radar_car,lidar_car,radar_size,stride,p0)
        p = p.astype(float)
        p = np.around(p, 4)
        optimized_p.append(p)
    initial_p0 = np.array(initial_p0)
    optimized_p = np.array(optimized_p)
    return initial_p0,optimized_p

def loss_p0(radar_occupancy_map,radar_car,lidar_car,radar_size,stride,p0):

    
    lower_bounds = [-2, -2,-2,-10,-10,-10]
    upper_bounds = [2, 2,2,10,10,10]
    # stride = [0.1,0.1,0.1,0.2,0.2,0.2]
   
    loss = residuals(p0,lidar_car,radar_occupancy_map,radar_size)
    
    print(loss)
    loss = opt.least_squares(residuals,p0,bounds=(lower_bounds, upper_bounds),diff_step = stride,args=(lidar_car,radar_occupancy_map,radar_size),method="trf",gtol=None,ftol=0.01,xtol=None,verbose=0)
    p = loss.x
    print(p)
    c = loss.cost
    print(c)
    
        #print(loss)
    path = 'vis_loss/loss.txt' 
    file = open(path,'a')
    file.write('\n')    
    T = p[:3]
    R = p[3:]
    return p


def save_loss_random_txt(initial_p0,optimized_p,fig_file):
    np.savetxt("cost_random/"+fig_file+"_p0.txt",initial_p0,fmt='%0.4f')
    np.savetxt("cost_random/"+fig_file+"_p.txt",optimized_p,fmt='%0.4f')
    
def draw_loss_random(initial_p0,optimized_p,fig_file):
    num = len(initial_p0)
    if num != len(optimized_p):
        print("erro")
        return 0
    p0_x = initial_p0[:,0]
    p_x = optimized_p[:,0]
    plt.figure(0)
    plt.scatter(p0_x,  # 横坐标
            p_x,  # 纵坐标
            c='red',  # 点的颜色
            label='function')  # 标签 即为点代表的意思
    plt.ylim((-1, 1))

    plt.xlabel(r"ininal guess of x ($[m]$)")
    plt.ylabel(r"result of x ($[m]$)")
    plt.savefig("cost_random/fig/"+  fig_file + "_x.jpg", bbox_inches='tight', pad_inches=0)
    
    plt.figure(1)
    p0_y = initial_p0[:,1]
    p_y = optimized_p[:,1]
    plt.scatter(p0_y,  # 横坐标
            p_y,  # 纵坐标
            c='red',  # 点的颜色
            label='function')  # 标签 即为点代表的意思
    plt.ylim((-1, 1))
    plt.xlabel(r"ininal guess of y ($[m]$)")
    plt.ylabel(r"result of y ($[m]$)")
    plt.savefig("cost_random/fig/"+  fig_file + "_y.jpg", bbox_inches='tight', pad_inches=0)
    
    plt.figure(2)
    p0_z = initial_p0[:,2]
    p_z = optimized_p[:,2]
    plt.scatter(p0_z,  # 横坐标
            p_z,  # 纵坐标
            c='red',  # 点的颜色
            label='function')  # 标签 即为点代表的意思
    plt.ylim((-1, 1))
    plt.xlabel(r"ininal guess of z ($[m]$)")
    plt.ylabel(r"result of z ($[m]$)")
    plt.savefig("cost_random/fig/"+  fig_file + "_z.jpg", bbox_inches='tight', pad_inches=0)
    
    plt.figure(3)
    p0_rx = initial_p0[:,3]
    p_rx = optimized_p[:,3]
    plt.scatter(p0_rx,  # 横坐标
            p_rx,  # 纵坐标
            c='red',  # 点的颜色
            label='function')  # 标签 即为点代表的意思
    plt.ylim((-5, 5))
    plt.xlabel(r"ininal guess of ${\Theta}_{x}$ ($[m]$)")
    plt.ylabel(r"result of ${\Theta}_{x}$ ($[m]$)")
    plt.savefig("cost_random/fig/"+  fig_file + "_rx.jpg", bbox_inches='tight', pad_inches=0)
    
    plt.figure(4)
    p0_ry = initial_p0[:,4]
    p_ry = optimized_p[:,4]
    plt.scatter(p0_ry,  # 横坐标
            p_ry,  # 纵坐标
            c='red',  # 点的颜色
            label='function')  # 标签 即为点代表的意思
    plt.ylim((-5, 5))
    plt.xlabel(r"ininal guess of ${\Theta}_{y}$ ($[m]$)")
    plt.ylabel(r"result of ${\Theta}_{y}$ ($[m]$)")
    plt.savefig("cost_random/fig/"+  fig_file + "_ry.jpg", bbox_inches='tight', pad_inches=0)
    
    plt.figure(5)
    p0_rz = initial_p0[:,5]
    p_rz = optimized_p[:,5]
    plt.scatter(p0_rz,  # 横坐标
            p_rz,  # 纵坐标
            c='red',  # 点的颜色
            label='function')  # 标签 即为点代表的意思
    plt.ylim((-5, 5))
    plt.xlabel(r"ininal guess of ${\Theta}_{z}$ ($[m]$)")
    plt.ylabel(r"result of ${\Theta}_{z}$ ($[m]$)")
    plt.savefig("cost_random/fig/"+  fig_file + "_rz.jpg", bbox_inches='tight', pad_inches=0)
    
def draw_loss_random_index(initial_p0,optimized_p,fig_file,medians):
    gt_boreas = ['0.00','0.00','0.21','0.00','0.00','-2.25']
    gt_orr = ['0.09','0.44','0.28','-0.17','-0.46','0.34']
    gt = gt_orr
    gt = gt_boreas
    num = len(initial_p0) 
    
    if num != len(optimized_p):
        print("erro")
        return 0
    index = np.arange(0,num)
    initial_p0 = np.array(initial_p0)
    optimized_p = np.array(optimized_p)
    
    
    p0_x = initial_p0[:,0] * 100
    p_x = optimized_p[:,0] * 100
    p0_y = initial_p0[:,1] * 100
    p_y = optimized_p[:,1] * 100
    p0_z = initial_p0[:,2] * 100
    p_z = optimized_p[:,2] * 100
    p0_rx = initial_p0[:,3]
    p_rx = optimized_p[:,3]
    p0_ry = initial_p0[:,4]
    p_ry = optimized_p[:,4]
    p0_rz = initial_p0[:,5]
    p_rz = optimized_p[:,5]
    
    # t_offset = 200
    # r_offset = 10
    
    # outlier_x = np.logical_or(p_x > medians[0]* 100 + t_offset ,p_x < medians[0]* 100 - t_offset)
    # outlier_y = np.logical_or(p_y > medians[1]* 100 + t_offset ,p_y < medians[1]* 100 - t_offset)
    # outlier_z = np.logical_or(p_z > medians[2]* 100 + t_offset ,p_z < medians[2]* 100 - t_offset)
    # outlier_t = np.logical_or(np.logical_or(outlier_x,outlier_y),outlier_z)
    # inlier_t = np.logical_not(outlier_t)
    
    # outlier_rx = np.logical_or(p_rx > medians[3] + r_offset ,p_rx < medians[3] - r_offset)
    # outlier_ry = np.logical_or(p_ry > medians[4] + r_offset ,p_ry < medians[4] - r_offset)
    # outlier_rz = np.logical_or(p_rz > medians[5] + r_offset ,p_rz < medians[5] - r_offset)
    # outlier_r = np.logical_or(np.logical_or(outlier_rx,outlier_ry),outlier_rz)
    # inlier_r = np.logical_not(outlier_r)
       
    # outlier = np.logical_or(outlier_t,outlier_r)
    # index_outlier = index[outlier]
    # outlier_point_x = p_x[outlier]
    # outlier_point_y = p_y[outlier]
    # outlier_point_z = p_z[outlier]
    # outlier_point_rx = p_rx[outlier]
    # outlier_point_ry = p_ry[outlier]
    # outlier_point_rz = p_rz[outlier]
    
    # inlier = np.logical_and(inlier_t,inlier_r)
    # index_inlier = index[inlier]
    # inlier_point_x = p_x[inlier]
    # inlier_point_y = p_y[inlier]
    # inlier_point_z = p_z[inlier]
    # inlier_point_rx = p_rx[inlier]
    # inlier_point_ry = p_ry[inlier]
    # inlier_point_rz = p_rz[inlier]
    
    
    plt.figure(12)
    plt.scatter(index,  # 横坐标
            p0_x,  # 纵坐标
            c='#84C3C6',  # 点的颜色
            marker="s",
            label="initial guess")  # 标签 即为点代表的意思
    plt.scatter(index_inlier,  # 横坐标
            inlier_point_x,  # 纵坐标
            c='#3F589B',  # 点的颜色
            label="optimized result")  # 标签 即为点代表的意思
#     plt.scatter(index_outlier,  # 横坐标
#             outlier_point_x,  # 纵坐标
#             c='#D24B4F',  # 点的颜色
#             marker='^',
#             label='outlier')  # 标签 即为点代表的意思
    plt.ylim((-110, 110))
    plt.legend(loc = 'upper right',fontsize=22)
    # 3.展示图形
    #plt.legend()  # 显示图例
    plt.tick_params(labelsize=22)
    plt.title(r'Ground Truth ${t}_{x}$: '+gt[0],fontsize=30) 
    plt.xlabel(r"index of initial guess",fontsize=36)
    plt.ylabel(r"${t}_{x}$ ($[cm]$)",fontsize=36)
    plt.savefig("cost_random/index/"+  fig_file + "_x_index.jpg", bbox_inches='tight', pad_inches=0)
    
    plt.figure(7)
    
    plt.scatter(index,  # 横坐标
            p0_y,  # 纵坐标
            c='#84C3C6',  # 点的颜色
            marker="s",
            label="initial guess")  # 标签 即为点代表的意思
    plt.scatter(index_inlier,  # 横坐标
            inlier_point_y,  # 纵坐标
            c='#3F589B',  # 点的颜色
            label="optimized result")  # 标签 即为点代表的意思
#     plt.scatter(index_outlier,  # 横坐标
#             outlier_point_y,  # 纵坐标
#             c='#D24B4F',  # 点的颜色
#             marker='^',
#             label='outlier')  # 标签 即为点代表的意思
    plt.legend(loc = 'upper right',fontsize=22)
    plt.tick_params(labelsize=22)
    #plt.xlim((-2, 2))
    plt.ylim((-110, 110))
    plt.title(r'Ground Truth ${t}_{y}$: '+gt[1],fontsize=30) 
    plt.xlabel(r"index of initial guess",fontsize=36)
    plt.ylabel(r"${t}_{y}$ ($[cm]$)",fontsize=36)
    plt.savefig("cost_random/index/"+  fig_file + "_y_index.jpg", bbox_inches='tight', pad_inches=0)
    
    plt.figure(8)
    
    plt.scatter(index,  # 横坐标
            p0_z,  # 纵坐标
            c='#84C3C6',  # 点的颜色
            marker="s",
            label="initial guess")  # 标签 即为点代表的意思
    plt.scatter(index_inlier,  # 横坐标
            inlier_point_z,  # 纵坐标
            c='#3F589B',  # 点的颜色
            label="optimized result")  # 标签 即为点代表的意思
#     plt.scatter(index_outlier,  # 横坐标
#             outlier_point_z,  # 纵坐标
#             c='#D24B4F',  # 点的颜色
#             marker='^',
#             label='outlier')  # 标签 即为点代表的意思
    plt.legend(loc = 'upper right',fontsize=22)
    plt.tick_params(labelsize=22)
    plt.ylim((-110, 110))
    plt.title(r'Ground Truth ${t}_{z}$: '+gt[2],fontsize=30) 
    plt.xlabel(r"index of initial guess",fontsize=36)
    plt.ylabel(r"${t}_{z}$ ($[cm]$)",fontsize=36)
    plt.savefig("cost_random/index/"+  fig_file + "_z_index.jpg", bbox_inches='tight', pad_inches=0)
    
    plt.figure(9)

    plt.scatter(index,  # 横坐标
            p0_rx,  # 纵坐标
            c='#84C3C6',  # 点的颜色
            marker="s",
            label="initial guess")  # 标签 即为点代表的意思
    plt.scatter(index_inlier,  # 横坐标
            inlier_point_rx,  # 纵坐标
            c='#3F589B',  # 点的颜色
            label="optimized result")  # 标签 即为点代表的意思
#     plt.scatter(index_outlier,  # 横坐标
#             outlier_point_rx,  # 纵坐标
#             c='#D24B4F',  # 点的颜色
#             marker='^',
#             label='outlier')  # 标签 即为点代表的意思
    plt.legend(loc = 'upper right',fontsize=22)
    plt.tick_params(labelsize=22)
    plt.ylim((-5.5, 5.5))
    plt.title(r'Ground Truth ${r}_{x}$: '+gt[3],fontsize=30) 
    plt.xlabel(r"index of initial guess",fontsize=36)
    plt.ylabel(r"${r}_{x}$ ($[deg]$)",fontsize=36)
    plt.savefig("cost_random/index/"+  fig_file + "_rx_index.jpg", bbox_inches='tight', pad_inches=0)
    
    plt.figure(10)
    
    plt.scatter(index,  # 横坐标
            p0_ry,  # 纵坐标
            c='#84C3C6',  # 点的颜色
            marker="s",
            label="initial guess")  # 标签 即为点代表的意思
    plt.scatter(index_inlier,  # 横坐标
            inlier_point_rx,  # 纵坐标
            c='#3F589B',  # 点的颜色
            label="optimized result")  # 标签 即为点代表的意思
#     plt.scatter(index_outlier,  # 横坐标
#             outlier_point_ry,  # 纵坐标
#             c='#D24B4F',  # 点的颜色
#             marker='^',
#             label='outlier')  # 标签 即为点代表的意思
    plt.legend(loc = 'upper right',fontsize=22)
    plt.tick_params(labelsize=22)
    plt.ylim((-5.5, 5.5))
    plt.title(r'Ground Truth ${r}_{y}$: '+gt[4],fontsize=30) 
    plt.xlabel(r"index of initial guess",fontsize=36)
    plt.ylabel(r"${r}_{y}$ ($[deg]$)",fontsize=36)
    plt.savefig("cost_random/index/"+  fig_file + "_ry_index.jpg", bbox_inches='tight', pad_inches=0)
    
    plt.figure(11)
    
    plt.scatter(index,  # 横坐标
            p0_rz,  # 纵坐标
            c='#84C3C6',  # 点的颜色
            marker="s",
            label="initial guess")  # 标签 即为点代表的意思
    plt.scatter(index_inlier,  # 横坐标
            inlier_point_rz,  # 纵坐标
            c='#3F589B',  # 点的颜色
            label="optimized result")  # 标签 即为点代表的意思
#     plt.scatter(index_outlier,  # 横坐标
#             outlier_point_rz,  # 纵坐标
#             c='#D24B4F',  # 点的颜色
#             marker='^',
#             label='outlier')  # 标签 即为点代表的意思
    plt.legend(loc = 'upper right',fontsize=22)
    plt.tick_params(labelsize=22)
    plt.ylim((-5.5, 5.5))
    plt.title(r'Ground Truth ${r}_{z}$: '+gt[5],fontsize=30) 
    plt.xlabel(r"index of initial guess",fontsize=36)
    plt.ylabel(r"${r}_{z}$ ($[deg]$)",fontsize=36)
    plt.savefig("cost_random/index/"+  fig_file + "_rz_index.jpg", bbox_inches='tight', pad_inches=0)

def read_txt(file_name_p0,file_name_p):
    initial_p0 = []
    optimized_p = []
    file_p0 = open(file_name_p0,'r')
    lines = file_p0.readlines()
    for line in lines:
        x = np.zeros(6)
        a = line.split()
        x = a[0:6]
        x = np.array(x)  
        initial_p0.append(x)
    file_p0.close()
    
    file_p = open(file_name_p,'r')
    lines = file_p.readlines()
    for line in lines:
        x = np.zeros(6)
        a = line.split()
        x = a[0:6] 
        x = np.array(x)  
        optimized_p.append(x)
    file_p.close()
    initial_p0 = np.array(initial_p0)
    optimized_p = np.array(optimized_p)
    initial_p0 = initial_p0.astype(float)
    optimized_p = optimized_p.astype(float)
    return initial_p0,optimized_p   
def draw_loss_box(initial_p0,optimized_p,fig_file):
    num = len(initial_p0)
    if num != len(optimized_p):
        print("erro")
        print("initial_p0",num)
        print("optimized_p",len(optimized_p))
        return 0
    initial_p0 = np.array(initial_p0)
    optimized_p = np.array(optimized_p)
    medians = np.zeros(6)
    
    plt.figure(0)
    p0_x = initial_p0[:,0]
    p_x = optimized_p[:,0]
    df = pd.DataFrame(p_x)
    x_medians = df.median()
    medians[0] = x_medians
    p_xx = -p_x
    p_x = np.concatenate((p_xx,p_x,p_x),axis=0)
    df = pd.DataFrame(p_x)
    df = df.astype(float)
    df.plot.box(title="x")
    plt.grid(linestyle="--", alpha=0.3)
    plt.ylim((-1.1, 1.1))
    plt.savefig("cost_random/box/"+  fig_file + "_x.jpg", bbox_inches='tight', pad_inches=0)
    
    
    plt.figure(1)
    p0_y = initial_p0[:,1]
    p_y = optimized_p[:,1]
    df = pd.DataFrame(p_y)
    y_medians = df.median()
    medians[1] = y_medians
    df = df.astype(float)
    df.plot.box(title="y")
    plt.grid(linestyle="--", alpha=0.3)
    plt.ylim((-1.1, 1.1))
    plt.savefig("cost_random/box/"+  fig_file + "_y.jpg", bbox_inches='tight', pad_inches=0)
    
    plt.figure(2)
    p0_z = initial_p0[:,2]
    p_z = optimized_p[:,2]
    df = pd.DataFrame(p_z)
    z_medians = df.median()
    medians[2] = z_medians
    df = df.astype(float)
    df.plot.box(title="z")
    plt.grid(linestyle="--", alpha=0.3)
    plt.ylim((-1.1, 1.1))
    plt.savefig("cost_random/box/"+  fig_file + "_z.jpg", bbox_inches='tight', pad_inches=0)
    
    plt.figure(3)
    p0_rx = initial_p0[:,3]
    p_rx = optimized_p[:,3]
    df = pd.DataFrame(p_rx)
    rx_medians = df.median()
    medians[3] = rx_medians
    df = df.astype(float)
    df.plot.box(title="rx")
    plt.grid(linestyle="--", alpha=0.3)
    plt.ylim((-5.5, 5.5))
    plt.savefig("cost_random/box/"+  fig_file + "_rx.jpg", bbox_inches='tight', pad_inches=0)
    
    
    plt.figure(4)
    p0_ry = initial_p0[:,4]
    p_ry = optimized_p[:,4]
    df = pd.DataFrame(p_ry)
    ry_medians = df.median()
    medians[4] = ry_medians
    df = df.astype(float)
    df.plot.box(title="ry")
    plt.grid(linestyle="--", alpha=0.3)
    plt.ylim((-5.5, 5.5))
    plt.savefig("cost_random/box/"+  fig_file + "_ry.jpg", bbox_inches='tight', pad_inches=0)
    
    
    plt.figure(5)
    p0_rz = initial_p0[:,5]
    p_rz = optimized_p[:,5]
    df = pd.DataFrame(p_rz)
    rz_medians = df.median()
    medians[5] = rz_medians
    df = df.astype(float)
    df.plot.box(title="rz")
    plt.grid(linestyle="--", alpha=0.3)
    plt.ylim((-5.5, 5.5))
    plt.savefig("cost_random/box/"+  fig_file + "_rz.jpg", bbox_inches='tight', pad_inches=0)
    
    return medians


def draw_loss_box_multi(initial_p0_1,optimized_p_1,initial_p0_2,optimized_p_2,initial_p0_3,optimized_p_3,fig_file):
    num = len(initial_p0_1)
    if num != len(optimized_p_1):
        print("erro")
        print("initial_p0",num)
        print("optimized_p",len(optimized_p))
        return 0
    
    column = ['1 frames','10 frames','28 frames']
    column = ['100 frames','200 frames','500 frames']
    p_x_1 = optimized_p_1[:,0]
    p_x_2 = optimized_p_2[:,0]
    p_x_3 = optimized_p_3[:,0]
    p_x = np.concatenate((np.expand_dims(p_x_1,axis=1),np.expand_dims(p_x_2,axis=1),np.expand_dims(p_x_3,axis=1)),axis=1)
    # m -> cm
    p_x = p_x * 100 
    df = pd.DataFrame(p_x,columns=column)
    df = df.astype(float)
    df.plot.box()
    plt.grid(linestyle="--", alpha=0.3)
    #plt.ylim((-1.1, 1.1))
    plt.ylim((-110, 110))
    plt.tick_params(labelsize=14)
    #plt.ylabel(r"${t}_{x}$ ($[m]$)",fontsize=16)
    plt.ylabel(r"${t}_{x}$ ($[cm]$)",fontsize=16)
    plt.savefig("cost_random/box/"+  fig_file + "_x.jpg", bbox_inches='tight', pad_inches=0)
    
    p_y_1 = optimized_p_1[:,1]
    p_y_2 = optimized_p_2[:,1]
    p_y_3 = optimized_p_3[:,1]
    p_y = np.concatenate((np.expand_dims(p_y_1,axis=1),np.expand_dims(p_y_2,axis=1),np.expand_dims(p_y_3,axis=1)),axis=1)
    p_y = p_y * 100
    df = pd.DataFrame(p_y,columns=column)
    df = df.astype(float)
    df.plot.box()
    plt.grid(linestyle="--", alpha=0.3)
    plt.ylim((-110, 110))
    plt.ylabel(r"${t}_{y}$ ($[cm]$)",fontsize=16)
    plt.tick_params(labelsize=14)
    plt.savefig("cost_random/box/"+  fig_file + "_y.jpg", bbox_inches='tight', pad_inches=0)
    
    p_z_1 = optimized_p_1[:,2]
    p_z_2 = optimized_p_2[:,2]
    p_z_3 = optimized_p_3[:,2]
    p_z = np.concatenate((np.expand_dims(p_z_1,axis=1),np.expand_dims(p_z_2,axis=1),np.expand_dims(p_z_3,axis=1)),axis=1)
    p_z = p_z * 100
    df = pd.DataFrame(p_z,columns=column)
    df = df.astype(float)
    df.plot.box()
    plt.grid(linestyle="--", alpha=0.3)
    plt.ylim((-110, 110))
    plt.ylabel(r"${t}_{z}$ ($[cm]$)",fontsize=16)
    plt.tick_params(labelsize=14)
    plt.savefig("cost_random/box/"+  fig_file + "_z.jpg", bbox_inches='tight', pad_inches=0)
    
    p_rx_1 = optimized_p_1[:,3]
    p_rx_2 = optimized_p_2[:,3]
    p_rx_3 = optimized_p_3[:,3]
    p_rx = np.concatenate((np.expand_dims(p_rx_1,axis=1),np.expand_dims(p_rx_2,axis=1),np.expand_dims(p_rx_3,axis=1)),axis=1)
    df = pd.DataFrame(p_rx,columns=column)
    df = df.astype(float)
    df.plot.box()
    plt.grid(linestyle="--", alpha=0.3)
    plt.ylim((-5.5, 5.5))
    plt.tick_params(labelsize=14)
    plt.ylabel(r"${r}_{x}$ ($[deg]$)",fontsize=16)
    plt.savefig("cost_random/box/"+  fig_file + "_rx.jpg", bbox_inches='tight', pad_inches=0)
    
    p_ry_1 = optimized_p_1[:,4]
    p_ry_2 = optimized_p_2[:,4]
    p_ry_3 = optimized_p_3[:,4]
    p_ry = np.concatenate((np.expand_dims(p_ry_1,axis=1),np.expand_dims(p_ry_2,axis=1),np.expand_dims(p_ry_3,axis=1)),axis=1)
    df = pd.DataFrame(p_ry,columns=column)
    df = df.astype(float)
    df.plot.box()
    plt.grid(linestyle="--", alpha=0.3)
    plt.ylim((-5.5, 5.5))
    plt.ylabel(r"${r}_{y}$ ($[deg]$)",fontsize=16)
    plt.tick_params(labelsize=14)
    plt.savefig("cost_random/box/"+  fig_file + "_ry.jpg", bbox_inches='tight', pad_inches=0)
    
    p_rz_1 = optimized_p_1[:,5]
    p_rz_2 = optimized_p_2[:,5]
    p_rz_3 = optimized_p_3[:,5]
    p_rz = np.concatenate((np.expand_dims(p_rz_1,axis=1),np.expand_dims(p_rz_2,axis=1),np.expand_dims(p_rz_3,axis=1)),axis=1)
    df = pd.DataFrame(p_rz,columns=column)
    df = df.astype(float)
    df.plot.box()
    plt.grid(linestyle="--", alpha=0.3)
    plt.ylim((-5.5, 5.5))
    plt.ylabel(r"${r}_{z}$ ($[deg]$)",fontsize=16)
    plt.tick_params(labelsize=14)
    plt.savefig("cost_random/box/"+  fig_file + "_rz.jpg", bbox_inches='tight', pad_inches=0)

    
    