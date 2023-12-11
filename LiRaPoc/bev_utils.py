import numpy as np
import torch
import cv2


def bev(points,save_name ='bev.jpg',mode = 'car', delta_l=0.4, pixel_l=500):
    
    r_min,r_max,theta_min,theta_max = 30,50,65,100
    if mode == 'car':
        #points[:,3] = points[:,3] / np.max(points[:,3])
        bev_image = lidar_pc2pixor_intensity(points,delta_l, pixel_l).astype(np.float32) 
        bev_image = (bev_image * 255).astype(np.uint8)
        bev_image = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2RGB)
        draw_line_car(bev_image,r_min,r_max,theta_min,theta_max,pixel_l)
    elif mode == 'polar':
        #points[:,3] = points[:,3] / np.max(points[:,3])
        bev_image = lidar_pc2pixor_intensity_polar(points).astype(np.float32) 
        bev_image = (bev_image * 255).astype(np.uint8)
        bev_image = cv2.cvtColor(bev_image, cv2.COLOR_GRAY2RGB)
        draw_line_polar(bev_image,r_min,r_max,theta_min,theta_max,pixel_l)
    # print(hmap_image.shape)
    #lidar_image = cv2.applyColorMap(lidar_image, cv2.COLORMAP_JET)
    # bin_path = os.path.join(processed_path, str(filename)+ '.jpg')
    cv2.imwrite(save_name, bev_image)
    return bev_image

def draw_line_polar(bev_image,r_min,r_max,theta_min,theta_max,pixel_l):
    # draw reference line in polar 
    w,h,_ = bev_image.shape
    r_min = int(r_min/100 * pixel_l)
    r_max = int(r_max/100 * pixel_l)
    theta_min = int(theta_min/360 * pixel_l)
    theta_max = int(theta_max/360 * pixel_l)
    #(63,188,233)
    #(233,188,63)
    #(255,122,69)
    #(69,122,255)
    #(255, 203, 5)
    #(5, 203, 255)
    cv2.line(bev_image, (0, r_min), (w, r_min), (233,188,63),2)
    cv2.line(bev_image, (0, r_max), (w, r_max), (233,188,63),2)
    cv2.line(bev_image, (theta_min, 0), (theta_min, h), (5, 203, 255),2)
    cv2.line(bev_image, (theta_max, 0), (theta_max, h), (5, 203, 255),2)
def draw_line_car(bev_image,r_min,r_max,theta_min,theta_max,pixel_l):
    # draw reference line in car
    w,h,_ = bev_image.shape
    r_min = int(r_min/200 * pixel_l)
    r_max = int(r_max/200 * pixel_l)
    theta_min_x = -int(pixel_l* np.sin(theta_min/180*np.pi)/2) + int(w/2)
    theta_min_y = -int(pixel_l* np.cos(theta_min/180*np.pi)/2) + int(w/2)
    theta_max_x = -int(pixel_l* np.sin(theta_max/180*np.pi)/2) + int(w/2)
    theta_max_y = -int(pixel_l* np.cos(theta_max/180*np.pi)/2) + int(w/2)
    cv2.circle(bev_image,(int(w/2),int(h/2)),r_min,(233,188,63),2)
    cv2.circle(bev_image,(int(w/2),int(h/2)),r_max,(233,188,63),2)
    cv2.line(bev_image, (int(w/2),int(h/2)), (theta_min_x, theta_min_y), (5, 203, 255),2)
    cv2.line(bev_image, (int(w/2),int(h/2)), (theta_max_x, theta_max_y), (5, 203, 255),2)
    
def fusion_bev(lidar,radar,save_name = 'fusion.jpg',mode = 'car',delta_l=0.2, pixel_l=1000):
       
    # fusion image # 
    
    # box position  
    # boreas 2   
    # 2x
    xmin = 350
    ymin = 450
    xmax = 550
    ymax = 650
    xmin = 400
    ymin = 550
    xmax = 500
    ymax = 650
    # orr sample 4
    # 2x
    # xmin = 350
    # ymin = 350
    # xmax = 550
    # ymax = 550
    # 4x
    # xmin = 370
    # ymin = 400
    # xmax = 470
    # ymax = 500
    bev_image_lidar = lidar_pc2pixor_intensity(lidar,delta_l, pixel_l).astype(np.float32) 
    bev_image_lidar = (bev_image_lidar * 255).astype(np.uint8)
    bev_image_radar = lidar_pc2pixor_intensity(radar,delta_l, pixel_l).astype(np.float32) 
    bev_image_radar = (bev_image_radar * 255).astype(np.uint8)
    img_fusion = np.ones((pixel_l,pixel_l,3))
    img_fusion = img_fusion*255
    for i in range(pixel_l):
        for j in range(pixel_l):
            if bev_image_radar[i][j] != 0:
                img_fusion[i][j] = [2,132,251] # orange 1          
            if bev_image_lidar[i][j] != 0:
                img_fusion[i][j] = [188,158,39] # blue 1
    
 
    cv2.rectangle(img_fusion,(xmin,ymin),(xmax,ymax),(0,0,0),2) 
    cv2.imwrite(save_name, img_fusion)
    
    ## part fusion image ##
    magnify_power = 4
    delta_l = delta_l/ magnify_power
    pixel_l = pixel_l * magnify_power
    bev_image_lidar = lidar_pc2pixor_intensity(lidar,delta_l, pixel_l).astype(np.float32) 
    bev_image_lidar = (bev_image_lidar * 255).astype(np.uint8)
    bev_image_radar = lidar_pc2pixor_intensity(radar,delta_l, pixel_l).astype(np.float32) 
    bev_image_radar = (bev_image_radar * 255).astype(np.uint8)
    img_fusion = np.ones((pixel_l,pixel_l,3))
    img_fusion = img_fusion*255
    for i in range(pixel_l):
        for j in range(pixel_l):
            if bev_image_radar[i][j] != 0:
                img_fusion[i][j] = [2,132,251] # orange 
            if bev_image_lidar[i][j] != 0:
                img_fusion[i][j] = [188,158,39] # blue 1
    xmin = xmin*magnify_power
    ymin = ymin*magnify_power
    xmax = xmax*magnify_power
    ymax = ymax*magnify_power
    x = xmax - xmin + 6
    y = ymax - ymin + 6
    img_part = np.ones((x,y,3))  
    img_part[3:-3,3:-3,:] = img_fusion[xmin:xmax,ymin:ymax,:]
    img_part[3:-3,3:-3,:] = img_fusion[ymin:ymax,xmin:xmax,:]
    xmin = 3
    ymin = 3
    xmax = x - 3
    ymax = y - 3
    cv2.rectangle(img_part,(xmin,ymin),(xmax,ymax),(0,0,0),2)
    cv2.imwrite(save_name[0:-4] + "_part.jpg", img_part)
    
    return img_fusion
#generate lidar intensity data (320*320*1)
def lidar_pc2pixor_intensity(lidar_data, delta_l=0.2, pixel_l=500, h1=-10, h2=10, delta_h=0.1):
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
    lidar_bev_idx = lidar_bev_idx.astype(np.int)

    lidar_intensity = np.zeros([pixel_l, pixel_l])

    for i in range(len(lidar_bev_idx)):
        #lidar_intensity[lidar_bev_idx[i,0], lidar_bev_idx[i,1]] = max(lidar_data[i,3], \
        #    lidar_intensity[lidar_bev_idx[i,0], lidar_bev_idx[i,1]])
        lidar_intensity[lidar_bev_idx[i,0], lidar_bev_idx[i,1]] = 1
    return lidar_intensity



def lidar_pc2pixor_intensity_polar(lidar_data_polar, delta_rho= 0.2, delta_phi = 0.72,pixel_rho= 500,pixel_phi= 500,h1=-3, h2=2.0, delta_h=0.1):
    # generate polar lidar intensity data (320*320*1)
    # pixel = max lidar intensity in this location

    # l: lidar range
    l = pixel_rho * delta_rho
    # lidar_data to lidar_data_polar
    #lidar_data[:, 0] = lidar_data[:, 0]
    #lidar_data_polar = np.ones_like(lidar_data)
    #lidar_data_polar[:,0] = np.sqrt(lidar_data[:,0]*lidar_data[:,0] + lidar_data[:,1]*lidar_data[:,1])
    #Alidar_data_polar[:,1] = np.arctan2(lidar_data[:,0],lidar_data[:,1])* 180 / np.pi
    #lidar_data_polar[lidar_data_polar[:,1]<0,1] = lidar_data_polar[lidar_data_polar[:,1]<0,1]+360
    #lidar_data_polar[:,2:] = lidar_data[:,2:]

    idx_r = np.logical_and(lidar_data_polar[:,0] >= 0, lidar_data_polar[:,0] < l)
    idx_z = np.logical_and(lidar_data_polar[:,2] >= h1, lidar_data_polar[:,2] < h2)
    idx_valid = np.logical_and(idx_r, idx_z)
    lidar_data_polar = lidar_data_polar[idx_valid, :]

    lidar_bev_idx = np.zeros([len(lidar_data_polar), 2])
    lidar_bev_idx[:,1] = np.floor(lidar_data_polar[:,1] / delta_phi)
    lidar_bev_idx[:,0] = np.floor(lidar_data_polar[:,0] / delta_rho)
    lidar_bev_idx[lidar_bev_idx == pixel_phi] = pixel_phi - 1
    lidar_bev_idx = lidar_bev_idx.astype(np.int)

    lidar_intensity = np.zeros([pixel_rho, pixel_phi])

    for i in range(len(lidar_bev_idx)):
        #lidar_intensity[lidar_bev_idx[i,0], lidar_bev_idx[i,1]] = max(lidar_data_polar[i,3], \
        #    lidar_intensity[lidar_bev_idx[i,0], lidar_bev_idx[i,1]])
        lidar_intensity[lidar_bev_idx[i,0], lidar_bev_idx[i,1]] = 1
    #print(lidar_intensity[100])

    return lidar_intensity

#generate lidar occuoancy data on height (320*320*35)
#if have lidar point, the pixel = 1
def lidar_pc2pixor_occupancy(lidar_data,delta_l=0.2, pixel_l=320, h1=-2.5, h2=1.0, delta_h=0.1):
    l1 = (-pixel_l/2) * delta_l
    l2 = (pixel_l/2) * delta_l

    pixel_h = np.int(np.round((h2 - h1) / delta_h))

    #lidar_data[:,0] = -lidar_data[:,0]
    #lidar_data_car[:,0] = -lidar_data_car[:,0]

    idx_x = np.logical_and(lidar_data[:,0] >= l1, lidar_data[:,0] < l2)
    idx_y = np.logical_and(lidar_data[:,1] >= l1, lidar_data[:,1] < l2)
    idx_z = np.logical_and(lidar_data[:,2] >= h1, lidar_data[:,2] < h2)
    idx_valid = np.logical_and(idx_z, np.logical_and(idx_y, idx_x))
    lidar_data = lidar_data[idx_valid, :]

    lidar_bev_idx = np.zeros([len(lidar_data), 2])
    lidar_bev_idx[:,0] = np.floor((-lidar_data[:,1] - l1) / delta_l)
    lidar_bev_idx[:,1] = np.floor((-lidar_data[:,0] - l1) / delta_l)
    lidar_bev_idx[lidar_bev_idx == pixel_l] = pixel_l - 1
    lidar_bev_idx = lidar_bev_idx.astype(np.int)


    lidar_height_idx = np.floor((lidar_data[:,2] - h1) / delta_h)
    lidar_height_idx[lidar_height_idx == pixel_h] = pixel_h - 1
    lidar_height_idx = lidar_height_idx.astype(np.int)

    lidar_occupancy = np.zeros([pixel_l, pixel_l, pixel_h])

    for i in range(len(lidar_bev_idx)):
        lidar_occupancy[lidar_bev_idx[i,0], lidar_bev_idx[i,1], lidar_height_idx[i]] = 1


    return lidar_occupancy

def lidar_pc2pixor_occupancy_polar(lidar_data,delta_l=0.2, pixel_l=320, h1=-2.5, h2=1.0, delta_h=0.1):
    # l: lidar range
    l = pixel_l * delta_rho
    # lidar_data to lidar_data_polar
    lidar_data[:, 0] = lidar_data[:, 0]
    lidar_data_polar = np.ones_like(lidar_data)
    lidar_data_polar[:, 0] = np.sqrt(lidar_data[:, 0] * lidar_data[:, 0] + lidar_data[:, 1] * lidar_data[:, 1])
    lidar_data_polar[:, 1] = np.arctan2(lidar_data[:, 0], lidar_data[:, 1]) * 180 / np.pi
    lidar_data_polar[lidar_data_polar[:, 1] < 0, 1] = lidar_data_polar[lidar_data_polar[:, 1] < 0, 1] + 360
    lidar_data_polar[:, 2:] = lidar_data[:, 2:]

    #idx_r = np.logical_and(lidar_data_polar[:, 0] >= 0, lidar_data_polar[:, 0] < l)
    #idx_z = np.logical_and(lidar_data_polar[:, 2] >= h1, lidar_data_polar[:, 2] < h2)
    #idx_valid = np.logical_and(idx_r, idx_z)
    #lidar_data_polar = lidar_data_polar[idx_valid, :]

    lidar_bev_idx = np.zeros([len(lidar_data_polar), 2])
    lidar_bev_idx[:, 1] = np.floor(lidar_data_polar[:, 1] / delta_phi)
    lidar_bev_idx[:, 0] = np.floor(lidar_data_polar[:, 0] / delta_rho)
    lidar_bev_idx[lidar_bev_idx == pixel_l] = pixel_l - 1
    lidar_bev_idx = lidar_bev_idx.astype(np.int)


    lidar_height_idx = np.floor((lidar_data[:,2] - h1) / delta_h)
    lidar_height_idx[lidar_height_idx == pixel_h] = pixel_h - 1
    lidar_height_idx = lidar_height_idx.astype(np.int)

    lidar_occupancy = np.zeros([pixel_l, pixel_l, pixel_h])

    for i in range(len(lidar_bev_idx)):
        lidar_occupancy[lidar_bev_idx[i,0], lidar_bev_idx[i,1], lidar_height_idx[i]] = 1

    return lidar_occupancy


def lidar_pc2pixor_occupancy_id(lidar_data,delta_l=0.2, pixel_l=320, h1=-2.5, h2=1.0, delta_h=0.1):
    # generate lidar occuoancy data on 32 ring id (320*320*32)
    # l: lidar range
    l1 = (-pixel_l/2) * delta_l
    l2 = (pixel_l/2) * delta_l

    idx_x = np.logical_and(lidar_data[:, 0] >= l1, lidar_data[:, 0] < l2)
    idx_y = np.logical_and(lidar_data[:, 1] >= l1, lidar_data[:, 1] < l2)
    idx_z = np.logical_and(lidar_data[:, 2] >= h1, lidar_data[:, 2] < h2)
    idx_valid = np.logical_and(idx_z, np.logical_and(idx_y, idx_x))
    lidar_data = lidar_data[idx_valid, :]

    lidar_occupancy = np.zeros([pixel_l, pixel_l, 32])
    for i in range(32):
        lidar_id_valid = (lidar_data[:,4] == i)
        lidar_id_data = lidar_data[lidar_id_valid]

        lidar_bev_idx = np.zeros([len(lidar_id_data), 2])
        lidar_bev_idx[:, 0] = np.floor((-lidar_id_data[:, 1] - l1) / delta_l)
        lidar_bev_idx[:, 1] = np.floor((-lidar_id_data[:, 0] - l1) / delta_l)
        lidar_bev_idx[lidar_bev_idx == pixel_l] = pixel_l - 1
        lidar_bev_idx = lidar_bev_idx.astype(np.int)
        for j in range(len(lidar_bev_idx)):
            lidar_occupancy[lidar_bev_idx[j, 0], lidar_bev_idx[j, 1], i] = 1


    return lidar_occupancy

def lidar_pc2pixor_occupancy_id_polar(lidar_data, delta_rho=0.2, delta_phi = 1.125,pixel_l=320, pixel_phi=320, h1=-2.5, h2=1.0, delta_h=0.1):
    # generate polar lidar occuoancy data on 32 ring id (320*320*1)
    id_num = 32
    # l: lidar range
    l = pixel_l * delta_rho
    # lidar_data to lidar_data_polar
    lidar_data[:, 0] = lidar_data[:, 0]
    lidar_data_polar = np.ones_like(lidar_data)
    lidar_data_polar[:, 0] = np.sqrt(lidar_data[:, 0] * lidar_data[:, 0] + lidar_data[:, 1] * lidar_data[:, 1])
    lidar_data_polar[:, 1] = np.arctan2(lidar_data[:, 0], lidar_data[:, 1]) * 180 / np.pi
    lidar_data_polar[lidar_data_polar[:, 1] < 0, 1] = lidar_data_polar[lidar_data_polar[:, 1] < 0, 1] + 360
    lidar_data_polar[:, 2:] = lidar_data[:, 2:]

    idx_r = np.logical_and(lidar_data_polar[:, 0] >= 0, lidar_data_polar[:, 0] < l)
    idx_z = np.logical_and(lidar_data_polar[:, 2] >= h1, lidar_data_polar[:, 2] < h2)
    idx_valid = np.logical_and(idx_r, idx_z)
    lidar_data_polar = lidar_data_polar[idx_valid, :]

    lidar_bev_idx = np.zeros([len(lidar_data_polar), 2])
    lidar_bev_idx[:, 1] = np.floor(lidar_data_polar[:, 1] / delta_phi)
    lidar_bev_idx[:, 0] = np.floor(lidar_data_polar[:, 0] / delta_rho)
    lidar_bev_idx[lidar_bev_idx == pixel_l] = pixel_l - 1
    lidar_bev_idx = lidar_bev_idx.astype(np.int)

    lidar_occupancy = np.zeros([pixel_l, pixel_l,id_num])
    for i in range(id_num):
        lidar_id_valid = (lidar_data_polar[:,4] == i)
        lidar_id_data = lidar_data_polar[lidar_id_valid]

        lidar_bev_idx = np.zeros([len(lidar_id_data), 2])
        lidar_bev_idx[:, 0] = np.floor(lidar_id_data[:, 0]/delta_rho)
        lidar_bev_idx[:, 1] = np.floor(lidar_id_data[:, 1] / delta_phi)
        lidar_bev_idx[lidar_bev_idx == pixel_l] = pixel_l - 1
        lidar_bev_idx = lidar_bev_idx.astype(np.int)
        for j in range(len(lidar_bev_idx)):
            lidar_occupancy[lidar_bev_idx[j, 0], lidar_bev_idx[j, 1], i] = 1


    return lidar_occupancy