import numpy as np
import open3d as o3d
def extract_euler_from_matrix(transformation_matrix):
    x, y, z = transformation_matrix[:3, 3]
    rotation_matrix = transformation_matrix[:3, :3]
    rx = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
    ry = np.degrees(-np.arctan2(rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)))
    rz = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    return x, y, z, rx, ry, rz

def icp(lidars,radars):
    frames = len(lidars)
    matrixs = np.zeros((4,4))
    for i in range(frames):
        lidar = lidars[i]
        radar = radars[i]
        radar[:,2] = 0
        pcd_radar= o3d.geometry.PointCloud()
        pcd_lidar = o3d.geometry.PointCloud()
        
        pcd_radar.points= o3d.utility.Vector3dVector(radar[:,0:3])
        pcd_lidar.points = o3d.utility.Vector3dVector(lidar[:,0:3])
        
        pcd_radar.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=5))
        threshold = 2.0  
        trans_init = np.asarray([[1, 0, 0, 0],  
                                    [0, 1, 0, 0], 
                                    [0, 0, 1, 0],  
                                    [0, 0, 0, 1]])

        # run icp
        reg_p2p = o3d.pipelines.registration.registration_icp(pcd_lidar, pcd_radar, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
        matrix = reg_p2p.transformation
        matrixs = matrixs+matrix
        #pcd_lidar.transform(reg_p2p.transformation)
    rotation_matrix = matrixs / frames

    print("icp result: ")
    print(rotation_matrix)
    x, y, z, rx, ry, rz = extract_euler_from_matrix(transformation_matrix=rotation_matrix)
    print(f"x:{x}, y:{y}, z:{z}, rx:{rx}, ry:{ry}, rz:{-rz}")
    # o3d.visualization.draw_geometries([pcd_lidar, pcd_radar])

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
