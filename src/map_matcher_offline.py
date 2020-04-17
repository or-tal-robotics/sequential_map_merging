#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors # for KNN algorithm
from scipy.optimize import differential_evolution
import copy
import pandas as pd
from nav_msgs.msg import OccupancyGrid, MapMetaData
from tf.transformations import quaternion_from_euler
import rosbag
import rospkg 
from map_matcher import send_map_ros_msg, rotate_map, ParticleFilterMapMatcher, likelihood, DEMapMatcher, RANSACMapMatcher, OccupancyGrid2LandmarksArray

rospack = rospkg.RosPack()
packadge_path = rospack.get_path('sequential_map_merging')
file_path = packadge_path + '/maps/map10v3.bag'
origin_publisher = rospy.Publisher('origin_map', OccupancyGrid, queue_size = 10) 
global_publisher = rospy.Publisher('global_map', OccupancyGrid, queue_size = 10) 
target_publisher = rospy.Publisher('target_map', OccupancyGrid, queue_size = 10) 

   
if __name__ == '__main__':
    bag = rosbag.Bag(file_path)
    init, init1, init2 = 1, 1, 1
    err_pf = []
    err_de = []
    rospy.init_node('offline_map_matcher')
    for topic, msg, t in bag.read_messages(topics=['/ABot1/map', '/ABot2/map']):
        if rospy.is_shutdown():
            break
        if topic == '/ABot1/map':
            map1_msg = msg
            landMarksArray1, landMarksArray1_empty = OccupancyGrid2LandmarksArray(map1_msg, filter_map = 1000)
            scale1 = msg.info.resolution
            if landMarksArray1 != "empty":
                if init1 == 1:
                    cm1 = np.sum(np.transpose(landMarksArray1),axis=1)/len(landMarksArray1)
                landMarksArray1 = landMarksArray1 - cm1
                landMarksArray1_empty = landMarksArray1_empty - cm1
                nbrs = NearestNeighbors(n_neighbors= 1, algorithm='kd_tree').fit(landMarksArray1)
                nbrs_empty = NearestNeighbors(n_neighbors= 1, algorithm='kd_tree').fit(landMarksArray1_empty)
                init1 = 0
            else:
                continue

        if topic == '/ABot2/map':
            map2_msg = msg
            landMarksArray2, landMarksArray2_empty = OccupancyGrid2LandmarksArray(map2_msg)
            scale2 = msg.info.resolution
            if landMarksArray2 != "empty":
                if init2 == 1:
                    cm2 = np.sum(np.transpose(landMarksArray2),axis=1)/len(landMarksArray2) 
                landMarksArray2 = landMarksArray2 - cm2
                landMarksArray2_empty = landMarksArray2_empty - cm2
                init2 = 0
            else:
                continue
        if init == 1 and init1 == 0 and init2 == 0:
            model = ParticleFilterMapMatcher(nbrs, landMarksArray2, Np = 1000)
            #X_de = DEMapMatcher(nbrs, landMarksArray2)
            init = 0
        elif init == 0 and init1 == 0 and init2 == 0:
            model.predict()
            #model.update(landMarksArray2, nbrs, nbrs_empty, scale1)
            model.update(landMarksArray2, nbrs, origin_empty_map_nbrs=None , res = scale1)
            
            #X_de = DEMapMatcher(nbrs, landMarksArray2, X_de)
            #X_ransac = RANSACMapMatcher(landMarksArray1, landMarksArray2)
            if model.indicate == model.N_history:
                model.resample()
                X_pf = model.refinement(landMarksArray2, nbrs, res = scale1, Np = 2000)
                print(X_pf)
                rotated_map = rotate_map(landMarksArray2, X_pf)
                rotated_empty_map = rotate_map(landMarksArray2_empty, X_pf)
                estimated_global_map = np.concatenate([landMarksArray1,rotated_map], axis=0)
                estimated_global_empty_map = np.concatenate([landMarksArray1_empty,rotated_empty_map], axis=0)
                send_map_ros_msg(estimated_global_map, estimated_global_empty_map, global_publisher,frame_id='pf_map', resolution=scale1)
                send_map_ros_msg(landMarksArray1, landMarksArray1_empty, origin_publisher,frame_id='/robot1/map', resolution=scale1)
                send_map_ros_msg(landMarksArray2,landMarksArray2_empty , target_publisher,frame_id='/robot2/map', resolution=scale2)

    

        
        
