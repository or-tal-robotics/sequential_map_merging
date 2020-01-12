#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors 
from scipy.optimize import differential_evolution
import copy
import pandas as pd
from nav_msgs.msg import OccupancyGrid, MapMetaData
from tf.transformations import quaternion_from_euler
import rospkg 
from map_matcher import send_map_ros_msg, rotate_map, ParticleFilterMapMatcher




rospack = rospkg.RosPack()
packadge_path = rospack.get_path('sequential_map_merging')
global_publisher = rospy.Publisher('global_map', OccupancyGrid, queue_size = 10) 

def OccupancyGrid2LandmarksArray(OccupancyGridMsg, filter_map = None):
    map = np.array(OccupancyGridMsg.data , dtype = np.float32)
    N = np.sqrt(map.shape)[0].astype(np.int32)
    Re = np.copy(map.reshape((N,N)))
    scale = OccupancyGridMsg.info.resolution
    landMarksArray = (np.argwhere( Re == 100 ) * scale)
    landMarksArray_empty = (np.argwhere( Re == 0 ) * scale)
    if landMarksArray.shape[0] != 0:
        if filter_map is not None:
            if len(landMarksArray1) > filter_map:
                a = len(landMarksArray1)//filter_map
            else:
                a = 1
            landMarksArray = landMarksArray[np.arange(0,len(landMarksArray),a)]
        return landMarksArray, landMarksArray_empty
    else:
        print("Error: Empty map!")
        return "empty" , "empty"



   
if __name__ == '__main__':
    rospy.init_node('sequential_map_matcher')

    robot1_topic = rospy.get_param("/sequential_map_matcher/origin_map",'origin_map')
    robot2_topic = rospy.get_param("/sequential_map_matcher/target_map",'target_map')
    Np = rospy.get_param("/sequential_map_matcher/n_particles",500)
    Nf = rospy.get_param("/sequential_map_matcher/n_observation",500)
    N_history = rospy.get_param("/sequential_map_matcher/n_history",5)
    N_theta = rospy.get_param("/sequential_map_matcher/n_theta",50)
    N_x = rospy.get_param("/sequential_map_matcher/n_x",20)
    N_y = rospy.get_param("/sequential_map_matcher/n_y",20)
    R_var = rospy.get_param("/sequential_map_matcher/R_var",0.1)

    print("Initilizing Sequential Map Matcher...")
    landMarksArray1 = "empty"
    landMarksArray2 = "empty"
    while landMarksArray1 == "empty" or landMarksArray2 == "empty":
        map1_msg = rospy.wait_for_message(robot1_topic, OccupancyGrid)
        map2_msg = rospy.wait_for_message(robot2_topic, OccupancyGrid)
        landMarksArray1, landMarksArray1_empty = OccupancyGrid2LandmarksArray(map1_msg, filter_map = Nf)
        landMarksArray2, landMarksArray2_empty = OccupancyGrid2LandmarksArray(map2_msg)
    cm1 = np.sum(np.transpose(landMarksArray1),axis=1)/len(landMarksArray1)
    cm2 = np.sum(np.transpose(landMarksArray2),axis=1)/len(landMarksArray2) 
    landMarksArray1 = landMarksArray1 - cm1
    landMarksArray1_empty = landMarksArray1_empty - cm1
    landMarksArray2 = landMarksArray2 - cm2
    landMarksArray2_empty = landMarksArray2_empty - cm2
    scale1 = map1_msg.info.resolution
    nbrs = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(landMarksArray1)
    nbrs_empty = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(landMarksArray1_empty)
    model = ParticleFilterMapMatcher(nbrs, landMarksArray2, Np, N_history, N_theta, N_x, N_y, R_var)
    print("Done Initilizing!")

    while not rospy.is_shutdown():
        map1_msg = rospy.wait_for_message('/ABot1/map', OccupancyGrid)
        map2_msg = rospy.wait_for_message('/ABot2/map', OccupancyGrid)
        landMarksArray1, landMarksArray1_empty = OccupancyGrid2LandmarksArray(map1_msg, filter_map = Nf)
        landMarksArray2, landMarksArray2_empty = OccupancyGrid2LandmarksArray(map2_msg)
        landMarksArray1 = landMarksArray1 - cm1
        landMarksArray1_empty = landMarksArray1_empty - cm1
        landMarksArray2 = landMarksArray2 - cm2
        landMarksArray2_empty = landMarksArray2_empty - cm2
        nbrs = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(landMarksArray1)
        nbrs_empty = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(landMarksArray1_empty)
        model.predict()
        model.update_parallel(landMarksArray2, nbrs, nbrs_empty, scale1)
        if model.indicate == model.N_history:
            model.resample()
            rotated_map = rotate_map(landMarksArray2, model.X_map)
            rotated_empty_map = rotate_map(landMarksArray2_empty, model.X_map)
            estimated_global_map = np.concatenate([landMarksArray1,rotated_map], axis=0)
            estimated_global_empty_map = np.concatenate([landMarksArray1_empty,rotated_empty_map], axis=0)
            send_map_ros_msg(estimated_global_map, estimated_global_empty_map, global_publisher, resolution=scale1)


    

    

        
        