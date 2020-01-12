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
from map_matcher import send_map_ros_msg, rotate_map, ParticleFilterMapMatcher, likelihood

ground_trouth_origin = np.array([-12,-5.0, 0])
ground_trouth_target = np.array([4.0, -8.0, -2.75])
ground_trouth_transformation_map5 = np.array([-6.94304748,  9.92673817,  3.56565882])
ground_trouth_transformation_map6 = np.array([4.34298586,  11.52861869,  1.58713136])
ground_trouth_transformation_map7 = np.array([-0.10729044,  4.94486143,  1.82609867])
ground_trouth_transformation_map9 = np.array([-8.13543295,  8.9289462,   1.83688384])
ground_trouth_transformation_map10 = np.array([ 1.33004618, 20.3074673,   1.83614012])
ground_trouth_transformation_map11 = np.array([5.24621998, 7.41091718, 3.16565656])
ground_trouth_transformation_map10_s1 = np.array([ 0.87207527, 20.5153013,   1.82142467])
ground_trouth_transformation_map10_s2 = np.array([13.84275813, 15.56581749,  2.84300749])













rospack = rospkg.RosPack()
packadge_path = rospack.get_path('sequential_map_merging')
file_path = packadge_path + '/maps/map10V3Disap.bag'
origin_publisher = rospy.Publisher('origin_map', OccupancyGrid, queue_size = 10) 
global_publisher = rospy.Publisher('global_map', OccupancyGrid, queue_size = 10) 
target_publisher = rospy.Publisher('target_map', OccupancyGrid, queue_size = 10) 


def get_error(T): 
    return  np.linalg.norm(T - ground_trouth_transformation_map7)


def DEMapMatcher(origin_map_nbrs, target_map, last_result = None):
    DE_func = lambda x: -likelihood(rotate_map(target_map,x),origin_map_nbrs, 0.3)
    if last_result is None:
        result = differential_evolution(DE_func, bounds = [(-15,15),(-15,15),(0,2*np.pi)] ,maxiter= 100 ,popsize=6,tol=0.0001, mutation=0.8)
        T_de = [result.x[0] , result.x[1] , min(result.x[2], 2*np.pi - result.x[2])]
    else:
        result = differential_evolution(DE_func, bounds = [(last_result[0]-10,last_result[0]+10),(last_result[1]-10,last_result[1]+10),(last_result[2]-0.5*np.pi,last_result[2]+0.5*np.pi)] ,maxiter= 100 ,popsize=6,tol=0.0001, mutation=0.8)
        T_de = [result.x[0] , result.x[1] , min(result.x[2], 2*np.pi - result.x[2])]
    return T_de



   
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
            map1 = np.array(msg.data , dtype = np.float32)
            N1 = np.sqrt(map1.shape)[0].astype(np.int32)
            Re1 = np.copy(map1.reshape((N1,N1)))
            scale1 = msg.info.resolution
            landMarksArray1 = (np.argwhere( Re1 == 100 ) * scale1)
            landMarksArray1_empty = (np.argwhere( Re1 == 0 ) * scale1)
            if landMarksArray1.shape[0] != 0:
                if init1 == 1:
                    cm1 = np.sum(np.transpose(landMarksArray1),axis=1)/len(landMarksArray1)
                landMarksArray1 = landMarksArray1 - cm1
                landMarksArray1_empty = landMarksArray1_empty - cm1
                #landMarksArray1_rc = landMarksArray1[np.random.choice(a = len(landMarksArray1), size = len(landMarksArray1)//1)]
                if len(landMarksArray1) > 500:
                    a = len(landMarksArray1)//500
                else:
                    a = 1
                landMarksArray1_rc = landMarksArray1[np.arange(0,len(landMarksArray1),a)]
                nbrs = NearestNeighbors(n_neighbors= 1, algorithm='kd_tree',n_jobs = -1).fit(landMarksArray1_rc)
                nbrs_empty = NearestNeighbors(n_neighbors= 1, algorithm='kd_tree',n_jobs = -1).fit(landMarksArray1_empty)
                init1 = 0
            else:
                continue

        if topic == '/ABot2/map':
            map2_msg = msg
            map2 = np.array(msg.data , dtype = np.float32)
            N2 = np.sqrt(map2.shape)[0].astype(np.int32)
            Re2 = np.copy(map2.reshape((N2,N2)))
            scale2 = msg.info.resolution
            landMarksArray2 = (np.argwhere( Re2 == 100 ) * scale2)
            landMarksArray2_empty = (np.argwhere( Re2 == 0 ) * scale2)
            if landMarksArray2.shape[0] != 0:
                if init2 == 1:
                    cm2 = np.sum(np.transpose(landMarksArray2),axis=1)/len(landMarksArray2) 
                landMarksArray2 = landMarksArray2 - cm2
                landMarksArray2_empty = landMarksArray2_empty - cm2
                init2 = 0
            else:
                continue
        if init == 1 and init1 == 0 and init2 == 0:
            model = ParticleFilterMapMatcher(nbrs, landMarksArray2, Np = 2000)
            #X_de = DEMapMatcher(nbrs, landMarksArray2)
            init = 0
        elif init == 0 and init1 == 0 and init2 == 0:
            model.predict()
            #model.update(landMarksArray2, nbrs, nbrs_empty, scale1)
            model.update_parallel(landMarksArray2, nbrs, nbrs_empty, scale1)
            #X_de = DEMapMatcher(nbrs, landMarksArray2, X_de)
            if model.indicate == model.N_history:
                model.resample()

                print(model.X_map)
                rotated_map = rotate_map(landMarksArray2, model.X_map)
                rotated_empty_map = rotate_map(landMarksArray2_empty, model.X_map)
                estimated_global_map = np.concatenate([landMarksArray1,rotated_map], axis=0)
                estimated_global_empty_map = np.concatenate([landMarksArray1_empty,rotated_empty_map], axis=0)
                send_map_ros_msg(estimated_global_map, estimated_global_empty_map, global_publisher, resolution=scale1)
                send_map_ros_msg(landMarksArray1, landMarksArray1_empty, origin_publisher, resolution=scale1)
                send_map_ros_msg(landMarksArray2,landMarksArray2_empty , target_publisher, resolution=scale2)
    raw_input("Press Enter to continue...")

    

        
        