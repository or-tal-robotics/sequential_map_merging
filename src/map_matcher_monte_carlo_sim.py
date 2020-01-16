#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors # for KNN algorithm
import copy
import pandas as pd
import rosbag
import rospkg 
from map_matcher import rotate_map, likelihood, DEMapMatcher, ParticleFilterMapMatcher, ICPMapMatcher, RANSACMapMatcher, OccupancyGrid2LandmarksArray

ground_trouth_transformation = np.array([-6.94304748,  9.92673817,  3.56565882])
ground_trouth_transformation_map7 = np.array([-0.10729044,  4.94486143,  1.82609867])
ground_trouth_transformation_map5 = np.array([-6.94304748,  9.92673817,  3.56565882])
ground_trouth_transformation_map10 = np.array([ 0.533004618, 20.78074673,   1.83614012])
ground_trouth_transformation_map3_v2 = np.array([5.24621998, 7.41091718, 3.16565656])
ground_trouth_transformation_map10Disap = np.array([ 1.33004618, 20.3074673,   1.83614012])
ground_trouth_transformation_map10V2Disap = np.array([-0.51853119, 20.39218468,  1.7672772 ])
ground_trouth_transformation_map10_s1 = np.array([ 0.87207527, 20.5153013,   1.82142467])
ground_trouth_transformation_map10_s2 = np.array([13.84275813, 15.56581749,  2.84300749])
ground_trouth_transformation_map5_s1 = np.array([6.70767783, 5.53418701, 1.56826218])
ground_trouth_transformation_map5_s2 = np.array([ 5.11960965, -8.43495044,  2.0768514 ])
ground_trouth_transformation_map1 = np.array([-4.34062588, 11.23243629,  1.60253341])
ground_trouth_transformation_map1v2 = np.array([-4.12802754, 12.34509795,  1.58907605])
ground_trouth_transformation_map4 = np.array([-3.32452954, 11.78747777,  1.58070866])











rospack = rospkg.RosPack()
packadge_path = rospack.get_path('sequential_map_merging')
file_path = packadge_path + '/maps/map4.bag'
stat_path_de =  packadge_path + '/statistics/csv/MonteCarloStatistics_de_map4.csv'
stat_path_pf =  packadge_path + '/statistics/csv/MonteCarloStatistics_pf_map4.csv'
#stat_path_ransac =  packadge_path + '/statistics/csv/MonteCarloStatistics_ransac_map5Disap.csv'
#stat_path_icp =  packadge_path + '/statistics/csv/MonteCarloStatistics_icp_map5Disap.csv'
monte_carlo_runs = 10
ground_trouth_transformation = ground_trouth_transformation_map4
kidnepped_flag = False

def save_data(file_path, data):               
        df = pd.DataFrame([data])
        df.to_csv(file_path, mode='a', sep='\t', header=False)

def get_error(T, ground_trouth): 
    e1 = np.linalg.norm(T[0:2] - ground_trouth[0:2])
    e2 = min([np.abs(T[2] - ground_trouth[2]), np.abs(2*np.pi - T[2] + ground_trouth[2])])
    return  np.sqrt(e1**2+e2**2)


   
if __name__ == '__main__':
    bag = rosbag.Bag(file_path)
    rospy.init_node('offline_map_matcher_monte_carlo_tester_kidneped')
    Np = rospy.get_param("/sequential_map_matcher/n_particles",1500)
    Nf = rospy.get_param("/sequential_map_matcher/n_observation",500)
    N_history = rospy.get_param("/sequential_map_matcher/n_history",5)
    N_theta = rospy.get_param("/sequential_map_matcher/n_theta",50)
    N_x = rospy.get_param("/sequential_map_matcher/n_x",20)
    N_y = rospy.get_param("/sequential_map_matcher/n_y",20)
    R_var = rospy.get_param("/sequential_map_matcher/R_var",0.1)
    Q_xy = rospy.get_param("/sequential_map_matcher/Q_xy", 0.1)
    Q_theta = rospy.get_param("/sequential_map_matcher/Q_theta",0.1)
    R_xy = rospy.get_param("/sequential_map_matcher/R_xy",0.2)
    R_theta = rospy.get_param("/sequential_map_matcher/R_theta",0.2)
    P_theta = rospy.get_param("/sequential_map_matcher/P_theta",[0.6, 0.1, 0.1, 0.2])
    P_xy = rospy.get_param("/sequential_map_matcher/P_xy",[0.7, 0.05, 0.1, 0.05, 0.1])
    xy_mul = rospy.get_param("/sequential_map_matcher/xy_mul",2.0)

    pf_stat = []
    de_stat = []
    for r in range(monte_carlo_runs):
        init, init1, init2 = 1, 1, 1
        err_pf = []
        err_de = []
        #err_ransac = []
        #err_icp = []
        iter = 0
        for topic, msg, t in bag.read_messages(topics=['/ABot1/map', '/ABot2/map']):
            if rospy.is_shutdown():
                print("Shutting down...")
                exit(0)

            if topic == '/ABot1/map':
                map1_msg = msg
                scale1 = msg.info.resolution
                landMarksArray1, landMarksArray1_empty = OccupancyGrid2LandmarksArray(map1_msg, filter_map = Nf)
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
                model = ParticleFilterMapMatcher(nbrs,
                                                 landMarksArray2,
                                                 Np,
                                                 N_history,
                                                 N_theta,
                                                 N_x,
                                                 N_y,
                                                 R_var,
                                                 Q_xy,
                                                 Q_theta,
                                                 R_xy,
                                                 R_theta,
                                                 P_theta,
                                                 P_xy,
                                                 xy_mul)

                X_de = DEMapMatcher(nbrs, landMarksArray2)
                #X_ransac = RANSACMapMatcher(landMarksArray2, landMarksArray1)
                #X_icp = ICPMapMatcher(landMarksArray2, landMarksArray1)
                init = 0
            elif init == 0 and init1 == 0 and init2 == 0:
                model.predict()
                model.update(landMarksArray2, nbrs,  origin_empty_map_nbrs=None , res = scale1)
                X_de = DEMapMatcher(nbrs, landMarksArray2, X_de)
                #X_ransac = RANSACMapMatcher(landMarksArray2, landMarksArray1)
                #X_icp = ICPMapMatcher(landMarksArray2, landMarksArray1)
                if model.indicate == model.N_history:
                    model.resample()
                    X_pf = model.refinement(landMarksArray2, nbrs, res = scale1, Np = 1000)
                    err_pf.append(get_error(X_pf, ground_trouth_transformation))
                    err_de.append(get_error(X_de, ground_trouth_transformation))
                    #err_ransac.append(get_error(X_ransac, ground_trouth_transformation))
                    #err_icp.append(get_error(X_icp, ground_trouth_transformation))
                    print("Monte Carlo run: "+str(r)+", step: "+str(iter))
                    print("PF: "+str(err_pf[iter])+" , DE: "+str(err_de[iter]))
                    #print("PF: "+str(err_pf[iter])+" , DE: "+str(err_de[iter]) +" , RANSAC: "+str(err_ransac[iter]) +" , ICP: "+str(err_icp[iter]))
                    print("-------------------------------------")
                    iter+=1
                    if iter == 25 and kidnepped_flag == True:
                        #model.X = model.X + 2.0
                        ground_trouth_transformation = ground_trouth_transformation_map5_s2
                        print("Kidenpping robot!")
        save_data(stat_path_pf, np.array(err_pf))
        save_data(stat_path_de, np.array(err_de))
        #save_data(stat_path_ransac, np.array(err_ransac))
        #save_data(stat_path_icp, np.array(err_icp))
        
            
    print("Done Simulation!")


    

        
        
