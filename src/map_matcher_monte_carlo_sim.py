#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors # for KNN algorithm
from scipy.optimize import differential_evolution
import copy
import pandas as pd
import rosbag
import rospkg 
from mcmc_offline2 import rotate_map, likelihood, DEMapMatcher, ParticleFilterMapMatcher

ground_trouth_transformation = np.array([-6.94304748,  9.92673817,  3.56565882])
ground_trouth_transformation_map7 = np.array([-0.10729044,  4.94486143,  1.82609867])
ground_trouth_transformation_map5 = np.array([-6.94304748,  9.92673817,  3.56565882])
ground_trouth_transformation_map10 = np.array([ 0.533004618, 20.78074673,   1.83614012])
rospack = rospkg.RosPack()
packadge_path = rospack.get_path('DMM')
file_path = packadge_path + '/maps/map10.bag'
stat_path_de =  packadge_path + '/statistics/csv/MonteCarloStatistics_de_map10.csv'
stat_path_pf =  packadge_path + '/statistics/csv/MonteCarloStatistics_pf_map10.csv'
monte_carlo_runs = 50

def get_error(T): 
    return  np.linalg.norm(T - ground_trouth_transformation_map10)


   
if __name__ == '__main__':
    bag = rosbag.Bag(file_path)
    rospy.init_node('offline_map_matcher_monte_carlo_tester')
    pf_stat = []
    de_stat = []
    for r in range(monte_carlo_runs):
        init, init1, init2 = 1, 1, 1
        err_pf = []
        err_de = []
        for topic, msg, t in bag.read_messages(topics=['/ABot1/map', '/ABot2/map']):
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
                    nbrs = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(landMarksArray1_rc)
                    nbrs_empty = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(landMarksArray1_empty)
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
                model = ParticleFilterMapMatcher(nbrs, landMarksArray2)
                X_de = DEMapMatcher(nbrs, landMarksArray2)
                init = 0
            elif init == 0 and init1 == 0 and init2 == 0:
                model.predict()
                model.update(landMarksArray2, nbrs, nbrs_empty, scale1)
                X_de = DEMapMatcher(nbrs, landMarksArray2, X_de)
                if model.indicate == model.N_history:
                    model.resample()
                    # map_star = rotate_map(landMarksArray2, model.X_map)
                    # map_de = rotate_map(landMarksArray2, X_de)
                    # plt.subplot(3,1,1)
                    # plt.axis([-22, 22, -22, 22])
                    # plt.scatter(map_star[: , 0] ,map_star[:,1] , color = 'b') # plot tPF map
                    # plt.scatter(map_de[: , 0] ,map_de[:,1] , color = 'g')
                    # plt.scatter(landMarksArray1[: , 0] ,landMarksArray1[:,1] ,color = 'r', marker=',', linewidths=0.01) # plot origin map

                    # plt.subplot(3,1,2)
                    # plt.scatter(model.X[:,0], model.X[:,1])

                    err_pf.append(get_error(model.X_map))
                    err_de.append(get_error(X_de ))
                    # plt.subplot(3,1,3)
                    # plt.plot(err_pf, color = 'b')
                    # plt.plot(err_de, color = 'r')
                    # plt.pause(0.05)
                    # plt.clf()
                    print("Monte Carlo run: "+str(r)+", step: "+str(len(err_pf)))
                    print("PF error: "+str(get_error(model.X_map))+" , DE error: "+str(get_error(X_de)))
                    print("-------------------------------------")

        pf_stat.append(err_pf)
        de_stat.append(err_de)
        if rospy.is_shutdown():
            print("User force exit, saving data...")
            pf_data = np.array(pf_stat)
            de_data = np.array(de_stat)
            pf_data = np.squeeze(pf_data)
            de_data = np.squeeze(de_data)
            np.savetxt(stat_path_pf, pf_data, delimiter=",")
            np.savetxt(stat_path_de, de_data, delimiter=",")
            raw_input("Done saving, press Enter to continue...")
            break
            
    print("Done Simulation, saving data...")
    pf_data = np.array(pf_stat)
    de_data = np.array(de_stat)
    pf_data = np.squeeze(pf_data)
    de_data = np.squeeze(de_data)
    np.savetxt(stat_path_pf, pf_data, delimiter=",")
    np.savetxt(stat_path_de, de_data, delimiter=",")
    raw_input("Done saving, press Enter to continue...")

    

        
        
