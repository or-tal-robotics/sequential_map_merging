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
ground_trouth_transformation_map3_v2 = np.array([5.24621998, 7.41091718, 3.16565656])
ground_trouth_transformation_map10Disap = np.array([ 1.33004618, 20.3074673,   1.83614012])
ground_trouth_transformation_map10V2Disap = np.array([-0.51853119, 20.39218468,  1.7672772 ])




rospack = rospkg.RosPack()
packadge_path = rospack.get_path('sequential_map_merging')
file_path = packadge_path + '/maps/map10V2Disap.bag'
stat_path_de =  packadge_path + '/statistics/csv/MonteCarloStatistics_de_map10V2Disap.csv'
stat_path_pf =  packadge_path + '/statistics/csv/MonteCarloStatistics_pf_map10V2Disap.csv'
monte_carlo_runs = 50
ground_trouth_transformation = ground_trouth_transformation_map10Disap
kidnepped_flag = True

def save_data(file_path, data):               
        df = pd.DataFrame([data])
        df.to_csv(file_path, mode='a', sep='\t', header=False)

def get_error(T, ground_trouth): 
    return  np.linalg.norm(T - ground_trouth)


   
if __name__ == '__main__':
    bag = rosbag.Bag(file_path)
    rospy.init_node('offline_map_matcher_monte_carlo_tester_kidneped')
    pf_stat = []
    de_stat = []
    for r in range(monte_carlo_runs):
        init, init1, init2 = 1, 1, 1
        err_pf = []
        err_de = []
        iter = 0
        for topic, msg, t in bag.read_messages(topics=['/ABot1/map', '/ABot2/map']):
            if rospy.is_shutdown():
                print("Shutting down...")
                exit(0)

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

                    err_pf.append(get_error(model.X_map, ground_trouth_transformation))
                    err_de.append(get_error(X_de, ground_trouth_transformation))
                    # plt.subplot(3,1,3)
                    # plt.plot(err_pf, color = 'b')
                    # plt.plot(err_de, color = 'r')
                    # plt.pause(0.05)
                    # plt.clf()
                    print("Monte Carlo run: "+str(r)+", step: "+str(iter))
                    print("PF error: "+str(err_pf[iter])+" , DE error: "+str(err_de[iter]))
                    print("-------------------------------------")
                    iter+=1
                    if iter == 50 and kidnepped_flag == True:
                        #model.X = model.X + 2.0
                        ground_trouth_transformation = ground_trouth_transformation_map10V2Disap
                        print("Kidenpping robot!")
        save_data(stat_path_pf, np.array(err_pf))
        save_data(stat_path_de, np.array(err_de))
        
            
    print("Done Simulation!")


    

        
        
