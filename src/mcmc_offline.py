#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
Created on Thu Sep  5 10:09:21 2019
@author: Matan Samina
"""
import rospy
import numpy as np
from rospy_tutorials.msg import Floats # for landmarks array
from rospy.numpy_msg import numpy_msg # for landmarks array 
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors # for KNN algorithm
from geometry_msgs.msg import Transform # for transpose of map
import operator
from scipy.optimize import differential_evolution
import copy
import pandas as pd
import os
from joblib import Parallel, delayed
import multiprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture

ground_trouth_origin = np.array([-11.0, 2.5, 0.0])
ground_trouth_target = np.array([15, 2.5, -3.14])
import rosbag
import rospkg 


rospack = rospkg.RosPack()
packadge_path = rospack.get_path('DMM')
file_path = packadge_path + '/maps/map5.bag'

def rotate_map(map, T):
    c ,s = np.cos(T[2]) , np.sin(T[2])
    R = np.array(((c,-s), (s, c))) #Rotation matrix
    rot_map = np.matmul(map,R) + T[0:2]

def likelihood(origin_map_nbrs, target_map_rotated, var):
    d, _ = origin_map_nbrs.kneighbors(target_map_rotated)
    p = np.sum((1/(np.sqrt(2*np.pi*var)))*np.exp(-np.power(d,2)/(2*var))) 
    return p

class ParticleFilterMapMatcher():
    def __init__(self,init_origin_map_nbrs, init_target_map, Np = 1000, N_history = 15,  N_theta = 30, N_x = 30, N_y = 30, R_var = 0.1):
        self.Np = Np
        self.R_var = R_var
        self.N_history = N_history
        self.filter = np.arange(1,N_history+1)/np.sum(np.arange(1,N_history+1))
        temp_X = []
        angles = np.linspace(0 , 2*np.pi ,N_theta )
        xRange = np.linspace(-15 , 15 , N_x)
        yRange = np.linspace(-15 , 15 ,N_y)
        x0 = [xRange[np.random.randint(N_x)] ,yRange[np.random.randint(N_y), angles[np.random.randint(N_theta)]
        tempMap = rotate_map(init_target_map, x0)
        w0 = likelihood(tempMap, init_origin_map_nbrs, self.R_var)
        temp_X.append(x0)
        for i in range(N_theta*N_x*N_y):
            xt = [xRange[np.random.randint(N_x)] +0.5  * np.random.randn(),
                yRange[np.random.randint(N_y)]+0.5  * np.random.randn(),
                angles[np.random.randint(N_theta)] +0.5  * np.random.randn()]
            tempMap = rotate_map(init_target_map, xt)
            wt = likelihood(tempMap, init_origin_map_nbrs, self.R_var)
            if wt>w0:
                temp_X.append(xt)
                x0 = xt
            elif np.random.binomial(1, wt/w0) == 1:
                temp_X.append(xt)
                x0 = xt
        self.X = np.array(temp_X[-Np:])
        self.W = np.ones((Np,N_history)
        self.indicate = 0
    def predict(self):
        self.X[:,0:2] = self.X[:,0:2] + np.random.normal(0.0, 0.1, size=self.X[:,0:2].shape)
        self.X[:,2] = self.X[:,2] + np.random.normal(0.0, 0.01, size=self.X[:,2].shape)

    def update(self, target_map, origin_map_nbrs):
        for i in range(self.Np):
            tempMap = rotate_map(target_map, self.X[i])
            self.W[i, self.indicate] = likelihood(tempMap, origin_map_nbrs, self.R_var)
        self.indicate += 1

    def resample(self):


   
if __name__ == '__main__':
    bag = rosbag.Bag(file_path)
    for topic, msg, t in bag.read_messages(topics=['/ABot1/map', '/ABot2/map']):
        #print(msg)
        print(topic)
        if topic == '/ABot1/map':
            map1 = np.array(msg.data , dtype = np.float32)
            N1 = np.sqrt(map1.shape)[0].astype(np.int32)
            Re1 = np.copy(map1.reshape((N1,N1)))
            scale1 = msg.info.resolution
            landMarksArray1 = (np.argwhere( Re1 == 100 ) * scale1) 
            nbrs = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(landMarksArray1)
            print(landMarksArray1.shape)
        if topic == '/ABot2/map':
            map2 = np.array(msg.data , dtype = np.float32)
            N2 = np.sqrt(map2.shape)[0].astype(np.int32)
            Re2 = np.copy(map2.reshape((N2,N2)))
            scale2 = msg.info.resolution
            landMarksArray2 = (np.argwhere( Re2 == 100 ) * scale2)
            print(landMarksArray2.shape)
        
        