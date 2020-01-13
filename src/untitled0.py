#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 17:30:29 2020

@author: or
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors 

def rotate_map_parallel(map, T):
    c ,s = np.cos(T[:,2]) , np.sin(T[:,2])
    R = np.array(((c,-s), (s, c)))
    Tmap = np.matmul(map,R)
    Tmap = np.transpose(Tmap, (1,2,0))
    rot_map = Tmap.reshape((Tmap.shape[0],-1))
    rot_map = rot_map + T[:,0:2].reshape((-1))
    rot_map = rot_map.reshape(Tmap.shape)
    return rot_map


T = np.random.uniform(-1,1,size=(1000,3))
map = np.random.uniform(-1,1,size=(2500,2))
target_map = np.random.uniform(-1,1,size=(1500,2))
c ,s = np.cos(T[:,2]) , np.sin(T[:,2])
R = np.array(((c,-s), (s, c)))
Tmap = np.matmul(map,R)
Tmap = np.transpose(Tmap, (1,2,0))
rot_map = Tmap.reshape((Tmap.shape[0],-1))
rot_map = rot_map + T[:,0:2].reshape((-1))
rot_map = rot_map.reshape(Tmap.shape)

var = 0.1
target_map_rotated = rotate_map_parallel(target_map, T)
nbrs = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(map)
d, _ = nbrs.kneighbors(target_map_rotated.reshape((-1,2)))
d = d.reshape((target_map_rotated.shape[0],target_map_rotated.shape[1]))
p = np.mean((1/(np.sqrt(2*np.pi*var)))*np.exp(-np.power(d,2)/(2*var)), axis=0)
p = p/np.sum(p)