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

class tPF():
    
    def __init__(self):

        self.pub = rospy.Publisher('/TM2', Transform , queue_size=1000 )
        rospy.init_node('listener', anonymous=True)
       
        # creat first particales 
        self.initialize_PF()

        # convert maps to landmarks arrays:
        self.oMap = maps("LM1") 
        self.tMap = maps("LM2")
        self.indicator = 0
        self.realT = np.array([2, 2, 90]) # real transformation
        self.K = 0 # time step for norm2
        counter = 0

        while not rospy.is_shutdown():

            a , b = self.oMap.started ,self.tMap.started  # self.maps.started : indicate if map recieved
        
            if a and b:
                
                #init nbrs for KNN
                self.nbrs = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(self.oMap.map)

                # DE algorithm for finding best match
                result = differential_evolution(self.func_de, bounds = [(-10,10),(-10,10),(0,360)] ,maxiter= 200 ,popsize=6,tol=0.0001)
                self.T_de = [result.x[0] , result.x[1] , min(result.x[2], 360 - result.x[2])] 

                self.likelihood_PF()
                
                counter +=1
                if counter >= 10 :
                    self.resampling()
                    counter = 0

                self.norm2() # finding norm2

                self.de_map = self.tMap.rotate2(result.x)
                #self.plotmaps() # plot landmarks of maps.

    def initialize_PF( self , angles = np.linspace(0 , 360 , 30) , xRange = np.linspace(-10 , 10 , 10) , yRange = np.linspace(-10 , 10 ,10) ):
       
        # make a list of class rot(s)
        self.Rot = []

        for i in range(len(angles)):
            for j in range(len(xRange)):
                for k in range(len(yRange)):
                    self.Rot.append(rot(angles[i] , xRange[j] , yRange[k]))
        
        print ("initialaize PF whith") , len(self.Rot) ,(" samples completed")

    def likelihood_PF(self):

        self.scores = []

        for i in self.Rot:    
            # 'tempMap' ->  map after transformation the secondery map [T(tMap)]:
            tempMap = self.tMap.rotate(i.x ,i.y , i.theta)
            i.weight(self.oMap.map , tempMap ,self.nbrs)
            self.scores.append(i.score) # add weights to array 

        maxt = max( self.Rot , key = operator.attrgetter('score') ) # finds partical with maximum score

        if maxt.score > self.indicator:          
        # check if there is a new partical thats better then previuos partical
            self.maxt = maxt
            print 'max W(tPF):' ,self.maxt.score ,self.maxt.theta
            self.maxMap = self.tMap.rotate(self.maxt.x ,self.maxt.y , self.maxt.theta )
            self.indicator =  maxt.score

            self.T_tPF = [self.maxt.x ,self.maxt.y , min(self.maxt.theta , 360 -self.maxt.theta)]

    def resampling(self):
        
        W = self.scores/np.sum(self.scores) # Normalized scores for resampling 
        Np = len(self.Rot)
        index = np.random.choice(a = Np ,size = Np ,p = W ) # resample by score
        Rot_arr = [] # creat new temporery array for new sampels 

        for i in index:
            tmp_rot = copy.deepcopy(self.Rot[i])
            tmp_rot.add_noise() # add noise to current sample and set score to 0
            Rot_arr.append(tmp_rot) # resample by weights

        self.Rot = Rot_arr
        self.indicator = 0
        print 'resample done'
 
    def func_de(self , T):

        X = self.tMap.rotate2(T)

        var = 0.16        
        # fit data of map 2 to map 1  
        distances, indices = self.nbrs.kneighbors(X)
        # find the propability 
        prob = (1/(np.sqrt(2*np.pi*var)))*np.exp(-np.power(distances,2)/(2*var)) 
        # returm the 'weight' of this transformation
        wiegth = np.sum((prob)/prob.shape[0])+0.000001 #np.sum(prob)

        return -wiegth # de algo minimized this value

    def norm2(self):
        
        normDE = np.linalg.norm(self.T_de - self.realT) 
        normtPF = np.linalg.norm(self.T_tPF - self.realT) 
       # print 'norm2 of de:' , normDE
       # print 'norm2 of tPF:'  , normtPF

        self.plot_norms(normDE , normtPF)

    def plot_norms(self , normDE , normtPF ):

        plt.axis([0 , 60, 0,  180])
        plt.scatter(self.K ,normDE , color = 'b') # plot tPF map
        plt.scatter(self.K ,normtPF ,color = 'r') # plot origin map
        plt.pause(0.05)
        self.K += 1

    def plotmaps(self):

        plt.axis([-60, 60, -60, 60])
        plt.axis([-30, 30, -30, 30])
        plt.scatter(self.maxMap[: , 0] ,self.maxMap[:,1] , color = 'b') # plot tPF map
        plt.scatter(self.oMap.map[: , 0] ,self.oMap.map[:,1] ,color = 'r') # plot origin map
        plt.scatter(self.de_map[: , 0] ,self.de_map[:,1] , color = 'g') # plot DE map
        plt.pause(0.05)
        plt.clf()

class rot(object):
    
    # define 'rot' to be the class of the rotation for resamplimg filter

    def __init__(self , theta , xShift , yShift):
        
         self.theta = theta
         self.x = xShift
         self.y = yShift
         self.score = 0 

    def weight(self , oMap , tMap , nbrs):
        
        var = 0.16
        # fit data of map 2 to map 1  
        distances, indices = nbrs.kneighbors(tMap)
        # find the propability 
        prob = (1/(np.sqrt(2*np.pi*var)))*np.exp(-np.power(distances,2)/(2*var)) 
        # returm the 'weight' of this transformation
        wiegth = np.sum((prob)/prob.shape[0])+0.000001 

        self.score += wiegth # sum up score
    
    def add_noise(self):
        
        factor = np.random.randn()
        self.x += 0.001 * factor
        self.y += 0.001 * factor
        self.theta += 10 * factor

        self.score=0

class maps:

    def __init__(self , topic_name ):
        
        self.check = True
        self.started = False # indicate if msg recived
        self.map = None #landmarks array
        self.cm = None
        self.name = topic_name

        rospy.Subscriber( topic_name , numpy_msg(Floats) , self.callback)
               
    def callback(self ,data):
  
        # reshape array from 1D to 2D
        landmarks = np.reshape(data.data, (-1, 2))
        # finding C.M for the first iterration

        if self.check:
            # determin the center of the map for initilaized map origin
            self.cm = np.sum(np.transpose(landmarks),axis=1)/len(landmarks)
            print ('set origin of: '), (self.name)
            self.check = False
            
        self.map = np.array(landmarks , dtype= "int32") - self.cm.T
        #print self.map
        self.started = True

    def rotate(self, xShift, yShift , RotationAngle): #rotat map for tPF
      
        theta = np.radians(RotationAngle) # angles to radians
        c ,s = np.cos(theta) , np.sin(theta)
        R = np.array(((c,-s), (s, c))) #Rotation matrix
        RotatedLandmarks = np.matmul( self.map , R ) + np.array([xShift , yShift ]) # matrix multiplation

        return  RotatedLandmarks
   
    def rotate2(self, T): #rotat map for DE
      
        theta = np.radians(T[2]) # angles to radians
        c ,s = np.cos(theta) , np.sin(theta)
        R = np.array(((c,-s), (s, c))) #Rotation matrix
        RotatedLandmarks = np.matmul( self.map , R ) + np.array([T[0] , T[1]]) # matrix multiplation

        return  RotatedLandmarks
   
if __name__ == '__main__':
   
    print ("Running")
    PFtry = tPF() # 'tPF' : Partical Filter
    rospy.spin()