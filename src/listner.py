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



class tPF():
    
    def __init__(self,Np = 100):

        # creat Np first particales 
        self.Np = Np
        self.initialize_PF()

        rospy.init_node('listener', anonymous=True)
        
        self.pub = rospy.Publisher('/TM2', Transform , queue_size=1000 )

        # convert maps to landmarks arrays:
        self.oMap = maps("LM1") 
        self.tMap = maps("LM2")
        self.bMap = []
        self.indicator = 0

        plt.axis([-60, 60, -60, 60])

        while not rospy.is_shutdown():

            a , b = self.oMap.started ,self.tMap.started  # self.maps.started : indicate if map recieved
            if a and b:

                self.likelihood_PF()

                plt.axis([-30, 30, -30, 30])
                plt.scatter(self.maxMap[: , 0] ,self.maxMap[:,1] , color = 'b')
                plt.scatter(self.oMap.map[: , 0] ,self.oMap.map[:,1] ,color = 'r')
                plt.pause(0.05)
                plt.clf()

                if self.maxt.score > 4 :
                    self.resample()



    def resample(self):

        mu_theta = self.maxt.theta
        mu_x = self.maxt.x
        mu_y = self.maxt.y
        
        sigmaT = 5
        sigmaS = 2

        #Draw random samples from a normal (Gaussian) distribution.
        angles = s = np.random.normal(mu_theta, sigmaT , 10)
        xRange = s = np.random.normal(mu_x, sigmaS, 10)
        yRange = s = np.random.normal(mu_y, sigmaS, 10)

        # make a list of class rot(s)
        self.Rot = []

        for i in range(len(angles)):
            for j in range(len(xRange)):
                for k in range(len(yRange)):
                    self.Rot.append(rot(angles[i] , xRange[j] , yRange[k]))
        
        print ("Resample PF whith") , len(self.Rot) ,(" samples completed")




    def initialize_PF( self , angles = np.linspace(0 , 360 , 40) , xRange = np.linspace(-10 , 10 , 10) , yRange = np.linspace(-10 , 10 ,10) ):
       
        # make a list of class rot(s)
        self.Rot = []

        for i in range(len(angles)):
            for j in range(len(xRange)):
                for k in range(len(yRange)):
                    self.Rot.append(rot(angles[i] , xRange[j] , yRange[k]))
        
        print ("initialaize PF whith") , len(self.Rot) ,(" samples completed")

    def likelihood_PF(self):

        for i in self.Rot:
            
            # 'tempMao' ->  map after transformation the secondery map [T(tMap)]:
            tempMap = self.tMap.rotate(i.x ,i.y , i.theta)
            i.weight(self.oMap.map , tempMap)
      
        
        maxt = max(self.Rot , key = operator.attrgetter('score')) # finds partical with maximum liklihood

        if maxt.w[0] > self.indicator: # check if the new partical is better then previuos partical

            self.maxt = maxt
            print 'max W:' ,self.maxt.score ,self.maxt.theta
            self.maxMap = self.tMap.rotate(self.maxt.x ,self.maxt.y , self.maxt.theta )
            self.indicator =  maxt.w[0]

    

    def startPF2(self):

    
        x = -1.45
        y = -3.5
        theta = 90
        self.bMap = self.tMap.rotate(x ,y , theta)
        
        i = rot(theta,x,y)
        i.weight(self.oMap.map , self.bMap )
        print 'True W:',(i.w)

        
        r = rospy.Rate(0.1) # 0.1 hz
        for i in self.Rot:
            
            # 'tempMao' ->  map after transformation the secondery map [T(tMap)]:
            tempMap = self.tMap.rotate(i.x ,i.y , i.theta)
            i.weight(self.oMap.map , tempMap)
        
        t = 100
        bestT = Transform()
        bestT.translation.x = self.Rot[t].x
        bestT.translation.y = self.Rot[t].y
        bestT.rotation.z = self.Rot[t].theta
        self.pub.publish(bestT)

class rot(object):
    
    # define 'rot' to be the class of the rotation for resamplimg filter

    def __init__(self , theta , xShift , yShift):
        
         self.theta = theta
         self.x = xShift
         self.y = yShift
         self.w = [0]
         self.score = 0 

         
    def weight(self , oMap , tMap):
        
        var = 0.16
        # initial KNN with the original map 
        nbrs = NearestNeighbors(n_neighbors= 2, algorithm='ball_tree').fit(oMap)
        # fit data of map 2 to map 1  
        distances, indices = nbrs.kneighbors(tMap)
        # find the propability 
        prob = (1/(np.sqrt(2*np.pi*var)))*np.exp(-np.power(distances,2)/(2*var)) 
        # returm the 'weight' of this transformation
        wiegth = np.sum((prob)/np.prod(distances.shape)) #np.sum(prob)
        
        self.w.insert(0,wiegth)

        self.score += wiegth

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
        #print landmarks
        # finding C.M for the first iterration

        if self.check:
            # determin the center of the map for initilaized map origin
            self.cm = np.sum(np.transpose(landmarks),axis=1)/len(landmarks)
            print ('set origin of: ' ), (self.name)
            self.check = False
            
        self.map = np.array(landmarks , dtype= "int32") - self.cm.T
        #print self.map
        self.started = True

    def rotate(self, xShift, yShift , RotationAngle): #rotat map
      
        theta = np.radians(RotationAngle) # angles to radians
        c ,s = np.cos(theta) , np.sin(theta)
        R = np.array(((c,-s), (s, c))) #Rotation matrix
        #print R
        #print np.shape(self.map) , np.shape(R)
        RotatedLandmarks = np.matmul( self.map , R ) + np.array([xShift , yShift ]) # matrix multiplation

        return  RotatedLandmarks
   
   
if __name__ == '__main__':
   
    print ("Running")
    PFtry = tPF() # 'tPF' : transportation Partical Filter
    rospy.spin()  
    