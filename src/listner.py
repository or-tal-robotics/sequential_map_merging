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

        while not rospy.is_shutdown():
            a , b = self.oMap.started ,self.tMap.started  # self.maps.started : indicate if map recieved
            if a and b:
                self.startPF()

                print "one iteration over"

    def initialize_PF( self , angles =np.linspace(0 , 360 , 40) , xRange = np.linspace(-10 , 10 , 10) , yRange = np.linspace(-10 , 10 ,10) ):
       
        # make a list of class rot(s)
        self.Rot = []

        for i in range(len(angles)):
            for j in range(len(xRange)):
                for k in range(len(yRange)):
                    self.Rot.append(rot(angles[i] , xRange[j] , yRange[k]))
        
        self.Rot.append(rot(30 , 10 , 10))
        
        print ("initialaize PF whith") , len(self.Rot) ,(" samples completed")

    def startPF(self):

        #temporry fubction for test
        x = 0.35
        y = -1.2
        theta = 89
        tempMap = self.tMap.rotate(x ,y , theta)


        #plt.scatter(tempMap[: , 0] ,tempMap[:,1])
        #plt.scatter(self.oMap.map[: , 0] ,self.oMap.map[:,1])
        #plt.show(block =True)
        
        i = rot(theta,x,y)
        i.weight(self.oMap.map , tempMap)
        print i.w

        bestT = Transform()
        bestT.translation.x = i.x + self.tMap.cm[0] + 4
        bestT.translation.y = i.y + self.tMap.cm[1] - 1
        bestT.rotation.z = i.theta  + np.radians(-4)
        self.pub.publish(bestT)

        #here is stoped - need to anderstand the rotation of the img 
    

    def startPF2(self):
        
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
        
         self.theta = np.radians(theta)
         self.x = xShift
         self.y = yShift
         self.w = []
         
    def weight(self , oMap , tMap):
        
        var = 0.16
        # initial KNN with the original map 
        nbrs = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(oMap)
        # fit data of map 2 to map 1  
        distances, indices = nbrs.kneighbors(tMap)
        # find the propability 
        prob = (1/(np.sqrt(2*np.pi*var)))*np.exp(-np.power(distances,2)/(2*var)) 
        # returm the 'weight' of this transformation
        wiegth =np.sum((prob)/np.prod(distances.shape)) #np.sum((prob)/np.prod(distances.shape)) 
        
        self.w.append(wiegth)

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
            print 'set origin of: ' ,  self.name
            self.check = False
            
        self.map = np.array(landmarks , dtype= "int32") - self.cm.T
        #print self.map
        self.started = True

    def plot(self):

        plt.scatter(self.map[: , 0] ,self.map[:,1])
        plt.show(block =True)

    def rotate(self, xShift, yShift , RotationAngle): #rotat map
      
        theta = np.radians(RotationAngle) # angles to radians
        c ,s = np.cos(theta) , np.sin(theta)
        R = np.array(((c,-s), (s, c))) #Rotation matrix
        #print R
        #print np.shape(self.map) , np.shape(R)
        RotatedLandmarks = np.matmul( self.map , R ) + np.array([xShift , yShift ]) # matrix multiplation

        return  RotatedLandmarks
   
   
if __name__ == '__main__':
   
    print "Running"
    PFtry = tPF() # 'tPF' : transportation Partical Filter
    rospy.spin()  
    