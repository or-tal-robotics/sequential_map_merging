#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
Created on Thu Sep  5 10:09:21 2019

@author: Matan Samina
"""

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats


class maps:

    def __init__(self):
        

        rospy.init_node('listener', anonymous=True)
        
        self.started1 = False  # 'started': if map recived -> true
        self.started2 = False

        self.pub1 = rospy.Publisher('LM1', numpy_msg(Floats),queue_size=1)  # publisher of landmarks of map 1
        self.pub1_anti = rospy.Publisher('LM1_anti', numpy_msg(Floats),queue_size=1)  # publisher of landmarks of map 2
        self.pub2 = rospy.Publisher('LM2', numpy_msg(Floats),queue_size=1)  # publisher of landmarks of map 1
        self.pub2_anti = rospy.Publisher('LM2_anti', numpy_msg(Floats),queue_size=1)  # publisher of landmarks of map 2
    
        self.map1_LM = None
        self.anti_map1_LM = None
        self.map2_LM = None 
        self.anti_map2_LM = None

        self.map1_OG = rospy.Subscriber("/ABot1/map", OccupancyGrid , self.callbackM1 )
        self.map2_OG = rospy.Subscriber("/ABot2/map", OccupancyGrid , self.callbackM2 )
        
        r = rospy.Rate(1) # 0.1 hz
     
        while not rospy.is_shutdown():

            if (self.started1):

                a = self.map1_LM.ravel()
                anti_a = self.anti_map1_LM.ravel()
                #print a.shape
                self.pub1.publish(a)
                self.pub1_anti.publish(anti_a)
                #print "landmarks array of map 1 is passed"
                self.started1 = False
            if (self.started2):

                a = self.map2_LM.ravel()
                anti_a = self.anti_map2_LM.ravel()
                #print a.shape
                self.pub2.publish(a)
                self.pub2_anti.publish(anti_a)
                #print "landmarks array of map 2 is passed"
                self.started2 = False
            r.sleep()        

    def callbackM1(self ,msg):
  
        
        maps = np.array(msg.data , dtype = np.float32)
        N = np.sqrt(maps.shape)[0].astype(np.int32)
        Re = np.copy(maps.reshape((N,N)))
        
        #convert to landmarks array
        scale = msg.info.resolution
        #CenterShift = msg.info.width/20   
        landMarksArray = (np.argwhere( Re == 100 ) * scale) # - np.array([CenterShift ,CenterShift]) 
        anti_landMarksArray = (np.argwhere( Re == 0 ) * scale)
       # print landMarksArray
        
        self.map1_LM = landMarksArray.astype(np.float32) #-np.array()
        #self.map1_LM = self.map1_LM[np.random.choice(len(self.map1_LM), min([len(self.map1_LM), 1000]))]
        self.anti_map1_LM = anti_landMarksArray.astype(np.float32)
        self.anti_map1_LM = self.anti_map1_LM[np.random.choice(len(self.anti_map1_LM), min([len(self.anti_map1_LM), 500]))]
        self.started1 = True   

    def callbackM2(self ,msg):

        maps = np.array(msg.data , dtype = np.float32)
        N = np.sqrt(maps.shape)[0].astype(np.int32)
        Re = np.copy(maps.reshape((N,N)))
        
        #convert to landmarks array
        scale = msg.info.resolution
        landMarksArray = np.argwhere( Re == 100 ) * scale
        anti_landMarksArray = (np.argwhere( Re == 0 ) * scale)
        
        self.map2_LM = landMarksArray.astype(np.float32)
        #self.map2_LM = self.map2_LM[np.random.choice(len(self.map2_LM), min([len(self.map2_LM), 1000]))]
        self.anti_map2_LM = anti_landMarksArray.astype(np.float32)
        self.anti_map2_LM = self.anti_map2_LM[np.random.choice(len(self.anti_map2_LM), min([len(self.anti_map2_LM), 500]))]
        self.started2 = True        

def listener():

    print "init     convert map to landmarks"
    LM_maps = maps() # convert maps to landmarks arrays
    rospy.spin()

if __name__ == '__main__':
    listener()
    