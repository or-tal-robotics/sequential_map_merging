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

        self.pub1 = rospy.Publisher('LM1', numpy_msg(Floats),queue_size=10)  # publisher of landmarks of map 1
        self.pub2 = rospy.Publisher('LM2', numpy_msg(Floats),queue_size=10)  # publisher of landmarks of map 2
    
        self.map1_LM = None
        self.map2_LM = None 

        self.map1_OG = rospy.Subscriber("/ABot1/map", OccupancyGrid , self.callbackM1 )
        self.map2_OG = rospy.Subscriber("/ABot2/map", OccupancyGrid , self.callbackM2 )
        
        r = rospy.Rate(0.1) # 0.1 hz
     
        while not rospy.is_shutdown():

            if (self.started1):

                a = self.map1_LM.ravel()
                print a.shape
                self.pub1.publish(a)
                print "landmarks array of map 1 is passed"
           
            if (self.started2):

                a = self.map2_LM.ravel()
                print a.shape
                self.pub2.publish(a)
                print "landmarks array of map 2 is passed"

            r.sleep()        

    def callbackM1(self ,msg):
  
        
        maps = np.array(msg.data , dtype = np.float32)
        N = np.sqrt(maps.shape)[0].astype(np.int32)
        Re = np.copy(maps.reshape((N,N)))
        
        #convert to landmarks array
        scale = msg.info.resolution
        CenterShift = msg.info.width/20   
        landMarksArray = (np.argwhere( Re == 100 ) * scale) # - np.array([CenterShift ,CenterShift]) 
       # print landMarksArray
        
        self.map1_LM = landMarksArray.astype(np.float32) #-np.array()
 
        self.started1 = True   

    def callbackM2(self ,msg):

        maps = np.array(msg.data , dtype = np.float32)
        N = np.sqrt(maps.shape)[0].astype(np.int32)
        Re = np.copy(maps.reshape((N,N)))
        
        #convert to landmarks array
        scale = msg.info.resolution
        landMarksArray = np.argwhere( Re == 100 ) * scale
    
        
        self.map2_LM = landMarksArray.astype(np.float32)
 
        self.started2 = True        

def listener():

    print "Running"
    LM_maps = maps() # convert maps to landmarks arrays
    rospy.spin()

if __name__ == '__main__':
    listener()
    