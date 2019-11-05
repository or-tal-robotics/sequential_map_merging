#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
Created on Thu Sep  5 10:09:21 2019

@author: matan
"""


import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path
import tf_conversions as tf
from geometry_msgs.msg import Transform 


newMap = ""
started = False
T = Transform()
pub = rospy.Publisher('/map2T', OccupancyGrid , queue_size=1000 )


def callback(data):
 
    
    print "New map recived"
    r = rospy.Rate(0.1) # 0.1 hz
    if started:

        global T ,pub
        newMap = data
        
        R = newMap.info.resolution

        d = (-512/2 )* R
        Dott = np.array([d,d])
        theta = T.rotation.z
        
        print theta

        # change map orientation
        x , y ,z , w = tf.transformations.quaternion_from_euler(0, 0, theta)    
        newMap.info.origin.orientation.x = x
        newMap.info.origin.orientation.y = y
        newMap.info.origin.orientation.z = z
        newMap.info.origin.orientation.w = w

        # change map origin 
        newPos =  rotate(theta , Dott )
        print newPos[0] , newPos[1] 
        newMap.info.origin.position.x = newPos[1]  + T.translation.x +d
        newMap.info.origin.position.y = newPos[0]  + T.translation.y +d

    

        pub.publish(newMap)
        #print newMap.info.origin
            
        
    


def callbackT(data):

    global T ,started
    T = data
    started = True

def listener():

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/ABot2/map", OccupancyGrid , callback)
    rospy.Subscriber("/TM2", Transform , callbackT)
    rospy.spin()   

def rotate(RotationAngle ,dot): #rotat map
    
    theta = RotationAngle # angles to radians
    c ,s = np.cos(theta) , np.sin(theta)
    R = np.array(((c,-s), (s, c))) #Rotation matrix
    RotatedLandmarks = np.matmul( dot , R )  # matrix multiplation

    return  RotatedLandmarks 

if __name__ == '__main__':

    print "Running"
    listener()
    