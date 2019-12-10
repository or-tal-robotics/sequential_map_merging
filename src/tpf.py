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



class tPF():
    
    def __init__(self):

        self.pub = rospy.Publisher('/TM2', Transform , queue_size=1000 )
        rospy.init_node('listener', anonymous=True)
       
        # creat first particales 
        self.initialize_PF()

        # convert maps to landmarks arrays:
        self.oMap = maps("LM1") 
        self.tMap = maps("LM2")
        self.best_score = 0
        self.realT = np.array([-1.5 , -2.5 , 90]) # real transformation

        self.N_eff = 1
        self.itr = 0
        self.prob = 0.1

        self.K = 1 # time step for norm2

        self.Nde = []
        self.NtPF = []
        self.resample_counter = 0

        while not rospy.is_shutdown():

            a , b = self.oMap.started ,self.tMap.started  # self.maps.started : indicate if map recieved
        
            if a and b:
                
                #init nbrs for KNN
                self.nbrs = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(self.oMap.map)

                # DE algorithm for finding best match
                #result = differential_evolution(self.func_de, bounds = [(-10,10),(-10,10),(0,360)] ,maxiter= 200 ,popsize=6,tol=0.0001)
                #self.T_de = [result.x[0] , result.x[1] , min(result.x[2], 360 - result.x[2])] 
                #print self.T_de

                self.likelihood_PF()
                print "N_eff=:"+str(self.N_eff)
                self.resample_counter +=1
                #print resample_counter

                if self.N_eff < 0.0001 :
                    self.resampling() # start re-sampling step 
                    self.resample_counter = 0

                #self.norm2() # finding norm2

               # if self.K == 80 :

                #    self.save_data_DE()

                #self.K += 1

                #self.de_map = self.tMap.rotate2(result.x)
                self.plotmaps() # plot landmarks of maps.




    def save_data_DE(self):

        if os.path.isfile('~/DE.csv'):
            with open('~/DE.csv', 'a') as f:
                self.save_stat_DE('~/DE.csv', ex = True, f=f)
        else:
            self.save_stat_DE('~/DE.csv', ex = False)

        print 'save data'

    def save_stat_DE(self,file_path, ex, f = None):

        data = {'De': self.Nde }
                
        df = pd.DataFrame(data, columns= ['De'])

        if ex==False:
            df.to_csv(file_path, sep='\t')
        else:
            df.to_csv(f, sep='\t', header=False)

    def save_data_tPF(self):

        if os.path.isfile('~/tPF.csv'):
            with open('~/tPF.csv', 'a') as f:
                self.save_stat_tPF('~/tPF.csv', ex = True, f=f)
        else:
            self.save_stat_tPF('~/tPF.csv', ex = False)

        print 'data saved'

    def save_stat_tPF(self,file_path, ex, f = None):

        data = {'tPF': self.NtPF}
                
        df = pd.DataFrame(data, columns= ['tPF'])

        if ex==False:
            df.to_csv(file_path, sep='\t')
        else:
            df.to_csv(f, sep='\t', header=False)

    def initialize_PF( self , angles = np.linspace(0 , 360 , 45) , xRange = np.linspace(-20 , 20 , 10) , yRange = np.linspace(-10 , 10 ,10) ):
       
        # make a list of class rot(s)
        self.Rot = []

        for i in range(len(angles)):
            for j in range(len(xRange)):
                for k in range(len(yRange)):
                    self.Rot.append(rot(angles[i] , xRange[j] , yRange[k]))
        
        print ("initialaize PF whith") , len(self.Rot) ,(" samples completed")

    def likelihood_PF(self):

        self.scores = []
        factor = np.power(1 - self.prob , self.resample_counter - self.itr )
        self.itr +=1 

        for i in self.Rot:    
            # 'tempMap' ->  map after transformation the secondery map [T(tMap)]:
            tempMap = self.tMap.rotate(i.x ,i.y , i.theta)
            i.weight(self.oMap.map , tempMap ,self.nbrs , factor)
            self.scores.append(i.score) # add weights to array 
        w = np.array(self.scores)
        self.N_eff = 1/np.sum(np.power(w,2))
        maxt = max( self.Rot , key = operator.attrgetter('score') ) # finds partical with maximum score

        if maxt.score > self.best_score:          
        # check if there is a new partical thats better then previuos partical
            self.maxt = maxt
          
            self.maxMap = self.tMap.rotate(self.maxt.x ,self.maxt.y , self.maxt.theta )
            self.best_score =  maxt.score

            self.T_tPF = [self.maxt.x ,self.maxt.y , np.radians(self.maxt.theta)]
            print 'max W(tPF):' ,self.T_tPF

    def resampling(self):

        self.itr = 0
        W = self.scores/np.sum(self.scores) # Normalized scores for resampling 
        Np = len(self.Rot)
        index = np.random.choice(a = Np ,size = Np ,p = W ) # resample by score
        Rot_arr = [] # creat new temporery array for new sampels 

        for i in index:
            tmp_rot = copy.deepcopy(self.Rot[i])
            tmp_rot.add_noise() # add noise to current sample and set score to 0
            Rot_arr.append(tmp_rot) # resample by weights

        self.Rot = Rot_arr
        self.best_score = 0
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
        
        # find norm 2 of transformation 
        normDE = np.linalg.norm(self.T_de - self.realT) 
       # normtPF = np.linalg.norm(self.T_tPF - self.realT) 
        self.Nde.append(normDE)
       # self.NtPF.append(normtPF)
        #self.plot_norms(normDE , normtPF)
        
    def plot_norms(self , normDE , normtPF ):

        plt.axis([0 , 60, 0,  180])
       # plt.scatter(self.K ,normDE , color = 'b') # plot tPF map
        plt.scatter(self.K ,normtPF ,color = 'r') # plot origin map
        plt.pause(0.05)
 
    def plotmaps(self):

        plt.axis([-60, 60, -60, 60])
        plt.axis([-30, 30, -30, 30])
        plt.scatter(self.maxMap[: , 0] ,self.maxMap[:,1] , color = 'b') # plot tPF map
        plt.scatter(self.oMap.map[: , 0] ,self.oMap.map[:,1] ,color = 'r') # plot origin map
        #plt.scatter(self.de_map[: , 0] ,self.de_map[:,1] , color = 'g') # plot DE map
        plt.pause(0.05)
        plt.clf()

class rot(object):
    
    # define 'rot' to be the class of the rotation for resamplimg filter

    def __init__(self , theta , xShift , yShift):
        
         self.theta = theta
         self.x = xShift
         self.y = yShift
         self.score = 0 



    def weight(self , oMap , tMap , nbrs , factor ):
        
        var = 0.16
        # fit data of map 2 to map 1  
        distances, indices = nbrs.kneighbors(tMap)
        # find the propability 
        prob = (1/(np.sqrt(2*np.pi*var)))*np.exp(-np.power(distances,2)/(2*var)) 
        # returm the 'weight' of this transformation
        wiegth = np.sum((prob)/prob.shape[0])+0.000001 #np.sum(prob) 
        
        self.score += wiegth * factor # sum up score
    
    def add_noise(self):
        
        self.x += 0.5 * np.random.randn()
        self.y += 0.5 * np.random.randn()
        self.theta += 0.5  * np.random.randn()
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