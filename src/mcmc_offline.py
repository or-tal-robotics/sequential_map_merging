#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors # for KNN algorithm
from scipy.optimize import differential_evolution
import copy
import pandas as pd
from nav_msgs.msg import OccupancyGrid, MapMetaData
from tf.transformations import quaternion_from_euler
import rosbag
import rospkg 

ground_trouth_origin = np.array([-12,-5.0, 0])
ground_trouth_target = np.array([4.0, -8.0, -2.75])
ground_trouth_transformation_map5 = np.array([-6.94304748,  9.92673817,  3.56565882])
ground_trouth_transformation_map6 = np.array([4.34298586,  11.52861869,  1.58713136])
ground_trouth_transformation_map7 = np.array([-0.10729044,  4.94486143,  1.82609867])

rospack = rospkg.RosPack()
packadge_path = rospack.get_path('DMM')
file_path = packadge_path + '/maps/map7.bag'
origin_publisher = rospy.Publisher('origin_map', OccupancyGrid, queue_size = 10) 
target_publisher = rospy.Publisher('target_map', OccupancyGrid, queue_size = 10) 

def send_map_ros_msg(landmarks, empty_landmarks, publisher, resolution = 0.01, width = 2048, height = 2048):
    map_msg = OccupancyGrid()
    map_msg.header.frame_id = 'map'
    map_msg.info.resolution = resolution
    map_msg.info.width = width
    map_msg.info.height = height

    data = -np.ones(shape = (width,height))
    for ii in range(len(landmarks)):
        on_x = landmarks[ii,0] // resolution + width  // 2
        on_y = landmarks[ii,1] // resolution + height // 2
        if on_x < width and on_x > 0 and on_y < height and on_y > 0:
            data[int(on_x), int(on_y)] = 100

    for ii in range(len(empty_landmarks)):
        off_x = empty_landmarks[ii,0] // resolution + width  // 2
        off_y = empty_landmarks[ii,1] // resolution + height // 2
        if off_x < width and off_x > 0 and off_y < height and off_y > 0:
            data[int(off_x), int(off_y)] = 0

    data_out = data.reshape((-1))
    map_msg.data = data_out
    publisher.publish(map_msg)
    
def rotate_map(map, T):
    c ,s = np.cos(T[2]) , np.sin(T[2])
    R = np.array(((c,-s), (s, c))) 
    rot_map = np.matmul(map,R) + T[0:2]
    return rot_map

def likelihood(target_map_rotated, origin_map_nbrs, var):
    d, _ = origin_map_nbrs.kneighbors(target_map_rotated)
    p = np.sum((1/(np.sqrt(2*np.pi*var)))*np.exp(-np.power(d,2)/(2*var))) + 1e-100
    return p

def get_error(T): 
    return  np.linalg.norm(T - ground_trouth_transformation)


def DEMapMatcher(origin_map_nbrs, target_map):
    DE_func = lambda x: -likelihood(rotate_map(target_map,x),origin_map_nbrs, 0.3)
    result = differential_evolution(DE_func, bounds = [(-15,15),(-15,15),(0,2*np.pi)] ,maxiter= 100 ,popsize=3,tol=0.0001)
    T_de = [result.x[0] , result.x[1] , min(result.x[2], 2*np.pi - result.x[2])]
    return T_de

class ParticleFilterMapMatcher():
    def __init__(self,init_origin_map_nbrs, init_target_map, Np = 1000, N_history = 7,  N_theta = 50, N_x = 20, N_y = 20, R_var = 0.03):
        self.Np = Np
        self.R_var = R_var
        self.N_history = N_history
        self.filter = np.arange(3,N_history+3,dtype=np.float32)
        temp_X = []
        angles = np.linspace(0 , 2*np.pi ,N_theta )
        xRange = np.linspace(-15 , 15 , N_x) 
        yRange = np.linspace(-15 , 15 ,N_y) 
        x0 = [xRange[np.random.randint(N_x)] ,yRange[np.random.randint(N_y)], angles[np.random.randint(N_theta)]]
        tempMap = rotate_map(init_target_map, x0)
        w0 = likelihood(tempMap, init_origin_map_nbrs, self.R_var)
        temp_X.append(x0)
        i = 0
        print("Initilizing particles...")
        while i < (N_theta*N_x*N_y):
            xt = [xRange[np.random.randint(N_x)],
                yRange[np.random.randint(N_y)],
                angles[np.random.randint(N_theta)]]
            tempMap = rotate_map(init_target_map, xt)
            wt = likelihood(tempMap, init_origin_map_nbrs, self.R_var)
            if wt>w0:
                temp_X.append(xt)
                x0 = xt
                w0 = wt
            elif np.random.binomial(1, wt/w0) == 1:
                temp_X.append(xt)
                x0 = xt
                w0 = wt
            elif np.random.binomial(1, 0.5) == 1:
                temp_X.append(xt)
                x0 = xt
                w0 = wt
            else:
                x = x0
                x[0] = x[0] + np.random.normal(0.0, 0.1)
                x[1] = x[1] + np.random.normal(0.0, 0.1)
                x[2] = x[2] + np.random.normal(0.0, 0.1) + np.random.choice(a = 4,p = [0.4,0.2,0.2,0.2] )*0.5*np.pi
                x[2] = np.remainder(x[2],2*np.pi)
                temp_X.append(x)
            i += 1
        self.X = np.array(temp_X[-Np:])
        self.W = np.ones((Np,N_history))
        self.indicate = 0
        print("Initilizing done with "+str(Np)+" samples out of "+str(len(temp_X)))
    def predict(self):
        self.X[:,0:2] = self.X[:,0:2] + np.random.normal(0.0, 0.07, size=self.X[:,0:2].shape)
        self.X[:,2] = self.X[:,2] + np.random.normal(0.0, 0.1, size=self.X[:,2].shape)
        self.X[:,2] = np.remainder(self.X[:,2],2*np.pi)

    def update(self, target_map, origin_map_nbrs):
        for i in range(self.Np):
            tempMap = rotate_map(target_map, self.X[i])
            if self.indicate > 0:
                self.W[i, self.indicate] = self.W[i, self.indicate - 1] * likelihood(tempMap, origin_map_nbrs, self.R_var)
            else:
                self.W[i, self.indicate] = likelihood(tempMap, origin_map_nbrs, self.R_var)
        self.indicate += 1
        p = np.dot(self.W, self.filter)
    def resample(self):
        print("performing resample!")
        p = np.dot(self.W, self.filter)
        p = p/np.sum(p)
        self.X_map = self.X[np.argmax(p)]
        idxs = np.random.choice(a = self.Np, size = self.Np,p = p)
        self.X = self.X[idxs]
        self.X[:,0] = self.X[:,0] + np.random.normal(0.0, 0.2, size=self.X[:,0].shape) + np.random.randint(-1,2) * np.random.choice(a = 5, size = self.X[:,0].shape,p = [0.6,0.1,0.1,0.1, 0.1] )*2.0
        self.X[:,1] = self.X[:,1] + np.random.normal(0.0, 0.2, size=self.X[:,1].shape) + np.random.randint(-1,2) * np.random.choice(a = 5, size = self.X[:,1].shape,p = [0.6,0.1,0.1,0.1, 0.1]  )*2.0
        self.X[:,2] = self.X[:,2] + np.random.normal(0.0, 0.01, size=self.X[:,2].shape) + np.random.choice(a = 4, size = self.X[:,2].shape,p = [0.4,0.2,0.2,0.2] )*0.5*np.pi
        self.X[:,2] = np.remainder(self.X[:,2],2*np.pi)
        self.indicate = 0

   
if __name__ == '__main__':
    bag = rosbag.Bag(file_path)
    init, init1, init2 = 1, 1, 1
    err_pf = []
    err_de = []
    rospy.init_node('offline_map_matcher')
    for topic, msg, t in bag.read_messages(topics=['/ABot1/map', '/ABot2/map']):
        if rospy.is_shutdown():
            break
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
                    ground_trouth_origin[0:2] = ground_trouth_origin[0:2] -  cm1
                landMarksArray1 = landMarksArray1 - cm1
                landMarksArray1_empty = landMarksArray1_empty - cm1
                landMarksArray1_rc = landMarksArray1[np.random.choice(a = len(landMarksArray1), size = len(landMarksArray1)//2)]
                
                nbrs = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(landMarksArray1_rc)
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
                    ground_trouth_target[0:2] = np.flip(ground_trouth_target[0:2])  -  cm2
                landMarksArray2 = landMarksArray2 - cm2
                landMarksArray2_empty = landMarksArray2_empty - cm2
                init2 = 0
            else:
                continue
        if init == 1 and init1 == 0 and init2 == 0:
            model = ParticleFilterMapMatcher(nbrs, landMarksArray2)
            
            init = 0
        elif init == 0 and init1 == 0 and init2 == 0:
            model.predict()
            model.update(landMarksArray2, nbrs)
            X_de = DEMapMatcher(nbrs, landMarksArray2)
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

                # err_pf.append(get_error(model.X_map))
                # err_de.append(get_error(X_de ))
                # plt.subplot(3,1,3)
                # plt.plot(err_pf, color = 'b')
                # plt.plot(err_de, color = 'r')
                # plt.pause(0.05)
                # plt.clf()
                print(model.X_map)
                send_map_ros_msg(landMarksArray1, landMarksArray1_empty, origin_publisher, resolution=scale1)
                send_map_ros_msg(rotate_map(landMarksArray2, model.X_map),rotate_map(landMarksArray2_empty, model.X_map) , target_publisher, resolution=scale2)
    raw_input("Press Enter to continue...")

    

        
        