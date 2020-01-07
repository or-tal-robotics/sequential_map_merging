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
import cv2
import imutils
import time
ground_trouth_origin = np.array([-12,-5.0, 0])
ground_trouth_target = np.array([4.0, -8.0, -2.75])
ground_trouth_transformation = np.array([-6.94304748,  9.92673817,  3.56565882])
rospack = rospkg.RosPack()
packadge_path = rospack.get_path('DMM')
file_path = packadge_path + '/maps/map5.bag'
origin_publisher = rospy.Publisher('origin_map', OccupancyGrid) 
target_publisher = rospy.Publisher('target_map', OccupancyGrid) 

def show_maps(T, origin, target):
    image_rot = rotate_image(target, T[2])
    image_t = translate_image(image_rot,T[0:2], 0.01)
    image_o = crop_image(origin, image_t.shape)
    if image_o.shape != image_t.shape:
        image_t = image_t[:-1,:-1]
    image_out = 0.5*image_o + 0.5*image_t
    image_out = cv2.resize(image_out, (600, 600))
    return image_out

def rotate_image(image, theta, convert = True, core = (2000, 2000)):
    image_temp = np.array(image)
    if convert == True:
        image_temp[image_temp == -1] = 0.5
        image_temp[image_temp == 100] = 1.0
        image_temp[image_temp == 0 ] = 0.1
        theta_degrees = theta*180/(np.pi)
        rotated_image = imutils.rotate_bound(image_temp, int(theta_degrees))
        image_shape = np.shape(image_temp)
        rotated_image_shape = np.shape(rotated_image)
        diff0 = (rotated_image_shape[0] - image_shape[0])//2
        diff1 = (rotated_image_shape[1] - image_shape[1])//2
        rotated_image = rotated_image[diff0:-diff0-1, diff1:-diff1-1]
        rotated_image[rotated_image == 0 ] = 0.5  
        diff0_core = (rotated_image_shape[0] - core[0])//2
        diff1_core = (rotated_image_shape[1] - core[1])//2
    return rotated_image[diff0_core:-diff0_core-1, diff1_core:-diff1_core-1]

def translate_image(image, T, res):
    image_temp = 0.5*np.ones_like(image)
    Tx = T[0]//res
    Ty = T[1]//res
    translation_matrix = np.float32([ [1,0,Tx], [0,1,Ty] ])
    num_rows, num_cols = image.shape
    img_translation = cv2.warpAffine(image, translation_matrix, (num_cols, num_rows))
    image_temp[image_temp == -1] = 0.5
    image_temp[image_temp == 100] = 1.0
    img_translation[img_translation == 0 ] = 0.5
    return img_translation

def crop_image(image, rotated_image_shape):
    image_temp = np.array(image)
    image_temp[image_temp == -1] = 0.5
    image_temp[image_temp == 100] = 1.0
    image_temp[image_temp == 0 ] = 0.1
    image_shape = np.shape(image)
    diff0 = (image_shape[0] - rotated_image_shape[0])//2
    diff1 = (image_shape[1] - rotated_image_shape[1])//2
    return image_temp[diff0:-diff0-1, diff1:-diff1-1]

def likelihood_images(T,origin, target):
    
    image_rot = rotate_image(target, T[2])
    image_t = translate_image(image_rot,T[0:2], 0.01)
    image_o = crop_image(origin, image_t.shape)
    if image_o.shape != image_t.shape:
        image_t = image_t[:-1,:-1]
    p = np.mean(np.multiply(image_o, image_t))
    return p


def send_map_ros_msg(landmarks, empty_landmarks, publisher, resolution = 0.01, width = 1024, height = 1024):
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
    def __init__(self,init_origin_map, init_target_map, Np = 100, N_history = 2,  N_theta = 10, N_x = 10, N_y = 10, R_var = 0.3):
        self.Np = Np
        self.R_var = R_var
        self.N_history = N_history
        self.filter = np.arange(3,N_history+3,dtype=np.float32)
        temp_X = []
        angles = np.linspace(0 , 2*np.pi ,N_theta )
        xRange = np.linspace(-1 , 1 , N_x) 
        yRange = np.linspace(-1 , 1 ,N_y) 
        x0 = [xRange[np.random.randint(N_x)] ,yRange[np.random.randint(N_y)], angles[np.random.randint(N_theta)]]
        w0 = likelihood_images(x0,init_origin_map, init_target_map )
        temp_X.append(x0)
        i = 0
        print("Initilizing particles...")
        while i < (N_theta*N_x*N_y) and not rospy.is_shutdown():
            xt = [xRange[np.random.randint(N_x)],
                yRange[np.random.randint(N_y)],
                angles[np.random.randint(N_theta)]]
            wt = likelihood_images(xt,init_origin_map, init_target_map )
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

    def update(self, origin_map, target_map):
        for i in range(self.Np):
            if self.indicate > 0:
                self.W[i, self.indicate] = self.W[i, self.indicate - 1] * likelihood_images(self.X[i],origin_map, target_map )
            else:
                self.W[i, self.indicate] =  likelihood_images(self.X[i],origin_map, target_map )
        self.indicate += 1
        p = np.dot(self.W, self.filter)
    def resample(self):
        print("performing resample!")
        p = np.dot(self.W, self.filter)
        p = p/np.sum(p)
        self.X_map = self.X[np.argmax(p)]
        idxs = np.random.choice(a = self.Np, size = self.Np,p = p)
        self.X = self.X[idxs]
        self.X[:,0] = self.X[:,0] + np.random.normal(0.0, 0.1, size=self.X[:,0].shape) + np.random.randint(-1,2) * np.random.choice(a = 5, size = self.X[:,0].shape,p = [0.6,0.1,0.1,0.1, 0.1] )*0.0
        self.X[:,1] = self.X[:,1] + np.random.normal(0.0, 0.1, size=self.X[:,1].shape) + np.random.randint(-1,2) * np.random.choice(a = 5, size = self.X[:,1].shape,p = [0.6,0.1,0.1,0.1, 0.1]  )*0.0
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
            if init1 == 1:
                cm1 = np.sum(np.transpose(landMarksArray1),axis=1)/len(landMarksArray1)
                ground_trouth_origin[0:2] = ground_trouth_origin[0:2] -  cm1
            landMarksArray1 = landMarksArray1 - cm1
            landMarksArray1_empty = landMarksArray1_empty - cm1
            nbrs = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(landMarksArray1)
            init1 = 0
        if topic == '/ABot2/map':
            map2_msg = msg
            map2 = np.array(msg.data , dtype = np.float32)
            N2 = np.sqrt(map2.shape)[0].astype(np.int32)
            Re2 = np.copy(map2.reshape((N2,N2)))
            scale2 = msg.info.resolution
            landMarksArray2 = (np.argwhere( Re2 == 100 ) * scale2)
            landMarksArray2_empty = (np.argwhere( Re2 == 0 ) * scale2)
            if init2 == 1:
                cm2 = np.sum(np.transpose(landMarksArray2),axis=1)/len(landMarksArray2) 
                ground_trouth_target[0:2] = np.flip(ground_trouth_target[0:2])  -  cm2
            landMarksArray2 = landMarksArray2 - cm2
            landMarksArray2_empty = landMarksArray2_empty - cm2
            init2 = 0
        if init == 1 and init1 == 0 and init2 == 0:
            model = ParticleFilterMapMatcher(Re1, Re2)
            
            init = 0
        elif init == 0 and init1 == 0 and init2 == 0:
            model.predict()
            model.update(Re1, Re2)
            X_de = DEMapMatcher(nbrs, landMarksArray2)
            if model.indicate == model.N_history:
                model.resample()
                print(likelihood_images(model.X_map,Re1, Re2))
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
                image_rot = rotate_image(Re2, model.X_map[2])
                image_t = translate_image(image_rot,model.X_map[0:2], 0.01)
                combined_map = show_maps(model.X_map,Re1, Re2)
                cv2.imshow("map_combined", combined_map)
                cv2.imshow("map1", cv2.resize(Re1, (600, 600)))
                cv2.imshow("map2", cv2.resize(Re2, (600, 600)))
                cv2.imshow("map2_T", cv2.resize(image_t, (600, 600)))
                time.sleep(0.01)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                #send_map_ros_msg(landMarksArray1, landMarksArray1_empty, origin_publisher, resolution=scale1)
                #send_map_ros_msg(rotate_map(landMarksArray2, model.X_map),rotate_map(landMarksArray2_empty, model.X_map) , target_publisher, resolution=scale2)
    raw_input("Press Enter to continue...")

    

        
        