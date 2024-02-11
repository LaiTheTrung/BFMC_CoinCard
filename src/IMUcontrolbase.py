#!/usr/bin/env python3

# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE


import rospy
import cv2
import time
import json
import numpy as np
from sensor_msgs.msg import Image
from utils.msg import localisation
from utils.msg import IMU
from cv_bridge import CvBridge, CvBridgeError
import heapq
from graph_data import graph_data
from neighbor_data import neighbor_data
from std_msgs.msg import String
from math import *
import os
import time 
from controlling_lib import *
from graph_data import graph_data
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from rotation import *

SIGMA = 6
def create_Gaussian_array(n,mean):
    output = np.arange(0,n).astype(np.float32)
    output = 1.0/np.sqrt(2*pi*SIGMA**2)*np.exp(-1/2*(output-mean)**2/SIGMA**2)
    mean = np.max(output)
    return output/mean

def searchNearyByCheckpoint(totalPoint,graphPoints,pos):
    pos = np.array((pos[0],pos[1]))
    EuclDistance = np.mean(np.square(graphPoints - pos),axis=1)
    min_idex = np.argmin(EuclDistance)
    return totalPoint[min_idex]

def interpolating_path(path,g_data):
    data = []
    for point in path:
        point_data = g_data[str(point)]
        data.append([point_data['x'],point_data['y']])
    n = len(data)
    data = np.array(data)
    # print(data.shape)
    # x = data[:,0]
    # y = data[:,1]
    # X = []
    # Y = []
    points_inter = data[0,:].reshape(1,2)
    for i in range(0,n-4,5):
        points = data[i:i+5]
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]

        # Build a list of the spline function, one for each dimension:
        splines = [UnivariateSpline(distance, coords, k=2, s=.7) for coords in points.T]

        # Computed the spline for the asked distances:
        alpha = np.linspace(0, 1, 5)
        points_fitted = np.vstack( spl(alpha) for spl in splines ).T
        points_inter = np.vstack((points_inter,points_fitted))

    # X = np.array(X)
    # Y = np.array(Y)
    return points_inter[1:]

########## UTILS FUNCTION FOR KARMAN ##################################################
#### Measurement Update #####################################################################

################################################################################################
# Since we'll need a measurement update for both the GPS data, let's make
# a function for it.
################################################################################################
def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):

    # construct H_k = [I, 0, 0] (size = 3 x 9)
    H_k = np.zeros([3, 9])
    H_k[0:3, 0:3] = np.identity(3)

    # 3.1 Compute Kalman Gain
    # evaluate size chain: (9 x 9) x (9 x 3) x ( (3 x 9) x (9 x 9) x (9 x 3) + (3 x 3) )
    # K_k should have a size: (9 x 3)
    K_k = p_cov_check @ H_k.T @ inv(H_k @ p_cov_check @ H_k.T + sensor_var)

    # 3.2 Compute error state
    # evaluate size chain: (9 x 3) x ( (3 x 1) - (3 x 1) )
    # delta_x_k should have a size: (9 x 1)

    delta_x_k = K_k @ (y_k - p_check)

    # 3.3 Correct predicted state
    p_hat = p_check + delta_x_k[0:3]
    v_hat = v_check + delta_x_k[3:6]
    ## use of self built function:
    # q_hat = quaternion_left_prod( delta_x_k[6:9] ) @ q_check
    # q_hat = normalize_quaternion(q_hat)
    ## use of pre-built functions:
    # print(' delta_x_k[6:9].shape:', delta_x_k.shape)
    q_obj = Quaternion( euler = delta_x_k[6:9] ).quat_mult_left(q_check)
    q_hat = Quaternion(*q_obj).normalize().to_numpy()
    # q_hat = Quaternion(*q_obj).to_numpy() # Note: after test, it tuns out we don't have to normalize the quaternion

    # 3.4 Compute corrected covariance
    # evaluate size chain: ( (9 x 9) - (9 x 3) x (3 x 9) ) x (9 x 9)
    p_cov_hat = ( np.identity(9) - K_k @ H_k ) @ p_cov_check
    # p_hat = p_hat.reshape(1,3)
    return p_hat, v_hat, q_hat, p_cov_hat


    
class MyCarHandler():
    # ===================================== INIT==========================================
    def __init__(self,start_point = 473, goal_point = 125):
        """
        Creates a bridge for converting the image from Gazebo image intro OpenCv image
        """
        # Constants
        ################################################################################################
        # Now that our data is set up, we can start getting things ready for our solver. One of the
        # most important aspects of a filter is setting the estimated sensor variances correctly.
        # We set the values here.
        ################################################################################################
        self.VAR_IMU_F = 0.10
        self.VAR_IMU_W = 0.25
        self.VAR_GPS = 20
        self.g = np.array([0, 0, -9.81])  # gravity
        self.l_jac = np.zeros([9, 6])
        self.l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
        self.h_jac = np.zeros([3, 9])
        self.h_jac[:, :3] = np.eye(3)  # measurement model jacobian


        ################################################################################################
        # PID control velocity 
        self.ROIs = create_Gaussian_array(12,7) 
        self.count = 0
        self.pids_kp = 0.8000
        self.pids_ki = 0.5000
        self.pids_kd = 0.000
        self.pids_tf = 0.040000
        self.speed = 20

        # self.PID = My_pid(self.pids_kp,self.pids_ki,self.pids_kd,self.pids_tf)
        self.PID = My_pid(self.pids_kp,0,0,self.pids_tf)

        ################################################################################################
        #Flag
        self.flag = 1
        self.GPS_update_flag = 0
        self.savePath = []

        ################################################################################################
        # For localisation
        totalPoint = graph_data.keys()
        points = []
        self.keys = []
        for i in totalPoint:
            self.keys.append(int(i))
            point_i = [graph_data[i]['x'], graph_data[i]['y']]
            points.append(point_i)
        self.GraphPoints = np.array(points)

        ################################################################################################
        # Map searching
        self.start_node_id = str(start_point)
        self.goal_node_id = str(goal_point)
        x_start = graph_data[self.start_node_id]['x']
        y_start = graph_data[self.start_node_id]['y']
        graph = {node_id: Node(node_id, data['x'], data['y'], neighbor_data['neighbors']) for (node_id, data),(node_id,neighbor_data) in zip(graph_data.items(),neighbor_data.items())}
        self.path = astar(graph, self.start_node_id, self.goal_node_id)

        self.i_points = interpolating_path(self.path,graph_data) #smoothing the path


        ################################################################################################
        # Ros command and call back function
        rospy.init_node('IMUnod', anonymous=True)
        self.publisher = rospy.Publisher('/automobile/command', String, queue_size=1)
        self.localisation = rospy.Subscriber("/automobile/localisation", localisation, self.callback_GPS)
        self.IMU = rospy.Subscriber("/automobile/IMU",IMU,self.callback_IMU)
        self.rate = rospy.Rate(100) #10Hz

        ################################################################################################
        # Initial value for IMU, 
        self.GPS_data = np.array([x_start,    y_start,     0]) #(3x1)
        self.IMU_data = np.array([0, 0, 0, 0, 0, 0]).reshape((6,1))

        # Intial value for our ES-EKF solver
        self.p_est = np.zeros(3)  # position estimates
        self.v_est = np.zeros(3)  # velocity estimates
        self.q_est = np.zeros(4)  # orientation estimates as quaternions
        self.p_cov = np.zeros([9, 9])  # covariance matrices at each timestep
        self.imu_f = np.zeros(3)

        # Set initial values.
        self.p_est = self.GPS_data
        self.v_est[0] = self.speed/100
        intial_angle = np.array([0,0,0])
        self.q_est = Quaternion( euler = intial_angle).to_numpy()
        self.C_ns_0 = Quaternion(*intial_angle).to_mat()
        self.p_cov[0] = np.zeros(9)  # covariance of estimate
        self.gps_i  = 0

        # Define some suppotive variables
        self.R_GPS      =  np.identity(3) *   self.VAR_GPS  # covariance matrix related to GNSS    
        self.t_imu      = 0.05                    # timestanps of imu
        self.t_gps      =  0.25                    # timestamps of gnss
        self.F_k         =  np.identity(9)         # 9x9 matrix
        self.L_k         =  np.zeros([9, 6])
        self.L_k[3:9, :] =  np.identity(6)
        self.Q           =  np.identity(6)               # covariance matrix related to noise of IMU
        self.Q[0:3, 0:3] =  self.Q[0:3, 0:3] * self.VAR_IMU_F      # covariance matrix related to special force of IMU
        self.Q[3:6, 3:6] =  self.Q[3:6, 3:6] * self.VAR_IMU_W      # covariance matrix related to rotational speed of IMU

        # ===== READY TO MOVE =========
        self.t_imu = time.time()
        self.t_gps = time.time()

    def moveP2P(self,pos,rot,p1,p2,velocity):
        """
        move from p1 to p2 with velocity
        this function input 3 points, current position, p1 and p2 and the velocity
        output is the steering angle
        """
        """
        pos,rot: current position and orientation measured by the systems of sensor ( this is the output of kalman filter)
        xo: current x
        yo: current y

        """
        xo = pos[0]
        yo = pos[1]
        ### the idea is finding the nearest point from the current position to p1 and p2, if the nearer is p1 => keep going, until the nearer is p2 => update the flag to update the new path.
        GraphPoints = np.array([[p1[0],p1[1]],
                                [p2[0],p2[1]]])
        keys = [1,2]
        nearest = searchNearyByCheckpoint(keys,GraphPoints,[xo,yo])
        if nearest ==2:
            print('set Flag to 1, update new value of p1 and p2')
            self.flag = 1
            return 0
        else:
            # get the present path, the path that the car need to follow at a moment have the equation described as ax + by + c =0.
            ly = GraphPoints[:,1]
            lx = GraphPoints[:,0]
            a,c = np.polyfit(lx, ly, 1) # ax-y+c = 0
            k_e = 0.05

            # =======lateral control=======
            #heading error
            yaw_path = atan2(p2[1]-p1[1],p2[0]-p1[0])
            heading_error = yaw_path + rot
            if heading_error > np.pi:
                heading_error -= 2*np.pi
            if heading_error < -np.pi: 
                heading_error += 2*np.pi

            #cross track error
            e = np.abs((a*xo - yo + c))/np.sqrt(a**2 +1)
            yaw_cross_track = atan2(yo-p1[1],xo-p1[0])
            yaw_path2ct = yaw_path - yaw_cross_track
            cr_track_steering = atan2(k_e*e,velocity)
            if yaw_path2ct  > 0:
                cr_track_steering = abs(cr_track_steering)
            if yaw_path2ct  < 0:
                cr_track_steering = -abs(cr_track_steering)


            return cr_track_steering + heading_error

    def run(self):
        while not rospy.is_shutdown():
            start = time.time()
            x = self.p_est[0]
            y = self.p_est[1]
            # print('self.q_est.shape:',self.q_est.shape)
            qc = Quaternion(*self.q_est)
            rot = self.GPS_data[2]
            print ('GPS data:',self.GPS_data)
            print ('IMU data:',self.imu_f)
            print ('fusion data:',self.p_est[:-1],qc.to_euler()[2],'\n')
            if self.flag == 1:
                m,n = self.i_points.shape
                if m >= 2:
                    self.start_point = self.i_points[0,:]
                    self.end_point = self.i_points[1,:]
                    print('startPoint:',self.path[0],'\t endPoint:',self.path[1])
                    # self.start_point = np.array([self.start_point['x'],self.start_point['y']])
                    # self.end_point = np.array([self.end_point['x'],self.end_point['y']])
                    print('startPoint:',self.start_point,'\t endPoint:',self.end_point)
                    self.i_points = self.i_points[1:,:]
                else:
                    self.speed =0
                self.flag =0


            angle = self.moveP2P([x,y],rot,self.start_point,self.end_point,self.speed/15)
            angle = 180*angle/pi
            angle = min(30,angle)
            angle = max(-30,angle)
            command_speed = '{"action":"1","speed":%f}'%(self.speed/100)
            command_stear = '{"action":"2","steerAngle":%.1f}'%angle
            # command = command_speed+','+command_stear


            self.publisher.publish(command_speed)
            self.publisher.publish(command_stear)

            self.rate.sleep()#10Hz
            # print('lps:',time.time()-start)



    def callback_GPS(self,data):
        self.GPS_data = np.array([data.posA, data.posB, data.rotA])
        self.GPS_update_flag = 1
        print('GPS call back')


    def callback_IMU(self,data):
        print('IMU call back')
        self.IMU_data = np.array([data.roll, data.pitch, data.yaw, data.accelx, data.accely, data.accelz])
        self.imu_f = np.array([data.accelx, data.accely, data.accelz])
        self.imu_w = np.array([data.roll, data.pitch, data.yaw])
        self.Fusion_IMU_and_GPS()


    def Fusion_IMU_and_GPS(self):
        #some preparations
        delta_t = time.time() - self.t_imu
        self.t_imu = time.time()
        Q_k = self.Q * delta_t * delta_t
        C_ns = Quaternion(*self.q_est).to_mat()
       

        #==== 1. Update state with IMU inputs====

        ## 1-1: update position states
        self.p_est = self.p_est + delta_t * self.v_est + delta_t**2/2 * (C_ns @ self.imu_f + self.g)
        ## 1-2: update velocity states
        self.v_est = self.v_est + delta_t * (C_ns @ self.imu_f + self.g)
        ## 1-3: update orientation states
        q_tmp = Quaternion( euler = (self.imu_w * delta_t) ).quat_mult_right( self.q_est )
        self.q_est = Quaternion(*q_tmp).normalize().to_numpy()
        # print('self.q_est.shape',self.q_est.shape)

        #==== 2.  Propagate uncertainty ====
        ## 2-1: Linearize the motion model and compute Jacobians
        self.F_k[0:3, 3:6] = np.identity(3) * delta_t
        self.F_k[3:6, 6:9] = - skew_operator( C_ns @ self.imu_f ) * delta_t
        ## 2-2 execute the propagate uncertainty process
        self.p_cov = self.F_k @ self.p_cov @ self.F_k.T + self.L_k @ Q_k @ self.L_k.T

        #==== 3. GPS measurements Fusion ====
        if self.GPS_update_flag == 1:
            self.GPS_update_flag = 0
            [self.p_est, self.v_est, self.q_est, self.p_cov] = measurement_update(self.R_GPS, self.p_cov, self.GPS_data, self.p_est[0], self.v_est, self.q_est)



if __name__ == '__main__':
    try:
        nod = MyCarHandler(goal_point=125)
        nod.run()
    except rospy.ROSInterruptException:
        pass
