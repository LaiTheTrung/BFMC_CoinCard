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
from neighbor_data import neighbor_data

SIGMA = 6
def create_Gaussian_array(n,mean):
    output = np.arange(0,n).astype(np.float32)
    output = 1.0/np.sqrt(2*pi*SIGMA**2)*np.exp(-1/2*(output-mean)**2/SIGMA**2)
    mean = np.max(output)
    return output/mean



def CameraProcessor(frame):
    frame = cv2.resize(frame, (640,480))
    ## Choosing points for perspective transformation
    tl = (184,250)
    bl = (15 ,450)
    tr = (458,250)
    br = (640,450)

    ## Aplying perspective transformation
    pts1 = np.float32([tl, bl, tr, br]) 
    pts2 = np.float32([[0, 0], [0, 480], [480, 0], [480, 480]]) 

    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    transformed_frame = cv2.warpPerspective(frame, matrix, (480,480))
    cv2.circle(frame, tl, 5, (0,0,255), -1)
    cv2.circle(frame, bl, 5, (0,0,255), -1)
    cv2.circle(frame, tr, 5, (0,0,255), -1)
    cv2.circle(frame, br, 5, (0,0,255), -1)
    ### Object Detection
    # Image Thresholding
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)
    return frame,transformed_frame, mask

class MyCarHandler():
    # ===================================== INIT==========================================
    def __init__(self,LaneDetector):
        """
        Creates a bridge for converting the image from Gazebo image intro OpenCv image
        """
        self.ROIs = create_Gaussian_array(12,7) 
        self.count = 0
        self.pids_kp = 0.8000
        self.pids_ki = 0.5000
        self.pids_kd = 0.000
        self.pids_tf = 0.040000
        # self.PID = My_pid(self.pids_kp,self.pids_ki,self.pids_kd,self.pids_tf)
        self.PID = My_pid(self.pids_kp,0,0,self.pids_tf)
        self.Lane = LaneDetector
        self.bridge = CvBridge()
        self.cv_image = np.zeros((640, 480))
        rospy.init_node('CAMnod', anonymous=True)
        self.publisher = rospy.Publisher('/automobile/command', String, queue_size=1)
        self.image_sub = rospy.Subscriber("/automobile/image_raw", Image, self.callback_Camera)
        self.localisation = rospy.Subscriber("/automobile/localisation", String, self.callback_IMU)
        rospy.spin()

    def callback_IMU(self,data):
        print('IMU:',data.data)

    def callback_Camera(self, data):
        """
        :param data: sensor_msg array containing the image in the Gazsbo format
        :return: nothing but sets [cv_image] to the usefull image that can be use in opencv (numpy array)
        """
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        img,transformed_frame,mask = CameraProcessor(self.cv_image)
        angle,msk = self.Lane.slidingWindowsDec(mask)
        angle = self.PID.get(angle)

        ori_point = (int(320+240*tan(angle*pi/180)),240)
        cv2.line(img,(320,640),ori_point,color = (0, 255, 0),thickness = 2)
        cv2.imshow("Original", img )
        cv2.imshow("Bird's Eye View", transformed_frame)
        cv2.imshow("Lane Detection - Sliding Windows", msk)
        # print('angle:',angle)
        speed = 10
        command_speed = '{"action":"1","speed":%f}'%(speed/100)
        command_stear = '{"action":"2","steerAngle":%.1f}'%angle
        # command = command_speed+','+command_stear
        # self.publisher.publish(command_speed)
        # self.publisher.publish(command_stear)


        # self.count +=1
        # if self.count >= 12:
        #     name = str(time.time())+'.jpg'
        #     path = '/home/thetrung/Documents/Simulator/image_raw_data'
        #     cv2.imwrite(os.path.join(path,name),self.cv_image)
        #     self.count = 0

        key = cv2.waitKey(1)


if __name__ == '__main__':
    try:
        laneDetector = LaneDetection()
        cv2.namedWindow("Trackbars")
        cv2.createTrackbar("L - H", "Trackbars", 0, 255, CameraProcessor)
        cv2.createTrackbar("L - S", "Trackbars", 0, 255, CameraProcessor)
        cv2.createTrackbar("L - V", "Trackbars", 200, 255, CameraProcessor)
        cv2.createTrackbar("U - H", "Trackbars", 255, 255, CameraProcessor)
        cv2.createTrackbar("U - S", "Trackbars", 50, 255, CameraProcessor)
        cv2.createTrackbar("U - V", "Trackbars", 255, 255,CameraProcessor)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        nod = MyCarHandler(laneDetector)
    except rospy.ROSInterruptException:
        pass
