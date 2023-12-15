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
from std_msgs.msg import String
from math import *
import os
import time 
SIGMA = 6


def create_Gaussian_array(n,mean):
    output = np.arange(0,n).astype(np.float)
    output = 1.0/np.sqrt(2*pi*SIGMA**2)*np.exp(-1/2*(output-mean)**2/SIGMA**2)
    mean = np.max(output)
    return output/mean
def imgProc(image,distribution):

    frame = cv2.resize(image, (640,480))

    ## Choosing points for perspective transformation
    tl = (250,151)
    bl = (0 ,317)
    tr = (380,151)
    br = (534,317)

    cv2.circle(frame, tl, 5, (0,0,255), -1)
    cv2.circle(frame, bl, 5, (0,0,255), -1)
    cv2.circle(frame, tr, 5, (0,0,255), -1)
    cv2.circle(frame, br, 5, (0,0,255), -1)

    ## Aplying perspective transformation
    pts1 = np.float32([tl, bl, tr, br]) 
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]]) 

    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))

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

    #Histogram
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint


    #Sliding Window
    y = 472
    lx = []
    rx = []
    count = 0
    msk = mask.copy()
    center_x = []
    while y>0:
        ## Left threshold
        left_x = 0
        right_x = 0
        img = mask[y-40:y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                left_x = (left_base-50 + cx)
                left_base = left_base-50 + cx
                count +=1
        
        ## Right threshold
        img = mask[y-40:y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                right_x = (right_base-50 + cx)
                right_base = right_base-50 + cx

        cv2.rectangle(msk, (left_base-50,y), (left_base+50,y-40), (255,255,255), 2)
        cv2.rectangle(msk, (right_base-50,y), (right_base+50,y-40), (255,255,255), 2)
        center_x.append(1/2*(left_x+right_x))
        y -= 40
    center_x= np.array(center_x)
    if count > 0:
        direction = np.sum(np.multiply(center_x,distribution))/count
    else:
        direction = 320
    angle = atan2((direction-320),320)*180/pi
    print('angle:',angle)
    return angle,frame,transformed_frame,msk
class My_pid:
    def __init__(self,kP,kI,kD,deltaT) -> None:
        self.KP = kP
        self.KI = kI
        self.KD = kD
        self.I = 0
        self.error = 0
        self.deltaT = deltaT
    def get(self,error):
        P = self.KP*error
        self.I = self.I*0.9 + self.KI*error*self.deltaT
        D = self.KD*(error-self.error)/self.deltaT
        self.error = error
        return  P + self.I + D


class CameraHandler():
    # ===================================== INIT==========================================
    def __init__(self):
        """
        Creates a bridge for converting the image from Gazebo image intro OpenCv image
        """
        self.ROIs = create_Gaussian_array(12,7)
        self.count = 0
        self.pids_kp = 0.55000
        self.pids_ki = 0.810000
        self.pids_kd = 0.000222
        self.pids_tf = 0.040000
        # self.PID = My_pid(self.pids_kp,self.pids_ki,self.pids_kd,self.pids_tf)
        self.PID = My_pid(self.pids_kp,0,0,self.pids_tf)

        self.bridge = CvBridge()
        self.cv_image = np.zeros((640, 480))
        rospy.init_node('CAMnod', anonymous=True)
        self.publisher = rospy.Publisher('/automobile/command', String, queue_size=1)
        self.image_sub = rospy.Subscriber("/automobile/image_raw", Image, self.callback)
        rospy.spin()

    def callback(self, data):
        """
        :param data: sensor_msg array containing the image in the Gazsbo format
        :return: nothing but sets [cv_image] to the usefull image that can be use in opencv (numpy array)
        """
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        angle,img,transformed_frame,msk = imgProc(self.cv_image,self.ROIs)
        # angle = self.PID.get(angle)
        cv2.imshow("Original", img )
        cv2.imshow("Bird's Eye View", transformed_frame)
        cv2.imshow("Lane Detection - Sliding Windows", msk)
        speed = 20
        command_speed = '{"action":"1","speed":%f}'%(speed/100)
        command_stear = '{"action":"2","steerAngle":%.1f}'%angle
        # command = command_speed+','+command_stear
        self.publisher.publish(command_speed)
        self.publisher.publish(command_stear)


        # self.count +=1
        # if self.count >= 12:
        #     name = str(time.time())+'.jpg'
        #     path = '/home/thetrung/Documents/Simulator/image_raw_data'
        #     cv2.imwrite(os.path.join(path,name),self.cv_image)
        #     self.count = 0

        key = cv2.waitKey(1)

            
if __name__ == '__main__':
    try:
        cv2.namedWindow("Trackbars")
        cv2.createTrackbar("L - H", "Trackbars", 0, 255, imgProc)
        cv2.createTrackbar("L - S", "Trackbars", 0, 255, imgProc)
        cv2.createTrackbar("L - V", "Trackbars", 200, 255, imgProc)
        cv2.createTrackbar("U - H", "Trackbars", 255, 255, imgProc)
        cv2.createTrackbar("U - S", "Trackbars", 50, 255, imgProc)
        cv2.createTrackbar("U - V", "Trackbars", 255, 255, imgProc)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        nod = CameraHandler()
    except rospy.ROSInterruptException:
        pass
