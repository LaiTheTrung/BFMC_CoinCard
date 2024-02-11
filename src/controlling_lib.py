
import numpy as np
import cv2
from math import *
import heapq
from graph_data import graph_data
from neighbor_data import neighbor_data

class LaneDetection:

    def __init__(self) -> None:
        self.MIN_margin = 10
        self.left_fit = np.array([0,0,0])
        self.right_fit = np.array([0,0,0])
        self.l_h = 0
        self.l_s = 0
        self.l_v = 200
        self.u_h = 255
        self.u_s = 50
        self.u_v = 255
        self.SECTION_LINE = 120 # from bottom
    
    def detCenter(self,section_line):
        y = section_line
        lx = self.left_fit[0]*y**2 + self.left_fit[1]*y + self.left_fit[2]
        rx = self.right_fit[0]*y**2 + self.right_fit[1]*y + self.right_fit[2]
        return (lx+rx)/2
    
    def intersectionDetection(self):
        
        pass
    def turnPriority(self,priority):
        L,R,Str,Stop = priority
        

    def singleWindowWorker(self,y,mask,right_base,left_base):
        left_x = None
        right_x = None
        img = mask[y-self.margin:y,max(0,left_base-50):min(480,left_base+50)]
        left_base = int(self.left_fit[0]*y**2 + self.left_fit[1]*y + self.left_fit[2])
        right_base = int(self.right_fit[0]*y**2 + self.right_fit[1]*y + self.right_fit[2])
        # print(left_base, right_base)
        m,n = img.shape
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                left_x = left_base - (n-50) + cx
                left_base = left_base/2 + left_x/2
        
        ## Right threshold
        img = mask[y-self.margin:y, right_base-50:right_base+50]
        m,n = img.shape
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                right_x = (right_base-50 + cx)
                right_base = right_base/2 + right_x/2
        return left_x,int(left_base),right_x ,int(right_base)

    # def ControllerRules(self):


    def slidingWindowsDec(self,mask):
        
        #Histogram
        histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
        midpoint = int(histogram.shape[0]/2)
        left_base = np.argmax(histogram[:midpoint-50])
        right_base = np.argmax(histogram[midpoint+50:]) + midpoint+50
        self.left_fit = np.array([0,0,left_base])
        self.right_fit = np.array([0,0,right_base])
        #Sliding Window
        y = 472
        ly =[]
        ry =[]
        lx = []
        rx = []
        #use for intersection detection
        steplx = [] # this just contain maximize 3 value
        steply = []
        steprx = []
        stepry = []

        #Flag
        count = 0
        msk = mask.copy()
        self.margin = self.MIN_margin
        while y>0:
            left_x,left_base,right_x,right_base = self.singleWindowWorker(y,mask,right_base,left_base)
            if count <= 3:
                if left_x is not None:
                    ly.append(y)
                    lx.append(left_x)
                if right_x is not None:
                    ry.append(y)
                    rx.append(right_x)
                cv2.rectangle(msk, (left_base-50,y), (left_base+50,y-self.margin), (255,255,255), 2)
                cv2.rectangle(msk, (right_base-50,y), (right_base+50,y-self.margin), (255,255,255), 2)
                if count == 3:
                    if len(lx)>=2:
                        firstLine = np.polyfit(ly, lx, 1)
                        self.left_fit = np.array([0,firstLine[0],firstLine[1]])
                    if len(rx)>=2:
                        firstLine =np.polyfit(ry, rx, 1)
                        self.right_fit = np.array([0,firstLine[0],firstLine[1]])

            else:
                if left_x is not None:
                    steply.append(y)
                    steplx.append(left_x)
                if right_x is not None:
                    stepry.append(y)
                    steprx.append(right_x)
                cv2.rectangle(msk, (left_base-50,y), (left_base+50,y-self.margin), (255,255,255), 2)
                cv2.rectangle(msk, (right_base-50,y), (right_base+50,y-self.margin), (255,255,255), 2)
                #update value for lx, rx
                if len(steplx) == 3:
                    Lline_poly = np.polyfit(steply,steplx,1) 
                    angle_const = Lline_poly[0]
                    polly_angle = self.left_fit[0]*steply[0]*2 + self.left_fit[1] # take derivative
                    if abs(angle_const-polly_angle) < 0.2:
                        lx +=steplx[1:]
                        ly +=steply[1:]
                    steplx = [steplx[-1]]
                    steply = [steply[-1]]
                
                if len(steprx) == 3:
                    Rline_poly = np.polyfit(stepry,steprx,1) 
                    angle_const = Rline_poly[0]
                    polly_angle = self.right_fit[0]*stepry[0]*2 + self.right_fit[1] # take derivative
                    if abs(angle_const-polly_angle) < 0.2:
                        rx +=steprx[1:]
                        ry +=stepry[1:]
                    steprx = [steprx[-1]]
                    stepry = [stepry[-1]]

                if count % 3 ==0:
                    if len(lx)>=3:
                        self.left_fit = np.polyfit(ly, lx, 2)
                    if len(rx)>=3:
                        self.right_fit = np.polyfit(ry, rx, 2)


            self.margin += 2
            y -= self.margin
            count +=1

        direction= self.detCenter(section_line=480-self.SECTION_LINE)
        angle = atan2((direction-240),self.SECTION_LINE)*180/pi
        
        return angle,msk


class IMU_pathPlanning():
    def __init__(self, graph, addNoise = False):
        self.graph = graph
        self.addNoise = True
        
    

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
    
class My_LateralControl:
    def __init__(self,k,deltaT) -> None:
        self.k = k #0.1
        self.deltaT = deltaT
    
    def get(self,cross_track_error,velocity,angle_error):
        cr_track_steering = atan2(self.k*cross_track_error/velocity)
        return angle_error + cr_track_steering
    
class Node:
    def __init__(self, node_id, x, y, neighbors):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.neighbors = neighbors
        self.g = float('inf')  # Initial g value set to infinity
        self.h = 0  # Heuristic value (to be implemented based on your requirements)
        self.parent = None

    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

def heuristic(node, goal):
    # Example: Euclidean distance as heuristic
    return ((node.x - goal.x)**2 + (node.y - goal.y)**2)/2

def astar(graph, start_id, goal_id):
    start_node = graph[start_id]
    goal_node = graph[goal_id]

    open_set = [start_node]
    closed_set = set()
    while open_set:
        current_node = heapq.heappop(open_set)
        # print(current_node.node_id)
        if current_node.node_id == goal_id:
            # Destination reached, reconstruct the path
            path = []
            while current_node:
                path.append(current_node.node_id)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(current_node.node_id)
        # print(current_node.node_id)
        # print(current_node.neighbors)
        for neighbor_id in graph[current_node.node_id].neighbors:
            # print(neighbor_id)
            neighbor_node = graph[neighbor_id]
            # print(neighbor_node.node_id)
            if neighbor_node.node_id in closed_set:
                continue

            tentative_g = current_node.g + 1  # Assuming unit cost for each step

            if tentative_g < neighbor_node.g or neighbor_node not in open_set:
                neighbor_node.g = tentative_g
                neighbor_node.h = heuristic(neighbor_node, goal_node)
                neighbor_node.parent = current_node

                if neighbor_node not in open_set:
                    heapq.heappush(open_set, neighbor_node)