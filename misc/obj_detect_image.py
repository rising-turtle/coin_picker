#!/usr/bin/env python2
import rospy 
import cv2 
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image 
from std_msgs.msg import String
import numpy as np

class ObjDetectImg(object):
    def __init__(self):
        rospy.init_node('obj_detect_node')

        rospy.loginfo("Hello I'm object detection node") 

        sub_img = rospy.Subscriber('/cam/color', Image, self.image_obj_bbs)
        self.pub_coords = rospy.Publisher('/cam/keypoints', Image, queue_size=3)
        
        self.cv_bridge = CvBridge()
        self.len_thick = 2

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200
        params.thresholdStep = 10


        # Filter by Area.
        params.filterByArea = True
        params.minArea = 1500
        params.maxArea = 100 * 100

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.05

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.65

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.01

        # Create a detector with the parameters
        # OLD: detector = cv2.SimpleBlobDetector(params)
        self.detector = cv2.SimpleBlobDetector_create(params)
        self.hue_stages = 6
        self.colr_stages = 16
        self.d = 256 / self.colr_stages
        self.frame_persistence = 10
        self.keypoints = []
        self.lives = []
        self.cvt_factor = 20.0 / 30
        self.scale_factor = 4

        rospy.spin()
    
    def image_obj_bbs(self, msg):

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        except CvBridgeError:
            rospy.logerr("CvBridge Error ")
        # print(cv_image.shape)
        cv_image = cv_image[0:-1,114:-114]
        cross_len = 30 # here is pixel need to find a good value after fix the camera 
        rec_len = cross_len * 5 
        # cv_image = self.threshold_img(cv_image)
        cv_image = cv_image // self.d * self.d + self.d / 4
        keypoints = self.get_keypoints(cv_image)
        doubles = set()
        self.lives = [l - 1 for l in self.lives]
        for i, k in enumerate(keypoints):
            for j, p in enumerate(self.keypoints):
                if cv2.KeyPoint_overlap(k, p) > 0:
                    doubles.add(k)
                    self.lives[j] = self.frame_persistence
        for k in doubles:
            keypoints.remove(k)
        for k in keypoints:
            self.keypoints.append(k)
            self.lives.append(self.frame_persistence)
        for i, l in list(enumerate(self.lives))[::-1]:
            if l <= 0:
                self.lives.pop(i)
                self.keypoints.pop(i)

        rob_cord = self.keypoints_2_robot(cv_image, self.keypoints)
        cv_image = self.draw_cross(cv_image, cross_len)
        cv_image = self.draw_rectangle(cv_image, rec_len)
        cv_image = self.draw_keypoints(cv_image, self.keypoints)
            
        msg = []
        for i, c in enumerate(rob_cord):
            x,y = self.keypoints[i].pt
            h,s,v = c[1]
            c = c[0]
            # print(c,x,y)
            cv_image = cv2.putText(cv_image, "%d - (%d, %d, %d): (%d, %d)" % (c[2], h,s,v, c[0],c[1]), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255))
            msg.extend(c)
        self.pub_coords.publish(data=msg, width=len(msg), height=self.scale_factor)
        cv2.imshow("image_show", cv_image)
        cv2.waitKey(3)

    def threshold_img(self, img):
        img2 = np.array(img)
        img2[...,0] = (img[...,0] // self.colr_stages) * self.colr_stages
        img2[...,1] = (img[...,1] // self.colr_stages) * self.colr_stages
        img2[...,2] = (img[...,2] // self.colr_stages) * self.colr_stages
        print(img2[50,50], img[50,50])
        return img2

    def get_keypoints(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # hsv[...,1] = hsv[...,1] * 1.2
        # hsv[...,2] = hsv[...,2] * 0.8
        keypoints = []
        for i in range(0, 180, 180 // self.hue_stages):
            hsv2 = np.array(hsv)
            hsv2[...,0] = (hsv[...,0] + i) % 180
            img2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
            keypoints = keypoints + self.detector.detect(img2)
        to_remove = []
        for i, kp1 in enumerate(keypoints):
            for j, kp2 in enumerate(keypoints[i+1:]):
                if cv2.KeyPoint_overlap(kp1, kp2):
                    new_pt = tuple( (np.array(kp1.pt) + np.array(kp1.pt)) / 2)
                    new_size = max(kp1.size, kp2.size) + np.linalg.norm((np.array(kp1.pt) - np.array(kp1.pt)) / 2)
                    keypoints[j].pt = new_pt
                    keypoints[j].size = new_size
                    to_remove.append(i)
                    break
        for i in to_remove[::-1]:
            keypoints = keypoints[:i] + keypoints[i+1:]
        return keypoints


    
    def draw_cross(self, img, len):
        # get dimensions of image
        dimensions = img.shape

        # height, width, number of channels in image
        height = img.shape[0]
        width = img.shape[1]
        

        cx = width / 2
        cy = height / 2
        half_len = len/2  
        cv2.line(img, (cx, cy - half_len), (cx, cy + half_len), (0, 0, 0), self.len_thick)
        cv2.line(img, (cx - half_len, cy), (cx + half_len, cy), (0, 0, 0), self.len_thick)
        
        return img
    
    def draw_rectangle(self, img, len): 
              # get dimensions of image
        dimensions = img.shape

        # height, width, number of channels in image
        height = img.shape[0]
        width = img.shape[1]

        cx = width / 2
        cy = height / 2
        half_len = len/2  
        # print(width, height)
        # cv2.putText(img, str(width) + " x " + str(height), (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0))
        cv2.line(img, (cx - half_len, cy - half_len), (cx - half_len, cy + half_len), (0, 0, 0), self.len_thick)
        cv2.line(img, (cx - half_len, cy - half_len), (cx + half_len, cy - half_len), (0, 0, 0), self.len_thick)
        cv2.line(img, (cx - half_len, cy + half_len), (cx + half_len, cy + half_len), (0, 0, 0), self.len_thick)
        cv2.line(img, (cx + half_len, cy - half_len), (cx + half_len, cy + half_len), (0, 0, 0), self.len_thick)
        return img 

    def draw_bb(self, img, x1, y1, x2, y2): 
        cv2.line(img, (x1, y1), (x1, y2), (0, 0, 0), self.len_thick)
        cv2.line(img, (x1, y1), (x2, y1), (0, 0, 0), self.len_thick)
        cv2.line(img, (x1, y2), (x2, y2), (0, 0, 0), self.len_thick)
        cv2.line(img, (x2, y1), (x2, y2), (0, 0, 0), self.len_thick)
        return img 
    
    def draw_keypoints(self, img, keypoints):
        return cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def keypoints_2_robot(self, img, keypoints):
        out = []
        height = img.shape[0]
        width = img.shape[1]
        i = 0
        for kp in keypoints:
            pt = kp.pt
            y = -self.cvt_factor * (kp.pt[0] - (width / 2)) + 200
            x = 210 - self.cvt_factor * (kp.pt[1] - (height / 2))
            x /=  self.scale_factor
            y /= self.scale_factor
            h,s,v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[int(pt[1]), int(pt[0])]
            color = 1 if (h < 20 or h > 160) and (s > 160) and (v > 140) else 2 if (h > 45 and h < 75) and (s > 80) and (v < 256) else 3 if (h > 60 and h < 140) and (s > 100) and (v < 256) else 0
            print(i, (int(x), int(y), color))
            i += 1
            out.append(((int(x), int(y), color), (h,s,v)))
        return out

if __name__=='__main__':
    try:
        ObjDetectImg()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start node')
   
        