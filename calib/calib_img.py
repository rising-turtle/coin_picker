#!/usr/bin/env python2
import rospy 
import cv2 
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image 

class CalbImg(object):
    def __init__(self):
        rospy.init_node('calib_node')

        rospy.loginfo("Hello I'm calibration node") 

        sub_img = rospy.Subscriber('/cam/color', Image, self.image_cb)
        
        self.cv_bridge = CvBridge()
        self.len_thick = 3

        rospy.spin()
    
    def image_cb(self, msg):

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        except CvBridgeError:
            rospy.logerr("CvBridge Error ")
        
        # cross len 20 cm 
        # rectangle len 100 cm 
        
        cross_len = 52 # here is pixel need to find a good value after fix the camera 
        # ratio = 20(cm)/cross_len(pixel)
        rec_len = cross_len * 5 
        cv_image = self.draw_cross(cv_image, cross_len)
        cv_image = self.draw_rectangle(cv_image, rec_len)
        
        cv2.imshow("image_show", cv_image)
        cv2.waitKey(3)
    
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
        cv2.line(img, (cx - half_len, cy - half_len), (cx - half_len, cy + half_len), (0, 0, 0), self.len_thick)
        cv2.line(img, (cx - half_len, cy - half_len), (cx + half_len, cy - half_len), (0, 0, 0), self.len_thick)
        cv2.line(img, (cx - half_len, cy + half_len), (cx + half_len, cy + half_len), (0, 0, 0), self.len_thick)
        cv2.line(img, (cx + half_len, cy - half_len), (cx + half_len, cy + half_len), (0, 0, 0), self.len_thick)
        return img 
        
        

if __name__=='__main__':
    try:
        CalbImg()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start node')
   
        