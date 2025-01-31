#!/usr/bin/env python2

import rospy 
import cv2 
import numpy as np
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 

class ImageSubscriber:
    def __init__(self):
        rospy.init_node("image_subscriber")
        self.bridge = CvBridge() 
        self.image_sub = rospy.Subscriber("/cam/color", Image, self.image_cb)
        self.image_count = 0
        rospy.loginfo("Image Subscriber Node Started")


    def image_cb(self, msg):
        try:
            # Convert ROS image msg to opencv format 
            # cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            #cv_image = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")

            # Display the image 
            cv2.imshow('Received Image', cv_image)
            key = cv2.waitKey(200) & 0xFF # Keep the image window responsive 

            if key == ord('s') or key == ord('S'):
                self.image_count += 1 
                 # Save the image 
                image_file = "../images/" + str(self.image_count) +"_rgb.png"
                cv2.imwrite(image_file, cv_image)
                rospy.loginfo("Image saved as " + image_file)
           
        except Exception as e:
            rospy.logerr("Error processing image: {}".format(e))

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = ImageSubscriber()
        node.run()
    except rospy.ROSInterruptException:
        pass





