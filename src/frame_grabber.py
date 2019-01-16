# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import math
import cv2
import imutils
import roslib
import rospy
import traceback
import logging
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from pv_estimator.msg import Meas

class frame_grabber:
    def __init__(self):
        self.prevFrame = np.zeros((2448, 2048, 3),np.uint8)
        self.tLastImg = rospy.Time()
        self.bridge = CvBridge()

        self.pub = rospy.Publisher("/tracker/meas", Meas, queue_size = 2)
        self.count = 0
        print('Frame Grabber Initialized')

    # callback for when an image is available
    def imageCallback(self, msg):
        found = False
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            print('failed to convert ')
            print(e)
            logging.error(traceback.format_exc())

        tCurrImg = msg.header.stamp

        
