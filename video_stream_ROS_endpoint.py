#!/usr/bin/env python3
from ultralytics import YOLO


import rclpy
from rclpy.node import Node
import sys, traceback, math, cv2, numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#from ultralytics import YOLO    



class ObjectDetection(Node):
    def __init__(self):
        super().__init__('object_detection')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/image_raw/uncompressed', self.callback, 10)
        self.orig_img = None
        self.params = [0]

    def callback(self, data):
        self.params = self.get_trackbar_params()
        try:
            cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.orig_img = cv_img
        except CvBridgeError as exc:
            self.get_logger().error(traceback.format_exc())

    def get_trackbar_params(self):
        return [cv2.getTrackbarPos('Param', 'Parameter-Trackbars')]

def show_images(obj):
    if obj.orig_img is not None:
        cv2.imshow('Original', obj.orig_img)



def handle_trackbar_changes(obj):
    params = obj.get_trackbar_params()

def trackbar_callback(x):
    pass

def create_windows_and_trackbars():     
    cv2.namedWindow('Parameter-Trackbars')
    cv2.createTrackbar('Param', 'Parameter-Trackbars', 0, 179, trackbar_callback)

def main(args=None):
    rclpy.init(args=args)
    od = ObjectDetection()
    create_windows_and_trackbars()
    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

    while rclpy.ok():
        rclpy.spin_once(od)
        #model=YOLO('yolov8m-pose.pt')
        #results=model(source=0, show=True, conf=0.3, save=True)

        show_images(od)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key to exit
            break
        handle_trackbar_changes(od)

    cv2.destroyAllWindows()
    od.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()