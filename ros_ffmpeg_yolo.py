

from ultralytics import YOLO
import cv2
import traceback
import math 
import numpy as np
import random

import random

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


#ONNX - CPU
#https://docs.ultralytics.com/integrations/onnx/?h=onnx#key-features-of-onnx-models
#TensorRT - GPU - Jetson ve Laptop
#https://docs.ultralytics.com/tr/integrations/tensorrt/#deployment-options-in-tensorrt
#Triton -GPU - Jetson ve LAPTOP
#https://docs.ultralytics.com/tr/guides/triton-inference-server/
#DeepStream - gpu -----
#https://docs.ultralytics.com/tr/yolov5/tutorials/running_on_jetson_nano/?h=deepstrea#install-necessary-packages


# model
print("load model")
model_old = YOLO("yolo-Weights/yolov8m-seg.pt")
#model_old.export(format='engine')  # creates 'yolov8n.engine'
model = YOLO('yolo-Weights/yolov8m-seg.engine')

print("loaded model")


# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
yolo_classes = list(model_old.names.values())

classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
colors = [random.choices(range(256), k=3) for _ in classes_ids]
alpha = 0.5 
conf = 0.5


class ObjectDetection(Node):
    def __init__(self):
        print("step 5.0")

        super().__init__('object_detection')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/image_raw/uncompressed', self.callback, 10)
        self.orig_img = None
        self.params = [0]
 
        print("step 5.1")


    def callback(self, data):
        print("step 5.2")
        try:
            cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.orig_img = cv_img
            print("step 5.2-1")
        except CvBridgeError as exc:
            self.get_logger().error(traceback.format_exc())


def show_images(obj):
    print("step 5.3")
    if obj.orig_img is not None:
        print("step 5.4")
        #cv2.imshow('Original', obj.orig_img)
        conf = 0.5
        #results = model.predict(obj.orig_img, conf=conf)
        results = model(obj.orig_img)
        print("step 5.5")
        img = obj.orig_img
    for r in results:
        #masks = r.masks.xy
 
        for box in r.boxes:
            for mask in zip(r.masks.xy):
                points = np.int32([mask])
                color_number = classes_ids.index(int(box.cls[0]))
                #cv2.fillPoly(img, points, colors[color_number])

                color = colors[color_number]
                overlay = img.copy()
                cv2.fillPoly(overlay, points, color)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                
                
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
          

        cv2.imshow("Image Processing", img)

        print("step 6")                

def main(args=None):
    rclpy.init(args=args)
    print("step 1")

    od = ObjectDetection()
    print("step 2")

    print("step 3")

    while rclpy.ok():
        rclpy.spin_once(od)
        #model=YOLO('yolov8m-pose.pt')
        #results=model(source=0, show=True, conf=0.3, save=True)
        print("step 4")

        show_images(od)
        print("step 5")

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()
    od.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()