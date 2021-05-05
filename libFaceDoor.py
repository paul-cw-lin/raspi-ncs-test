#libFaceDoor 
import time, datetime
import imutils
import math
import cv2
import numpy as np

#--------------------------------------------------------#
# 1. wget --no-check-certificate https://download.01.org/opencv/2019/open_model_zoo/R1/models_bin/face-detection-adas-0001/FP16/face-detection-adas-0001.bin
# 2. wget --no-check-certificate https://download.01.org/opencv/2019/open_model_zoo/R1/models_bin/face-detection-adas-0001/FP16/face-detection-adas-0001.xml
#
class ov_FaceDect:
    def __init__(self, bin_path, xml_path):
        #load the model
        net = cv2.dnn.readNet(xml_path, bin_path)
        self.net = net

    def detect_face(self, frame, score=0.5, target_device=cv2.dnn.DNN_TARGET_MYRIAD):
        net = self.net
        # Specify target device.
        net.setPreferableTarget(target_device)

        #Prepare input blob and perform an inference.
        blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
        net.setInput(blob)
        out = net.forward()

        faces, scores = [], []
        for detection in out.reshape(-1, 7):
            confidence = float(detection[2])
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            if confidence > score:
                faces.append([xmin,ymin,xmax-xmin,ymax-ymin])
                scores.append(confidence)

        return faces, scores

class ov_FaceRecognize:
    def __init__(self, bin_path, xml_path):
        #load the model
        net = cv2.dnn.readNet(xml_path, bin_path)
        self.net = net

    def detect_face(self, face_img, target_device=cv2.dnn.DNN_TARGET_MYRIAD):
        net = self.net
        # Specify target device.
        net.setPreferableTarget(target_device)

        #Prepare input blob and perform an inference.
        blob = cv2.dnn.blobFromImage(face_img, size=(128, 128), ddepth=cv2.CV_8U)
        net.setInput(blob)
        out = net.forward()

        return out

class ov_Face5Landmarks:
    def __init__(self, bin_path, xml_path):
        #load the model
        net = cv2.dnn.readNet(xml_path, bin_path)
        self.net = net

    def getLandmarks(self, face_img, target_device=cv2.dnn.DNN_TARGET_MYRIAD):
        net = self.net
        # Specify target device.
        net.setPreferableTarget(target_device)

        #Prepare input blob and perform an inference.
        try:
            blob = cv2.dnn.blobFromImage(face_img, size=(48, 48), ddepth=cv2.CV_8U)
            net.setInput(blob)
            out = net.forward()
        except:
            out = None

        return out

    def renderFace(self, im, landmarks, color=(0, 255, 0), radius=5):
        for p in landmarks:
            cv2.circle(im, (p[0], p[1]), radius, color, -1)

        return im

    def angle_2_points(self, p1, p2):
        r_angle = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
        rotate_angle = r_angle * 180 / math.pi

        return rotate_angle