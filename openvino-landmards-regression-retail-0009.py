#from libFaceDoor import ov_Face5Landmarks
#import time, datetime
#import imutils
#import math
import cv2
import numpy as np


bin_path = '/home/pi/Documents/build/open_model_zoo/tools/downloader/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.bin'
xml_path = '/home/pi/Documents/build/open_model_zoo/tools/downloader/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml'

net = cv2.dnn.readNet(xml_path, bin_path)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

#faceLandmarks = ov_Face5Landmarks(bin_path, xml_path)

cap = cv2.VideoCapture(0)

#while True:
ret, frame = cap.read()
    
    #try:
blob = cv2.dnn.blobFromImage(frame, size=(48, 48), ddepth=cv2.CV_8U)
net.setInput(blob)
out = net.forward()
    #except:
        #out = None

out = np.array(out)
a = out.resize(out, (10,1))

print('a : ', a)

for p in a:
    x1 = int(p[0] * frame.shape[1])
    y1 = int(p[1] * frame.shape[0])
    x2 = int(p[2] * frame.shape[1])
    y2 = int(p[3] * frame.shape[0])
    print('x1', x1)
    print('y1', y1)
    print('x2', x2)
    print('y2', y2)
    #cv2.circle(frame, (p[0], p[1]), 5, (0,255,0), -1)
    
#cv2.imshow('frame', frame)    
cv2.waitKey()
    #if key == 27 or key ==ord('q'):
cap.release()
cv2.destroyAllWindows()
    #    break

#points = faceLandmarks.getLandmarks(face, target_device=cv2.dnn.DNN_TARGET_MYRIAD)
