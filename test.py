import cv2
import numpy as np
import matplotlib.pyplot as plt

#Face detection
bin_path = '/home/pi/Documents/build/open_model_zoo/tools/downloader/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.bin'
xml_path = '/home/pi/Documents/build/open_model_zoo/tools/downloader/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml'

#Landmarks 35
bin_path_35 = '/home/pi/Documents/build/open_model_zoo/tools/downloader/intel/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.bin'
xml_path_35 = '/home/pi/Documents/build/open_model_zoo/tools/downloader/intel/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml'

net = cv2.dnn.readNet(xml_path, bin_path)
net_35 = cv2.dnn.readNet(xml_path_35, bin_path_35)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
net_35.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

o = cv2.imread('paul-3.jpg')
img = o

blob_35 = cv2.dnn.blobFromImage(img, size=(60, 60), ddepth=cv2.CV_8U) #Landmark 35
net_35.setInput(blob_35)
out_35 = net_35.forward()

for i in enumerate(out_35.reshape(35,2)):
    k,j = i
    cv2.circle(img, (int(j[0]*img.shape[1]),int(j[1]*img.shape[0])), 2, (0,255,0), -1)

'''
for detection_35 in out_35.reshape(1,70):
    x0 = int(detection_35[0]*img.shape[1])
    y0 = int(detection_35[1]*img.shape[0])
    x1 = int(detection_35[2]*img.shape[1])
    y1 = int(detection_35[3]*img.shape[0])
    x2 = int(detection_35[4]*img.shape[1])
    y2 = int(detection_35[5]*img.shape[0])
    x3 = int(detection_35[6]*img.shape[1])
    y3 = int(detection_35[7]*img.shape[0])
    x4 = int(detection_35[8]*img.shape[1])
    y4 = int(detection_35[9]*img.shape[0])
    x5 = int(detection_35[10]*img.shape[1])
    y5 = int(detection_35[11]*img.shape[0])
    x6 = int(detection_35[12]*img.shape[1])
    y6 = int(detection_35[13]*img.shape[0])
    x7 = int(detection_35[14]*img.shape[1])
    y7 = int(detection_35[15]*img.shape[0])
    x8 = int(detection_35[16]*img.shape[1])
    y8 = int(detection_35[17]*img.shape[0])
    x9 = int(detection_35[18]*img.shape[1])
    y9 = int(detection_35[19]*img.shape[0])
    cv2.circle(img, (x0,y0), 2, (0,255,0), -1)
    cv2.circle(img, (x1,y1), 2, (0,255,0), -1)
    cv2.circle(img, (x2,y2), 2, (0,255,0), -1)
    cv2.circle(img, (x3,y3), 2, (0,255,0), -1)
    cv2.circle(img, (x4,y4), 2, (0,255,0), -1)
    cv2.circle(img, (x5,y5), 2, (0,255,0), -1)
    cv2.circle(img, (x6,y6), 2, (0,255,0), -1)
    cv2.circle(img, (x7,y7), 2, (0,255,0), -1)
    cv2.circle(img, (x8,y8), 2, (0,255,0), -1)
    cv2.circle(img, (x9,y9), 2, (0,255,0), -1)
'''    
cv2.imshow('frame', img)
cv2.waitKey()
cv2.destroyAllWindows()
