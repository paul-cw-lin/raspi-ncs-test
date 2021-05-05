import cv2
import numpy as np

#Landmarks 5
bin_path = '/home/pi/Documents/build/open_model_zoo/tools/downloader/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.bin'
xml_path = '/home/pi/Documents/build/open_model_zoo/tools/downloader/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml'

net = cv2.dnn.readNet(xml_path, bin_path)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print('frame.shape', frame.shape)

blob = cv2.dnn.blobFromImage(frame, size=(48, 48), ddepth=cv2.CV_8U) #Landmark 5
net.setInput(blob)
out = net.forward()

for detection in out.reshape(1,10):
    print('detection.shape =', detection.shape)
    print('detection[0]', detection[0])
    print('detection[1]', detection[1])
    print('detection[2]', detection[2])
    print('detection[3]', detection[3])
    print('detection[4]', detection[4])
    print('detection[5]', detection[5])
    print('detection[6]', detection[6])
    print('detection[7]', detection[7])
    print('detection[8]', detection[8])
    print('detection[9]', detection[9])
    x0 = int(detection[0]*frame.shape[1])
    y0 = int(detection[1]*frame.shape[0])
    x1 = int(detection[2]*frame.shape[1])
    y1 = int(detection[3]*frame.shape[0])
    x2 = int(detection[4]*frame.shape[1])
    y2 = int(detection[5]*frame.shape[0])
    x3 = int(detection[6]*frame.shape[1])
    y3 = int(detection[7]*frame.shape[0])
    x4 = int(detection[8]*frame.shape[1])
    y4 = int(detection[9]*frame.shape[0])
    print('x0 =', x0, 'y0 =', y0)
    print('x1 =', x1, 'y1 =', y1)
    print('x2 =', x2, 'y2 =', y2)
    print('x3 =', x3, 'y3 =', y3)
    print('x4 =', x4, 'y4 =', y4)
    cv2.circle(frame, (x0,y0), 5, (255,0,0), -1)
    cv2.circle(frame, (x1,y1), 5, (0,255,0), -1)
    cv2.circle(frame, (x2,y2), 5, (0,0,255), -1)
    cv2.imshow('frame', frame)
    

cv2.waitKey()
cap.release()
cv2.destroyAllWindows()