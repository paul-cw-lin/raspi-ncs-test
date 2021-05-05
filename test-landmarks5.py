import cv2
import numpy as np

#Face detection
bin_path = '/home/pi/Documents/build/open_model_zoo/tools/downloader/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.bin'
xml_path = '/home/pi/Documents/build/open_model_zoo/tools/downloader/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml'

#Landmarks 5
bin_path_5 = '/home/pi/Documents/build/open_model_zoo/tools/downloader/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.bin'
xml_path_5 = '/home/pi/Documents/build/open_model_zoo/tools/downloader/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml'

net = cv2.dnn.readNet(xml_path, bin_path)
net_5 = cv2.dnn.readNet(xml_path_5, bin_path_5)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
net_5.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

while True:
    ret, frame = cap.read()
        
    blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U) #Face detection
    net.setInput(blob)
    out = net.forward()
    
    for detection in out.reshape(-1,7):
        confidence = float(detection[2])
        if confidence > 0.5:
            xmin = int(detection[3]*frame.shape[1])
            ymin = int(detection[4]*frame.shape[0])
            xmax = int(detection[5]*frame.shape[1])
            ymax = int(detection[6]*frame.shape[0])
            cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), (0,255,0), 2)
            roiframe = frame[ymin:ymax, xmin:xmax]
                        
            blob_5 = cv2.dnn.blobFromImage(roiframe, size=(48, 48), ddepth=cv2.CV_8U) #Landmark 5
            net_5.setInput(blob_5)
            out_5 = net_5.forward()
    
            for detection_5 in out_5.reshape(1,10):
                x0 = int(detection_5[0]*roiframe.shape[1])
                y0 = int(detection_5[1]*roiframe.shape[0])
                x1 = int(detection_5[2]*roiframe.shape[1])
                y1 = int(detection_5[3]*roiframe.shape[0])
                x2 = int(detection_5[4]*roiframe.shape[1])
                y2 = int(detection_5[5]*roiframe.shape[0])
                x3 = int(detection_5[6]*roiframe.shape[1])
                y3 = int(detection_5[7]*roiframe.shape[0])
                x4 = int(detection_5[8]*roiframe.shape[1])
                y4 = int(detection_5[9]*roiframe.shape[0])
                cv2.circle(roiframe, (x0,y0), 5, (0,255,0), -1)
                cv2.circle(roiframe, (x1,y1), 5, (0,255,0), -1)
                cv2.circle(roiframe, (x2,y2), 5, (0,255,0), -1)
                cv2.circle(roiframe, (x3,y3), 5, (0,255,0), -1)
                cv2.circle(roiframe, (x4,y4), 5, (0,255,0), -1)
        
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
