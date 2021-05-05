import cv2
import numpy as np

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
            roiframe = frame[ymin+20:ymax+20, xmin+20:xmax+20]
                        
            blob_35 = cv2.dnn.blobFromImage(roiframe, size=(60, 60), ddepth=cv2.CV_8U) #Landmark 35
            net_35.setInput(blob_35)
            out_35 = net_35.forward()
    
            for detection_35 in enumerate(out_35.reshape(35,2)):
                k,j = detection_35
                cv2.circle(roiframe, (int(j[0]*roiframe.shape[1]),int(j[1]*roiframe.shape[0])), 2, (0,255,0), -1)
                                       
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
