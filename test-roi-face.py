import cv2
import numpy as np

net = cv2.dnn.readNet(
    '/home/pi/opencv/opencv-master/samples/dnn/face_detector/opencv_face_detector.pbtxt',
    '/home/pi/opencv/opencv-master/samples/dnn/face_detector/opencv_face_detector_uint8.pb')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(300,300), scale=1.0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    b = np.zeros(frame.shape, dtype=np.uint8)
    classIds, confidences, boxes = model.detect(frame, 0.5)
    
    for (classid, conf, box) in zip(classIds, confidences, boxes):
        x,y,w,h = box
        b[y:y+h, x:x+w]=255
        c = cv2.bitwise_and(frame,b)
        face = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        #cv2.imshow('video', frame)
        cv2.imshow('c', c)
        cv2.imshow('face', face)
        
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break