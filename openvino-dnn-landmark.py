import cv2
import time

#intel model
net = cv2.dnn.readNet(
    '/home/pi/Documents/build/open_model_zoo/tools/downloader/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml',
    '/home/pi/Documents/build/open_model_zoo/tools/downloader/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.bin'
    )

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(300,300), scale=1.0)

cap = cv2.VideoCapture(0)
ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH)/cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
WIDTH = 480
HEIGHT = int(WIDTH/ratio)
FONT = cv2.FONT_HERSHEY_SIMPLEX

while True:
    begin_time = time.time()
    ret, frame = cap.read()
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.flip(frame, 1)
    
    classes, confs, boxes = model.detect(frame, 0.5)
    for (classid, conf, box) in zip(classes, confs, boxes):
        x,y,w,h = box
        text = '%2f' % conf
        
        if y - 20 < 0:
            y1 = y+20
        else:
            y1=y-10
            
        fps = 1/ (time.time() - begin_time)
        text = "fps: {:.1f}" .format(fps)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, text, (10,20), FONT, 0.8, (255,0,0), 2)
        
        cv2.imshow('video', frame)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break