import cv2
import numpy as np

a = np.random.randint(0,50,(10,1,1))

for d in a.reshape(1,10):
    print('d = ', d)
    print('d[0] = ', d[0])