import numpy as np
import cv2

cap = cv2.VideoCapture('test_video/1.mp4')
# record the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test.avi', fourcc, 15.0, (640, 480), True)
while True:
    ret,image = cap.read()
    image = cv2.resize(image,(640,480))
    out.write(image)
