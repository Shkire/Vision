__author__ = 'shkire'

import os

import cv2

import numpy as np

pathbase = os.path.dirname(os.path.abspath(__file__))
# DETECCION HAAR
clasCas = cv2.CascadeClassifier()
clasCas.load(pathbase+'/haar/coches.xml')

vid = cv2.VideoCapture(pathbase+"Videos/video1.wmv")
while(vid.isOpened()):
    ret, frame = vid.read()
    img2=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rectangulos= clasCas.detectMultiScale(frame,1.1,10)
    for (x,y,w,h) in rectangulos:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()