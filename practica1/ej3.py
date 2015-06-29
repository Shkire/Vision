import cv2

import numpy as np

lista=[]
clasCas = cv2.CascadeClassifier()
clasCas.load('haar/coches.xml')
vid = cv2.VideoCapture('Videos/video1.wmv')
while(vid.isOpened()):
    ret, frame = vid.read()
    if (frame==None):
	break
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rectangulos= clasCas.detectMultiScale(img,1.2,5)
    for (x,y,w,h) in rectangulos:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    lista.append(frame)
    cv2.imshow('frame',frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
vid.release()
cv2.destroyAllWindows()
vid = cv2.VideoCapture('Videos/video2.wmv')
while(vid.isOpened()):
    ret, frame = vid.read()
    if (frame==None):
	break
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rectangulos= clasCas.detectMultiScale(img,1.1,7)
    for (x,y,w,h) in rectangulos:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    lista.append(frame)
    cv2.imshow('frame',frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
vid.release()
cv2.destroyAllWindows()
