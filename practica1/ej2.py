import cv2

import numpy as np

import matplotlib as mplt

import matplotlib.pyplot as plt

import math

path= 'test/test'
extension = '.jpg'
lista=[]
clasCas = cv2.CascadeClassifier()
clasCas.load('haar/coches.xml')

for i in xrange(1, 34):
    img = path + str(i) + extension
    temp = cv2.imread(img,0)
    lista.append(temp)

for i in xrange(0, 33):
    img=cv2.cvtColor(lista[i],cv2.COLOR_GRAY2BGR)
    rectangulos= clasCas.detectMultiScale(lista[i],1.1,10)
    for (x,y,w,h) in rectangulos:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    plt.imshow(img)
    plt.show()