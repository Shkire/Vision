__author__ = 'shkire'

import os

import cv2

import numpy as np

import matplotlib as mplt

import matplotlib.pyplot as plt

import math

#------------------------------------------ENTRENAMIENTO----------------------------

#CREACION ORB

orb = cv2.ORB(500,1.5,5)

#CREACION INDICE FLANN

FLANN_INDEX_LSH = 6

index_params= dict(algorithm = FLANN_INDEX_LSH,

                   table_number = 6, # 12

                   key_size = 12,     # 20

                   multi_probe_level = 1) #2

search_params = dict(checks=50)   # or pass empty dictionar

flann = cv2.FlannBasedMatcher(index_params,search_params)

lista = []

vectores = []

keypoints = []

vect=[]

pathbase = os.path.dirname(os.path.abspath(__file__))
path = pathbase + '/training/frontal_'

print path

extension = '.jpg'

#CARGA DE IMAGENES DE ENTRENAMIENTO

for i in xrange(1, 49):

    img = path + str(i) + extension

    print img

    temp = cv2.imread(img,0)


    lista.append(temp)

#DETECCION DE KEYPOINTS Y SUS DESCRIPTORES CORRESPONDIENTES

    kp1,des1 = orb.detectAndCompute(lista[i-1], None)

    flann.add([des1])

    centro = (lista[i-1].shape[1]/2,lista[i-1].shape[0]/2) #x,y

    #aniadir vectores de votacion y kps

    for j in xrange(0,len(kp1)):
        vector = (centro[0]-kp1[j].pt[0],centro[1]-kp1[j].pt[1]) #x,y
        vect.append(vector)

    vectores.append(vect)
    keypoints.append(kp1)


#ENTRENAMIENTO

flann.train()

#-----------------------------------TESTEO------------------------------

path2 = pathbase + '/test/test'

lista2=[]


#CARGA DE IMAGENES DE TEST

for i in xrange(1, 34):

    img = path + str(i) + extension



    print img

    temp = cv2.imread(img,0)

    votaciones = np.zeros((round(temp.shape[0]/10),round(temp.shape[1]/10)))#,np.uint8)



    lista2.append(temp)

    kp2,des2 = orb.detectAndCompute(lista2[i-1], None)

    match=flann.knnMatch(des2,50)

    for j in xrange(0,len(match)):

        for k in xrange(0,len(match[j])):

            imgpos=match[j][k].imgIdx
            despos=match[j][k].trainIdx
            rescale=kp2[j].size/keypoints[imgpos][despos].size
            vectorRescala=(vectores[imgpos][despos][0]*rescale,vectores[imgpos][despos][1]*rescale)
            #print vectorRescala
            if (((round((kp2[j].pt[0]+vectorRescala[0])/10))<(round(temp.shape[1]/10))) & ((round((kp2[j].pt[1]+vectorRescala[1])/10))<(round(temp.shape[0]/10))) & ((round((kp2[j].pt[0]+vectorRescala[0])/10))>=0) & ((round((kp2[j].pt[1]+vectorRescala[1])/10))>=0)):
                if match[j][k].distance>0:
                    votaciones[round((kp2[j].pt[1]+vectorRescala[1])/10)][round((kp2[j].pt[0]+vectorRescala[0])/10)]=votaciones[round((kp2[j].pt[1]+vectorRescala[1])/10)][round((kp2[j].pt[0]+vectorRescala[0])/10)]+(1/(match[j][k]).distance)
                else:
                    votaciones[round((kp2[j].pt[1]+vectorRescala[1])/10)][round((kp2[j].pt[0]+vectorRescala[0])/10)]=votaciones[round((kp2[j].pt[1]+vectorRescala[1])/10)][round((kp2[j].pt[0]+vectorRescala[0])/10)]+1
                print "("+str(round((kp2[j].pt[1]+vectorRescala[1])/10))+","+str(round((kp2[j].pt[0]+vectorRescala[0])/10))+")="+str(votaciones[round((kp2[j].pt[1]+vectorRescala[1])/10)][round((kp2[j].pt[0]+vectorRescala[0])/10)])


    res = cv2.resize(votaciones,None,fx=10,fy=10,interpolation=cv2.INTER_NEAREST)
    #color=cv2.cvtColor(temp,cv2.COLOR_GRAY2BGR)
    plt.imshow(temp, cmap=mplt.cm.get_cmap('gray'))
    plt.figure()
    plt.imshow(res)#, cmap=mplt.cm.get_cmap('gray'))
    plt.show()
