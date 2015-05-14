__author__ = 'shkire'

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

path = '/home/shkire/Escritorio/Vision/MaterialP3/training/frontal_'

extension = '.jpg'

#CARGA DE IMAGENES DE ENTRENAMIENTO

for i in xrange(1, 49):

    img = path + str(i) + extension

    print img

    temp = cv2.imread(img,0)
    #plt.imshow(temp, cmap=mplt.cm.get_cmap('binary'))
    #plt.show()


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

    #img2 = cv2.drawKeypoints(lista[i-1], kp1, color=(0,255,0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #plt.imshow(img2)

    #plt.show()

#ENTRENAMIENTO

flann.train()

#-----------------------------------TESTEO------------------------------

path2 = '/home/shkire/Escritorio/Vision/MaterialP3/test/test'

lista2=[]


#CARGA DE IMAGENES DE TEST

for i in xrange(1, 34):

    img = path + str(i) + extension



    print img

    temp = cv2.imread(img,0)

    #print type(temp)

    votaciones = np.zeros((round(temp.shape[0]/10),round(temp.shape[1]/10)))#,np.uint8)
    #print (temp.shape[0]/10,temp.shape[1]/10)


    #plt.imshow(temp, cmap=mplt.cm.get_cmap('gray'))
    #plt.show()

    lista2.append(temp)

    #print "Imagen guardada"

    kp2,des2 = orb.detectAndCompute(lista2[i-1], None)

    #print "Obtenidos des y kp"

    match=flann.knnMatch(des2,50)

    #print "obtenidos matchs"

    for j in xrange(0,len(match)):

        #print   "paso "+str(j+1)+" elemento numero "+str(j)+" de la lista de matches"

        for k in xrange(0,len(match[j])):

            #print   "paso "+str(k+1)+" elemento numero "+str(k)+" de la lista de vecinos"

            imgpos=match[j][k].imgIdx

            #print   "obtenido posicion imagen: "+str(imgpos)

            despos=match[j][k].trainIdx

            #print   "obtenido posicion descriptor: "+str(despos)

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
    #color[np.argmax(np.amax(res,1))][np.argmax(np.amax(res,0))]=(255,0,0)
    #color[np.argmax(np.amax(res,1))+1][np.argmax(np.amax(res,0))+1]=(255,0,0)
    #color[np.argmax(np.amax(res,1))+2][np.argmax(np.amax(res,0))+2]=(255,0,0)
    #color[np.argmax(np.amax(res,1))+3][np.argmax(np.amax(res,0))+3]=(255,0,0)
    #color[np.argmax(np.amax(res,1))+4][np.argmax(np.amax(res,0))+4]=(255,0,0)
    #color[np.argmax(np.amax(res,1))+5][np.argmax(np.amax(res,0))+5]=(255,0,0)
    #color[np.argmax(np.amax(res,1))+6][np.argmax(np.amax(res,0))+6]=(255,0,0)
    #color[np.argmax(np.amax(res,1))+7][np.argmax(np.amax(res,0))+7]=(255,0,0)
    #color[np.argmax(np.amax(res,1))+8][np.argmax(np.amax(res,0))+8]=(255,0,0)
    #color[np.argmax(np.amax(res,1))+9][np.argmax(np.amax(res,0))+9]=(255,0,0)
    plt.imshow(temp, cmap=mplt.cm.get_cmap('gray'))
    plt.figure()
    plt.imshow(res)#, cmap=mplt.cm.get_cmap('gray'))
    plt.show()
    #print np.argmax(np.amax(res,0))
    #print np.argmax(np.amax(res,1))





    #print match