import cv2
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import math

def AisinsideB((Ax,Ay,Aw,Ah), (Bx,By,Bw,Bh)): #el primer rectangulo esta dentro del segundo
	if (Ax>By) & (Ay>By) & ((Ax-Bx+Aw) < Bw) & ((Ay-By+Ah) < Bh):
		return True
	return False


numerokeypoints = 100
orb = cv2.ORB(numerokeypoints,1,1)
lista = []
pathcars = '/home/danielduque/Escritorio/Vision/MaterialP3/training/frontal_'
haarClassifier = '/home/danielduque/Escritorio/Vision/MaterialP3/haar/matriculas.xml'
extension = '.jpg'
pathchars = '/home/danielduque/Escritorio/Vision/MaterialP4/training_ocr/frontal_'

for i in xrange(1, 2):
	#carga de la imagen
	imgpath = pathcars + str(i) + extension
	temp = cv2.imread(imgpath,cv2.CV_LOAD_IMAGE_GRAYSCALE) #cv2.CV_LOAD_IMAGE_GRAYSCALE
	plt.imshow(temp, cmap=mpl.cm.get_cmap('gray'))
	#plt.imshow(temp)
	plt.show()
	#lista.append(temp)

	#umbralizado
	imgthreshold = cv2.adaptiveThreshold(temp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 7)
	imgcontours = imgthreshold.copy() #es necesario, luego temp se destruye al umbralizar
	#plt.imshow(imgthreshold, cmap=mpl.cm.get_cmap('gray'))
	#plt.imshow(imgthreshold)
	#plt.show()

	#deteccion de matricula
	cascade = cv2.CascadeClassifier(haarClassifier)
	matriculas = cascade.detectMultiScale(
		imgthreshold,
      		scaleFactor=1.1,
        	minNeighbors=3,
        	minSize=(5,5),
        	flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
	imgthreshold = cv2.cvtColor(imgthreshold, cv2.COLOR_GRAY2BGR)
	print matriculas
	for (x,y,w,h) in matriculas:
		cv2.rectangle(imgthreshold, (x,y), (x+w, y+h), (255,0,0), 2)
	plt.imshow(imgthreshold)
	plt.show()
	print AisinsideB(matriculas[0], matriculas[1])
	print AisinsideB(matriculas[1], matriculas[0])




	#busqueda de contornos
	contours,hierarchy = cv2.findContours(imgcontours,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #antes habia 1,2
	top = len(contours)
	proportions = []
	for j in xrange(0,top):
		cnt = contours[j]
		x,y,w,h = cv2.boundingRect(cnt)
		#cv2.rectangle(imgthreshold,(x,y),(x+w,y+h),(0,255,0),2)
		proportion = h/float(w)
		#print proportion
		if proportion > 1:
			for matricula in matriculas:
				if AisinsideB((x,y,w,h), matricula):
					cv2.rectangle(imgthreshold,(x,y),(x+w,y+h),(0,255,0),2)
			#proportions.append(proportion)
			#cv2.rectangle(imgthreshold,(x,y),(x+w,y+h),(0,255,0),2)
	plt.imshow(imgthreshold)
	plt.show()


	cascade = cv2.CascadeClassifier(haarClassifier);
