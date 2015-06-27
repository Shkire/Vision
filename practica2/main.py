import os
import cv2
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import math
from sklearn.lda import LDA

#----------------------------Funciones auxiliares----------------------------
#el primer rectangulo esta dentro del segundo
def AisinsideB((Ax,Ay,Aw,Ah), (Bx,By,Bw,Bh)):
	if (Ax>Bx) & (Ay>By) & ((Ax-Bx+Aw) < Bw) & ((Ay-By+Ah) < Bh):
		return True
	return False

#calcula el area del rectangulo pasado
def rectArea ((x,y,w,h)):
	return w*h

#se utiliza si de un caracter se reconocen varios contornos
def erosionar (imgthreshold, finalcontours):
	while (len(finalcontours) > 1):
		#comprobar si no esta dentro de ningun otro
		finalcontours1 = []
		for cnt in finalcontours:
			fuera = True
			for cont in finalcontours:
				if (cnt != cont):
					fuera = fuera and not AisinsideB(cnt, cont)
			if (fuera):
				finalcontours1.append(cnt)
				if (len(finalcontours1)) > 1:
					break
		if (len(finalcontours1)) == 1:
			return finalcontours1
			break
		#hacer erosion
		imgthreshold = cv2.erode(imgthreshold,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),iterations=1)
		#buscar contornos
		imgcontours = imgthreshold.copy() 
		contours,hierarchy = cv2.findContours(imgcontours,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		finalcontours1 = []
		#resetear finalcontours y meter en el los nuevos rectangulos
		for cnt in contours:
			x,y,w,h = cv2.boundingRect(cnt) 
			if (h < imgheight-5):
				finalcontours1.append((x,y,w,h))
		return finalcontours1

#dada una imagen y sus rectangulos, la recorta, redimensiona, y devuelve el vector de caracteristicas
def obtenercaracteristicas(imagen,rectangulos):
	(x,y,w,h) = rectangulos[0]
	imagen = imagen[y:y+h,x:x+w]
	imagen = cv2.resize(imagen, (10,10))
	#imagen = np.asarray(imagen)
	imagen = imagen.reshape(1,100)
	return imagen[0]



#----------------------------Programa principal----------------------------
numerokeypoints = 100
orb = cv2.ORB(numerokeypoints,1,1)
lista = []
pathbase = os.path.dirname(os.path.abspath(__file__))
pathcars = pathbase + '/testing_ocr/frontal_'
haarClassifier = pathbase + '/haar/matriculas.xml'
extension = '.jpg'
pathchars = pathbase + '/training_ocr/frontal_'

for i in xrange(6, 6):
	#carga de la imagen
	imgpath = pathcars + str(i) + extension
	print imgpath
	temp = cv2.imread(imgpath,cv2.CV_LOAD_IMAGE_GRAYSCALE) #cv2.CV_LOAD_IMAGE_GRAYSCALE
	plt.imshow(temp, cmap=mpl.cm.get_cmap('gray'))
	#plt.imshow(temp)
	plt.title("Imagen original")
	plt.show()
	#lista.append(temp)

	#umbralizado
	imgthreshold = cv2.adaptiveThreshold(temp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 7)
	imgcontours = imgthreshold.copy() #es necesario, luego imgcontours se destruye
	plt.imshow(imgthreshold, cmap=mpl.cm.get_cmap('gray'))
	#plt.imshow(imgthreshold)
	plt.title("Imagen umbralizada")
	plt.show()

	#deteccion de matricula
	cascade = cv2.CascadeClassifier(haarClassifier)
	matriculas = cascade.detectMultiScale(
		imgthreshold,
      		scaleFactor=1.1,
        	minNeighbors=3,
        	minSize=(5,5),
        	flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
	imgthreshold = cv2.cvtColor(imgthreshold, cv2.COLOR_GRAY2BGR) #se transforma para pintar encima en color
	print matriculas
	for (x,y,w,h) in matriculas:
		cv2.rectangle(imgthreshold, (x,y), (x+w, y+h), (255,0,0), 2)
	plt.imshow(imgthreshold)
	plt.title("Matricula(s) detectada(s)")
	plt.show()


	#busqueda de contornos
	contours,hierarchy = cv2.findContours(imgcontours,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #antes habia 1,2
	top = len(contours)
	#proportions = []
	i = iter(contours)
	finalcontours = []
	for cnt in i:
		#cnt = contours[j]
		x,y,w,h = cv2.boundingRect(cnt)
		proportion = h/float(w)
		#si el rectangulo es vertical
		if (proportion > 1):
			for matricula in matriculas:
				#si esta dentro de alguna de las matriculas encontradas
				if AisinsideB((x,y,w,h), matricula):
					#si tiene un tamano minimo
					if h > 0.5*matricula[3]:
						finalcontours.append((x,y,w,h))
						cv2.rectangle(imgthreshold,(x,y),(x+w,y+h),(0,255,0),2)
			#proportions.append(proportion)

	plt.imshow(imgthreshold)
	plt.title("Caracteres encontrados")
	plt.show()
#termina el for de imagenes



caracteres = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A"]#, "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
pathtraining = pathbase + "/training_ocr/"
C = [] #matriz de caracteristicas
E = [] #vector de enteros. Fila i de la matriz C -> caracter caracteres[E[i]]
pos = -1
for caracter in caracteres:
	pos = pos+1
	for i in xrange(1,11):
		#carga de la imagen
		imgpath = pathtraining + caracter + "_" + str(i) + extension
		temp = cv2.imread(imgpath,cv2.CV_LOAD_IMAGE_GRAYSCALE)
		imgheight, imgwidth = temp.shape

		#umbralizado
		imgthreshold = cv2.adaptiveThreshold(temp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 81, 1)

		#busqueda de contornos
		imgcontours = imgthreshold.copy() 
		contours,hierarchy = cv2.findContours(imgcontours,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		i = iter(contours)
		finalcontours = []
		for cnt in i:
			x,y,w,h = cv2.boundingRect(cnt)
			if (h < imgheight-5):
				finalcontours.append((x,y,w,h))
			elif (len(contours) == 1):
				finalcontours.append((x,y,w,h))
				#si se detecta solo un rectangulo, o uno muy pequeno y es la imagen entera, se deberia dilatar y volver a buscar contornos. Es el caso inverso a que detecte el caracter en varios rectangulos


		#si se divide el caracter en varios rectangulos
		if (len(finalcontours) > 1):
			finalcontours = erosionar(imgthreshold,finalcontours)

		#si al final no conseguimos un buen contorno, tomamos la imagen entera
		if (len(finalcontours) == 0 or (len(finalcontours) >= 1 and finalcontours[0][3] < 0.35*imgheight)):
			finalcontours = [(2,2,imgwidth-4,imgheight-4)]

		
		#obtener vector de caracteristicas
		caracteristicas = obtenercaracteristicas(temp,finalcontours)
		C.append(caracteristicas)
		E.append(pos)
lda = LDA()
lda.fit(C, E)
CR = lda.transform(C)

CR = CR.astype(np.float32) 
E = np.array(E)
#probar: cv2.NormalBayesClassifier, cv2.EM, cv2.Knearest
clasificador = cv2.NormalBayesClassifier()
clasificador.train(CR, E)

caracRed = lda.transform([caracteristicas])
retval, results = clasificador.predict(np.float32(caracRed))
print retval











