#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import shape, size
from math import floor, ceil, sqrt, pi, e, pow, atan
from MyThread import ReturnValueThread as Thread
from statistics import median


'''
Los métodos implementados tendrán imágenes como entrada y salida. Asumimos que estas
imágenes son matrices que cumplen:
    · Tienen un único valor de intensidad por cada punto (escala de grises).
    · El tipo de datos es de punto flotante de doble o simple precisión.
    · El valor de intensidad se encuentra, como norma general, en el rango [0, 1].
    · El tamaño de la imagen M × N es arbitrario y en general M 6=/N.
'''

class ImageProcessing():

    def __normalizeImage(self, image):
        max, min = np.max(image), np.min(image)
        return np.float32([(x-min)/(max-min) for x in image])

    def openImage(self, image, bin=False):
        img = cv.imread(image, cv.IMREAD_GRAYSCALE)
        if (bin):
            img = np.where((img <= 127), 255, 0)
        return np.float32([x/255 for x in img])

    def showImage(self, image):
        cv.imshow("Image", image)
        cv.waitKey(0)

    def saveImage(self, path, image):
        cv.imwrite(path, image*255)
    
    def plotHistograms(self, h1, h2):
        plt.subplot(1, 2, 1)
        plt.hist((h1*255).ravel(),256,[0,256])
        plt.title("Antes de la mod. del rango.")
        plt.subplot(1, 2, 2)
        plt.hist((h2*255).ravel(),256,[0,256])
        plt.title("Después de la mod. del rango.")
        plt.show()

    def binaryImage(self, image, thres_val=0.5, negative=False):
        newImage = np.zeros(shape=(image.shape[0], image.shape[1]))
        for x in range(len(image)):
            for y in range(len(image[0])):
                if negative:
                    if (image[x][y] < thres_val):
                        newImage[x][y] = 1.0
                else:
                    if (image[x][y] >= thres_val):
                        newImage[x][y] = 1.0
        return np.float32(newImage)

    def __invertBinImage(self, image):
        newImage = np.zeros(shape=(image.shape[0], image.shape[1]))
        for x in range(len(image)):
            for y in range(len(image[0])):
                if (image[x][y] >= 0.5):
                    newImage[x][y] = 0.0
                else:
                    newImage[x][y] = 1.0
        return np.float32(newImage)

    
    '''
    Función que permite hacer una compresión (o estiramiento) lineal de histograma mediante la introducción de nuevos lı́mites inferior y superior.

    Entrada:
        · inImage: Matriz MxN con la imagen de entrada.
        · inRange: Vector 1x2 con el rango de niveles de intensidad [imin, imax] de entrada. Si el vector está vacı́o (por defecto), el mı́nimo 
        y máximo de la imagen de entrada se usan como imin e imax.
        · outRange: Vector 1x2 con el rango de niveles de instensidad [omin, omax] de salida. El valor por defecto es [0 1].
    Salida:
        · outImage: Matriz MxN con la imagen de salida.
    '''
    def adjustIntensity(self, inImage, inRange=[], outRange=[0,1]):
        if (inRange==[]):
            inRange = [np.min(inImage), np.max(inImage)]
        
        f = lambda x: outRange[0] + ((outRange[1]-outRange[0])*(x-inRange[0]))/(inRange[1]-inRange[0])
        
        return np.float32([f(x) for x in inImage])
        
    
    '''
    Algoritmo de ecualización de histograma.

    Entrada:
        · inImage: Matriz MxN con la imagen de entrada.
        · nBins: Número de bins utilizados en el procesamiento. Se asume que el intervalo de entrada [0 1] se divide en nBins intervalos 
        iguales para hacer el procesamiento, y que la imagen de salida vuelve a quedar en el intervalo [0 1]. Por defecto 256.
    Salida:
        · outImage: Matriz MxN con la imagen de salida.
    '''
    def equalizeIntensity(self, inImage, nBins=256):
        
        hist = np.zeros(shape=(nBins))
        for x in range(inImage.shape[0]):
            for y in range(inImage.shape[1]):
                bin = nBins-1 if (inImage[x][y]==1.0) else int(inImage[x][y]*nBins)
                hist[bin] += 1
        
        acchist = np.zeros(shape=hist.shape)
        acc = 0
        for x in range(hist.shape[0]):
            acc += hist[x]
            acchist[x] = acc
        
        cdf_m = np.ma.masked_equal(acchist*255,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        
        aux=np.zeros(shape=inImage.shape)
        for x in range(inImage.shape[0]):
            for y in range(inImage.shape[1]):
                bin = nBins-1 if (inImage[x][y]==1.0) else int(inImage[x][y]*nBins)
                aux[x][y] = cdf[bin]
        
        return self.__normalizeImage(aux)


    '''
    Filtrado espacial mediante convolución sobre una imagen.

    Entrada:
        · inImage: Matriz MxN con la imagen de entrada.
        · kernel: Matriz PxQ con el kernel del filtro de entrada. Se asume que la posición central del filtro está en (P/2 + 1, Q/2 + 1).
    Salida:
        · outImage: Matriz MxN con la imagen de salida.
    '''
    def __filterImage(self, inImage, kernel, center=[]):
        # inImage[α][γ]
        α, μ = inImage.shape
        # kernel[λ][μ]
        λ, γ = kernel.shape
        p, q = floor(λ/2)+1, floor(γ/2)+1 if (center==[]) else center
        outImage = np.zeros(shape=(α,μ))
        
        for i in range(α):
            for j in range(μ):
                aux=0
                for x in range(λ):
                    for y in range(γ):
                        k_i = i-p+x
                        k_j = j-q+y
                        if not ((k_i>=α) or (k_j>=μ) or (k_i<0) or (k_j<0)):
                            aux += kernel[x][y]*inImage[k_i][k_j]
                outImage[i][j]=aux

        return outImage
    def filterImage(self, inImage, kernel):    
        t = Thread(target=self.__filterImage, args=(inImage, kernel))
        t.start()
        return t.join()
    

    '''
    Función que devuelve un kernel Gaussiano unidimensional con σ dado.

    Entrada
        · sigma: Parámetro σ de entrada.
    Salida:
        · kernel: Vector 1xN con el kernel de salida, teniendo en cuenta que el centro (x = 0) de la Gaussiana está en la posición N/2 + 1 y 
        N se calcula a partir de σ como N = 2*3σ + 1.
    '''    
    def gaussKernel1D(self, σ):
        # N = 2[3σ↑] + 1
        N = 2*ceil(3*σ) + 1
        # [N/2↓] + 1
        center = floor(N/2) + 1
        size = np.zeros(N)

        for x in range(N):
            size[x] = center - N + x

        α = 1/(sqrt(2*pi)*σ)        
        f = lambda x: α * pow(e, - pow(x, 2)/(2*pow(σ, 2)))
        
        return np.float32([f(x) for x in size]).reshape(1,N)
    

    '''
    Función que permite realizar un suavizado Gaussiano bidimensional usando un filtro N × N de parámetro σ, donde N se calcula igual 
    que en la función anterior.

    Entrada
        · inImage: Matriz MxN con la imagen de entrada.
        · sigma: Parámetro σ de entrada.
    Salida:
        · outImage: Matriz MxN con la imagen de salida.
    '''
    def gaussianFilter(self, inImage, σ):
        kernel = self.gaussKernel1D(σ)
        t = Thread(target=self.filterImage, args=(inImage, kernel))
        t.start()
        kernelT = np.transpose(kernel)
        return self.filterImage(t.join(), kernelT)

    
    '''
    Filtro de medianas bidimensional.

    Entrada
        · inImage: Matriz MxN con la imagen de entrada.
        · filterSice: Valor entero N indicando que el tamaño de ventana es de NxN. La posición central de la ventana es 
        ([N/2] + 1, [N/2] + 1).
    Salida:
        · outImage: Matriz MxN con la imagen de salida.
    '''
    def medianFilter(self, inImage, filterSize):
        α, η = inImage.shape
        p, q = floor(filterSize/2)+1, floor(filterSize/2)+1
        outImage = np.zeros(shape=(α,η))
        μ = np.median(inImage)

        for i in range(α):
            for j in range(η):
                aux=list()
                for x in range(filterSize):
                    for y in range(filterSize):
                        k_i = i-p+x
                        k_j = j-q+y
                        if ((k_i>=α) or (k_j>=η) or (k_i<0) or (k_j<0)):
                            aux.append(μ)
                        else:
                            aux.append(inImage[k_i, k_j])
                outImage[i][j]=median(aux)
        
        return outImage


    '''
    Filtro de realce High Boost.

    Entrada
        · inImage: Matriz MxN con la imagen de entrada.
        · A: Factor de amplificación del filtro high-boost.
        · method: Método de suavizado. Los valores pueden ser:
            · ’gaussian’, indicando que usará la función gaussianFilter.
            · ’median’, indicando que se usará la función medianFilter.
        · param: Valor del parámetro del filtro de suavizado. Proporcionará el valor de σ en el caso del filtro Gaussiano,
        y el tamaño de ventana en el caso del filtro de medianas.
    Salida:
        · outImage: Matriz MxN con la imagen de salida.
    '''
    def highBoost(self, inImage, A, method, param):
        A += 1
        if (method=="gaussian"):
            t = Thread(target=self.gaussianFilter, args=(inImage, param))
            t.start()
        elif (method=="median"):
            t = Thread(target=self.medianFilter, args=(inImage, param))
            t.start()
        else:
            raise Exception("Método incorrecto")
        convImage = inImage*A
        return self.__normalizeImage(convImage - t.join())
    

    '''
    Operador de erosión.

    Entrada
        · inImage: Matriz MxN con la imagen de entrada.
        · SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
        · center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es la esquina superior izquierda.
        Si es un vector vacío (valor por defecto), el centro se calcula como ([P/2] + 1, [Q/2] + 1).
    Salida:
        · outImage: Matriz MxN con la imagen de salida.
    '''
    def __basicOperator(self, inImage, SE, is_erosion, center):
        outImage = inImage
        α, μ = inImage.shape
        λ, γ = SE.shape

        if (center==[]):
            center = [floor(λ/2)+1, floor(γ/2)+1]

        padded_img = np.pad(array=inImage, pad_width=(λ-1 if (λ>γ) else γ-1), mode='minimum')
        
        for i in range(α+1):
            for j in range(μ+1):
                k = padded_img[i:i+λ,j:j+γ]
                if (np.all(k==SE) if is_erosion else np.any(k==SE)):
                    outImage[i-1,j-1]=1
                else:
                    outImage[i-1,j-1]=0
        return outImage
    def erode(self, inImage, SE, center=[]):
        t = Thread(target=self.__basicOperator, args=(inImage, SE, True, center))
        t.start()
        return t.join()
    
    
    '''
    Operador de dilatación.

    Entrada
        · inImage: Matriz MxN con la imagen de entrada.
        · SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
        · center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es la esquina superior izquierda.
        Si es un vector vacío (valor por defecto), el centro se calcula como ([P/2] + 1, [Q/2] + 1).
    Salida:
        · outImage: Matriz MxN con la imagen de salida.
    '''
    def dilate(self, inImage, SE, center=[]):
        t = Thread(target=self.__basicOperator, args=(inImage, SE, False, center))
        t.start()
        return t.join()


    '''
    Operador de apertura.

    Entrada
        · inImage: Matriz MxN con la imagen de entrada.
        · SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
        · center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es la esquina superior izquierda.
        Si es un vector vacío (valor por defecto), el centro se calcula como ([P/2] + 1, [Q/2] + 1).
    Salida:
        · outImage: Matriz MxN con la imagen de salida.
    '''
    def opening(self, inImage, SE, center=[]):
        return self.dilate(self.erode(inImage, SE, center), SE, center)


    '''
    Operador de cierre.

    Entrada
        · inImage: Matriz MxN con la imagen de entrada.
        · SE: Matriz PxQ de zeros y unos definiendo el elemento estructurante.
        · center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es la esquina superior izquierda.
        Si es un vector vacío (valor por defecto), el centro se calcula como ([P/2] + 1, [Q/2] + 1).
    Salida:
        · outImage: Matriz MxN con la imagen de salida.
    '''
    def closing(self, inImage, SE, center=[]):
        return self.erode(self.dilate(inImage, SE, center), SE, center)
    

    '''
    Operador de cierre.

    Entrada
        · inImage: Matriz MxN con la imagen de entrada.
        · center: Vector 1x2 con las coordenadas del centro de SE. Se asume que el [0 0] es la esquina superior izquierda.
        · objSE: Matriz PxQ de zeros y unos definiendo el elemento estructurante del objeto.
        · bgSE: Matriz PxQ de zeros y unos definiendo el elemento estructurante del fondo.
    Salida:
        · outImage: Matriz MxN con la imagen de salida.
    '''
    def hit_or_miss(self, inImage, objSEj, bgSE, center=[]):

        if (objSEj.shape!=bgSE.shape):
            raise Exception("Error: elementos estructurantes incoherentes")
        
        for x in range(len(objSEj)):
            for y in range(len(objSEj[0])):
                if (objSEj[x][y]==1) and (bgSE[x][y]==1):
                    raise Exception("Error: elementos estructurantes incoherentes")
        
        t1 = Thread(target=self.__invertBinImage, args=([inImage]))
        t1.start()
        t2 = Thread(target=self.erode, args=(inImage,objSEj,center))
        t2.start()
        t3 = Thread(target=self.erode, args=(t1.join(),bgSE,center))
        t3.start()

        objIm = t2.join()
        bgIm = t3.join()

        outImage = np.zeros(shape=(inImage.shape[0],inImage.shape[1]))

        for x in range(len(inImage)):
            for y in range(len(inImage[0])):
                if (objIm[x][y]==1.0) and (bgIm[x][y]==1.0):
                    outImage[x][y] = 1.0
        
        return outImage
    

    '''
    Obtener las componentes Gx y Gy del gradiente de una imagen.

    Entrada
        · inImage: Matriz MxN con la imagen de entrada.
        · operator: Permite seleccionar el operador utilizado mediante los valores: ’Roberts’, 
        ’CentralDiff’, ’Prewitt’ o ’Sobel’.
    Salida:
        · gx, gy: Componentes Gx y Gy del gradiente.
    '''
    def gradientImage(self, inImage, operator):
        if (operator=="Roberts"):
            Gx = np.array([[-1, 0], [0, 1]])
            Gy = np.array([[0, -1], [1, 0]])
        elif (operator=="CentralDiff"):
            Gx = np.array([-1/2, 0, 1/2])
            Gy = np.array([[-1/2], [0], [1/2]])
        elif (operator=="Prewitt"):
            Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            Gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        elif (operator=="Sobel"):
            Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        else:
            raise Exception("Método incorrecto")
        
        t = Thread(target=self.filterImage, args=(inImage, Gx))
        t.start()
        t2 = Thread(target=self.filterImage, args=(inImage, Gy))
        t2.start()
        return [t.join(), t2.join()]

    
    '''
    Obtener las componentes Gx y Gy del gradiente de una imagen.

    Entrada
        · inImage: Matriz MxN con la imagen de entrada.
        · operator: Permite seleccionar el operador utilizado mediante los valores: ’Roberts’, 
        ’CentralDiff’, ’Prewitt’ o ’Sobel’.
    Salida:
        · gx, gy: Componentes Gx y Gy del gradiente.
    '''
    def gradientImage(self, inImage, operator):
        if (operator=="Roberts"):
            Gx = np.array([[-1, 0], [0, 1]])
            Gy = np.array([[0, -1], [1, 0]])
            #TODO que es ese operador
        elif (operator=="CentralDiff"):
            Gx = np.array([-1/2, 0, 1/2])
            Gy = np.array([[-1/2], [0], [1/2]])
        elif (operator=="Prewitt"):
            Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            Gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        elif (operator=="Sobel"):
            Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        else:
            raise Exception("Método incorrecto")
        
        t = Thread(target=self.filterImage, args=(inImage, Gx))
        t.start()
        t2 = Thread(target=self.filterImage, args=(inImage, Gy))
        t2.start()
        return [t.join(), t2.join()]



    def __calcMag(self, Gx, Gy):
        f = lambda x, y: sqrt(pow(x,2)+pow(y,2))
        mag = np.zeros(shape=(Gx.shape))
        for x in range(len(Gx)):
            for y in range(len(Gx[0])):
                mag[x][y] = f(Gx[x][y],Gy[x][y])
        return mag
    def __nonMaxSup(self, Em, Eo):
        M, N = Eo.shape
        Non_max = np.zeros(shape=(M,N))
        Eo = (np.degrees(Eo))
        
        for i in range(1,M-1):
            for j in range(1,N-1):
                # Horizontal 0
                if (0 <= Eo[i,j] < 22.5) or (157.5 <= Eo[i,j] <= 180) or (-22.5 <= Eo[i,j] < 0) or (-180 <= Eo[i,j] < -157.5):
                    b = Em[i, j+1]
                    c = Em[i, j-1]
                    Eo[i,j] = 0
                # Diagonal 45
                elif (22.5 <= Eo[i,j] < 67.5) or (-157.5 <= Eo[i,j] < -112.5):
                    b = Em[i+1, j+1]
                    c = Em[i-1, j-1]
                    Eo[i,j] = 45
                # Vertical 90
                elif (67.5 <= Eo[i,j] < 112.5) or (-112.5 <= Eo[i,j] < -67.5):
                    b = Em[i+1, j]
                    c = Em[i-1, j]
                    Eo[i,j] = 90
                # Diagonal 135
                elif (112.5 <= Eo[i,j] < 157.5) or (-67.5 <= Eo[i,j] < -22.5):
                    b = Em[i+1, j-1]
                    c = Em[i-1, j+1]  
                    Eo[i,j] = 135         
                    
                # Non-max Suppression
                if (Em[i,j] >= b) and (Em[i,j] >= c):
                    Non_max[i,j] = Em[i,j]
                else:
                    Non_max[i,j] = 0
        return np.float32(Non_max), Eo

    def __thresholdImage(self, img, tl, th, Eo):
        thresholdedImg = np.zeros(shape=(img.shape))
        h, w = thresholdedImg.shape
        lst = []
        for i in range(1, h-1):
            for j in range(1, w-1):
                if (img[i][j] > th):
                    thresholdedImg[i][j] = 1.0
        for x,y in lst:
            if (Eo[x][y]==0):
                i = 0
                j = 1
            elif (Eo[x][y]==45):
                i = 1
                j = 1
            elif (Eo[x][y]==90):
                i = 1
                j = 0
            elif (Eo[x][y]==135):
                i = 1
                j = -1
            i2 = -i
            j2 = -j
            while (thresholdedImg[x+i][y+j]>tl) and (thresholdedImg[x+i][y+j]>th):
                thresholdedImg[x+i][y+j] = 1.0
                if (i>0):
                    i+=1
                else:
                    i-=1
                if (j>0):
                    j+=1
                else:
                    j-=1
            i = i2
            j = j2
            while (thresholdedImg[x+i][y+j]>tl) and (thresholdedImg[x+i][y+j]>th):
                thresholdedImg[x+i][y+j] = 1.0
                if (i>0):
                    i+=1
                else:
                    i-=1
                if (j>0):
                    j+=1
                else:
                    j-=1
                
        return thresholdedImg


    '''
    Detector de bordes de Canny.

    Entrada
        · inImage: Matriz MxN con la imagen de entrada.
        · σ: Parámetro σ del filtro Gaussiano.
        · tlow, thigh: Umbrales de histéresis bajo y alto, respectivamente.
    Salida:
        · outImage: Matriz MxN con la imagen de salida.
    '''
    def edgeCanny(self, inImage, σ, tlow, thigh):
        
        if not (tlow<thigh):
            raise Exception("El valor de 'tlow' ha de ser inferior al de 'thigh'")
        
        t1 = Thread(target=self.gaussianFilter, args=(inImage, σ))
        t1.start()
        filteredImage = t1.join()
        self.saveImage("canny/filteredImage.png",filteredImage)
        
        t2 = Thread(target=self.gradientImage, args=(filteredImage, "Sobel"))
        t2.start()
        Gx, Gy = t2.join()

        t3 = Thread(target=self.__calcMag, args=(Gx,Gy))
        t3.start()
        Eo = np.zeros(shape=(inImage.shape))
        for x in range(len(inImage)):
            for y in range(len(inImage[0])):
                if (Gy[x][y]==0.0) or (Gx[x][y]==0.0):
                    continue
                Eo[x][y] = atan(Gy[x][y]/Gx[x][y])
        Em = t3.join()
        
        self.saveImage("canny/magnitud.png",self.__normalizeImage(Em))
        self.saveImage("canny/orientation.png",Eo)

        noMaxSupImg, Eo = self.__nonMaxSup(Em, Eo)
        noMaxSupImg = self.__normalizeImage(noMaxSupImg)
        self.saveImage("canny/nomaxsup.png",noMaxSupImg)
        outImage = self.__thresholdImage(noMaxSupImg, tlow, thigh, Eo)
        return self.__normalizeImage(outImage)

    

    def __createCircularMask(self, radius):
        λ = radius*2
        α = radius
        mask = np.ones(shape=(λ+1, λ+1))

        for i in range(radius):
            for x in range(α):
                mask[i][x]=0.
            y = λ+1
            while (λ-α+1<y):
                mask[i][y-1]=0.
                y -= 1
            α -= 1

        for i in range(radius, λ+1):
            for x in range(α):
                mask[i][x]=0.
            y = λ+1
            while (λ-α+1<y):
                mask[i][y-1]=0.
                y -= 1
            α += 1
        
        return mask

    '''
    Detector de esquinas basado en SUSAN con máscara circular de radio arbitrario

    Entrada
        · inImage: Matriz MxN con la imagen de entrada.
        · r: Radio de la máscara circular.
        · t: Umbral de diferencia de intensidad respecto al núcleo de la máscara.
    Salida:
        · outCorners, usanArea: La función devolverá el cálculo de área USAN para cada punto en usanArea, 
        y el mapa umbralizado con respecto al criterio geométrico en outCorners.
    '''
    def cornerSusan(self, inImage, r, t):
        α, β = inImage.shape
        mask = self.__createCircularMask(r)
        g = 3/4*pow(r*2,2)
        outImage = np.zeros(shape=(inImage.shape))

        for x in range(r, α-r):
            for y in range(r, β-r):
                aux = 0
                for i in range(r*2):
                    for j in range(r*2):
                        if (abs(mask[i][j]-inImage[x][y])<t):
                            aux += 1
                if (aux<g):
                    outImage[x][y]=g-aux
        img = self.__normalizeImage(outImage)
        return self.binaryImage(img,thres_val=0.52), img