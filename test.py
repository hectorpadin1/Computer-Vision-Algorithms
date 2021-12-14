#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import p1
import numpy as np

imgprocesser = p1.ImageProcessing()


#==================== ADJUST INTENSITY ==================================

'''
img =imgprocesser.openImage("pruebas/grays.png")
h = imgprocesser.adjustIntensity(img, inRange=[], outRange=[0, 255/255])
imgprocesser.saveImage("result.png", h)
imgprocesser.plotHistograms(img, h)
'''

#==================== EQUALIZE INTENSITY ==================================

'''
img = imgprocesser.openImage("pruebas/eq.png")
# PROBAR CON 256 y 10.
h = imgprocesser.equalizeIntensity(img, 128)
imgprocesser.saveImage("result.png", h)
imgprocesser.plotHistograms(img, h)
'''

#==================== FILTER IMAGE ==================================

'''
kernel = imgprocesser.openImage("pruebas/kernel.png")
img = imgprocesser.openImage("pruebas/deltas.png")
img2 = imgprocesser.filterImage(img, kernel)
imgprocesser.saveImage("filter.png", img2)
'''

#==================== GAUSIAN KERNEL ==================================

'''
img = imgprocesser.openImage("gausianFilter/grayscale.jpeg")
img2 = imgprocesser.medianFilter(img, 5)
imgprocesser.saveImage("result.png", img2)
'''

#==================== GAUSIAN FILTER ==================================

'''
img = imgprocesser.openImage("pruebas/kernel.png")
img2 = imgprocesser.gaussianFilter(img, 2)
imgprocesser.saveImage("result.png", img2)
'''

#==================== MEDIAN FILTER ==================================

'''
img = imgprocesser.openImage("pruebas/kernel.png")
img2 = imgprocesser.medianFilter(img, 3)
imgprocesser.saveImage("result.png", img2)
'''

#==================== HIGH BOOST ==================================

'''
img = imgprocesser.openImage("pruebas/circles.png")
img2 = imgprocesser.highBoost(img, 2, "gaussian", 2)
imgprocesser.saveImage("result.png", img2)
'''

#==================== ERODE ==================================

'''
img = imgprocesser.openImage("operadores/test6.png")
bin_img = imgprocesser.binaryImage(img)
EE = np.array([1., 1.]).reshape(1,2)
#img = imgprocesser.openImage("operadores/bird.jpeg")
#bin_img = imgprocesser.binaryImage(img)
#EE = np.ones(shape=(3,3))
#img = imgprocesser.openImage("operadores/test6.png")
#EE = np.array([[1., 0.], [0., 1.]])
#bin_img = imgprocesser.binaryImage(img)
img2 = imgprocesser.erode(bin_img, EE, center=[0,0])
imgprocesser.saveImage("result.png", img2)
'''

#==================== DILATE ==================================

'''
#img = imgprocesser.openImage("operadores/testDilate.png")
#bin_img = imgprocesser.binaryImage(img)
#EE = np.array([1., 1.]).reshape(1,2)
img = imgprocesser.openImage("operadores/bird.jpeg")
bin_img = imgprocesser.binaryImage(img)
EE = np.ones(shape=(3,3))
img2 = imgprocesser.dilate(bin_img, EE, center=[0,0])
imgprocesser.saveImage("result.png", img2)
'''

#==================== OPENING ==================================


img = imgprocesser.openImage("operadores/stars.png")
bin_img = imgprocesser.binaryImage(img)
EE = np.ones(shape=(3,3))
img2 = imgprocesser.opening(bin_img, EE)
imgprocesser.saveImage("result.png", img2)


#==================== CLOSING ==================================

'''
img = imgprocesser.openImage("operadores/figures.png")
bin_img = imgprocesser.binaryImage(img)
EE = np.ones(shape=(3,3))
img2 = imgprocesser.dilate(bin_img, EE)
imgprocesser.saveImage("result.png", img2)
'''

#==================== HIT OR MISS ==================================

'''
#img = imgprocesser.openImage("hitOrMiss/figure.png")
#bin_img = imgprocesser.binaryImage(img)
## Buscamos el píxel negro rodeado de píxeles blancos en la figura
#objSEj = np.array([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]])
#bgSE = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
img = imgprocesser.openImage("pruebas/morph.png")
bin_img = imgprocesser.binaryImage(img)
objSEj = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
bgSE = np.array([[1., 0., 1.], [0., 0., 0.], [1., 0., 1.]])
img2 = imgprocesser.hit_or_miss(bin_img, objSEj=objSEj, bgSE=bgSE, center=[])
imgprocesser.saveImage("result.png", img2)
'''

#==================== GRADIENT IMAGE ==================================

'''
img = imgprocesser.openImage("gradient/chica.png")
imgx,imgy = imgprocesser.gradientImage(img, "Sobel")
imgprocesser.saveImage("gradient/Sobelx.jpeg", imgx)
imgprocesser.saveImage("gradient/Sobely.jpeg", imgy)
'''

#========================= CANNY ===============================

'''
img = imgprocesser.openImage("pruebas/circles2.png")
img2 = imgprocesser.edgeCanny(img, 1.2, 0.07, 0.15)
imgprocesser.saveImage("result.png", img2)
'''
'''
img = imgprocesser.openImage("canny/chica.png")
img2 = imgprocesser.edgeCanny(img, 1.2, 0.07, 0.15)
imgprocesser.saveImage("canny/canny.png", img2)
'''

#========================= SUSAN ===============================

'''
img = imgprocesser.openImage("susan/figures.png")
points, borders = imgprocesser.cornerSusan(img, 4, 0.05)
imgprocesser.saveImage("susan/points.png", points)
imgprocesser.saveImage("susan/borders.png", borders)
'''