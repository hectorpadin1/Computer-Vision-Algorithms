# Visión Artificial.

* Surname: Padín Torrente.
* Name: Héctor.

The project is consists in three files:
* MyThread.py: A own implementation of Thread class of python, in which we are able to retrieve the value that computes the target function.
* p1.py: Class that implements the Computer Vision algorithms.
* test.py: Script to test those implemented algorithms.

Some images were also provided to test the algorithms, and their results.

Algorithms: 
### Contrast enhancement
* **adjustIntensity**: alteration of the dynamic range of an image that allows linear histogram compression (or stretching) by introducing new lower and upper limits.
```python3
img =imgprocesser.openImage("pruebas/grays.png")
h = imgprocesser.adjustIntensity(img, inRange=[], outRange=[0, 255/255])
imgprocesser.saveImage("result.png", h)
imgprocesser.plotHistograms(img, h)
```
* **equalizeIntensity**: algorithm for histogram equalization.
```python3
img = imgprocesser.openImage("pruebas/eq.png")
h = imgprocesser.equalizeIntensity(img, 128)
imgprocesser.saveImage("result.png", h)
imgprocesser.plotHistograms(img, h)
```
### Spatial filtering: smoothing and enhancement
* **filterImage**: function to perform spatial filtering by convolution on an image with an arbitrary kernel passed as a parameter.
```python3
kernel = np.ones(shape=(3,3)
img = imgprocesser.openImage("pruebas/deltas.png")
img2 = imgprocesser.filterImage(img, kernel)
imgprocesser.saveImage("filter.png", img2)
```
* **gaussKernel1D**: function that computes a one-dimensional Gaussian kernel with a given σ.
* **gaussianFilter**: function to perform a two-dimensional Gaussian smoothing using an N × N parameter σ filter.
```python3
img = imgprocesser.openImage("pruebas/kernel.png")
img2 = imgprocesser.gaussianFilter(img, 2)
imgprocesser.saveImage("result.png", img2)
```
* **medianFilter**: two-dimensional median filter, specifying the size of the filter.
```python3
img = imgprocesser.openImage("pruebas/kernel.png")
img2 = imgprocesser.medianFilter(img, 3)
imgprocesser.saveImage("result.png", img2)
```
* **highBoost**: High Boost filter that allows to specify, in addition to the amplification factor A, the smoothing method used and its parameter.
```python3
img = imgprocesser.openImage("pruebas/circles.png")
img2 = imgprocesser.highBoost(img, 2, "gaussian", 2)
imgprocesser.saveImage("result.png", img2)
```
### Morphological operators
* **erode**, **dilate**, **opening** and **closing**: morphological operators of erosion, dilation, opening and closing for binary images and arbitrary structuring element.
```python3
img = imgprocesser.openImage("operadores/bird.jpeg")
bin_img = imgprocesser.binaryImage(img)
EE = np.ones(shape=(3,3))
img2 = imgprocesser.erode(bin_img, EE, center=[0,0])
img2 = imgprocesser.dilate(bin_img, EE, center=[0,0])
imgprocesser.saveImage("result.png", img2)

img = imgprocesser.openImage("operadores/stars.png")
bin_img = imgprocesser.binaryImage(img)
EE = np.ones(shape=(3,3))
img2 = imgprocesser.opening(bin_img, EE)
img2 = imgprocesser.closing(bin_img, EE)
imgprocesser.saveImage("result.png", img2)
```
* **highBoost**: hit-or-miss transformation of an image, given two structuring elements, one for the object and the other one for background.
```python3
img = imgprocesser.openImage("hitOrMiss/figure.png")
bin_img = imgprocesser.binaryImage(img)
# Looking for black pixel surrounded of white pixels
objSEj = np.array([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]])
bgSE = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
img2 = imgprocesser.hit_or_miss(bin_img, objSEj=objSEj, bgSE=bgSE, center=[])
imgprocesser.saveImage("result.png", img2)
```
### Edge detection
* **gradientImage**: function to obtain the Gx and Gy components of the gradient of an image, choosing between Roberts, CentralDiff , Prewitt and Sobel operators.
```python3
img = imgprocesser.openImage("gradient/chica.png")
imgx,imgy = imgprocesser.gradientImage(img, "Sobel")
imgprocesser.saveImage("gradient/Sobelx.jpeg", imgx)
imgprocesser.saveImage("gradient/Sobely.jpeg", imgy)
```
* **edgeCanny**: Canny edge detector.
```python3
img = imgprocesser.openImage("pruebas/circles2.png")
img2 = imgprocesser.edgeCanny(img, 1.2, 0.07, 0.15)
imgprocesser.saveImage("result.png", img2)
```
### Corner detection
* **cornerSusan**: corner detector based on SUSAN with a circular mask of arbitrary radius.
```python3
img = imgprocesser.openImage("susan/figures.png")
points, borders = imgprocesser.cornerSusan(img, 4, 0.05)
imgprocesser.saveImage("susan/points.png", points)
imgprocesser.saveImage("susan/borders.png", borders)
```
