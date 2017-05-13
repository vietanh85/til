from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Goal
# In this tutorial, you will learn Simple thresholding, Adaptive thresholding, Otsu's thresholding etc.
# You will learn these functions : cv2.threshold, cv2.adaptiveThreshold etc.
# http://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

def plot(titles, images):
  for i in range(len(titles)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
  plt.show()

# Simple Thresholding
img = cv2.imread('data/gradient.png') # should be a grayscale image
(ret, threshold1) = cv2.threshold(img, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
(ret, threshold2) = cv2.threshold(img, thresh=127, maxval=255, type=cv2.THRESH_BINARY_INV)
(ret, threshold3) = cv2.threshold(img, thresh=127, maxval=255, type=cv2.THRESH_TRUNC)
(ret, threshold4) = cv2.threshold(img, thresh=127, maxval=255, type=cv2.THRESH_TOZERO)
(ret, threshold5) = cv2.threshold(img, thresh=127, maxval=255, type=cv2.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, threshold1, threshold2, threshold3, threshold4, threshold5]
# plot(titles, images)

# Adaptive Thresholding
img = cv2.imread('data/sudoku.png', 0) # should be a grayscale image
# img = cv2.medianBlur(img, 5)
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
th3 = cv2.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

img1 = cv2.medianBlur(img, 5)
ret,th4 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
th5 = cv2.adaptiveThreshold(img1, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
th6 = cv2.adaptiveThreshold(img1, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)

titles = [
          'Original Image',
          'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding',
          'Adaptive Gaussian Thresholding',
          'Original Image (blur)',
          'Global Thresholding (v = 127) (blur)',
          'Adaptive Mean Thresholding (blur)',
          'Adaptive Gaussian Thresholding (blur)'
]
images = [img, th1, th2, th3, img1, th4, th5, th6]
# plot(titles, images)

# Otsu's Binarization
# How Otsu's Binarization Works?
img = cv2.imread('data/pic2.png', 0) # should be a grayscale image

# global threshold
(ret1, threshold1) = cv2.threshold(img, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

# otsu's threshold
(ret2, threshold2) = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# otsu's threshold after blur
blur = cv2.GaussianBlur(img, (5, 5), sigmaX=0)
(ret3, threshold3) = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, threshold1,
          img, 0, threshold2,
          blur, 0, threshold3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in range(3):
  # img
  plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
  plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

  # histogram
  plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
  plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])

  # threshold
  plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
  plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()

plot(titles, images)


# cap = cv2.VideoCapture(0)
# while 1:
#   _, frame = cap.read()
#
#   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#   frame = cv2.medianBlur(frame, 5)
#   # frame = cv2.GaussianBlur(frame, (9, 9), 0)
#   frame = cv2.adaptiveThreshold(frame, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
#   cv2.imshow('th3', frame)
#   if cv2.waitKey(1) & 0xff == ord('q'):
#     break
# cap.release()
# cv2.destroyAllWindows()









# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
