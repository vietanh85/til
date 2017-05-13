from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Goals
# Blur the images with various low pass filters
# Apply custom-made filters to images (2D convolution)
# http://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html

# 2D Convolution ( Image Filtering )
# As in one-dimensional signals, images also can be filtered with various low-pass filters(LPF), high-pass filters(HPF) etc. LPF helps in removing noises, blurring the images etc. HPF filters helps in finding edges in the images.

img = cv2.imread('data/opencv-logo.png')

# manually average blur
# kernel = np.ones((10, 10), np.float32)/100
# blur = cv2.filter2D(img, -1, kernel)

# average blur
# blur = cv2.blur(img, ksize=(10, 10)) # ksize is kernel size

# gaussian blur
# blur = cv2.GaussianBlur(img, ksize=(21,21), sigmaX=0)

# gaussian blur with kernel
# kernel = cv2.getGaussianKernel(ksize=(21, 21), sigma=0)
# blur = cv2.filter2D(img, -1, kernel)

# median blur
# blur = cv2.medianBlur(img, 21)

# bilateral blur
blur = cv2.bilateralFilter(img, d=21, sigmaColor=125, sigmaSpace=125)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()