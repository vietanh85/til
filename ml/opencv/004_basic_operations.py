from __future__ import print_function
import numpy as np
import cv2
# import matplotlib
# matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

# Goal
# Learn to:
# Access pixel values and modify them
# Access image properties
# Setting Region of Interest (ROI)
# Splitting and Merging images
# Almost all the operations in this section is mainly related to Numpy rather than OpenCV. A good knowledge of Numpy is required to write better optimized code with OpenCV.
# *( Examples will be shown in Python terminal since most of them are just single line codes )*

img = cv2.imread('jurassic_world.jpg')

# access BGR pixel value
bgr = img[100, 100]
print(bgr)

# access blue pixel value
blue = img[100, 100, 0]
print(blue)

# modify the pixel value
img[100, 100] = [255, 255, 255]
print(img[100, 100])
# WARN: Numpy is a optimized library for fast array calculations. So simply accessing each and every pixel values and modifying it will be very slow and it is discouraged.
# NOTE: Above mentioned method is normally used for selecting a region of array, say first 5 rows and last 3 columns like that. For individual pixel access, Numpy array methods, array.item() and array.itemset() is considered to be better. But it always returns a scalar. So if you want to access all B,G,R values, you need to call array.item() separately for all.

# get red value using np.item
print(img.item(10, 10, 2))
# set red value using np.itemset
img.itemset((10, 10, 2), 100)
print(img.item(10, 10, 2))

# access image properties (rows, columns, channels - if image is color)
print(img.shape)
# NOTE: If image is grayscale, tuple returned contains only number of rows and columns. So it is a good method to check if loaded image is grayscale or color image.

# total pixels
print(img.size)

# data type
print(img.dtype)
# NOTE: img.dtype is very important while debugging because a large number of errors in OpenCV-Python code is caused by invalid datatype.

# image ROI
face = img[60:90, 220:250]
# img[30:60, 170:200] = face
# cv2.imshow('face', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# split and merge
(b, g, r) = cv2.split(img)
img = cv2.merge((b, g, r))
b = img[:, :, 0]
# img[:, :, 2] = 0
# WANR: cv2.split() is a costly operation (in terms of time). So do it only if you need it. Otherwise go for Numpy indexing.

# borders
from matplotlib import pyplot as plt

replicated = cv2.copyMakeBorder(img, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_REFLECT101)
reflect_101 = cv2.copyMakeBorder(img, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_WRAP)


plt.subplot(231), plt.imshow(img,'gray'),plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicated,'gray'),plt.title('replicated')
plt.subplot(233), plt.imshow(reflect,'gray'),plt.title('reflect')
plt.subplot(234), plt.imshow(reflect101,'gray'),plt.title('reflect101')
plt.subplot(235), plt.imshow(reflect_101,'gray'),plt.title('reflect_101')
plt.subplot(236), plt.imshow(wrap,'gray'),plt.title('wrap')

plt.show()
