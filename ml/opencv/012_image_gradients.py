from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Goal
# Find Image gradients, edges etc
# We will see following functions : cv2.Sobel(), cv2.Scharr(), cv2.Laplacian() etc
# http://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html

img = cv2.imread('data/sudoku.png', 0)
#
# laplacian = cv2.Laplacian(img, ddepth=cv2.CV_64F)
# sobelx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
# sobely = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
#
# plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#
# plt.show()

# Output dtype = cv2.CV_8U
# sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
# # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
# sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# abs_sobel64f = np.absolute(sobelx64f)
# sobel_8u = np.uint8(abs_sobel64f)
# plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
# plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
# plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
# plt.show()


cap = cv2.VideoCapture(0)
# kernel = np.ones((10, 10), np.uint8)
while 1:
  _, img = cap.read()
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  img = cv2.GaussianBlur(img, (5, 5), 10)

  img = cv2.Laplacian(img, ddepth=cv2.CV_64F)
  # imgx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
  # imgy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
  # img = np.sqrt(np.square(imgx) + np.square(imgy))

  cv2.imshow('img', img)
  if cv2.waitKey(1) & 0xff == ord('q'):
    break

