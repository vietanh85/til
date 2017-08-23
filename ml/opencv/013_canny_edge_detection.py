from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Goal
# Concept of Canny edge detection
# OpenCV functions for that : cv2.Canny()
# http://docs.opencv.org/master/da/d22/tutorial_py_canny.html

# img = cv2.imread('data/messi5.jpg', 0)
# edges = cv2.Canny(img,100,200)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

cap = cv2.VideoCapture(0)
while 1:
  _, img = cap.read()
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.Canny(img, 100, 150)

  cv2.imshow('img', img)
  if cv2.waitKey(1) & 0xff == ord('q'):
    break
