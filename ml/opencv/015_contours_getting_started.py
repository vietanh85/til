from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Goal
#
# Understand what contours are.
# Learn to find contours, draw contours etc
# You will see these functions : cv2.findContours(), cv2.drawContours()
# http://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html

# img = cv2.imread('data/messi5.jpg', 0)
#
# ret, thr = cv2.threshold(img, 127, 155, 0)
# img2, contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # there are three arguments in cv2.findContours() function:
# #    first one is source image,
# #    second is contour retrieval mode,
# #    third is contour approximation method
#
# cv2.drawContours(img, contours, -1, (0,255,0), 2)
# cv2.imshow('img', img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
while 1:
  _, img = cap.read()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  ret, thr = cv2.threshold(gray, 127, 155, 0)
  img2, contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

  cv2.imshow('img', img)
  if cv2.waitKey(1) & 0xff == ord('q'):
    break

