from __future__ import print_function
import numpy as np
import cv2

# Goal
# In this tutorial, you will learn how to convert images from one color-space to another, like BGR <-> Gray, BGR <-> HSV etc.
# In addition to that, we will create an application which extracts a colored object in a video
# You will learn following functions : cv2.cvtColor(), cv2.inRange() etc.

# print all the color-spaces converter flags
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
for f in flags:
  print(f)

# get hsv color range
# green = np.uint8([[[255, 0, 0]]])
# hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
# print(hsv_green)
# [[[ 60 255 255]]]
# Now you take [H-10, 100,100] and [H+10, 255, 255] as lower bound and upper bound respectively


# NOTE: For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]. Different softwares use different scales. So if you are comparing OpenCV values with them, you need to normalize these ranges.

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('Predix_drone_demo.mp4')

while(1):

  # take each frame
  _, frame = cap.read()

  # convert BGR to HSV (https://en.wikipedia.org/wiki/HSL_and_HSV)
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # define range of blue color in HSV
  lower = np.array([110, 50, 50])
  upper = np.array([130, 255, 255])
  # lower = np.array([50, 50, 50])
  # upper = np.array([70, 255, 255])

  # threshold the HSV image to get only blue colors
  mask = cv2.inRange(hsv, lower, upper)

  # bitwise-and mask and original
  res = cv2.bitwise_and(frame, frame, mask=mask)

  cv2.imshow('frame', frame)
  cv2.imshow('hsv', hsv)
  cv2.imshow('mask', mask)
  cv2.imshow('res', res)
  k = cv2.waitKey(5) & 0xFF
  if k == 27:
    break

cv2.destroyAllWindows()

# There are some noises in the image. We will see how to remove them in later chapters.
# This is the simplest method in object tracking. Once you learn functions of contours, you can do plenty of things like find centroid of this object and use it to track the object, draw diagrams just by moving your hand in front of camera and many other funny stuffs.