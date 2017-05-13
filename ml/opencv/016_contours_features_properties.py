from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Goal
# To find the different features of contours, like area, perimeter, centroid, bounding box etc
# You will see plenty of functions related to contours.
# http://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html

img = cv2.imread('data/j.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thr = cv2.threshold(gray, 127, 155, 0)
img2, contours, hierarchy = cv2.findContours(thr, 1, 2)

# for cnt in contours:
#   cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
#   cv2.imshow('img', img)
#   cv2.waitKey(0)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cnt = contours[0]

# Moments
# https://en.wikipedia.org/wiki/Image_moment
M = cv2.moments(cnt)
print(M)

# Centroid
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

# Contour Area
area = cv2.contourArea(cnt)
print(area)

# Contour Perimeter
perimeter = cv2.arcLength(cnt, True)
print(perimeter)

# Contour Approximation
# https://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
epsilon = .05 * cv2.arcLength(contours[2], True)
approx = cv2.approxPolyDP(contours[2], epsilon, True)
# cv2.drawContours(img, [approx], 0, (255, 0, 0), 2)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print(approx)

# Convex Hull
hull = cv2.convexHull(contours[2])
# cv2.drawContours(img, [hull], 0, (0, 0, 255), 2)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print(hull)

# Checking Convexity
k = cv2.isContourConvex(cnt)
print(k)

# cnt = contours[2]
# Straight Bounding Rectangle
x,y,w,h = cv2.boundingRect(cnt)
# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Rotated Rectangle
rec = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rec)
box = np.int0(box)
# cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Minimum Enclosing Circle
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
# cv2.circle(img, center, radius, (0, 255, 0), 2)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Fitting an Ellipse
ellipse = cv2.fitEllipse(cnt)
# cv2.ellipse(img, ellipse, (0, 0, 255), 2)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Fitting a Line
rows,cols = img.shape[:2]
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
# cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Here we will learn to extract some frequently used properties of objects like Solidity, Equivalent Diameter, Mask image, Mean Intensity etc. More features can be found at Matlab regionprops documentation.
# http://www.mathworks.com/help/images/ref/regionprops.html
# http://docs.opencv.org/master/d1/d32/tutorial_py_contour_properties.html

# Aspect Ratio
x,y,w,h = cv2.boundingRect(cnt)
aspect_ratio = float(w)/h
print(aspect_ratio)

# Extend = obj area / bounding rect area
area = cv2.contourArea(cnt)
x,y,w,h = cv2.boundingRect(cnt)
bounding_area = w*h
extend = float(area)/bounding_area
print(extend)

# Solidity = obj area / convex hull area
area = cv2.contourArea(cnt)
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = float(area)/bounding_area
print(solidity)

# Equivalent Diameter is the diameter of the circle whose area is same as the contour area.
area = cv2.contourArea(cnt)
equi_diameter = np.sqrt(4*area/np.pi)
print(equi_diameter)

# Orientation is the angle at which object is directed. Following method also gives the Major Axis and Minor Axis lengths.
(x,y),(MA, ma),angle = cv2.fitEllipse(cnt)

# Mask and Pixel Points
mask = np.zeros(gray.shape,np.uint8)
cv2.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
# pixelpoints = cv2.findNonZero(mask)
# Numpy gives coordinates in **(row, column)** format, while OpenCV gives coordinates in **(x,y)** format.
print(pixelpoints)

# Maximum Value, Minimum Value and their locations
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray, mask = mask)
print(min_val, max_val, min_loc, max_loc)

# Mean Color or Mean Intensity
mean_val = cv2.mean(img, mask = mask)
print(mean_val)

# Extreme Points
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
print(leftmost, rightmost, topmost, bottommost)

# Goal
# Convexity defects and how to find them.
# Finding shortest distance from a point to a polygon
# Matching different shapes
