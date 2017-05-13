from __future__ import print_function
import numpy as np
import cv2

# Goal
# Learn to draw different geometric shapes with OpenCV
# You will learn these functions : cv2.line(), cv2.circle() , cv2.rectangle(), cv2.ellipse(), cv2.putText() etc.

# create black img
img = np.zeros((512, 512, 3), np.uint8)

# draw diagonal blue line with thickness = 5px, color in BGR
cv2.line(img, pt1=(0, 0), pt2=(511, 511), color=(255, 0, 0), thickness=5)

# draw a rectangle
cv2.rectangle(img, pt1=(384,0), pt2=(510,128), color=(0,255,0), thickness=3)

# draw a circle, thickness = -1 to fill the closed shape (such as circle)
cv2.circle(img, center=(447,63), radius=63, color=(0, 0, 255), thickness=-1)

# draw an ellipse
# axes: axes lengths (major axis length, minor axis length)
# angle: the angle of rotation of ellipse in anti-clockwise direction
# startAngle and endAngle denotes the starting and ending of ellipse arc measured in clockwise direction from major axis
cv2.ellipse(img, center=(256, 256), axes=(100, 50), angle=0, startAngle=0, endAngle=180, color=(255, 255, 0), thickness=-1)

# draw a polygon
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
# cv2.polylines() can be used to draw multiple lines. Just create a list of all the lines you want to draw and pass it to the function.
# All lines will be drawn individually. It is a much better and faster way to draw a group of lines than calling cv2.line() for each line.
cv2.polylines(img, pts=[pts], isClosed=True, color=(0, 255, 255), thickness=2)

# put text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
cv2.putText(img, text='OpenCV', org=(10, 500), fontFace=font, fontScale=4, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

# show image
cv2.imshow('image', img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()