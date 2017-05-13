from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Goal
# We will learn different morphological operations like Erosion, Dilation, Opening, Closing etc.
# We will see different functions like : cv2.erode(), cv2.dilate(), cv2.morphologyEx() etc.

# img = cv2.imread('data/j.png')
# img = cv2.imread('data/messi5.jpg')
kernel = np.ones((5, 5), np.uint8)

# erosion
# img = cv2.erode(img, kernel, iterations=1)

# dilation
# img = cv2.dilate(img, kernel, iterations=1)

# opening
# Opening is just another name of erosion followed by dilation. It is useful in removing noise, as we explained above. Here we use the function,
# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# closing
# Closing is reverse of Opening, Dilation followed by Erosion. It is useful in closing small holes inside the foreground objects, or small black points on the object.
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# morphological gradient
# It is the difference between dilation and erosion of an image. The result will look like the outline of the object.
# img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# top hat
# It is the difference between input image and Opening of the image. Below example is done for a 9x9 kernel.
# img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# black hat
# It is the difference between the closing of the input image and input image.
# img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)


cap = cv2.VideoCapture(0)
# kernel = np.ones((10, 10), np.uint8)
while 1:
  _, img = cap.read()
  # erosion
  # img = cv2.erode(img, kernel, iterations=1)

  # dilation
  # img = cv2.dilate(img, kernel, iterations=1)

  # opening
  # Opening is just another name of erosion followed by dilation. It is useful in removing noise, as we explained above. Here we use the function,
  # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

  # closing
  # Closing is reverse of Opening, Dilation followed by Erosion. It is useful in closing small holes inside the foreground objects, or small black points on the object.
  # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

  # morphological gradient
  # It is the difference between dilation and erosion of an image. The result will look like the outline of the object.
  # img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

  # top hat
  # It is the difference between input image and Opening of the image. Below example is done for a 9x9 kernel.
  # img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

  # black hat
  # It is the difference between the closing of the input image and input image.
  # img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

  cv2.imshow('img', img)
  if cv2.waitKey(1) & 0xff == ord('q'):
    break

cv2.destroyAllWindows()
