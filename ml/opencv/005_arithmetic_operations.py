from __future__ import print_function
import numpy as np
import cv2

# Goal
# Learn several arithmetic operations on images like addition, subtraction, bitwise operations etc.
# You will learn these functions : cv2.add(), cv2.addWeighted() etc.

# adding
x = np.uint8([250])
y = np.uint8([10])

print(cv2.add(x, y)) # 250+10 = 260 => 255 (uint8)
print(x + y) # 250+10 = 260 % 256 = 4
# NOTE: There is a difference between OpenCV addition and Numpy addition. OpenCV addition is a saturated operation while Numpy addition is a modulo operation.

# image blending
img1 = cv2.imread('data/apple.jpg')
img2 = cv2.imread('data/baboon.jpg')

dst = cv2.addWeighted(img1, .5, img2, .5, gamma=0)


# Load two images
img1 = cv2.imread('data/messi5.jpg')
img2 = cv2.imread('data/opencv-logo-white.png')

# create ROI,
(rows, columns, channels) = img2.shape
roi = img1[0:rows, 0:columns]

# create the mask of logo and its invert mask
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
(ret, mask) = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# black out the logo in roi
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# take region of logo
img2_bg = cv2.bitwise_and(img2, img2, mask=mask)

# add 2
dst = cv2.add(img1_bg, img2_bg)
img1[0:rows, 0:columns] = dst


cv2.imshow('res', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()