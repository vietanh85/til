from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Goals
# Learn to apply different geometric transformation to images like translation, rotation, affine transformation etc.
# You will see these functions: cv2.getPerspectiveTransform
# http://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html

img = cv2.imread('data/messi5.jpg')
# cv2.imshow('original', img)

# scaling
res1 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# cv2.imshow('res1', res1)
(height, width) = img.shape[:2]
res2 = cv2.resize(img, (2*width, 2*height), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('res2', res2)

# translation
M = np.float32([[1, 0, 100], [0, 1, 50]])
trans = cv2.warpAffine(img, M, (width, height))
# cv2.imshow('trans', trans)

# rotation
M = cv2.getRotationMatrix2D((width/2, height/2), angle=90, scale=1)
rota = cv2.warpAffine(img, M, (width, height))
# cv2.imshow('rota', rota)


# Affine Transformation
# img = cv2.imread('data/drawing.png')
# cv2.imshow('original', img)
(height, width) = img.shape[:2]

# how to get these points ???
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1, pts2)
afftrans = cv2.warpAffine(img, M, (width, height))
# cv2.imshow('afftrans', afftrans)

# Perspective Transformation
img = cv2.imread('data/sudoku.png')
(height, width, ch) = img.shape

# cv2.imshow('original', img)

# how to get these points ???
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1, pts2)
# pertrans = cv2.warpPerspective(img, M, (width, height))
pertrans = cv2.warpPerspective(img, M, (300, 300))
# cv2.imshow('pertrans', pertrans)


# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(pertrans),plt.title('Output')
# plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

