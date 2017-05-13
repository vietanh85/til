from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Goal
#
# We will learn about Image Pyramids
# We will use Image pyramids to create a new fruit, "Orapple"
# We will see these functions: cv2.pyrUp(), cv2.pyrDown()
# http://docs.opencv.org/master/dc/dff/tutorial_py_pyramids.html

# im = cv2.imread('data/messi5.jpg')
# lower = cv2.pyrDown(im)
# higher = cv2.pyrUp(lower)
# lap = cv2.subtract(im, higher)
#
# cv2.imshow('im', im)
# cv2.imshow('lower', lower)
# cv2.imshow('higher', higher)
# cv2.imshow('lap', lap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

A = cv2.imread('data/apple.jpg')
B = cv2.imread('data/orange.jpg')

# generate gaussian pyramid of apple
G = A.copy()
gpA = [G]
for _ in range(6):
  G = cv2.pyrDown(G)
  gpA.append(G)

# generate gaussian pyramid of orange
G = B.copy()
gpB = [G]
for _ in range(6):
  G = cv2.pyrDown(G)
  gpB.append(G)

# A level in Laplacian Pyramid is formed by the difference between that level in Gaussian Pyramid and expanded version of its upper level in Gaussian Pyramid
# generate laplacian pyramid for apple
lpA = [gpA[5]]
for i in range(5, 0, -1):
  GE = cv2.pyrUp(gpA[i])
  L = cv2.subtract(gpA[i-1], GE)
  lpA.append(L)

# generate laplacian pyramid for orange
lpB = [gpB[5]]
for i in range(5, 0, -1):
  GE = cv2.pyrUp(gpB[i])
  L = cv2.subtract(gpB[i-1], GE)
  lpB.append(L)

# add left and right halves of images in each level
LS = []
for la, lb in zip(lpA, lpB):
  rows, cols, dpt = la.shape
  ls = np.hstack((la[:, :cols/2], lb[:, cols/2:]))
  LS.append(ls)

# reconstruct
ls_ = LS[0]
for i in range(1, 6):
  ls_ = cv2.pyrUp(ls_)
  ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
real = np.hstack((A[:,:cols/2],B[:,cols/2:]))
# cv2.imwrite('Pyramid_blending2.jpg',ls_)
# cv2.imwrite('Direct_blending.jpg',real)

cv2.imshow('ls_', ls_)
cv2.imshow('real', real)
cv2.waitKey(0)
cv2.destroyAllWindows()

# more about blending image: http://pages.cs.wisc.edu/~csverma/CS766_09/ImageMosaic/imagemosaic.html





