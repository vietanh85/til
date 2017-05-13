
from __future__ import print_function
import cv2

# Goals
# Here, you will learn how to read an image, how to display it and how to save it back
# You will learn these functions : cv2.imread(), cv2.imshow() , cv2.imwrite()
# Optionally, you will learn how to display images with Matplotlib

# load image in gray scale
img = cv2.imread('jurassic_world.jpg')#, cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', img)
k = cv2.waitKey(0)
if k == 27:
  cv2.destroyAllWindows()
elif k == ord('s'):
  cv2.imwrite('jurassic_world_gray.jpg', img)
  cv2.destroyAllWindows()

from matplotlib import pyplot as plt

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# OpenCV follows BGR order, while matplotlib likely follows RGB order.
(b, g, r) = cv2.split(img)
img2 = cv2.merge([r, g, b])
plt.imshow(img2, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()