import numpy as np
import cv2

# Goal
#
# In image processing, since you are dealing with large number of operations per second, it is mandatory that your code is not only providing the correct solution, but also in the fastest manner. So in this chapter, you will learn
#
# To measure the performance of your code.
# Some tips to improve the performance of your code.
# You will see these functions : cv2.getTickCount, cv2.getTickFrequency etc.
# Apart from OpenCV, Python also provides a module time which is helpful in measuring the time of execution. Another module profile helps to get detailed report on the code, like how much time each function in the code took, how many times the function was called etc. But, if you are using IPython, all these features are integrated in an user-friendly manner. We will see some important ones, and for more details, check links in Additional Resouces section.

# measuring performance
# cv2.setUseOptimized(False)
print(cv2.useOptimized())
img1 = cv2.imread('data/messi5.jpg')
e1 = cv2.getTickCount()
for i in range(5, 49, 2):
  img1 = cv2.medianBlur(img1, i)
  # cv2.imshow('blur', img1)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
e2 = cv2.getTickCount()
time = (e2 - e1)/cv2.getTickFrequency()
print(time)

# Avoid using loops in Python as far as possible, especially double/triple loops etc. They are inherently slow.
# Vectorize the algorithm/code to the maximum possible extent because Numpy and OpenCV are optimized for vector operations.
# Exploit the cache coherence.
# Never make copies of array unless it is needed. Try to use views instead. Array copying is a costly operation.
