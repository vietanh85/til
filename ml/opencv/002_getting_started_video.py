from __future__ import print_function
import numpy as np
import cv2

# Goal
# Learn to read video, display video and save video.
# Learn to capture from Camera and display it.
# You will learn these functions : cv2.VideoCapture(), cv2.VideoWriter()

# creating the VideoCapture object to read video from camera
cap = cv2.VideoCapture(0)

# read video from file
# cap = cv2.VideoCapture('Predix_drone_demo.mp4')

# FourCC is a 4-byte code used to specify the video codec. The list of available codes can be found in fourcc.org
# In Fedora: DIVX, XVID, MJPG, X264, WMV1, WMV2. (XVID is more preferable. MJPG results in high size video. X264 gives very small size video)
# In Windows: DIVX (More to be tested and added)
# In OSX : *(I don't have access to OSX. Can some one fill this?)*
# define video codec and creating VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# isColor default is True, need video with 3 channels of colors
out = cv2.VideoWriter('output.avi', fourcc, fps=20.0, frameSize=(1280, 720), isColor=False)

while cap.isOpened(): #True:
  # capture frame-by-frame
  # ret value is boolean, if end of video, it will be false
  (ret, frame) = cap.read()
  # print(frame.shape)
  # frame operations
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(frame, (5, 5), 0)
  edged = cv2.Canny(blurred, 10, 40, 10)
  # print(gray.shape)
  # write the video
  out.write(edged)

  # display the result
  cv2.imshow('frame', edged)

  if cv2.waitKey(1) & 0xff == ord('q'):
    break

# get capture object information, prodId from 0 to 18
# for i in range(0, 19):
#   print(cap.get(i))
# set 320x240
# cap.set(3,320)
# cap.set(4,240)

# release the capture
cap.release()
cv2.destroyAllWindows()