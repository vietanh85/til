# -*- coding: utf-8 -*-


# MNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mnist import MNIST
from display_network import *

mndata = MNIST("../mnist")
mndata.load_testing()

X = mndata.test_images
K = 10

kmeans = KMeans(n_clusters=K).fit(X)

pred_label = kmeans.predict(X)

print(type(kmeans.cluster_centers_.T))
print(kmeans.cluster_centers_.T.shape)
#A = display_network(kmeans.cluster_centers_.T, K, 1)


# a colormap and a normalization instance
#cmap = plt.cm.jet
#norm = plt.Normalize(vmin=A.min(), vmax=A.max())

# map the normalized data to colors
# image is now RGBA (512x512x4) 
#image = cmap(norm(A))
#image = cmap(norm(A))
#img = kmeans.cluster_centers_.T.reshape((280, 28, 4))
#plt.imshow(A)

#import scipy.misc
#scipy.misc.imsave('aa.png', image)

for i in range(10):
#  print(kmeans.cluster_centers_.T[:, i].shape)
  img = kmeans.cluster_centers_.T[:, i].reshape(28, 28)
  plt.imshow(img)
  plt.show()





# Object Segmentation

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

img = mpimg.imread('avatar_small.png')
plt.imshow(img)
imgplot = plt.imshow(img)
plt.axis('off')
plt.show() 
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))

# Image Compression 

for K in [2, 3, 4, 5, 10, 15, 20]:
  kmeans = KMeans(n_clusters=K).fit(X)
  label = kmeans.predict(X)

  img4 = np.zeros_like(X)
  # replace each pixel by its center
  for k in range(K):
      img4[label == k] = kmeans.cluster_centers_[k]
  # reshape and display output image
  img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
  plt.imshow(img5, interpolation='nearest')
  plt.axis('off')
  plt.show()



for K in [3]:
    kmeans = KMeans(n_clusters=K).fit(X)
    label = kmeans.predict(X)

    img4 = np.zeros_like(X)
    # replace each pixel by its center
    for k in range(K):
        img4[label == k] = kmeans.cluster_centers_[k]
    # reshape and display output image
    img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
    plt.imshow(img5, interpolation='nearest')
    plt.axis('off')
    plt.show()

