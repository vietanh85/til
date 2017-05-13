
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

iris_X = iris['data']
iris_y = iris['target']

print('Number of classes: %d' % len(np.unique(iris_y)))
print('Number of data points: %d' % len(iris_X))

X0 = iris_X[iris_y == 0, :]
print('Sample of class 0: \n', X0[:5 ,:])

X1 = iris_X[iris_y == 1, :]
print('Number of class 1: \n', X1[:5 ,:])

X2 = iris_X[iris_y == 2, :]
print('Number of class 2: \n', X2[:5 ,:])

(X_train, X_test, y_train, y_test) = train_test_split(
    iris_X, iris_y, test_size=50)

print('Training size: %d' % len(X_train))
print('Test size: %d' % len(X_test))

def myweights(distance):
  signma2 = .5
  return np.exp(-distance**2/signma2)


clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2)
# clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights='distance')
# clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights=myweights)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Print results for 20 test data points:')
print('Predicted results: ', y_pred[30:50])
print('Ground truth     : ', y_test[30:50])
print('Accuracy of 1NN: %.2f %%' % (100*accuracy_score(y_test, y_pred)))

# KNN with MNIST
import numpy as np 
from mnist import MNIST # require `pip install python-mnist`
# https://pypi.python.org/pypi/python-mnist/

import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import time

# you need to download the MNIST dataset first
# at: http://yann.lecun.com/exdb/mnist/
mndata = MNIST('../mnist/') # path to your MNIST folder 
mndata.load_testing()
mndata.load_training()
X_test = mndata.test_images
X_train = mndata.train_images
y_test = np.asarray(mndata.test_labels)
y_train = np.asarray(mndata.train_labels)


start_time = time.time()
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end_time = time.time()
print("Accuracy of 1NN for MNIST: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
print("Running time: %.2f (s)" % (end_time - start_time))