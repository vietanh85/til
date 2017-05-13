# -*- coding: utf-8 -*-

# gradient descent with linear regression
# L(w) = 1/2N * ||y - Xbar * w||**2
# --> L'(w) = 1/N * Xbar.T * (Xbar * w - y)

import math
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(11)

X = np.random.rand(1000, 1)
# y = 4 + 3x
y = 4 + 3*X + .2*np.random.rand(1000, 1)

# building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)

w_lr = np.dot(np.linalg.pinv(A), b)
print('Solution found by formula: w = ',w_lr.T)

# display result
w = w_lr
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1*x0

# draw the fitting line
plt.plot(X.T, y.T, 'b.', markersize=4) # data
plt.plot(x0, y0, linewidth=2) #fitting line
plt.axis([0, 1, 4, 7])
plt.show()

# Stochastic Gradient Descent
def sgrad(w, i):
  xi = Xbar[i, :]
  yi = y[i]
  return (xi*(xi.dot(w) - yi)).reshape(2, 1)

def cost(w):
  N = Xbar.shape[0]
  return 1/(2*N) * np.linalg.norm(y - Xbar.dot(w), 2)**2


def SGD(w_init, grad, eta):
  w = [w_init]
  w_last_check = w_init
  it_check_w = 10
  N = X.shape[0]
  count = 0
  for it in range(it_check_w):
    # shuffle
    ids = np.random.permutation(N)
    for i in range(N):
      count += 1
      w_new = w[-1] - eta*grad(w[-1], ids[i])
      w.append(w_new)
      if count % it_check_w == 0:
        w_this_check = w_new
        if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:
          return (w, count)
        w_last_check = w_this_check
  return (w, count)

def SGD_momentum(w_init, grad, eta, gamma):
  w = [w_init]
  v = np.zeros_like(w_init)
  w_last_check = w_init
  it_check_w = 10
  N = X.shape[0]
  count = 0
  for it in range(it_check_w):
    # shuffle
    ids = np.random.permutation(N)
    for i in range(N):
      count += 1
      v_new = gamma * v + eta * grad(w[-1],  ids[i])
      w_new = w[-1] - v_new
      w.append(w_new)
      if count % it_check_w == 0:
        w_this_check = w_new
        if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:
          return (w, count, v)
        w_last_check = w_this_check
  return (w, count, v)
  
def SGD_NAG(w_init, grad, eta, gamma):
  w = [w_init]
  v = np.zeros_like(w_init)
  w_last_check = w_init
  it_check_w = 10
  N = X.shape[0]
  count = 0
  for it in range(it_check_w):
    # shuffle
    ids = np.random.permutation(N)
    for i in range(N):
      count += 1
      v_new = gamma * v + eta * grad(w[-1] - gamma * v,  ids[i])
      w_new = w[-1] - v_new
      w.append(w_new)
      if count % it_check_w == 0:
        w_this_check = w_new
        if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:
          return (w, count, v)
        w_last_check = w_this_check
  return (w, count, v)

w_init = np.array([[1], [2]])

(w4, it4) = SGD(w_init, sgrad, .1)
(w5, it5, v5) = SGD_momentum(w_init, sgrad, .1, .9)
(w6, it6, v6) = SGD_NAG(w_init, sgrad, .1, .9)



print('Solution found by SGD: w = ', w4[-1].T, ', after %d steps.' %(it4+1))
print('Solution found by SGD momentum: w = ', w5[-1].T, ', after %d steps.' %(it5+1))
print('Solution found by SGD NAG: w = ', w6[-1].T, ', after %d steps.' %(it6+1))
