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

def batch(iterable, n=1):
  l = len(iterable)
  for ndx in range(0, l, n):
    yield iterable[ndx:min(ndx + n, l)]

def grad(w):
  N = Xbar.shape[0]
  return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

# Stochastic Gradient Descent
def sgrad(w, i):
  xi = Xbar[i, :]
  yi = y[i]
  return (xi*(xi.dot(w) - yi)).reshape(2, 1)

# Mini Batch Gradient Descent
def mgrad(w, ids):
#  print(ids)
  xi = Xbar[ids, :]
  yi = y[ids]
#  print(xi.T)
#  print(yi)
  print(xi.T.dot(xi.dot(w) - yi))
  return xi.T.dot(xi.dot(w) - yi)
#  return (xi*(xi.dot(w) - yi)).reshape(2, 1)


def cost(w):
  N = Xbar.shape[0]
  return 1/(2*N) * np.linalg.norm(y - Xbar.dot(w), 2)**2


def numeric_grad(w, cost):
  eps = 1e-4
  g = np.zeros_like(w)
  for i in range(len(w)):
    w_p = w.copy()
    w_n = w.copy()
    w_p[i] += eps
    w_n[i] -= eps
    g[i] = (cost(w_p) - cost(w_n))/(2*eps)
  return g

def check_grad(w, cost, grad):
  w = np.random.rand(w.shape[0], w.shape[1])
  grad1 = grad(w)
  grad2 = numeric_grad(w, cost)
  return np.linalg.norm(grad1 - grad2) < 1e-6

print( 'Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))

def myGD(w_init, grad, eta):
  w = [w_init]
  for it in range(1000):
    w_new = w[-1] - eta*grad(w[-1])
    if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:
      break
    w.append(w_new)
  return (w, it)

def GD_momentum(theta_init, grad, eta, gamma):
  theta = [theta_init]  
  v = np.zeros_like(theta_init)
  for it in range(200):
    v_new = gamma * v + eta*grad(theta[-1])
    theta_new = theta[-1] - v_new
    if np.linalg.norm(grad(theta_new)) / len(theta_new) < 1e-3:
      break
    theta.append(theta_new)
    v = v_new
  return (theta, it, v)

# Nesterov accelerated gradient (NAG)
def GD_NAG(theta_init, grad, eta, gamma):
  theta = [theta_init]  
  v = np.zeros_like(theta_init)
  for it in range(200):
    v_new = gamma * v + eta * grad(theta[-1] - gamma * v)
    theta_new = theta[-1] - v_new
    if np.linalg.norm(grad(theta_new)) / len(theta_new) < 1e-3:
      break
    theta.append(theta_new)
    v = v_new
  return (theta, it, v)

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


def MGD(w_init, grad, eta):
  w = [w_init]
  w_last_check = w_init
  it_check_w = 10
  N = X.shape[0]
  count = 0
  for it in range(it_check_w):
    # shuffle
    ids = np.random.permutation(N)
    for b in batch(ids, 5):
      count += 1
      print(b)
      w_new = w[-1] - eta*grad(w[-1], b)
      w.append(w_new)
      if count % it_check_w == 0:
        w_this_check = w_new
        if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:
          print('hoi tu')
          return (w, count)
        w_last_check = w_this_check
  print('ko hoi tu')
  return (w, count)


w_init = np.array([[1], [2]])
#(w1, it1) = myGD(w_init, grad, .1)
#(w2, it2, v2) = GD_momentum(w_init, grad, .1, .9)
#(w3, it3, v3) = GD_NAG(w_init, grad, .1, .9)

#(w4, it4) = MGD(w_init, mgrad, .1) #SGD(w_init, sgrad, .1)
#(w5, it5, v5) = SGD_momentum(w_init, sgrad, .1, .9)
#(w6, it6, v6) = SGD_NAG(w_init, sgrad, .1, .9)

(w7, it7) = MGD(w_init, mgrad, .1)
#(w5, it5, v5) = SGD_momentum(w_init, sgrad, .1, .9)
#(w6, it6, v6) = SGD_NAG(w_init, sgrad, .1, .9)

#print('Solution found by GD: w = ', w1[-1].T, ', after %d iterations.' %(it1+1))
#print('Solution found by GD momentum: w = ', w2[-1].T, ', after %d iterations.' %(it2+1))
#print('Solution found by GD NAG: w = ', w3[-1].T, ', after %d iterations.' %(it3+1))
#print('='*50)
#print('Solution found by SGD: w = ', w4[-1].T, ', after %d steps.' %(it4+1))
#print('Solution found by SGD momentum: w = ', w5[-1].T, ', after %d steps.' %(it5+1))
#print('Solution found by SGD NAG: w = ', w6[-1].T, ', after %d steps.' %(it6+1))
#print('='*50)
print('Solution found by mini batch GD: w = ', w7[-1].T, ', after %d steps.' %(it7+1))
#print('Solution found by mini batch GD momentum: w = ', w5[-1].T, ', after %d steps.' %(it5+1))
#print('Solution found by mini batch GD NAG: w = ', w6[-1].T, ', after %d steps.' %(it6+1))




#Solution found by GD: w =  [[ 4.08763103  3.02534397]] , after 459 iterations.
#Solution found by GD momentum: w =  [[ 4.10620715  3.00363112]] , after 99 iterations.
#Solution found by GD NAG: w =  [[ 4.11639668  2.96991619]] , after 57 iterations.
#Solution found by SGD: w =  [[ 4.07950192  3.06705219]] , after 311 steps.
#Solution found by SGD momentum: w =  [[ 4.07453271  3.01477986]] , after 391 steps.
#Solution found by SGD NAG: w =  [[ 4.11321939  3.00961644]] , after 811 steps.