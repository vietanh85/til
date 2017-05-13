# -*- coding: utf-8 -*-

import numpy as np

# f(x) = x^2 + 10 sin(x)
def cost(theta):
#  return theta.T.dot(theta) + 10 * np.sin(theta)
  return theta**2 + 10 * np.sin(theta)

# f'(x) = 2x + 10 cos(x)
def grad(theta):
  return 2 * theta + 10 * np.cos(theta)


def GD_momentum(theta_init, eta, gamma):
  theta = [theta_init]  
  v = np.zeros_like(theta_init)
  for it in range(200):
    v_new = gamma * v + eta * grad(theta[-1])
    theta_new = theta[-1] - v_new
    if np.linalg.norm(grad(theta_new)) < 1e-3:
      break
    theta.append(theta_new)
    v = v_new
  return (theta, it, v)

# Nesterov accelerated gradient (NAG)
def NAG(theta_init, eta, gamma):
  theta = [theta_init]  
  v = np.zeros_like(theta_init)
  for it in range(200):
    v_new = gamma * v + eta * grad(theta[-1] - gamma * v)
    theta_new = theta[-1] - v_new
    if np.linalg.norm(grad(theta_new)) < 1e-3:
      break
    theta.append(theta_new)
    v = v_new
  return (theta, it, v)

(x1, it1, v1) = GD_momentum(5, .1, .9)
print(x1[-1], it1, v1)
(x2, it2, v2) = NAG(5, .1, .9)
print(x2[-1], it2, v2)
#(x2, it2, v2) = GD_momentum([[1], [2]], 5, .9)
