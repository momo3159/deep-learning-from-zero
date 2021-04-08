import numpy as np
import sys, os
sys.path.append(os.pardir)

def softmax(x):
  c = np.max(x)
  exp_x = np.exp(x - c)
  sum_exp_x = np.sum(exp_x)

  return exp_x / sum_exp_x

def cross_entropy_error(y, t):
  delta = 1e-7
  return -np.sum(t * np.log(y - delta))

def numerical_gradient(f, x):
  h = 1e-4
  grad = np.zeros_like(x)

  for idx in range(x.size):
    tmp_val = x[idx]
    x[idx] = tmp_val + h
    fxh1 = f(x)

    x[idx] = tmp_val - h
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2*h)
    x[idx] = tmp_val
  
  return grad

# 1層のネットワーク
class simpleNet:
  def __init__(self):
    self.W = np.random.randn(2, 3)

  def predict(self, x):
    return np.dot(x, self.W)
  
  def loss(self, x, t):
    z = self.predict(x)
    y = softmax(z)
    loss = cross_entropy_error(y, t)

    return loss

net = simpleNet()
print(net.W)
print(net.W.size)

x = np.array([0.6, 0.9])
p = net.predict(x)
np.argmax(p)

t = np.array([0, 0, 1])
net.loss(x, t)

def f(W):
  return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)