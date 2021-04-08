import sys, os
sys.path.append(os.pardir)
from util.gradient import numerical_gradient
from util.activate_functions import *
from util.loss_functions import *

import numpy as np

class TwoLayerNet:
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def predict(self, x):
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']

    z1 = sigmoid(np.dot(x, W1) + b1)
    y = softmax(np.dot(z1, W2) + b2)

    return y

  def loss(self, x, t):
    y = self.predict(x)
    
    return cross_entropy_error(y, t)

  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1) # 各行での最大値となるindexを取得
    t = np.argmax(t, axis=1)

    accuracy = np.sum(y == t) / float(x.shape[0])    
    return accuracy

  def numerical_gradient(self, x, t):
    loss_W = lambda W:self.loss(x, t)
    grads = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    return grads


def do(f, x):
  x[0] += 10000
  print(f(x))

def add(x, y):
   # xは自由変数（ => その時点で一番近い値が使われる）
  a = lambda x: x[0] + y
  do(a, x)

add([1, 2], 100)