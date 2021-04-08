import numpy as np 
from util.activate_functions import sigmoid, identity_function

def init_network():
  network = {}
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.1, 0.2, 0.3]])
  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6], [0.1, 0.2]])
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4], [0.1, 0.2]])
  
  return network

def forward(network, X):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']

  # 入力 -> 1層
  Z1 = sigmoid(np.dot(X, W1))
  A1 = np.append(Z1, 1.0)

  # 1 -> 2層
  Z2 = sigmoid(np.dot(A1, W2))
  A2 = np.append(Z2, 1.0)

  # 2 -> 出力
  W3 = np.array([[0.1, 0.3], [0.2, 0.4], [0.1, 0.2]])
  Y = identity_function(np.dot(A2, W3))

  return Y


network = init_network()
X = np.array([1.0, 0.5, 1.0])
Y = forward(network, X)
print(Y)