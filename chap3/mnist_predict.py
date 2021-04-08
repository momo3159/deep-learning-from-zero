import sys, os
sys.path.append("/usr/local/lib/python3.8/site-packages")
sys.path.append(os.pardir)
print(sys.path)
import pickle
import numpy as np
from util.activate_functions import sigmoid
from dataset.mnist import load_mnist

BATCH_SIZE = 100

def get_data():
  (x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=True, one_hot_label=False)

  return x_test, t_test

def init_network():
  with open("sample_weight.pkl", "rb") as f:
    network = pickle.load(f)

  return network

def predict(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)

  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  
  y = np.dot(z2, W3) + b3
  
  return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0


for i in range(0, len(x), BATCH_SIZE):
  y = predict(network, x[i:i+BATCH_SIZE])

  res = np.argmax(y, axis=1)

  accuracy_cnt += np.sum(res== t[i:i+BATCH_SIZE])
print("Accuracy:" + str(accuracy_cnt / len(x))