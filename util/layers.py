import numpy as np 
import softmax, cross_entropy_error_with_batch

class Relu:
  def __init__(self):
    self.mask = None
    pass
  
  def forward(self, x):
    self.mask = (x <= 0)
    out = x.copy()
    out[self.mask] = 0

    return out
      
    
  def backward(self, dout):
    dout[self.mask] = 0
    dx = dout
    
    return dx

class Sigmoid:
  def __init__(self):
    self.out = None
    pass
  
  def forward(self, x):
    self.out = 1 / (1 + np.exp(-x))
    return out
  
  def backward(self, dout):
    dx = dout * (1.0 - self.out) * self.out
    return dx

def Affine:
  def __init__(self, W, B):
    self.X = None
    self.W = W
    self.B = B
    self.dw = None
    self.db = None

  def forward(self, X):
    self.X = X 
    return np.dot(X, self.W) + self.B 
  
  def backword(self, dout):
    dx = np.dot(dout, self.W.T)
    self.dw = np.dot(self.X.T, dout)
    self.db = np.sum(dout, axis=0)

    return dx

def SoftmaxWithLoss: 
  def __init__(self):
    self.t = None
    self.y = None
    pass
  
  def forward(self, x, t):
    self.t = t
    self.y = softmax(x)
    loss = cross_entropy_error_with_batch(self.y, self.t)

    return loss

  def backward(self, dout=1):
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) / batch_size

    return dx    
