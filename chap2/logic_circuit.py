import numpy as np 

def AND(x1, x2):
  # 重み・・・入力信号の重要度
  # バイアス・・・発火のしやすさ 
  w1, w2, theta = 0.5, 0.5, 0.7
  b = -theta

  x = np.array([1, x1, x2])
  w = np.array([b, w1, w2])
  
  if np.dot(x, w) <= 0:
    return 0
  else:
    return 1

def NAND(x1, x2):
  if AND(x1, x2) == 1:
    return 0
  else:
    return 1

def OR(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.2

  if np.dot(x, w) + b <= 0:
    return 0
  else:
    return 1

def XOR(x1, x2):
  t1 = OR(x1, x2)
  t2 = NAND(x1, x2)
  y =  AND(t1, t2)
  
  return y 

