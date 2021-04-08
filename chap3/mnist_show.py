import sys
import sys, os
sys.path.append("/usr/local/lib/python3.8/site-packages")
sys.path.append(os.pardir)
print(sys.path)

import numpy as np
from dataset.mnist import load_mnist
from PIL import Image 

def img_show(img):
  print(img.dtype)
  pil_img = Image.fromarray(img)
  pil_img.show()

(x_train, t_train), (x_test, t_test) = \
  load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
img_show(img)