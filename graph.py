import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10,10,10000)
y1 = (-1/8000) * x ** 7 + (1/120) * x ** 5 - (1/ 6) * x ** 3  + x
y2 = 0.001 * x ** 7 + 0.1 * x ** 2 + x + 1
plt.plot(x,y1)
plt.plot(x,y2)
plt.show()