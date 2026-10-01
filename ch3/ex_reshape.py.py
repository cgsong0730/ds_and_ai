import numpy as np

x = np.array([[1, 2, 5, 8],[1, 2, 5, 8]])
print(x.shape)
print(x.reshape(8,))

y = np.array(range(1, 9))
print(y.shape)
print(y.reshape(2, 4))
