import numpy as np

x = np.array([[1, 2, 5, 8],[1, 2, 5, 8]])
print(x.shape)
print(x.reshape(-1,))

# (2, 4)
# [1 2 5 8 1 2 5 8]
