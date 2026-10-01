import numpy as np

test_array = np.arange(1, 11)
print(test_array.sum())

test_array = np.arange(1, 13).reshape(3, 4)
print(test_array)
print(test_array.sum())
print(test_array.sum(axis=0))
print(test_array.sum(axis=1))
