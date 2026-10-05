import numpy as np

# 2차원 넘파이 배열 연산
test_array = np.arange(1, 13).reshape(3, 4)
# print(test_array)
# print(test_array.sum())
# print(test_array.sum(axis=0))
print(test_array.sum(axis=1))


