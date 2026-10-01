import matplotlib.pyplot as plt
import numpy as np

X_1 = range(100)
Y_1 = [np.cos(value) for value in X_1]

X_2 = range(200)
Y_2 = [np.sin(value) for value in X_2]

plt.plot(X_1, Y_1)
plt.plot(X_2, Y_2)

plt.show()
