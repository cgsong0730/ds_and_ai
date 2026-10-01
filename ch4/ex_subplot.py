import matplotlib.pyplot as plt
import numpy as np

X_1 = range(100)
Y_1 = [np.cos(value) for value in X_1]

X_2 = range(200)
Y_2 = [np.sin(value) for value in X_2]

fig = plt.figure()
fig.set_size_inches(10, 10)

ax_1 = fig.add_subplot(1, 2, 1)
ax_2 = fig.add_subplot(1, 2, 2)

ax_1.plot(X_1, Y_1, c="b")
ax_2.plot(X_2, Y_2, c="g")
plt.show()