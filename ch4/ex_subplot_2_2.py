import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 100)
y_1 = np.sin(x)
y_2 = np.cos(x)
y_3 = np.tan(x)
y_4 = np.exp(x)

fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(x, y_1)
ax[0, 1].plot(x, y_2)
ax[1, 0].plot(x, y_3)
ax[1, 1].scatter(x, y_4)

plt.show()
