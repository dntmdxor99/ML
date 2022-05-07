import numpy as np
import matplotlib.pyplot as plt


def step(x):
    return np.array(x > 0, dtype=np.int64)


x = np.arange(-5.0, 5.0, 0.1)
y = step(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
