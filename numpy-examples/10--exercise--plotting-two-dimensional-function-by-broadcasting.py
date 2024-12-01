import numpy as np
import matplotlib.pyplot as plt

# Example: we have a function z = f(x, y)
# We want to compute the function across the grid

# x and y have 50 steps from 0 to 5
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
# Broadcast x and y
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

plt.imshow(z, origin='lower', extent=[0, 5, 0, 5])
plt.colorbar()
plt.savefig('../numpy-examples-figures/plotting-two-dimensional-function-by-broadcasting.svg')
plt.close()
