import numpy as np
import matplotlib.pyplot as plt

plt.style.use('classic')

x = np.linspace(-np.pi, np.pi, 100)
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--')
# Use
# plt.show()
# or
# %gui
# to disable a GUI event loop, then
# %matplotlib
# Installed qt6 event loop hook.
# Using matplotlib backend: QtAgg
plt.savefig('../matplotlib-examples-figures/sin-cos-1.svg')
plt.close()

# MATLAB-style interface

# Create a plot figure
plt.figure()
# Create the first of two panels and set current axis
plt.subplot(2, 1, 1)  # (rows, columns, panel number)
plt.plot(x, np.sin(x))
# Create the second panel and set current axis
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
plt.savefig('../matplotlib-examples-figures/sin-cos-2.svg')
plt.close()

# object-oriented interface

# Each Figure can contain one or more Axes instances.
# Each axes instance has attributes xaxis and yaxis.
# ax is an array of two Axes objects.
fig, ax = plt.subplots(2)
# Call plot() method on the appropriate object.
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
plt.savefig('../matplotlib-examples-figures/sin-cos-3.svg')
plt.close()
