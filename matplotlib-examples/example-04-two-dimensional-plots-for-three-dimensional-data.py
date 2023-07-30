"""
Display three-dimensional data within a two-dimensional plot.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')


def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


x = np.linspace(0, 5, 3)
y = np.linspace(0, 5, 3)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

X
# array([[0. , 2.5, 5. ],
#        [0. , 2.5, 5. ],
#        [0. , 2.5, 5. ]])

Y
# array([[0. , 0. , 0. ],
#        [2.5, 2.5, 2.5],
#        [5. , 5. , 5. ]])

Z
# array([[-0.83907153,  0.6781112 ,  0.41940746],
#        [-0.83907153,  0.69220188,  0.40969682],
#        [-0.83907153,  0.70553683,  0.40107702]])


x = np.linspace(0, 5, 100)
y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z, colors='black')
plt.savefig('../matplotlib-examples-figures/two-dimensional-plots-for-three-dimensional-data--counter-1.svg')
plt.close()
# Negative values are represented by dashed lines and positive values by solid lines.

# more lines and color by the RdGy (Red–Gray) colormap

plt.contour(X, Y, Z, 20, cmap='RdGy')
plt.savefig('../matplotlib-examples-figures/two-dimensional-plots-for-three-dimensional-data--counter-2.svg')
plt.close()

# Discover more colormaps in IPython:
# plt.cm.<TAB>

plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()
plt.savefig('../matplotlib-examples-figures/two-dimensional-plots-for-three-dimensional-data--counterf.svg')
plt.close()
# The black regions are "peaks", while the red regions are "valleys".
# The colors are discrete rather than continuous.

# Plot smooth colors by plt.imshow
plt.imshow(
    Z,
    extent=[0, 5, 0, 5],
    origin='lower',
    cmap='RdGy',
    interpolation='gaussian',
    aspect='equal'
)
plt.colorbar()
plt.savefig('../matplotlib-examples-figures/two-dimensional-plots-for-three-dimensional-data--imshow.svg')
plt.close()

# Plot colors by plt.imshow and overplot with counters by plt.clabel
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.imshow(
    Z,
    extent=[0, 5, 0, 5],
    origin='lower',
    cmap='RdGy',
    alpha=0.5
)
plt.colorbar()
plt.savefig('../matplotlib-examples-figures/two-dimensional-plots-for-three-dimensional-data--imshow--counter--clabel.svg')
plt.close()
