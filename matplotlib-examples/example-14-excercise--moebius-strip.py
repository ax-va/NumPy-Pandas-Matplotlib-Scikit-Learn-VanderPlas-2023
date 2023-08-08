import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# position of the loop about its center
theta = np.linspace(0, 2 * np.pi, 30)
# width of the strip
w = np.linspace(-0.25, 0.25, 8)
W, Theta = np.meshgrid(w, theta)
# twisting of the strip about its axis
Phi = 0.5 * Theta

# radius in x-y plane
R = 1 + W * np.cos(Phi)
X = np.ravel(R * np.cos(Theta))
Y = np.ravel(R * np.sin(Theta))
Z = np.ravel(W * np.sin(Phi))

# Triangulate in the underlying parametrization
tri = Triangulation(np.ravel(W), np.ravel(Theta))

ax = plt.axes(projection='3d')
ax.plot_trisurf(X, Y, Z, triangles=tri.triangles, cmap='Greys', linewidths=0.2)
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
ax.axis('off')
plt.savefig('../matplotlib-examples-figures/moebius-strip.svg')
plt.close()
