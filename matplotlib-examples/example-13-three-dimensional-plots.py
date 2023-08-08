import numpy as np
import matplotlib.pyplot as plt

# # # three-dimensional points and lines

# ax.plot3D and ax.scatter3D

fig = plt.figure()
ax = plt.axes(projection='3d')
# Data for a three-dimensional line
z_line = np.linspace(0, 15, 1000)
x_line = np.sin(z_line)
y_line = np.cos(z_line)
ax.plot3D(x_line, y_line, z_line, 'gray')

# Data for three-dimensional scattered points
z_data = 15 * np.random.random(100)
x_data = np.sin(z_data) + 0.1 * np.random.randn(100)
y_data = np.cos(z_data) + 0.1 * np.random.randn(100)
ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Reds')
plt.savefig('../matplotlib-examples-figures/three-dimensional-plots-1--plot3D--scatter3D.svg')
plt.close()

# # # three-dimensional contour plots

# ax.contour3D

x = np.linspace(-3, 3, 7)
# array([-3., -2., -1.,  0.,  1.,  2.,  3.])
y = np.linspace(-3, 3, 7)
# array([-3., -2., -1.,  0.,  1.,  2.,  3.])
X, Y = np.meshgrid(x, y)
X
# array([[-3., -2., -1.,  0.,  1.,  2.,  3.],
#        [-3., -2., -1.,  0.,  1.,  2.,  3.],
#        [-3., -2., -1.,  0.,  1.,  2.,  3.],
#        [-3., -2., -1.,  0.,  1.,  2.,  3.],
#        [-3., -2., -1.,  0.,  1.,  2.,  3.],
#        [-3., -2., -1.,  0.,  1.,  2.,  3.],
#        [-3., -2., -1.,  0.,  1.,  2.,  3.]])

Y
# array([[-3., -3., -3., -3., -3., -3., -3.],
#        [-2., -2., -2., -2., -2., -2., -2.],
#        [-1., -1., -1., -1., -1., -1., -1.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
#        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
#        [ 2.,  2.,  2.,  2.,  2.,  2.,  2.],
#        [ 3.,  3.,  3.,  3.,  3.,  3.,  3.]])


# Show a three-dimensional contour diagram of a three-dimensional sinusoidal function
def f(x, y):
    """
    three-dimensional sinusoidal function
    """
    return np.sin(np.sqrt(x ** 2 + y ** 2))


x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 40, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig('../matplotlib-examples-figures/three-dimensional-plots-2--contour3D.svg')
plt.close()

# Set the elevation and azimuthal angles by using ax.view_init
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 40, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# 60 degrees above the x-y plane,
# rotated 35 degrees counter-clockwise about the z-axis:
ax.view_init(60, 35)
plt.savefig('../matplotlib-examples-figures/three-dimensional-plots-3--view_init.svg')
plt.close()

# # # wireframes and surface plots

ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z)
ax.set_title('wireframe')
plt.savefig('../matplotlib-examples-figures/three-dimensional-plots-4--plot_wireframe.svg')
plt.close()

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title('surface')
plt.savefig('../matplotlib-examples-figures/three-dimensional-plots-5--plot_surface-1.svg')
plt.close()

# slice into the function that is visualized

r = np.linspace(0, 6, 20)
theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)
# partial polar grid
R, Theta = np.meshgrid(r, theta)
X = R * np.sin(Theta)
Y = R * np.cos(Theta)
Z = f(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
plt.savefig('../matplotlib-examples-figures/three-dimensional-plots-6--plot_surface-2.svg')
plt.close()

# # # surface triangulations

# set of random points
theta = 2 * np.pi * np.random.random(1000)
r = 6 * np.random.random(1000)
x = np.ravel(r * np.sin(theta))
y = np.ravel(r * np.cos(theta))
z = f(x, y)

ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)
plt.savefig('../matplotlib-examples-figures/three-dimensional-plots-7--points-to-triangulate.svg')
plt.close()

ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
plt.savefig('../matplotlib-examples-figures/three-dimensional-plots-8--plot_trisurf.svg')
plt.close()
