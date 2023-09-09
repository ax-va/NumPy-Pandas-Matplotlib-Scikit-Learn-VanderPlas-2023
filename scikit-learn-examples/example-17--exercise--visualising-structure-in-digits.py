import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import fetch_openml
from sklearn.manifold import Isomap

# 70,000 images of size of 28 x 28 with handwritten digits
mnist = fetch_openml('mnist_784', parser='auto')
mnist.data.shape
# (70000, 784)
type(mnist.data)
# pandas.core.frame.DataFrame

mnist_data = np.asarray(mnist.data)
mnist_data.shape
# (70000, 784)
type(mnist_data)
# numpy.ndarray
mnist_target = np.asarray(mnist.target, dtype=int)

fig, ax = plt.subplots(nrows=6, ncols=8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(mnist_data[1250 * i].reshape(28, 28), cmap='gray_r')
plt.savefig('../scikit-learn-examples-figures/visualising-structures-in-digits-1--mnist-handwritten-digits.svg')
plt.close()

cmap = matplotlib.colormaps.get_cmap("jet")
colors = cmap(np.arange(cmap.N))
cmap = LinearSegmentedColormap.from_list("jet", colors, N=6)

# Use only 1/30 of the data: full dataset takes a long time!
data = mnist_data[::30]
target = mnist_target[::30]
model = Isomap(n_components=2)
proj = model.fit_transform(data)
plt.scatter(
    proj[:, 0],
    proj[:, 1],
    c=target,
    cmap=cmap
)
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.savefig('../scikit-learn-examples-figures/visualising-structures-in-digits-2--isomap.svg')
plt.close()


def plot_components(data, model, images=None, ax=None, thumb_frac=0.05, cmap='gray'):
    ax = ax or plt.gca()
    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 1], '.k')
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # Don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), proj[i])
            ax.add_artist(imagebox)


# Choose 1/4 of the "1" digits to project
data = mnist_data[mnist_target == 1][::4]
fig, ax = plt.subplots(figsize=(10, 10))
model = Isomap(n_neighbors=5, n_components=2, eigen_solver='dense')
plot_components(
    data,
    model,
    images=data.reshape((-1, 28, 28)),
    ax=ax,
    thumb_frac=0.05,
    cmap='gray_r'
)
plt.savefig('../scikit-learn-examples-figures/visualising-structures-in-digits-3--isomap-with-ones.svg')
plt.close()
