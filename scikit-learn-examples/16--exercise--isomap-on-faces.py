import matplotlib.pyplot as plt
import numpy as np
from matplotlib import offsetbox
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

plt.style.use('seaborn-v0_8-whitegrid')

# # # Isomap on faces

faces = fetch_lfw_people(min_faces_per_person=30)
faces.data.shape
# (2370, 2914)  #  2,370 images, each with 2,914 pixels

fig, ax = plt.subplots(nrows=4, ncols=8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='gray')
plt.savefig('../scikit-learn-examples-figures/isomap-on-faces-1--data.svg')
plt.close()

# Look at the explained variance ratio
model = PCA(100, svd_solver='randomized').fit(faces.data)
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel('n components')
plt.ylabel('cumulative variance')
plt.savefig('../scikit-learn-examples-figures/isomap-on-faces-2--cumulative-variance-from-pca-projection.svg')
plt.close()

# PCA needs at least 100 linear components.
# Isomap uses only 2 components
model = Isomap(n_components=2)
proj = model.fit_transform(faces.data)
proj.shape
# (2370, 2)


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


fig, ax = plt.subplots(figsize=(10, 10))
plot_components(faces.data,
model=Isomap(n_components=2),
images=faces.images[:, ::2, ::2])
plt.savefig('../scikit-learn-examples-figures/isomap-on-faces-3--isomap-interpretation.png')
plt.close()

# The first two Isomap dimensions seem to describe global image features:
# the overall brightness of the image from left to right, and
# the general orientation of the face from bottom to top.
