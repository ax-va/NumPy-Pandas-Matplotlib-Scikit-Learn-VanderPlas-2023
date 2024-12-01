import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding

plt.style.use('seaborn-v0_8-whitegrid')

# Manifold learning:
# Given high-dimensional embedded data, a manifold learning estimator
# seeks a low-dimensional representation of the data that
# preserves certain relationships within the data.

# Methods:
# - multidimensional scaling (MDS) preserves the distance between every pair of points;
# - locally linear embedding (LLE) preserves the distance between neighboring points;
# - isometric mapping (Isomap)

# # # manifold learning: “HELLO”

HELLO_PATH = '../scikit-learn-examples-figures/manifold-learning-01--hello.png'


def save_hello():
    # Make a plot with "HELLO" text; save as PNG
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.text(0.5, 0.4, 'HELLO', va='center', ha='center', weight='bold', size=85)
    fig.savefig(HELLO_PATH)
    plt.close(fig)


def make_2d_points(N=1000, rseed=42):
    # Open this PNG and draw random points from it
    from matplotlib.image import imread
    data = imread(HELLO_PATH)[::-1, :, 0].T
    rng = np.random.RandomState(rseed)
    X = rng.rand(4 * N, 2)
    i, j = (X * data.shape).astype(int).T
    mask = (data[i, j] < 1)
    X = X[mask]
    X[:, 0] *= (data.shape[0] / data.shape[1])
    X = X[:N]
    return X[np.argsort(X[:, 0])]


cmap = matplotlib.colormaps.get_cmap("rainbow")
colors = cmap(np.arange(cmap.N))
cmap = LinearSegmentedColormap.from_list("rainbow", colors, N=5)

save_hello()

X = make_2d_points(1000)
colorize = dict(c=X[:, 0], cmap=cmap)
plt.scatter(X[:, 0], X[:, 1], **colorize)
plt.axis('equal')
plt.savefig('../scikit-learn-examples-figures/manifold-learning-02--data.svg')
plt.close()

# # # multidimensional scaling


def rotate_2d_points(X, angle):
    theta = np.deg2rad(angle)
    R = [[np.cos(theta), np.sin(theta)],
         [-np.sin(theta), np.cos(theta)]]
    return X @ R


# Rotate and translate
X2 = rotate_2d_points(X, 20) + 5
plt.scatter(X2[:, 0], X2[:, 1], **colorize)
plt.axis('equal')
plt.savefig('../scikit-learn-examples-figures/manifold-learning-03--rotated-data.svg')
plt.close()

# What is fundamental, in this case, is the
# distance between each point within the dataset.

D = pairwise_distances(X)
D.shape
# (1000, 1000)

plt.imshow(D, zorder=2, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.savefig('../scikit-learn-examples-figures/manifold-learning-04--pairwise-distance.svg')
plt.close()

D2 = pairwise_distances(X2)
np.allclose(D, D2)
# True

model = MDS(n_components=2, dissimilarity='precomputed', random_state=1701, normalized_stress="auto")
# The MDS algorithm uses only the N x N distance matrix
out = model.fit_transform(D)
plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis('equal')
plt.savefig('../scikit-learn-examples-figures/manifold-learning-05--multidimensional-scaling-1.svg')
plt.close()

# # # MDS as manifold learning


def place_into_higher_space(X, dimension=3, rseed=42):
    assert dimension >= X.shape[1]
    rng = np.random.RandomState(rseed)
    C = rng.randn(dimension, dimension)
    e, V = np.linalg.eigh(C @ C.T)
    return X @ V[:X.shape[1]]


X3 = place_into_higher_space(X, 3)
X3.shape
# (1000, 3)

ax = plt.axes(projection='3d')
ax.scatter3D(X3[:, 0], X3[:, 1], X3[:, 2], **colorize)
plt.savefig('../scikit-learn-examples-figures/manifold-learning-06--mds-2--data-placed-into-3d.svg')
plt.close()

# MDS makes 2D data from the 3D data
model = MDS(n_components=2, random_state=1701, normalized_stress="auto")
out3 = model.fit_transform(X3)
plt.scatter(out3[:, 0], out3[:, 1], **colorize)
plt.axis('equal')
plt.savefig('../scikit-learn-examples-figures/manifold-learning-07--mds-3--2d-data-from-3d-data.svg')
plt.close()

# # # nonaffine / nonlinear embeddings: where MDS fails


def make_hello_s_curve(X):
    """
    Takes the input and contorts it into an “S” shape in three dimensions.
    The fundamental relationships between the data points still remain.
    """
    t = (X[:, 0] - 2) * 0.75 * np.pi
    x = np.sin(t)
    y = X[:, 1]
    z = np.sign(t) * (np.cos(t) - 1)
    return np.vstack((x, y, z)).T


# data embedded nonaffinely into three dimensions
X_s = make_hello_s_curve(X)
ax = plt.axes(projection='3d')
ax.scatter3D(X_s[:, 0], X_s[:, 1], X_s[:, 2], **colorize)
plt.savefig('../scikit-learn-examples-figures/manifold-learning-08--mds-4--3d-hello-s-curve.svg')
plt.close()

# MDS fails to recover the underlying structure
model = MDS(n_components=2, random_state=2, normalized_stress="auto")
outS = model.fit_transform(X_s)
plt.scatter(outS[:, 0], outS[:, 1], **colorize)
plt.axis('equal')
plt.savefig('../scikit-learn-examples-figures/manifold-learning-09--mds-5--mds-fails.svg')
plt.close()

# # # nonlinear manifolds: locally linear embedding

# Solution:
# Modify the algorithm such that it only preserves distances
# between nearby points instead of each pair of points
# ->
# locally linear embedding (LLE)

# modified LLE
model = LocallyLinearEmbedding(
    n_neighbors=100,
    n_components=2,
    method='modified',
    eigen_solver='dense'
)
out = model.fit_transform(X_s)
fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1], **colorize)
ax.set_ylim(0.15, -0.15)
plt.savefig('../scikit-learn-examples-figures/manifold-learning-10--modified-locally-linear-embedding.svg')
plt.close()
# The result remains somewhat distorted compared to the original manifold.

# # # some thoughts on manifold methods

# The only clear advantage of manifold learning methods over
# PCA is their ability to preserve nonlinear relationships in the data.

# For data that is highly clustered, see t-distributed stochastic
# neighbor embedding (t-SNE) implemented in sklearn.manifold.TSNE.

# # # Isomap

# See Tsomap in the next exercise.
