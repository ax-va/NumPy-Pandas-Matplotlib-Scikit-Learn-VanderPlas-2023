import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.style.use('seaborn-v0_8-whitegrid')

# PCA is fundamentally a dimensionality reduction algorithm,
# but it can also be useful as a tool for visualization,
# noise filtering, feature extraction and engineering, and much more.

# # # introduction to principal component analysis

rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
X.shape
# (200, 2)
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal')
plt.savefig('../scikit-learn-examples-figures/principal-component-analysis-1--data.svg')
plt.close()

# Goal: learn about the relationship between the x and y values, i.e.
# find pricipal axes in the data and the data using those axes.

pca = PCA(n_components=2)
pca.fit(X)
pca.components_
# array([[-0.94446029, -0.32862557],
#        [-0.32862557,  0.94446029]])

pca.explained_variance_
# array([0.7625315, 0.0184779])

pca.mean_
# array([ 0.03351168, -0.00408072])


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(
        arrowstyle='->',
        linewidth=2,
        shrinkA=0,
        shrinkB=0
    )
    ax.annotate('', v1, v0, arrowprops=arrowprops)


# Plot data
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')
plt.savefig('../scikit-learn-examples-figures/principal-component-analysis-2--data-interpretation.svg')
plt.close()
# The length of each vector it is a measure of the variance of the data when projected onto that principal axis.

# # # PCA as dimensionality reduction

pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:", X.shape)
# original shape: (200, 2)
print("transformed shape:", X_pca.shape)
# transformed shape: (200, 1)

X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.25, color="r")
plt.axis('equal')
plt.savefig('../scikit-learn-examples-figures/principal-component-analysis-3--inverse_transform.svg')
plt.close()

# # # choosing the number of components
# cumulative explained variance ratio
# See the next exercise: PCA for classifying handwritten digits

# # # PCA as noise filtering
# See the next exercise: PCA for classifying handwritten digits

# advantages:
# - flexible
# - fast
# - easily interpretable
#
# disadvantages:
# does not perform so well when there are nonlinear relationships within the data
