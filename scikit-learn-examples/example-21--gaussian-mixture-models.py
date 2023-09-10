import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from sklearn.datasets import make_moons

plt.style.use('seaborn-v0_8-whitegrid')

# # # motivation: weaknesses of k-means

# k-means clusters must be circular.

# Generate some data
X, y_true = make_blobs(
    n_samples=400,
    centers=4,
    cluster_std=0.60,
    random_state=0
)
X = X[:, ::-1]  # Flip axes for better plotting
kmeans = KMeans(4, random_state=0, n_init="auto")
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.savefig('../scikit-learn-examples-figures/gaussian-mixture-models-01--motivation-1--k-means.svg')
plt.close()


def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    """
    Plots labeled data within circular clusters.
    """
    labels = kmeans.fit_predict(X)
    # Plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    # Plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max() for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, ec='black', fc='lightgray', lw=3, alpha=0.5, zorder=1))


# circular clusters implied by the k-means model
kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
plot_kmeans(kmeans, X)
plt.savefig('../scikit-learn-examples-figures/gaussian-mixture-models-02--motivation-2--k-means-circles.svg')
plt.close()

# muddled clusters
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))
# poor performance of k-means for noncircular clusters
kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
plot_kmeans(kmeans, X_stretched)
plt.savefig('../scikit-learn-examples-figures/gaussian-mixture-models-03--motivation-3--poor-k-means-clusters.svg')
plt.close()

# # # generalizing EM: Gaussian mixture models

gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.savefig('../scikit-learn-examples-figures/gaussian-mixture-models-04--gaussian-mixture-model.svg')
plt.close()

# probability that some point belongs to the clusters
probs = gmm.predict_proba(X)
probs[:5].round(3)
# array([[0.   , 0.463, 0.537, 0.   ],
#        [0.   , 0.   , 0.   , 1.   ],
#        [0.   , 0.   , 0.   , 1.   ],
#        [0.   , 0.   , 1.   , 0.   ],
#        [0.   , 0.   , 0.   , 1.   ]])

probs.max(1)[:5].round(3)
# array([0.537, 1.   , 1.   , 1.   , 1.   ])

# uncertainty reflected in the size
size = 50 * probs.max(1) ** 2  # Square emphasizes differences
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size)
plt.savefig('../scikit-learn-examples-figures/gaussian-mixture-models-05--uncertainty-reflected-in-size.svg')
plt.close()

# expectation–maximization algorithm:
# 1. Choose starting guesses for the location and shape;
# 2. Repeat until converged:
#   2a. E-step: For each point, find weights encoding
#       the probability of membership in each cluster;
#   2b. M-step: For each cluster, update its location, normalization,
#       and shape based on *all* data points, making use of the weights.


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Draws an ellipse with a given position and covariance.
    """
    ax = ax or plt.gca()
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    # Draw the ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle=angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40)
    ax.axis('equal')
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


gmm = GaussianMixture(n_components=4, random_state=42)  # default: covariance_type=full
plot_gmm(gmm, X)
plt.savefig('../scikit-learn-examples-figures/gaussian-mixture-models-06--gmm-as-circular-clusters.svg')
plt.close()

gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
plot_gmm(gmm, X_stretched)
plt.savefig('../scikit-learn-examples-figures/gaussian-mixture-models-07--gmm-as-stretched-out-clusters.svg')
plt.close()

# # # covariance types

# covariance_type="diag": ellipse, but not arbitrarily oriented
# covariance_type="spherical": sphere
# covariance_type="full": arbitrarily oriented ellipse

# # # Gaussian mixture models as density estimation

X_moon, y_moon = make_moons(n_samples=200, noise=.05, random_state=0)
plt.scatter(X_moon[:, 0], X_moon[:, 1])
plt.savefig('../scikit-learn-examples-figures/gaussian-mixture-models-08--clusters-with-nonlinear-boundaries.svg')
plt.close()

# two-component GMM fit to nonlinear clusters
gmm2 = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
plot_gmm(gmm2, X_moon)
plt.savefig('../scikit-learn-examples-figures/gaussian-mixture-models-09--two-gmm-clusters-for-nonlinear-boundaries.svg')
plt.close()

# many GMM clusters to model the distribution of points
gmm16 = GaussianMixture(n_components=16, covariance_type='full', random_state=0)
plot_gmm(gmm16, X_moon, label=False)
plt.savefig('../scikit-learn-examples-figures/gaussian-mixture-models-10--many-gmm-clusters-for-nonlinear-boundaries.svg')
plt.close()

# Draw 200 new points from that 16-component GMM fit
X_new, y_new = gmm16.sample(200)
plt.scatter(X_new[:, 0], X_new[:, 1])
plt.savefig('../scikit-learn-examples-figures/gaussian-mixture-models-11--new-data-from-gmm-clusters.svg')
plt.close()

# Use the Akaike information criterion (AIC) and
# the Bayesian information criterion (BIC) to avoid overfitting
n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full',random_state=0).fit(X_moon) for n in n_components]
plt.plot(n_components, [m.bic(X_moon) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X_moon) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.savefig('../scikit-learn-examples-figures/gaussian-mixture-models-12--aic--bic.svg')
plt.close()
# The optimal number of clusters is the value that minimizes the AIC or BIC.
# The choice of number of components measures how well a GMM works
# as *a density estimator*, not how well it works as *a clustering algorithm*.
