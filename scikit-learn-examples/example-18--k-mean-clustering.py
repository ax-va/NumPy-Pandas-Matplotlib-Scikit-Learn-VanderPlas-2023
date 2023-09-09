import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering

plt.style.use('seaborn-v0_8-whitegrid')

# k-means clustering:
# - The cluster center is the arithmetic mean of all the points belonging to the cluster;
# - Each point is closer to its own cluster center than to other cluster centers.

# Generate a two-dimensional dataset containing four distinct blobs
X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.60,
    random_state=0
)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.savefig('../scikit-learn-examples-figures/k-mean-clustering-1--data.svg')
plt.close()

k_means = KMeans(
    n_clusters=4,
    n_init="auto"  # multiple starting guesses
)
k_means.fit(X)
y_k_means = k_means.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_k_means, s=50, cmap='viridis')
centers = k_means.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=500, alpha=0.5)
plt.savefig('../scikit-learn-examples-figures/k-mean-clustering-2--k-mean-clustering.svg')
plt.close()

# Uses the expectation–maximization algorithm:
# 1. Guess some cluster centers;
# 2. Repeat until converged:
#   2a. E-step: Assign points to the nearest cluster center;
#   2b. M-step: Set the cluster centers to the mean of their assigned points.


# very basic implementation
def find_clusters(X, n_clusters, rseed=2):
    """
    Implements a k-mean clustering algorithm.
    """
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    ind = rng.permutation(X.shape[0])[:n_clusters]
    # for the above data:
    # array([ 98, 259, 184, 256])
    centers = X[ind]
    # for the above data:
    # array([[ 0.27239604,  5.46996004],
    #        [-1.36999388,  7.76953035],
    #        [ 0.08151552,  4.56742235],
    #        [-0.6149071 ,  3.94963585]])
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        # for the above data in the first iteration:
        # array([3, 1, 0, 1, 3, 2, 3, 2, 1, 1, 3, 1, 2, 1, 3, 2, 2, 3, 3, 3, 2, 3,
        #        0, 3, 3, 2, 3, 2, 3, 2, 1, 1, 2, 1, 1, 1, 1, 1, 3, 3, 2, 3, 0, 2,
        #        3, 3, 1, 3, 1, 3, 3, 3, 1, 3, 2, 3, 1, 3, 1, 3, 1, 2, 1, 3, 3, 3,
        #        1, 3, 1, 3, 2, 3, 1, 3, 3, 1, 3, 2, 3, 1, 3, 2, 3, 3, 1, 0, 3, 0,
        #        1, 1, 2, 3, 1, 3, 3, 2, 3, 2, 0, 3, 1, 3, 1, 3, 2, 3, 3, 0, 1, 0,
        #        3, 3, 3, 1, 3, 2, 1, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3,
        #        3, 3, 1, 3, 3, 1, 3, 1, 1, 3, 2, 3, 0, 3, 1, 2, 1, 1, 1, 2, 0, 2,
        #        3, 3, 1, 3, 3, 0, 1, 2, 2, 3, 2, 3, 3, 2, 3, 2, 2, 1, 3, 0, 3, 1,
        #        2, 3, 2, 3, 3, 2, 3, 3, 2, 2, 0, 2, 2, 1, 3, 3, 2, 2, 3, 3, 3, 2,
        #        3, 1, 2, 3, 3, 3, 2, 1, 3, 1, 2, 1, 2, 3, 2, 0, 1, 3, 3, 3, 3, 2,
        #        1, 3, 3, 3, 3, 3, 3, 1, 1, 0, 2, 1, 2, 3, 3, 2, 3, 3, 1, 3, 2, 2,
        #        2, 1, 1, 1, 1, 3, 3, 0, 0, 3, 3, 2, 3, 3, 3, 3, 3, 1, 2, 2, 3, 3,
        #        1, 3, 2, 1, 0, 3, 3, 3, 3, 2, 2, 3, 3, 0, 1, 1, 2, 2, 2, 3, 3, 2,
        #        1, 3, 1, 2, 3, 3, 1, 1, 1, 3, 2, 2, 1, 3])
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        # for the above data in the first iteration:
        # array([[ 0.996244  ,  5.28262763],
        #        [-1.39262004,  7.7943098 ],
        #        [ 1.26533912,  3.65658566],
        #        [-0.03528305,  1.93369899]])
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels


centers, labels = find_clusters(X, 4)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.savefig('../scikit-learn-examples-figures/k-mean-clustering-3--manual-implementation-of-k-mean-clustering.svg')
plt.close()

# A few caveats to be aware:
# 1) The globally optimal result may not be achieved
# -> multiple starting guesses, as indeed Scikit-Learn does by default;

centers, labels = find_clusters(X, 4, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.savefig('../scikit-learn-examples-figures/k-mean-clustering-4--poor-results-of-em-algorithm.svg')
plt.close()

# 2) The number of clusters must be selected beforehand
# -> silhouette analysis,
# Gaussian mixture models, or
# DBSCAN, mean-shift, or affinity propagation, all available in sklearn.cluster;

labels = KMeans(n_clusters=6, random_state=0, n_init="auto").fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.savefig('../scikit-learn-examples-figures/k-mean-clustering-5--another-number-of-clusters.svg')
plt.close()

# 3) k-means is limited to linear cluster boundaries
# -> kernel transformation to project the data into a higher dimension
# where a linear separation is possible -> kernelized k-means;

X, y = make_moons(n_samples=200, noise=.05, random_state=0)
labels = KMeans(n_clusters=2, random_state=0, n_init="auto").fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.savefig('../scikit-learn-examples-figures/k-mean-clustering-6--poor-results-for-nonlinear-boundaries.svg')
plt.close()

# kernelized k-means in SpectralClustering
model = SpectralClustering(
    n_clusters=2,
    affinity='nearest_neighbors',
    assign_labels='kmeans'
)
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.savefig('../scikit-learn-examples-figures/k-mean-clustering-7--SpectralClustering.svg')
plt.close()

# 4) k-means can be slow for large numbers of samples
# -> batch-based k-means algorithms -> sklearn.cluster.MiniBatchKMeans.
