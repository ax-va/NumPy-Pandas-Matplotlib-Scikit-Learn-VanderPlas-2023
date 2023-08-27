import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC  # support vector classifier
from sklearn.datasets import make_circles

plt.style.use('seaborn-v0_8-whitegrid')

# Naive Bayes: generative classification = to model each class
# Support vector machine (SVM): discriminative classification = to find a line curve or manifold to divide the classes

X, y = make_blobs(
    n_samples=50,
    centers=2,
    random_state=0,
    cluster_std=0.60
)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.savefig('../scikit-learn-examples-figures/support-vector-machine-01--training-data.svg')
plt.close()

# 1) The idea of SVM:
# There are continuously many straight lines separating two classes
x = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)
for slope, intercept in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    f_x = slope * x + intercept
    plt.plot(x, f_x, '-k')
    plt.xlim(-1, 3.5)
plt.savefig('../scikit-learn-examples-figures/support-vector-machine-02--svm-idea-1.svg')
plt.close()


# 2) The idea of SVM:
# Draw around each line a margin of some width up to the nearest point.
# The margin with the maximum width is optimal.
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
for slope, intercept, width in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    f_x = slope * x + intercept
    plt.plot(x, f_x, '-k')
    plt.fill_between(
        x, f_x - width, f_x + width,
        edgecolor='none',
        color='lightgray',
        alpha=0.5
    )
plt.xlim(-1, 3.5)
plt.savefig('../scikit-learn-examples-figures/support-vector-machine-03--svm-idea-2.svg')
plt.close()

# # # fitting a support vector machine

model = SVC(kernel='linear', C=1E10)
model.fit(X, y)


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """ Plot the decision function for a 2D SVC """
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    x_grid = np.linspace(xlim[0], xlim[1], 30)
    y_grid = np.linspace(ylim[0], ylim[1], 30)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
    xy = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T
    P = model.decision_function(xy).reshape(X_mesh.shape)

    # Plot decision boundary and margins
    ax.contour(
        X_mesh, Y_mesh, P,
        colors='k', levels=[-1, 0, 1],
        alpha=0.5, linestyles=['--', '-', '--']
    )

    # Plot support vectors
    if plot_support:
        ax.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=300, linewidth=1,
            edgecolors='black',
            facecolors='none'
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)
plt.savefig('../scikit-learn-examples-figures/support-vector-machine-04--margin-and-support-vectors.svg')
plt.close()

# support vectors
model.support_vectors_
# array([[0.44359863, 3.11530945],
#        [2.33812285, 3.43116792],
#        [2.06156753, 1.96918596]])


def plot_svm(N=10, ax=None):
    X_, y_ = make_blobs(
        n_samples=300,
        centers=2,
        random_state=0,
        cluster_std=0.60
    )
    X_ = X_[:N]
    y_ = y_[:N]
    model = SVC(kernel='linear', C=1E10)
    model.fit(X_, y_)
    ax = ax or plt.gca()
    ax.scatter(X_[:, 0], X_[:, 1], c=y_, s=50, cmap='autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_function(model, ax)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
    axi.set_title(f'N = {N}')
plt.savefig('../scikit-learn-examples-figures/support-vector-machine-05--positions-of-support-vectors-matter.svg')
plt.close()

# # # beyond linear boundaries: kernel SVM

# Look at some data that is not linearly separable
X, y = make_circles(n_samples=100, factor=.1, noise=.1)
clf = SVC(kernel='linear').fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False)
plt.savefig('../scikit-learn-examples-figures/support-vector-machine-06--not-linearly-separable-data.svg')
plt.close()

# a radial basis function (RBF) centered on the middle clump
r = np.exp(-(X ** 2).sum(1))

X.shape
# (100, 2)
(X ** 2).shape
# (100, 2)
(X ** 2).sum(1).shape
# (100,)
r
# (100,)

ax = plt.subplot(projection='3d')
ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
ax.view_init(elev=20, azim=30)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('r')
plt.savefig('../scikit-learn-examples-figures/support-vector-machine-07--new-dimension-using-rbf.svg')
plt.close()
# A third dimension added to the data using RBF allows for linear separation.

# Compute a basis function centered at *every* point in the dataset -> kernel transformation.

# Change our linear kernel to an RBF kernel, using the kernel model hyperparameter
clf = SVC(kernel='rbf', C=1E6)
clf.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(
    clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
    s=300, lw=1, facecolors='none'
)
plt.savefig('../scikit-learn-examples-figures/support-vector-machine-08--rbf-kernel.svg')
plt.close()

# # # tuning the SVM: softening margins

# data with some level of overlap
X, y = make_blobs(
    n_samples=100,
    centers=2,
    random_state=0,
    cluster_std=1.2
)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.savefig('../scikit-learn-examples-figures/support-vector-machine-09--data-with-overlap.svg')
plt.close()

# Use the tuning parameter, most often known as C.
# For a very large C, the margin is hard.
# For a smaller C, the margin is softer.

X, y = make_blobs(
    n_samples=100, centers=2,
    random_state=0, cluster_std=1.2
)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel='linear', C=C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, axi)
    axi.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=300, lw=1, facecolors='none'
    )
    axi.set_title(f'C = {C:.1f}', size=14)
plt.savefig('../scikit-learn-examples-figures/support-vector-machine-10--hard-and-soft-margins.svg')
plt.close()
