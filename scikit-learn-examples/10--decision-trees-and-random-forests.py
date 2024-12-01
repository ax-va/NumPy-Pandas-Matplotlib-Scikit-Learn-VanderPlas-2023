import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

plt.style.use('seaborn-v0_8-whitegrid')

# A random forest is an ensemble of randomized decision trees.

# # # decision tree

# Create data for the decision tree classifier
X, y = make_blobs(
    n_samples=300,
    centers=4,
    random_state=0,
    cluster_std=1.0
)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')
plt.savefig('../scikit-learn-examples-figures/decision-trees-and-random-forests-1--data-for-decision-tree-classifier.svg')
plt.close()

# DecisionTreeClassifier classifier
tree = DecisionTreeClassifier().fit(X, y)


def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    # Plot the training points
    ax.scatter(
        X[:, 0], X[:, 1],
        c=y, s=30, cmap=cmap,
        clim=(y.min(), y.max()),
        zorder=3
    )
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
    np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(
        xx, yy, Z,
        alpha=0.3,
        levels=np.arange(n_classes + 1) - 0.5,
        cmap=cmap, zorder=1
    )
    ax.set(xlim=xlim, ylim=ylim)


visualize_classifier(DecisionTreeClassifier(), X, y)
plt.savefig('../scikit-learn-examples-figures/decision-trees-and-random-forests-2--decision-tree-classification.svg')
plt.close()
# The decision tree classifier has resulted in overfitting, for example,
# in the skinny purple region between the yellow and blue regions.

# # # decision trees and overfitting

# Overfitting is a general property of decision trees.

# # # ensembles of estimators: random forests

# Bagging: multiple overfitting estimators can be combined to reduce the overfitting.
# An ensemble of randomized decision trees is known as a random forest.

# bagging classification manually in Scikit-Learn
tree = DecisionTreeClassifier()
# Fit each estimator with a random subset of 80% of the training points
bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8, random_state=1)
bag.fit(X, y)
# Plot decision boundaries for an ensemble of random decision trees
visualize_classifier(bag, X, y)
plt.savefig('../scikit-learn-examples-figures/decision-trees-and-random-forests-3--bagging-classification.svg')
plt.close()

# random forests within the context of classification
# RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)
visualize_classifier(model, X, y)
# Plot decision boundaries for a random forest, which is an optimized ensemble of decision trees
plt.savefig('../scikit-learn-examples-figures/decision-trees-and-random-forests-4--random-forest-classification.svg')
plt.close()

# # # random forest regression
# random forests within the context of regression

# combination of a fast and slow oscillation with noise
rng = np.random.RandomState(42)
x = 10 * rng.rand(200)


def model(x, sigma=0.3):
    fast_oscillation = np.sin(5 * x)
    slow_oscillation = np.sin(0.5 * x)
    noise = sigma * rng.randn(len(x))
    return slow_oscillation + fast_oscillation + noise


y = model(x)
plt.errorbar(x, y, 0.3, fmt='o')
plt.savefig('../scikit-learn-examples-figures/decision-trees-and-random-forests-5--fast-and-slow-oscillation-with-noise.svg')
plt.close()


forest = RandomForestRegressor(200)
forest.fit(x[:, None], y)
x_fit = np.linspace(0, 10, 1000)
y_fit = forest.predict(x_fit[:, None])
y_true_noiseless = model(x_fit, sigma=0)
plt.errorbar(x, y, 0.3, fmt='o', alpha=0.3)
plt.plot(x_fit, y_fit, '-r')
plt.plot(x_fit, y_true_noiseless, '-k', alpha=0.5)
plt.savefig('../scikit-learn-examples-figures/decision-trees-and-random-forests-6--random-forest-regression.svg')
plt.close()

# The multiple trees allow for a probabilistic classification:
# a majority vote among estimators gives an estimate of the probability
# (predict_proba in Scikit-Learn).
