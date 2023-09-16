from typing import Self

import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


class BayesianKDEClassifier(ClassifierMixin, BaseEstimator):
    """
    Bayesian generative classification based on KDE.

    General approach for generative classification:
    1. Split the training data by label;
    2. For each set, fit a KDE to obtain a generative model of the data
    (to compute a likelihood P(x | y) for any observation x and label y);
    3. From the number of examples of each class in the training set,
    compute the *class prior*, P(y);
    4. For an unknown point x, the posterior probability for each class is
                        P(y | x) ∝ P(x | y) * P(y) or
                        log P(y | x) ∝ log P(x | y) + log P(y).
    The class that maximizes this posterior is the label assigned to the point.
    """

    def __init__(self, bandwidth: float = 1.0, kernel: str = 'gaussian') -> None:
        """
        Args:
            bandwidth: the kernel bandwidth within each class
            kernel: the kernel name, passed to KernelDensity
        """
        # special syntax in Scikit-Learn
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.classes = None
        self.features_of_each_class = None
        self.models = None
        self.log_priors = None

    def fit(self, X: np.array, y: np.array) -> Self:
        """
        Args:
            X: features
            y: labels (targets, classes)
        Returns:
             this KDEClassifier instance
        """
        self.classes = np.sort(np.unique(y))
        self.features_of_each_class = [X[y == y_i] for y_i in self.classes]
        # Fit the kernel density for each class
        self.models = [
            KernelDensity(
                bandwidth=self.bandwidth,
                kernel=self.kernel,
            ).fit(X_i) for X_i in self.features_of_each_class
        ]
        self.log_priors = [np.log(X_i.shape[0] / X.shape[0]) for X_i in self.features_of_each_class]
        return self

    def predict_proba(self, X: np.array) -> np.array:
        """
        Args:
            X: features
        Returns:
             np.array with class probabilities of shape [n_samples, n_classes];
             entry (i, j) is the posterior probability that sample i is a member of class j
        """
        log_probs = np.array([model.score_samples(X) for model in self.models]).T
        result = np.exp(log_probs + self.log_priors)
        return result / result.sum(axis=1, keepdims=True)

    def predict(self, X: np.array) -> str:
        """
        Args:
            X: features
        Returns:
            most probable label (target, class)
        """
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]


# Use our custom estimator for the classification of handwritten digits
digits = load_digits()
grid = GridSearchCV(
    BayesianKDEClassifier(),
    {'bandwidth': np.logspace(start=0, stop=2, num=100)}
)
grid.fit(digits.data, digits.target)

# Plot the cross-validation score as a function of bandwidth
fig, ax = plt.subplots()
ax.semilogx(
    np.array(grid.cv_results_['param_bandwidth']),
    grid.cv_results_['mean_test_score']
)
ax.set(title='KDE Model Performance', ylim=(0, 1), xlabel='bandwidth', ylabel='accuracy')
plt.savefig('../scikit-learn-examples-figures/bayesian-kde-for-classifying-handwritten-digits--cross-validation-for-bandwidth.svg')
plt.close()
print(f'best param: {grid.best_params_}')
# best param: {'bandwidth': 6.135907273413174}
print(f'accuracy = {grid.best_score_}')
# accuracy = 0.9677298050139276

# Compare with GaussianNB
gaussian_nb_accuracy = cross_val_score(GaussianNB(), digits.data, digits.target).mean()
print("accuracy=", gaussian_nb_accuracy)
# accuracy= 0.8069281956050759

# possible improvements:
# - allow the bandwidth in each class to vary independently;
# - optimize these bandwidths not based on their prediction score,
# but on the likelihood of the training data under the generative model
# within each class (i.e. use the scores from KernelDensity
# instead of the global prediction accuracy)
