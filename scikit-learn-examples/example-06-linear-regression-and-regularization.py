import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

plt.style.use('seaborn-v0_8-whitegrid')

# # # linear regression by the straight-line fit
# y = a * x + b
# a: slope
# b: intercept

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
plt.scatter(x, y, color="b")

model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
x_fit = np.linspace(0, 10, 1000)
y_fit = model.predict(x_fit[:, np.newaxis])

plt.plot(x_fit, y_fit, color="r")

plt.savefig('../scikit-learn-examples-figures/linear-regression-and-regularization-1--straight-line-fit.svg')
plt.close()

print("Model slope:", model.coef_)
# Model slope: [2.02720881]
# close to 2
print("Model intercept:", model.intercept_)
# Model intercept: -4.998577085553204
# close to -5

# multidimensional linear models: fitting a hyperplane to points in higher dimensions
# y = a_0 + a_1 * x_1 + a_2 + x_2 + ...

rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X, [1.5, -2., 1.])

model.fit(X, y)
print(model.intercept_)
# 0.5000000000000033
print(model.coef_)
# [ 1.5 -2.   1. ]

# # # basis function regression

# # # polynomial basis functions

# y = a_0 + a_1 * x + a_2 * x² + a_3 * x³ + ...
# is equivalent to
# y = a_0 + a_1 * x_1 + a_2 * x_2 + a_3 * x_3 + ...
# with x = x_1, x² = x_2, x³ = x_3, ...

# The linearity is in a_0, a_1, a_2, ...

x = np.array([2, 3, 4])
poly = PolynomialFeatures(degree=3, include_bias=False)
# Convert the one-dimensional array into the three-dimensional array
poly.fit_transform(x[:, None])
# array([[ 2.,  4.,  8.],
#        [ 3.,  9., 27.],
#        [ 4., 16., 64.]])

# 7th-degree polynomial model
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)
poly_model.fit(x[:, np.newaxis], y)
y_fit = poly_model.predict(x_fit[:, np.newaxis])
plt.scatter(x, y, color="b")
plt.plot(x_fit, y_fit, color="r")
plt.savefig('../scikit-learn-examples-figures/linear-regression-and-regularization-2--polynomial-basis-functions.svg')
plt.close()

 # # # Gaussian basis functions

 # The Gaussian basis functions are not built into Scikit-Learn.
 # Build these bsais functions:


class GaussianFeatures(BaseEstimator, TransformerMixin):
    """ Uniformly spaced Gaussian features for one-dimensional input """

    def __init__(self, N, width_factor=2.0):
        self._N = N
        self._width_factor = width_factor
        self._centers = None
        self._width = None

    @property
    def centers_(self):
        return self._centers

    @property
    def width_(self):
        return self._width

    def fit(self, X, y=None):
        # Create N centers spread along the data range
        self._centers = np.linspace(X.min(), X.max(), self._N)
        self._width = self._width_factor * (self._centers[1] - self._centers[0])
        return self

    def transform(self, X):
        return self._gauss_basis(
            X[:, :, np.newaxis],
            self._centers,
            self._width,
            axis=1
        )

    @staticmethod
    def _gauss_basis(x, mean, width, axis=None):
        arg = (x - mean) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))


gauss_model = make_pipeline(GaussianFeatures(20), LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
y_fit = gauss_model.predict(x_fit[:, np.newaxis])
plt.scatter(x, y, color="b")
plt.plot(x_fit, y_fit, color="r")
plt.xlim(0, 10)
plt.savefig('../scikit-learn-examples-figures/linear-regression-and-regularization-3--gaussian-basis-functions.svg')
plt.close()

# # # regularization

# overfitting with a larger number of Gaussian basis functions
model = make_pipeline(GaussianFeatures(30), LinearRegression())
model.fit(x[:, np.newaxis], y)
y_fit = model.predict(x_fit[:, np.newaxis])
plt.scatter(x, y, color="b")
plt.plot(x_fit, y_fit, color="r")
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)
plt.savefig('../scikit-learn-examples-figures/linear-regression-and-regularization-4--overfitting-with-gaussian-basis-functions.svg')
plt.close()


def plot_basis(model, title=None):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    model.fit(x[:, np.newaxis], y)
    ax[0].scatter(x, y, color="b")
    ax[0].plot(x_fit, model.predict(x_fit[:, np.newaxis]), color="r")
    ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
    if title:
        ax[0].set_title(title)
    # Plot the amplitude of the basis function at each location
    ax[1].plot(
        model.steps[0][1].centers_,
        model.steps[1][1].coef_
    )
    ax[1].set(
        xlabel='basis location',
        ylabel='coefficient',
        xlim=(0, 10)
    )


model = make_pipeline(GaussianFeatures(30), LinearRegression())
# ax[1]: plot the coefficients of the Gaussian bases with respect to their locations
plot_basis(model)
plt.savefig('../scikit-learn-examples-figures/linear-regression-and-regularization-5--gaussian-basis-coefficients.svg')
plt.close()

# # # ridge regression (also L_2 regularization, also Tikhonov regularization)

# The penalty is given by
# $P = \alpha * \sum_{n = 1}^N \theta_n^2$,
# where $\alpha$ controls the strength of the penalty.

model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
plot_basis(model, title='Ridge Regression ($L_2$ Regularization)')
plt.savefig('../scikit-learn-examples-figures/linear-regression-and-regularization-6--gaussian-basis-coefficients-with-ridge-regression.svg')
plt.close()

 # # # lasso regression (L_1 regularization)

# The penalty is given by
# $P = \alpha * \sum_{n = 1}^N |\theta_n|$,
# where $\alpha$ controls the strength of the penalty.

# Lasso regression tends to favor sparse models where possible.

model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001, max_iter=2000))
plot_basis(model, title='Lasso Regression ($L_1$ Regularization)')
plt.savefig('../scikit-learn-examples-figures/linear-regression-and-regularization-7--gaussian-basis-coefficients-with-lasso-regression.svg')
plt.close()



