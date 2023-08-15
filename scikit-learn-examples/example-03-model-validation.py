import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV

plt.style.use('seaborn-v0_8-whitegrid')

# # # model validation

# # # wrong way

iris = load_iris()
X = iris.data
y = iris.target

# k-nearest neighbors classifier:
# the label of an unknown point is the same
# as the label of its closest training point

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)
y_model = model.predict(X)
accuracy_score(y, y_model)
# 1.0

# It trains and evaluates the model on the same data.
# This model is an instance-based estimator that simply stores the training data.

# # # right way: holdout sets (testing data)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=0,
    train_size=0.5
)

model.fit(X_train, y_train)
y_model = model.predict(X_test)
accuracy_score(y_test, y_model)
# 0.9066666666666666

# # # cross-validation

# # # two-fold cross-validation

X1 = X_train
X2 = X_test
y1 = y_train
y2 = y_test

y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
accuracy_score(y1, y1_model), accuracy_score(y2, y2_model)
# (0.96, 0.9066666666666666)

# # # five-fold cross-validation

cross_val_score(model, X, y, cv=5)
# array([0.96666667, 0.96666667, 0.93333333, 0.93333333, 1.        ])

# # # leave-one-out cross validation

# We train on all points but one in each trial
scores = cross_val_score(model, X, y, cv=LeaveOneOut())
# array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1.,
#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1.,
#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

# 150 points -> 150 scores

# Estimate the error rate
scores.mean()
# 0.96

"""
The bias-variance trade-off

high-bias model (very low model complexity) -> underfitting
high-variance model (very high model complexity) -> overfitting

R² score = coefficient of determination
R² score = 1 -> perfect match
R² score = 0 -> not better than the mean of the data
R² score negative -> a worse model

Compare the training and validation scores depending on model complexity (given training data)
to find the trade-off between underfitting and overfitting -> the validation curve

Compare the training and validation scores depending on training data (given model complexity)
to stop to increase the number of training data -> the learning curve
"""

# # # validation curves in Scikit-Learn

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),  LinearRegression(**kwargs))


def make_data(N, err=1.0, rseed=1):
    # randomly sample the data
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y


X, y = make_data(40)

# polynomial regression model
X_test = np.linspace(-0.1, 1.1, 500)[:, None]
plt.scatter(X.ravel(), y, color='black')
axis = plt.axis()
for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
    plt.plot(X_test.ravel(), y_test, label=f'degree={degree}')
plt.xlim(-0.1, 1.0)
plt.ylim(-2, 12)
plt.legend(loc='best')
plt.savefig('../scikit-learn-examples-figures/model-validation-1--polynomial-regression.svg')
plt.close()

# the validation curve:
# validation_curve
degree = np.arange(0, 21)
train_score, val_score = validation_curve(
    PolynomialRegression(), X, y,
    param_name='polynomialfeatures__degree',
    param_range=degree, cv=7
)
plt.plot(
    degree, np.median(train_score, 1),
    color='blue', label='training score'
)
plt.plot(
    degree, np.median(val_score, 1),
    color='red', label='validation score'
)
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')
plt.savefig('../scikit-learn-examples-figures/model-validation-2--validation_curve.svg')
plt.close()

# # # the learning curves

X2, y2 = make_data(200)
plt.scatter(X2.ravel(), y2)
plt.savefig('../scikit-learn-examples-figures/model-validation-3--more-training-data.svg')
plt.close()

# validation_curve
degree = np.arange(0, 21)
train_score2, val_score2 = validation_curve(
    PolynomialRegression(), X2, y2,
    param_name='polynomialfeatures__degree',
    param_range=degree, cv=7
)
plt.plot(
    degree, np.median(train_score2, 1),
    color='blue', label='training score'
)
plt.plot(
    degree, np.median(val_score2, 1),
    color='red', label='validation score'
)
plt.plot(
    degree, np.median(train_score, 1),
    color='blue', alpha=0.3, linestyle='dashed'
)
plt.plot(
    degree, np.median(val_score, 1),
    color='red', alpha=0.3, linestyle='dashed'
)
plt.legend(loc='lower center')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')
plt.savefig('../scikit-learn-examples-figures/model-validation-4--more-training-data-does-not-help.svg')
plt.close()

# the learning curve:
# learning_curve
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
# learning curves for a low-complexity model and a high-complexity model
for i, degree in enumerate([2, 9]):
    N, train_lc, val_lc = learning_curve(
        PolynomialRegression(degree), X, y,
        cv=7, train_sizes=np.linspace(0.3, 1, 25)
    )
    ax[i].plot(
        N, np.mean(train_lc, 1),
        color='blue', label='training score'
    )
    ax[i].plot(
        N, np.mean(val_lc, 1),
        color='red', label='validation score'
    )
    ax[i].hlines(
        np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1],
        color='gray', linestyle='dashed')
    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].set_title('degree = {0}'.format(degree), size=14)
    ax[i].legend(loc='best')
plt.savefig('../scikit-learn-examples-figures/model-validation-5--learning_curve.svg')
plt.close()
# The training and validation curves are already close to each other ->
# Adding more training data does not significantly improve the fit

# # # validation in practice: grid search

# two parameters:
# - polynomial degree -> into PolynomialFeatures()
# - whether to fit the intercept -> into LinearRegression()
param_grid = {
    'polynomialfeatures__degree': np.arange(21),
    'linearregression__fit_intercept': [True, False]
}
grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
grid.fit(X, y)
grid.best_params_
# {'linearregression__fit_intercept': False, 'polynomialfeatures__degree': 4}

model = grid.best_estimator_
y_test = model.fit(X, y).predict(X_test)

plt.scatter(X.ravel(), y)
lim = plt.axis()
# (-0.04687651021505175,
#  0.9844070023112612,
#  -0.7308177116555796,
#  10.902869392322714)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)
plt.savefig('../scikit-learn-examples-figures/model-validation-6--grid-search.svg')
plt.close()
